"""evaluation.metrics of detection and control: every metric on hand-made data with known
answers, and the metrics of a whole session."""

import json

import numpy as np
import pytest

from adamus import constants as c
from adamus.errors import AdamusError
from adamus.evaluation import metrics
from adamus.evaluation.session_reader import load_session_folder
from adamus.session_format import RAW_SEPARATOR
from tests.support.handmade_session_folders import (
    CONTEXT,
    ELECTRODES,
    FS,
    NO_F,
    H,
    N,
    session_builder,
    tick,
    ticks_with,
    write_session_folder,
)


@pytest.fixture
def folder(tmp_path, raw_config):
    """Build a session folder of the protocol of the metric tests and load it."""
    return session_builder(tmp_path, raw_config)


class TestSampling:
    def test_sampling_rate_rate_from_first_and_last_timestamp(self):
        assert metrics.sampling_rate(np.arange(513) / 256) == pytest.approx(256.0)
        assert metrics.sampling_rate(np.array([0.0, 1.0, 4.0])) == 0.5

    def test_longest_gap_longest_gap(self):
        assert metrics.longest_gap(np.array([0.0, 0.004, 0.008, 0.7, 0.704])) == pytest.approx(
            0.692
        )

    def test_two_samples_are_enough(self):
        assert metrics.sampling_rate(np.array([10.0, 10.5])) == 2.0
        assert metrics.longest_gap(np.array([10.0, 10.5])) == 0.5

    def test_only_the_first_and_the_last_timestamp_decide_the_rate(self):
        assert metrics.sampling_rate(np.array([5.0, 7.0, 6.0])) == 2.0
        assert metrics.sampling_rate(np.array([5.0, 5.0, 6.0])) == 2.0

    @pytest.mark.parametrize("stamps", [[], [1.0], [2.0, 2.0], [3.0, 2.0]])
    def test_sampling_rate_needs_two_samples_and_time_that_advances(self, stamps):
        assert metrics.sampling_rate(np.array(stamps)) is None

    @pytest.mark.parametrize("stamps", [[], [1.0]])
    def test_longest_gap_needs_two_samples(self, stamps):
        assert metrics.longest_gap(np.array(stamps)) is None

    def test_spread_is_median_95th_percentile_and_maximum(self):
        assert metrics.summary_of(list(range(1, 101))) == {
            "median": 50.5, "p95": pytest.approx(95.05), "max": 100.0,
        }  # fmt: skip
        assert metrics.summary_of([]) is None


class TestValidShare:
    def test_valid_share_counts_valid_among_full_steps_of_each_block(self, folder):
        session = folder(ticks_with(NO_F, valid=lambda k: k % 4 != 0))
        blocks = metrics.valid_share(session)
        assert [block["index"] for block in blocks] == [0, 1, 2, 3, 4]
        assert [block["label"] for block in blocks] == ["REST", "FOCUS", "REST", "FOCUS", "REST"]
        # The first step ends when two windows have arrived: block 0 has fewer full steps.
        assert [block["full"] for block in blocks] == [13, 17, 17, 17, 17]
        # Full steps of block 0 are k = 0..12; k = 0, 4, 8, 12 are invalid.
        assert blocks[0] == {"index": 0, "label": "REST", "full": 13, "valid": 9, "ratio": 9 / 13}

    def test_block_without_steps_has_no_ratio(self, folder):
        blocks = metrics.valid_share(folder(ticks_with(NO_F, steps=5)))
        assert blocks[1] == {"index": 1, "label": "FOCUS", "full": 0, "valid": 0, "ratio": None}


class TestStateSeparation:
    def test_separation_uses_every_rest_and_focus_block(self, folder):
        """There is no calibration: the first blocks count as much as the others."""
        session = folder(ticks_with({0: 0.1, 1: 0.9, 2: 0.2, 3: 0.9, 4: 0.1}))
        auc, difference = metrics.state_separation(session)
        assert auc == 1.0
        assert difference == pytest.approx(0.9 - (13 * 0.1 + 17 * 0.2 + 17 * 0.1) / 47)
        inverted_first = folder(ticks_with({0: 0.9, 1: 0.0, 2: 0.2, 3: 0.9, 4: 0.1}))
        inverted_auc = metrics.state_separation(inverted_first)[0]
        assert inverted_auc is not None and inverted_auc < 1.0

    def test_auc_is_one_half_for_identical_distributions(self, folder):
        assert metrics.state_separation(folder(ticks_with({**NO_F, 2: 0.4, 3: 0.4, 4: 0.4}))) == (
            0.5,
            0.0,
        )

    def test_auc_is_zero_when_focus_is_always_lower(self, folder):
        auc, difference = metrics.state_separation(
            folder(ticks_with({**NO_F, 2: 0.8, 3: 0.1, 4: 0.9}))
        )
        assert auc == 0.0 and difference == pytest.approx(0.1 - 0.85)

    def test_auc_counts_pairs_where_focus_is_higher(self, folder):
        """FOCUS f alternates 0.2 / 0.8, REST is 0.5: about half of the pairs are won."""
        focus = lambda k: 0.2 if k % 2 else 0.8  # noqa: E731
        auc, _ = metrics.state_separation(folder(ticks_with({**NO_F, 2: 0.5, 3: focus, 4: 0.5})))
        assert auc == pytest.approx(9 / 17)  # 17 full steps, k = 56 to 72: 9 even, 8 odd

    def test_invalid_and_not_full_steps_do_not_count(self, folder):
        first_of_block_3 = (3 * 10 * FS - CONTEXT) // H + 1
        not_full = N // H - 1  # steps whose window still reaches into block 2
        ticks = ticks_with(
            {**NO_F, 2: 0.2, 3: lambda k: 0.0 if k < first_of_block_3 + not_full else 0.9, 4: 0.2},
            valid=lambda k: not first_of_block_3 + 5 <= k < first_of_block_3 + 8,
        )
        for row in ticks:
            if not row["valid"]:
                row["f"] = 0.0  # a held value
        assert metrics.state_separation(folder(ticks))[0] == 1.0

    def test_no_data(self, folder):
        assert metrics.state_separation(folder(ticks_with(NO_F))) == (None, None)
        assert metrics.state_separation(folder(ticks_with({**NO_F, 3: 0.9}))) == (None, None)


class TestResponseTimes:
    def test_response_time_is_counted_from_the_start_of_the_block(self, folder):
        first = {block: (block * 10 * FS - CONTEXT) // H + 1 for block in range(5)}

        def on_after(block, steps):
            return lambda k: 1.0 if k - first[block] >= steps else 0.0

        def off_after(block, steps):
            return lambda k: 0.0 if k - first[block] >= steps else 1.0

        session = folder(
            ticks_with({**NO_F, 2: off_after(2, 3), 3: on_after(3, 6), 4: off_after(4, 0)})
        )
        assert metrics.response_times(session) == {
            # blocks 0 and 1 have no f yet; the 7th step of block 3 ends 3.5 s after its start
            "rise_s": [None, 3.5],
            "fall_s": [None, 2.0, 0.5],
            "rise_median_s": 3.5,
            "fall_median_s": 1.25,
        }

    def test_f_of_exactly_one_half_counts_as_on(self, folder):
        session = folder(ticks_with({**NO_F, 2: 0.5, 3: 0.5, 4: 0.49}))
        times = metrics.response_times(session)
        assert times["rise_s"] == [None, 0.5] and times["fall_s"] == [None, None, 0.5]

    def test_block_that_never_crosses_the_threshold(self, folder):
        assert metrics.response_times(folder(ticks_with({**NO_F, 2: 0.9, 3: 0.1, 4: 0.9}))) == {
            "rise_s": [None, None], "fall_s": [None, None, None],
            "rise_median_s": None, "fall_median_s": None,
        }  # fmt: skip

    def test_invalid_steps_are_skipped(self, folder):
        first = (3 * 10 * FS - CONTEXT) // H + 1
        ticks = ticks_with(
            {**NO_F, 2: 0.1, 3: 0.9, 4: 0.1}, valid=lambda k: not first <= k < first + 4
        )
        assert metrics.response_times(folder(ticks))["rise_s"] == [None, 2.5]


class TestLoop:
    def test_control_latency_is_time_from_the_last_eeg_sample_to_the_sent_message(self, folder):
        session = folder(ticks_with({**NO_F, 2: 0.2, 3: 0.9, 4: 0.2}))
        latency = metrics.summary_of(session.control_latencies())
        assert latency == {
            "median": pytest.approx(0.030),
            "p95": pytest.approx(0.030),
            "max": pytest.approx(0.030),
        }

    def test_step_interval_is_the_gap_between_neighbouring_sent_steps(self, folder):
        every_step = metrics.step_regularity(folder(ticks_with({**NO_F, 2: 0.2, 3: 0.9, 4: 0.2})))
        assert every_step == {"median": 0.5, "p95": 0.5, "max": 0.5}
        every_other = ticks_with({**NO_F, 2: 0.2, 3: 0.9, 4: 0.2}, valid=lambda k: k % 2 == 0)
        regularity = metrics.step_regularity(folder(every_other))
        assert regularity is not None and regularity["median"] == pytest.approx(1.0)

    def test_two_sent_steps_are_enough_for_an_interval(self, folder):
        ticks = [tick(k, block=2, f=0.2) for k in range(2)]
        assert metrics.step_regularity(folder(ticks)) == {"median": 0.5, "p95": 0.5, "max": 0.5}
        assert metrics.step_regularity(folder(ticks[:1])) is None

    def test_nothing_sent(self, folder):
        session = folder(ticks_with(NO_F))
        assert metrics.step_regularity(session) is None
        assert metrics.summary_of(session.control_latencies()) is None


class TestCompleteness:
    def test_completeness_complete_record(self, folder):
        session = folder([tick(k) for k in range(10)])
        assert metrics.completeness(session, CONTEXT + 9 * H) == {
            "ticks": 10, "filtered_records": 10, "expected": 10, "passed": True,
        }  # fmt: skip

    def test_samples_that_do_not_fill_a_hop_add_no_step(self, folder):
        assert metrics.completeness(folder([tick(k) for k in range(10)]), CONTEXT + 10 * H - 1)[
            "passed"
        ]

    def test_missing_steps_are_detected(self, folder):
        session = folder([tick(k) for k in range(10)])
        assert metrics.completeness(session, CONTEXT + 12 * H) == {
            "ticks": 10, "filtered_records": 10, "expected": 13, "passed": False,
        }  # fmt: skip

    def test_missing_filtered_record_is_detected(self, folder):
        session = folder([tick(k) for k in range(10)])
        path = session.directory / c.FILE_FILTERED
        path.write_bytes(path.read_bytes()[: 9 * 4 * ELECTRODES * N])
        result = metrics.completeness(session, CONTEXT + 9 * H)
        assert (result["filtered_records"], result["passed"]) == (9, False)

    def test_truncated_filtered_record_is_detected(self, folder):
        session = folder([tick(k) for k in range(10)])
        path = session.directory / c.FILE_FILTERED
        path.write_bytes(path.read_bytes() + b"\0")
        assert metrics.completeness(session, CONTEXT + 9 * H)["passed"] is False

    @pytest.mark.parametrize("samples", [0, 500, CONTEXT - 1])
    def test_session_shorter_than_the_signal_of_one_step_has_no_steps(self, folder, samples):
        assert metrics.completeness(folder([]), samples) == {
            "ticks": 0, "filtered_records": 0, "expected": 0, "passed": True,
        }  # fmt: skip


class TestComputeMetrics:
    def test_metrics_file_has_every_metric_and_is_plain_json(self, folder):
        session = folder(ticks_with({**NO_F, 2: 0.2, 3: 0.9, 4: 0.2}), [])
        path = metrics.analyze(session.directory)
        assert path == session.directory / "metrics.json"
        stored = json.loads(path.read_text())
        assert path.read_text() == json.dumps(stored, indent=2) + "\n"  # readable by people too
        hidden = [file.name for file in session.directory.iterdir() if file.name.startswith(".")]
        assert hidden == []  # no file it was written through is left behind
        assert list(stored) == [
            "session", "sampling_rate_hz", "longest_sample_gap_s", "valid_steps_per_block",
            "processing_time_s", "state_separation_auc", "mean_f_difference", "response_time_s",
            "control_latency_s", "step_interval_s", "audio_dropouts",
            "recording_length_error_s", "music_response", "total_response_s",
            "record_completeness",
        ]  # fmt: skip
        assert stored["session"] == "20261110-150412_X"
        assert stored["sampling_rate_hz"] == pytest.approx(256.0, rel=1e-4)
        assert stored["longest_sample_gap_s"] == pytest.approx(1 / 256, rel=1e-3)
        assert stored["processing_time_s"] == {
            "median": pytest.approx(0.001),
            "p95": pytest.approx(0.001),
            "max": pytest.approx(0.001),
        }
        assert stored["state_separation_auc"] == 1.0
        assert stored["record_completeness"]["passed"] is True

    def test_folder_that_is_not_a_session(self, tmp_path):
        with pytest.raises(AdamusError) as error:
            metrics.compute_metrics(tmp_path)
        assert str(error.value).startswith(
            f"{tmp_path} is not a complete session folder: cannot read {tmp_path / 'config.yaml'}: "
        )
        with pytest.raises(AdamusError, match="is not a complete session folder"):
            metrics.compute_metrics(tmp_path / "missing")

    @pytest.mark.parametrize("raw", ["empty", "missing"])  # BrainFlow made it, or not even that
    def test_session_in_which_no_sample_arrived(self, tmp_path, raw_config, raw):
        """It ended before its first sample, e.g. a headset that sends nothing: it has
        metrics all the same, which say that no sample and no step was recorded."""
        directory = write_session_folder(
            tmp_path / "20261110-150412_X", raw_config, [], [], samples=0
        )
        if raw == "missing":
            (directory / c.FILE_RAW).unlink()
        stored = metrics.compute_metrics(directory)
        assert stored["record_completeness"] == {
            "ticks": 0, "filtered_records": 0, "expected": 0, "passed": True,
        }  # fmt: skip
        assert (stored["sampling_rate_hz"], stored["longest_sample_gap_s"]) == (None, None)

    def test_raw_file_that_cannot_be_read_is_reported(self, folder):
        session = folder([tick(0)])
        (session.directory / c.FILE_RAW).write_text("not samples\n")
        with pytest.raises(AdamusError) as error:
            metrics.compute_metrics(session.directory)
        incomplete = f"{session.directory} is not a complete session folder"
        assert str(error.value) == f"{incomplete}: {error.value.__cause__}"

    def test_a_last_row_cut_short_as_the_program_ended_is_left_out(self, folder):
        session = folder([tick(k) for k in range(3)])
        whole = load_session_folder(session.directory).raw_samples()
        with (session.directory / c.FILE_RAW).open("a") as raw:
            raw.write(f"1.0{RAW_SEPARATOR}2.")
        cut = load_session_folder(session.directory).raw_samples()
        assert np.array_equal(cut, whole, equal_nan=True)

    def test_samples_that_are_not_numbers_do_not_cost_the_metrics(self, folder):
        """A recording may hold nan or inf; played back, they go into raw.csv as they are."""
        session = folder([tick(k) for k in range(3)])
        raw = session.directory / c.FILE_RAW
        table = np.loadtxt(raw, delimiter="\t", ndmin=2)
        table[5, 1], table[6, 2] = np.nan, np.inf
        np.savetxt(raw, table, delimiter="\t", fmt="%.6f")
        result = metrics.compute_metrics(session.directory)
        assert result["record_completeness"]["passed"] is True
        assert result["sampling_rate_hz"] == pytest.approx(256)

    def test_session_without_its_filtered_file(self, folder):
        session = folder([tick(0)])
        (session.directory / c.FILE_FILTERED).unlink()
        with pytest.raises(AdamusError) as error:
            metrics.compute_metrics(session.directory)
        incomplete = f"{session.directory} is not a complete session folder"
        assert str(error.value) == f"{incomplete}: {error.value.__cause__}"

    def test_damaged_ticks_file(self, folder):
        session = folder([tick(0)])
        (session.directory / c.FILE_TICKS).write_text('{"k": 0,\n')
        with pytest.raises(AdamusError, match="is not a complete session folder"):
            metrics.compute_metrics(session.directory)
