"""A session of the default protocol over the synthetic recording: its known results.

The synthetic recording has a ground truth (the powers of the typical person of
the training data, at rest and during the task), so the whole chain (source,
processing, detector, control, record, metrics, statistics) must give these
numbers. The detector controls from the first step on.
"""

import numpy as np
import pytest

from adamus import constants as c
from adamus.config import parse_config, read_yaml
from adamus.evaluation import metrics
from adamus.evaluation.session_reader import load_session_folder
from adamus.evaluation.session_statistics import statistics_of
from adamus.session_format import REASON_END_OF_STREAM
from tests.support.finished_sessions import CONTEXT, FS, FinishedSession, H, synthetic_matrix
from tests.support.project_paths import CONFIG_FILE

BLOCK = 60 * FS


@pytest.fixture(scope="module")
def session(tmp_path_factory, default_protocol):
    """A finished session of the default protocol over its synthetic recording."""
    root = tmp_path_factory.mktemp("synthetic_default")
    config = parse_config(read_yaml(CONFIG_FILE))
    return FinishedSession(root, config, synthetic_matrix(default_protocol))


@pytest.fixture(scope="module")
def session_metrics(session):
    return metrics.compute_metrics(session.directory)


class TestSession:
    def test_every_step_is_valid(self, session):
        assert len(session.ticks) == (8 * BLOCK - CONTEXT) // H + 1 == 953
        assert all(tick["valid"] for tick in session.ticks)

    def test_full_steps_of_every_block(self, session):
        """Block 0 has fewer: the first step ends when two windows have arrived."""
        full = [
            sum(1 for t in session.ticks if t["block_index"] == block and t["full_window"])
            for block in range(8)
        ]
        assert full == [113] + [117] * 7

    def test_the_first_step_already_controls(self, session):
        assert session.ticks[0]["f"] is not None and session.ticks[0]["t_sent"] is not None

    def test_mean_f_is_low_in_rest_blocks_and_high_in_focus_blocks(self, session):
        def mean_f(block):
            ticks = [t for t in session.ticks if t["block_index"] == block and t["full_window"]]
            return float(np.mean([t["f"] for t in ticks]))

        rest = [mean_f(block) for block in (0, 2, 4, 6)]
        focus = [mean_f(block) for block in (1, 3, 5, 7)]
        assert rest == pytest.approx([0.286, 0.093, 0.177, 0.198], abs=0.002)
        assert focus == pytest.approx([0.597, 0.729, 0.752, 0.722], abs=0.002)

    def test_session_ends_with_the_recording_in_the_free_play_block(self, session):
        assert session.result.reason == REASON_END_OF_STREAM
        assert [value for _, value in session.source.markers] == [1, 2] * 4 + [3]


class TestMetrics:
    def test_states_are_separated(self, session_metrics):
        assert session_metrics["state_separation_auc"] == pytest.approx(0.893, abs=0.001)
        assert session_metrics["mean_f_difference"] == pytest.approx(0.512, abs=0.001)

    def test_focus_is_reached_and_left_within_seconds_of_a_block_start(self, session_metrics):
        assert session_metrics["response_time_s"] == {
            "rise_s": [2.5, 10.5, 0.5, 6.0], "fall_s": [4.0, 2.0, 4.5, 1.0],
            "rise_median_s": 4.25, "fall_median_s": 3.0,
        }  # fmt: skip

    def test_all_steps_of_all_blocks_are_valid(self, session_metrics):
        blocks = session_metrics["valid_steps_per_block"]
        assert [(b["full"], b["valid"], b["ratio"]) for b in blocks[:8]] == [(113, 113, 1.0)] + [
            (117, 117, 1.0)
        ] * 7
        assert blocks[8] == {"index": 8, "label": "FREE", "full": 0, "valid": 0, "ratio": None}

    def test_sampling_and_record(self, session_metrics):
        assert session_metrics["sampling_rate_hz"] == pytest.approx(256.0, rel=1e-6)
        assert session_metrics["longest_sample_gap_s"] == pytest.approx(1 / 256, rel=1e-3)
        assert session_metrics["record_completeness"] == {
            "ticks": 953, "filtered_records": 953, "expected": 953, "passed": True,
        }  # fmt: skip


class TestStatistics:
    def test_verdict_and_focus(self, session):
        statistics = statistics_of(load_session_folder(session.directory))
        assert (statistics.verdict, statistics.usable_share) == ("OK", 1.0)
        focus = statistics.focus
        assert focus is not None
        assert (focus.focus_blocks_on, focus.focus_blocks) == (4, 4)
        assert (focus.rest_blocks_off, focus.rest_blocks) == (4, 4)
        assert focus.switched_on == 30
        assert 0.0 < focus.f_min < 0.01 and 0.99 < focus.f_max < 1.0


class TestTheRawFileIsEnough:
    def test_processing_the_raw_file_again_gives_every_step_again(self, session, tmp_path):
        """What a session stored can later be analysed without it: raw.csv leads to the same steps.

        The raw file keeps six decimals, so the values differ a little, but within 1e-6.
        """
        from brainflow.data_filter import DataFilter

        from adamus.config import load_config
        from tests.support.stand_in_devices import StandInSource

        raw = session.directory / c.FILE_RAW
        again = FinishedSession(
            tmp_path,
            load_config(session.directory / c.FILE_CONFIG),
            None,
            name="again",
            source=StandInSource(DataFilter.read_file(str(raw))),
        )
        assert len(again.ticks) == len(session.ticks) == 953
        for ours, theirs in zip(again.ticks, session.ticks, strict=True):
            for key in ("k", "n_end", "valid", "block_index", "full_window", "f_held", "told"):
                assert ours[key] == theirs[key]
            assert (ours["score"] is None) is (theirs["score"] is None)
            for key in ("score", "f"):
                if ours[key] is not None:
                    assert ours[key] == pytest.approx(theirs[key], abs=1e-4)
