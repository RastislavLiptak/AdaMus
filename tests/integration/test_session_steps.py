"""realtime.session.Session, step by step: what each step computes from the signal, what
it sends to the generator, and the blocks it follows; run with stand-in devices."""

import json
import re

import numpy as np
import pytest
from brainflow.data_filter import DataFilter

from adamus import constants as c
from adamus.detection.focus_control import focus_of, next_level
from adamus.detection.focus_detector import Detector, load_models
from adamus.detection.signal_processing import design_filters
from adamus.detection.signal_processing import process as process_some
from adamus.errors import AdamusError
from adamus.session_format import REASON_END_OF_STREAM, Tick
from tests.support.finished_sessions import (
    CONTEXT,
    FS,
    FinishedSession,
    H,
    N,
    config_with,
    synthetic_matrix,
)
from tests.support.short_sessions import (
    CONTROL,
    CONTROL_STEPS,
    FOCUS_STYLE,
    IDLE_STYLE,
    SHORT,
    STEPS,
    short_session,
)
from tests.support.stand_in_devices import (
    StandInSource,
)

DETECTOR_ROWS = [2, 3, 1, 4]  # AF7, AF8, TP9, TP10 in the Muse 2 board matrix
EVERY = c.DETECTOR_ELECTRODES
FILTERS = design_filters(FS)


def raw_of(session):
    """raw.csv of a session, as BrainFlow reads it back."""
    raw = DataFilter.read_file(str(session.directory / c.FILE_RAW))
    assert raw is not None
    return raw


def process(context, window, filters, electrodes=EVERY):
    return process_some(context, window, filters, electrodes)


def scores_over(raw_rows, ticks, models, electrodes=EVERY):
    """The detector over the rows of a raw file, step by step as the session ran it."""
    detector = Detector(models.by_electrodes[electrodes], c.PRIOR_STEPS, c.MEMORY_STEPS)
    out = []
    for tick in ticks:
        step = process(raw_rows[:, tick["n_end"] - CONTEXT : tick["n_end"]], N, FILTERS, electrodes)
        out.append(detector.step(step.covariances) if step.valid else None)
    return out


def mean_f(session, block):
    return np.mean(
        [t["f"] for t in session.ticks if t["block_index"] == block and t["full_window"]]
    )


class TestSteps:
    def test_step_k_ends_with_sample_context_plus_k_times_h(self, short):
        """The first step needs its window and the window before it, for its filters."""
        assert [tick["k"] for tick in short.ticks] == list(range(STEPS))
        assert [tick["n_end"] for tick in short.ticks] == [CONTEXT + k * H for k in range(STEPS)]

    def test_tick_has_its_documented_fields_in_order(self, short):
        kinds = {
            "k": int, "n_end": int, "t_eeg": float, "t_proc_start": float, "t_proc_end": float,
            "t_sent": (float, type(None)), "block_index": int, "block_label": str,
            "full_window": bool, "max_abs_uv": list, "valid": bool,
            "score": (float, type(None)), "f": (float, type(None)),
            "f_held": bool, "w_idle": float, "w_focus": float,
        }  # fmt: skip
        assert list(kinds) == list(Tick.__annotations__)  # what the readers are typed against
        for tick in (short.ticks[0], short.ticks[-1]):
            assert list(tick) == list(kinds)
            for key, kind in kinds.items():
                assert isinstance(tick[key], kind), key
            assert len(tick["max_abs_uv"]) == 4

    def test_tick_values_are_those_of_processing_the_raw_window(self, short):
        """Row k of ticks belongs to raw.csv rows n_end - CONTEXT .. n_end - 1 of the
        detector's electrodes, in its order."""
        raw = raw_of(short)
        for k in (0, 31, 32, 72):
            tick = short.ticks[k]
            expected = process(
                raw[DETECTOR_ROWS, tick["n_end"] - CONTEXT : tick["n_end"]], N, FILTERS
            )
            assert tick["max_abs_uv"] == pytest.approx(expected.max_abs_uv.tolist(), rel=1e-6)
            assert tick["valid"] is expected.valid

    def test_filtered_record_k_is_the_filtered_window_of_step_k(self, short):
        stored = np.fromfile(short.directory / c.FILE_FILTERED, dtype="<f4").reshape(STEPS, 4, N)
        raw = raw_of(short)
        for k in (0, 40):
            expected = process(raw[DETECTOR_ROWS, k * H : k * H + CONTEXT], N, FILTERS).filtered
            assert np.allclose(stored[k], expected, atol=1e-3)

    def test_times_of_a_step(self, short):
        matrix = synthetic_matrix(SHORT)
        for tick in short.ticks:
            assert tick["t_eeg"] == matrix[6, tick["n_end"] - 1]  # time of the last sample
            assert tick["t_proc_start"] < tick["t_proc_end"]
            assert tick["t_sent"] is None or tick["t_sent"] > tick["t_proc_end"]

    def test_block_and_full_window_of_each_step(self, short):
        for tick in short.ticks:
            block = (tick["n_end"] - 1) // (10 * FS)
            assert (tick["block_index"], tick["block_label"]) == (block, SHORT[block][0])
            assert tick["full_window"] is (tick["n_end"] - N >= block * 10 * FS)

    @pytest.mark.parametrize("chunk", [1, 7, 128, 512, 513, 5000, 100_000])
    def test_steps_do_not_depend_on_how_the_samples_arrive(self, tmp_path, control, chunk):
        other = FinishedSession(
            tmp_path, config_with(CONTROL), synthetic_matrix(CONTROL), chunk=chunk
        )
        keys = (
            "k",
            "n_end",
            "valid",
            "score",
            "f",
            "f_held",
            "w_idle",
            "w_focus",
            "block_index",
            "full_window",
        )
        assert [[t[key] for key in keys] for t in other.ticks] == [
            [t[key] for key in keys] for t in control.ticks
        ]
        assert other.generator.weights == control.generator.weights

    def test_no_step_before_its_whole_signal_has_arrived(self, tmp_path):
        matrix = synthetic_matrix(SHORT)[:, : CONTEXT - 1]
        session = short_session(tmp_path, matrix)
        assert session.ticks == [] and session.result.reason == REASON_END_OF_STREAM
        assert (session.directory / c.FILE_FILTERED).stat().st_size == 0

    def test_only_the_electrodes_of_the_detector_are_processed(self, tmp_path):
        """The source delivers all it has; the session takes the detector's electrodes by name."""
        matrix = synthetic_matrix(SHORT)
        matrix[5] = 9000.0 * np.sign(np.sin(np.arange(matrix.shape[1])))  # another input: garbage
        session = short_session(tmp_path, matrix)
        assert all(tick["valid"] for tick in session.ticks)
        assert max(max(tick["max_abs_uv"]) for tick in session.ticks) < 200  # not 9000

    @pytest.mark.parametrize(("row", "valid"), [(2, False), (3, False), (1, True), (4, True)])
    def test_only_the_frontal_electrodes_decide_validity(self, tmp_path, row, valid):
        """AF7 and AF8 must be clean; TP9 and TP10 may be loud, as the model was trained."""
        matrix = synthetic_matrix(SHORT)
        matrix[row] *= 20.0
        session = short_session(tmp_path, matrix)
        assert {tick["valid"] for tick in session.ticks} == {valid}
        assert max(tick["max_abs_uv"][DETECTOR_ROWS.index(row)] for tick in session.ticks) > 400

    @pytest.mark.parametrize("lacking", [["TP9"], ["AF8", "TP10"], ["AF7", "AF8", "TP9"]])
    def test_an_electrode_the_source_lacks_is_zeros_and_the_others_run_the_detector(
        self, tmp_path, lacking
    ):
        matrix = synthetic_matrix(SHORT)
        source = StandInSource(matrix)
        for name in lacking:
            del source.electrode_rows[name]
        session = short_session(tmp_path, source=source)
        present = tuple(name for name in EVERY if name not in lacking)
        meta = json.loads((session.directory / c.FILE_META).read_text())
        assert meta["detector_electrodes"] == list(present)
        missing = [EVERY.index(name) for name in lacking]
        assert all(tick["max_abs_uv"][k] == 0.0 for tick in session.ticks for k in missing)
        assert all(tick["valid"] for tick in session.ticks)
        rows = matrix[DETECTOR_ROWS].copy()
        rows[missing] = 0.0
        expected = scores_over(rows, session.ticks, load_models(), present)
        assert [t["score"] for t in session.ticks] == pytest.approx(expected, rel=1e-4, abs=1e-6)

    def test_source_without_any_electrode_of_the_detector_cannot_run_a_session(self, tmp_path):
        source = StandInSource(synthetic_matrix(SHORT))
        for name in EVERY:
            del source.electrode_rows[name]
        with pytest.raises(AdamusError) as error:
            short_session(tmp_path, source=source)
        assert str(error.value) == (
            "the source has none of the detector's electrodes AF7, AF8, TP9, TP10"
        )


class TestControl:
    def test_the_first_clean_step_already_controls_the_music(self, short):
        first = short.ticks[0]
        assert first["valid"] and first["f"] is not None and first["t_sent"] is not None
        assert first["f"] == pytest.approx(focus_of(first["score"]))

    def test_f_is_the_chance_of_focus_after_the_scores_so_far(self, control):
        level = 0.0
        for tick in control.ticks:
            level = next_level(level, tick["score"])
            assert tick["f"] == pytest.approx(focus_of(level))
            assert (tick["w_focus"], tick["w_idle"]) == (tick["f"], 1.0 - tick["f"])

    def test_scores_are_those_of_the_detector_over_the_raw_file(self, control):
        """The record alone gives the scores again: raw.csv and the model copy."""
        raw = raw_of(control)
        models = load_models(control.directory / c.FOCUS_MODEL_FILE)
        expected = scores_over(raw[DETECTOR_ROWS], control.ticks, models)
        assert [t["score"] for t in control.ticks] == pytest.approx(expected, rel=1e-4, abs=1e-6)

    def test_music_follows_the_blocks(self, control):
        """The synthetic FOCUS blocks have the powers of the n-back task: every FOCUS block
        is fuller than every REST block, and most of its steps fuller than those of a REST."""
        assert min(mean_f(control, 1), mean_f(control, 3)) > 0.15 + max(
            mean_f(control, 0), mean_f(control, 2)
        )
        focus = [t["f"] for t in control.ticks if t["block_label"] == "FOCUS" and t["full_window"]]
        rest = [t["f"] for t in control.ticks if t["block_label"] == "REST" and t["full_window"]]
        above = np.mean([[a > b for b in rest] for a in focus])
        assert above > 0.75, above

    def test_one_pair_of_weights_is_sent_for_every_valid_step(self, control):
        sent = [tick for tick in control.ticks if tick["t_sent"] is not None]
        assert [tick["k"] for tick in sent] == list(range(CONTROL_STEPS))
        assert control.generator.weights == [
            (round(tick["w_idle"], 3), round(tick["w_focus"], 3)) for tick in sent
        ]

    def test_weights_of_a_step_are_sent_together(self, control):
        step_messages = [m for m in control.generator.messages if m.count("prompt") == 2]
        assert len(step_messages) == CONTROL_STEPS
        assert re.fullmatch(
            rf"prompt 0 {IDLE_STYLE} \d\.\d{{3}}; prompt 1 {FOCUS_STYLE} \d\.\d{{3}}; "
            r"volume -?\d+\.\d\d; drumless [01];",
            step_messages[0],
        )

    def test_invalid_step_sends_nothing_and_holds_f_and_the_weights(self, tmp_path):
        matrix = synthetic_matrix(CONTROL)
        bad = 150
        matrix[2, CONTEXT + bad * H - 10] += 5000.0  # a spike on AF7 in the last hop of step 150
        session = FinishedSession(tmp_path, config_with(CONTROL), matrix)
        invalid = [tick["k"] for tick in session.ticks if not tick["valid"]]
        assert invalid == list(range(bad, bad + len(invalid))) and len(invalid) >= N // H
        before = session.ticks[bad - 1]
        for k in invalid:
            tick = session.ticks[k]
            assert (tick["t_sent"], tick["f_held"], tick["score"]) == (None, True, None)
            assert (tick["f"], tick["w_idle"], tick["w_focus"]) == (
                before["f"],
                before["w_idle"],
                before["w_focus"],
            )
            assert tick["max_abs_uv"][0] > c.ARTIFACT_UV
        after = session.ticks[invalid[-1] + 1]
        assert (after["f_held"], after["valid"]) == (False, True) and after["t_sent"] is not None
        assert len(session.generator.weights) == CONTROL_STEPS - len(invalid)

    def test_without_clean_signal_the_music_stays_quiet(self, tmp_path):
        matrix = synthetic_matrix(CONTROL)
        matrix[2] += 1000.0 * np.sign(np.sin(2 * np.pi * 3 * np.arange(matrix.shape[1]) / FS))
        session = FinishedSession(tmp_path, config_with(CONTROL), matrix)
        assert session.result.reason == REASON_END_OF_STREAM  # nothing ends it early
        assert session.generator.weights == []
        assert all(t["f"] is None and t["w_focus"] == 0.0 for t in session.ticks)

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_samples_that_are_not_numbers_only_spoil_the_steps_that_hold_them(
        self, tmp_path, value
    ):
        """The record stays plain JSON, and the session goes on to the end."""
        matrix = synthetic_matrix(SHORT)
        matrix[DETECTOR_ROWS[2], 12 * FS] = value  # one sample on TP9
        session = short_session(tmp_path, matrix)
        assert session.result.reason == REASON_END_OF_STREAM
        holding = [t for t in session.ticks if t["n_end"] - 2 * N <= 12 * FS < t["n_end"]]
        assert len(holding) == 2 * N // H and not any(t["valid"] for t in holding)
        assert all(t["score"] is None for t in holding)
        assert {t["max_abs_uv"][2] for t in holding if t["n_end"] - N <= 12 * FS} == {None}
        assert any(t["valid"] for t in session.ticks if t not in holding)
        for written in session.ticks:
            assert json.loads(json.dumps(written, allow_nan=False)) == written

    def test_a_value_too_large_to_compute_with_spoils_only_its_steps(self, tmp_path):
        matrix = synthetic_matrix(SHORT)
        matrix[DETECTOR_ROWS[2], 12 * FS] = 1e170  # one sample on TP9
        session = short_session(tmp_path, matrix)
        assert session.result.reason == REASON_END_OF_STREAM  # the detector did not fail
        holding = [t for t in session.ticks if t["n_end"] - 2 * N <= 12 * FS < t["n_end"]]
        assert not any(t["valid"] for t in holding)
        assert any(t["valid"] for t in session.ticks if t not in holding)

    def test_flat_signal_is_invalid_and_has_no_score(self, tmp_path):
        matrix = synthetic_matrix(SHORT)
        matrix[DETECTOR_ROWS, : 6 * FS] = 0.0
        session = short_session(tmp_path, matrix)
        first = session.ticks[0]
        assert (first["valid"], first["score"], first["f"]) == (False, None, None)
        assert first["max_abs_uv"] == [0.0, 0.0, 0.0, 0.0]
        assert json.loads(json.dumps(first)) == first  # plain JSON, no NaN
        controlled = next(t for t in session.ticks if t["valid"])
        assert controlled["f"] == pytest.approx(focus_of(controlled["score"]))


class TestBlocks:
    def test_block_start_puts_a_marker_into_the_stream_and_an_event_into_the_record(self, short):
        starts = short.events_of("BLOCK_START")
        assert [(e["index"], e["label"], e["marker"]) for e in starts] == [
            (0, "REST", 1), (1, "FOCUS", 2), (2, "REST", 1), (3, "FOCUS", 2),
        ]  # fmt: skip
        assert [value for _, value in short.source.markers] == [1, 2, 1, 2]

    def test_block_start_fires_as_soon_as_its_first_sample_has_arrived(self, short):
        for event in short.events_of("BLOCK_START"):
            start = event["index"] * 10 * FS
            assert start <= event["n"] <= start + 64  # within one chunk of the source
        assert short.events_of("BLOCK_START")[0]["n"] == 64  # block 0: with the first samples

    def test_free_play_block_gets_marker_3(self, tmp_path):
        protocol = [["REST", 10], ["FOCUS", 10], ["FREE", 0]]
        session = FinishedSession(tmp_path, config_with(protocol), synthetic_matrix(SHORT))
        assert session.events_of("BLOCK_START")[-1]["marker"] == 3
        after_block_1 = (20 * FS - CONTEXT) // H + 1
        assert {t["block_label"] for t in session.ticks[after_block_1:]} == {"FREE"}

    def test_markers_are_stored_in_the_raw_file(self, short):
        raw = np.loadtxt(short.directory / c.FILE_RAW, delimiter="\t")
        positions = np.flatnonzero(raw[:, 7])
        assert raw[positions, 7].tolist() == [1.0, 2.0, 1.0, 2.0]
