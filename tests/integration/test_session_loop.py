"""realtime.session.Session: the main loop, run with stand-in devices."""

import json
import logging
import os
import re
import signal
import time
from pathlib import Path

import numpy as np
import pytest
from brainflow.data_filter import DataFilter

from adamus import constants as c
from adamus.config import load_config
from adamus.detection import focus_control
from adamus.detection.focus_control import focus_of, next_level
from adamus.detection.focus_detector import Detector, load_models
from adamus.detection.signal_processing import Steps, design_filters
from adamus.detection.signal_processing import process as process_some
from adamus.errors import AdamusError
from adamus.realtime import session as session_module
from adamus.realtime import session_recorder
from adamus.realtime.eeg_source import HeadsetSource, RecordingSource
from adamus.session_files import REASON_END_OF_STREAM, REASON_ERROR, REASON_USER, Event, Meta, Tick
from tests.support.finished_sessions import (
    CONTEXT,
    FS,
    FinishedSession,
    H,
    N,
    config_with,
    read_lines,
    synthetic_matrix,
    synthetic_signals,
)
from tests.support.stand_in_devices import (
    FakeClock,
    FakeGenerator,
    FakeSource,
    StopKey,
)
from tests.support.synthetic_dataset import write_recording

SHORT = [["REST", 10], ["FOCUS", 10], ["REST", 10], ["FOCUS", 10]]  # 40 s, 73 steps
STEPS = (40 * FS - CONTEXT) // H + 1
CONTROL = [["REST", 30], ["FOCUS", 30], ["REST", 30], ["FOCUS", 30]]
CONTROL_STEPS = (120 * FS - CONTEXT) // H + 1  # 233
DETECTOR_ROWS = [2, 3, 1, 4]  # AF7, AF8, TP9, TP10 in the Muse 2 board matrix
EVERY = c.DETECTOR_ELECTRODES
FILTERS = design_filters(FS)


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


IDLE_STYLE = "quiet slow solo piano"
FOCUS_STYLE = "fast loud piano concerto with full symphony orchestra brass and timpani"


def short_session(tmp_path, matrix=None, **options):
    """A session over the protocol SHORT, by default over its synthetic signal."""
    matrix = synthetic_matrix(SHORT) if matrix is None else matrix
    return FinishedSession(tmp_path, config_with(SHORT), matrix, **options)


@pytest.fixture(scope="module")
def short(tmp_path_factory):
    """A finished session over 40 s of synthetic signal."""
    return FinishedSession(
        tmp_path_factory.mktemp("short"), config_with(SHORT), synthetic_matrix(SHORT)
    )


@pytest.fixture(scope="module")
def control(tmp_path_factory):
    """A finished session over 120 s of synthetic signal, in blocks of 30 s."""
    return FinishedSession(
        tmp_path_factory.mktemp("control"), config_with(CONTROL), synthetic_matrix(CONTROL)
    )


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
        raw = DataFilter.read_file(str(short.directory / c.FILE_RAW))
        for k in (0, 31, 32, 72):
            tick = short.ticks[k]
            expected = process(
                raw[DETECTOR_ROWS, tick["n_end"] - CONTEXT : tick["n_end"]], N, FILTERS
            )
            assert tick["max_abs_uv"] == pytest.approx(expected.max_abs_uv.tolist(), rel=1e-6)
            assert tick["valid"] is expected.valid

    def test_filtered_record_k_is_the_filtered_window_of_step_k(self, short):
        stored = np.fromfile(short.directory / c.FILE_FILTERED, dtype="<f4").reshape(STEPS, 4, N)
        raw = DataFilter.read_file(str(short.directory / c.FILE_RAW))
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
        source = FakeSource(matrix)
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
        source = FakeSource(synthetic_matrix(SHORT))
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
        raw = DataFilter.read_file(str(control.directory / c.FILE_RAW))
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
            rf"prompt 0 {IDLE_STYLE} \d\.\d{{3}}; prompt 1 {FOCUS_STYLE} \d\.\d{{3}};",
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


class TestSessionFrame:
    def test_events_in_order(self, short):
        kinds = [event["type"] for event in short.events]
        assert kinds[:8] == ["SESSION_START"] + ["FUDI"] * 6 + ["STREAM_START"]
        assert kinds[8] == "BLOCK_START"
        assert set(kinds) == {
            "SESSION_START", "FUDI", "STREAM_START", "BLOCK_START", "SESSION_END",
        }  # fmt: skip
        assert kinds[-3:] == ["FUDI", "FUDI", "SESSION_END"]
        times = [event["t"] for event in short.events]
        assert times == sorted(times)
        assert all(list(event)[:3] == ["t", "n", "type"] for event in short.events)
        assert all(set(event) <= set(Event.__annotations__) for event in short.events)

    def test_generator_messages_of_a_session(self, short):
        audio = short.directory / "audio.wav"
        assert short.session_messages() == [
            "reset;", f"prompt 0 {IDLE_STYLE} 1.000;", f"prompt 1 {FOCUS_STYLE} 0.000;",
            f"rec open {audio};", "rec start;", "bypass 0;", "bypass 1;", "rec stop;",
        ]  # fmt: skip
        sent = [m.strip() for m in short.generator.messages if m.count("prompt") != 2]
        assert sent == short.session_messages()  # what is logged is what was sent

    def test_stream_starts_only_after_the_generator_was_woken_up(self, short):
        assert short.generator.messages[5] == "bypass 0;"
        (stream,) = short.events_of("STREAM_START")
        assert (stream["board_id"], stream["fs"], stream["n"]) == (38, 256, 0)

    def test_pd_stderr_is_copied_to_the_session_only_while_it_runs(self, short):
        assert short.generator.stderr_copies == [short.directory / "pd.log", None]

    def test_pd_stays_with_the_session_until_it_finished_the_recording(self, tmp_path, monkeypatch):
        """What Pd prints while it closes audio.wav belongs to the pd.log of this session,
        and the metrics read audio.wav only when it is whole."""
        generator, looks = FakeGenerator(), []

        def finished(path):
            looks.append((path, generator.stderr_copies[-1]))
            return len(looks) == 3

        monkeypatch.setattr(session_recorder, "recording_finished", finished)
        session = short_session(tmp_path,
                                  generator=generator)  # fmt: skip
        pd_log = session.directory / "pd.log"
        assert looks == [(session.directory / "audio.wav", pd_log)] * 3
        assert generator.stderr_copies == [pd_log, None]
        assert generator.messages[-1] == "rec stop;"  # it waited after the last message

    def test_a_recording_that_is_never_finished_holds_the_session_up_only_briefly(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(session_recorder, "recording_finished", lambda path: False)
        clock = FakeClock()
        session = short_session(tmp_path,
                                  clock=clock)  # fmt: skip
        assert session.result.reason == REASON_END_OF_STREAM and session.error is None
        (end,) = session.events_of("SESSION_END")
        (stop,) = [e for e in session.events_of("FUDI") if e["msg"] == "rec stop;"]
        assert end["t"] - stop["t"] == pytest.approx(session_recorder.RECORDING_CLOSE_S, abs=0.1)

    def test_session_end_carries_reason_and_sample_count(self, short):
        (end,) = short.events_of("SESSION_END")
        assert (end["reason"], end["n"]) == (REASON_END_OF_STREAM, 40 * FS)
        assert short.result.reason == REASON_END_OF_STREAM
        assert (short.error, short.result.interrupted) == (None, False)

    def test_folder_holds_every_file_adamus_writes(self, short):
        assert {path.name for path in short.directory.iterdir()} == {
            "config.yaml", "focus_model.json", "meta.json", "raw.csv", "filtered.f32",
            "ticks.jsonl", "events.jsonl", "adamus.log",
        }  # fmt: skip
        assert short.directory.name.endswith("_S01")
        assert "DEBUG" in (short.directory / c.FILE_LOG).read_text()

    def test_model_copy_is_the_model_the_detector_ran_with(self, short):
        copied = json.loads((short.directory / c.FOCUS_MODEL_FILE).read_text())
        assert copied == load_models().raw
        assert load_models(short.directory / c.FOCUS_MODEL_FILE).raw == copied

    def test_config_copy_is_the_configuration_used(self, short):
        assert load_config(short.directory / c.FILE_CONFIG) == config_with(SHORT)

    def test_meta_describes_what_the_session_ran_on(self, short):
        meta = json.loads((short.directory / c.FILE_META).read_text())
        assert list(meta) == list(Meta.__annotations__)  # what the readers are typed against
        assert meta["pd"] == "Pd-0.0-fake"
        assert (meta["board_id"], meta["fs"]) == (38, 256)
        # Where the raw file holds what: every electrode of the source, the time, the markers.
        assert meta["electrode_rows"] == {"TP9": 1, "AF7": 2, "AF8": 3, "TP10": 4}
        assert meta["detector_electrodes"] == ["AF7", "AF8", "TP9", "TP10"]
        assert (meta["timestamp_row"], meta["marker_row"]) == (6, 7)
        assert (meta["mode"], meta["playback_path"]) == ("playback", "fake.csv")
        assert meta["started"] == pytest.approx(short.events[0]["t"], abs=0.01)

    def test_two_sessions_get_two_folders(self, tmp_path):
        first = short_session(tmp_path, synthetic_matrix(SHORT[:2]), name="A")
        second = short_session(tmp_path, synthetic_matrix(SHORT[:2]), name="B")
        assert first.directory != second.directory
        assert {path.name[-2:] for path in (tmp_path / "sessions").iterdir()} == {"_A", "_B"}

    def test_sessions_folder_may_lie_anywhere(self, tmp_path):
        session = FinishedSession(
            tmp_path / "my data; 2026", config_with(SHORT), synthetic_matrix(SHORT)
        )
        assert session.result.reason == REASON_END_OF_STREAM
        audio = str(session.directory / "audio.wav").replace(" ", "\\ ").replace(";", "\\;")
        assert f"rec open {audio};" in session.session_messages()

    def test_keys_pressed_before_the_stream_started_are_forgotten(self, short):
        """Otherwise a second press of [SPACE] at the start would end the session at once."""
        assert short.keyboard.discarded_at == ["STREAM_START"]


class TestEndings:
    def test_space_ends_the_session(self, tmp_path):
        session = short_session(tmp_path, stop_after_steps=20)
        assert session.result.reason == REASON_USER and session.result.interrupted is False
        assert session.events[-1]["reason"] == REASON_USER
        assert session.session_messages()[-2:] == ["bypass 1;", "rec stop;"]
        assert session.source.stopped

    def test_samples_received_before_the_stream_stopped_are_recorded_but_not_sent(self, tmp_path):
        """raw.csv holds every sample of the board; the step files must cover them all."""
        matrix = synthetic_matrix(CONTROL)[:, : 70 * FS]
        session = FinishedSession(tmp_path, config_with(CONTROL), matrix, held_back=3 * H)
        assert len(session.ticks) == (70 * FS - CONTEXT) // H + 1
        assert session.ticks[-4]["t_sent"] is not None
        for tick in session.ticks[-3:]:
            assert tick["t_sent"] is None and tick["valid"] and tick["f"] is not None
        assert len(session.generator.weights) == sum(t["t_sent"] is not None for t in session.ticks)
        assert session.events[-1]["n"] == 70 * FS

    def test_playback_ends_when_no_sample_arrived_for_the_stream_timeout(self, tmp_path):
        config = config_with(SHORT, timeouts={"stream_s": 2})
        session = FinishedSession(tmp_path, config, synthetic_matrix(SHORT)[:, : 6 * FS])
        duration = session.events[-1]["t"] - session.events_of("STREAM_START")[0]["t"]
        # 6 s of samples arrive in 24 polls of 64 samples; then 2 s without any.
        assert duration == pytest.approx(24 * 0.05 + 2.0, abs=0.01)  # not one poll more
        assert session.result.reason == REASON_END_OF_STREAM

    def test_live_headset_that_stops_sending_is_an_error(self, tmp_path):
        session = short_session(tmp_path, synthetic_matrix(SHORT)[:, : 6 * FS], live=True)
        assert session.result.reason == REASON_ERROR
        (error,) = session.events_of("ERROR")
        assert (error["where"], error["message"]) == ("stream", "no samples for 5 s")
        assert session.error == "no samples for 5 s"
        meta = json.loads((session.directory / c.FILE_META).read_text())
        assert (meta["mode"], meta["playback_path"]) == ("live", None)

    def test_generator_that_fails_during_a_step_is_an_error(self, tmp_path):
        session = FinishedSession(
            tmp_path,
            config_with(CONTROL),
            synthetic_matrix(CONTROL),
            generator=FakeGenerator(fail_after_steps=3),
        )
        assert session.result.reason == REASON_ERROR
        (error,) = session.events_of("ERROR")
        assert (error["where"], error["message"]) == ("loop", "OSError: broken pipe")
        assert session.events[-1]["type"] == "SESSION_END" and session.source.stopped
        assert session.session_messages()[-2:] == ["bypass 1;", "rec stop;"]

    def test_samples_that_came_with_a_failing_block_start_are_still_recorded(self, tmp_path):
        """The marker of the second block cannot be put into the stream: the session fails,
        but every step of the samples it received is recorded, at its place in raw.csv."""

        class BrokenMarkers(FakeSource):
            def insert_marker(self, value):
                if self.markers:
                    raise OSError("the marker was refused")
                super().insert_marker(value)

        session = short_session(tmp_path, source=BrokenMarkers(synthetic_matrix(SHORT)))
        assert session.error == "OSError: the marker was refused"
        received = np.loadtxt(session.directory / c.FILE_RAW, delimiter="\t", ndmin=2).shape[0]
        assert [t["k"] for t in session.ticks] == list(range(Steps.at(FS).count(received)))

    def test_steps_a_failure_left_are_taken_up_before_the_last_samples_push_them_out(
        self, tmp_path
    ):
        """A read of four windows makes four steps ready at once; the second fails and
        leaves the other two waiting. The read after the failure brings all the rest."""
        source = FakeSource(synthetic_matrix(CONTROL), chunk=4 * Steps.at(FS).window)
        failing = FakeGenerator(fail_after_steps=1)
        session = FinishedSession(tmp_path, config_with(CONTROL), source=source, generator=failing)
        assert session.error == "OSError: broken pipe"
        received = np.loadtxt(session.directory / c.FILE_RAW, delimiter="\t", ndmin=2).shape[0]
        assert [t["k"] for t in session.ticks] == list(range(Steps.at(FS).count(received)))

    def test_of_several_failures_the_first_one_is_recorded(self, tmp_path):
        """The generator fails during a step, then the stream fails to stop as well."""

        class StuckSource(FakeSource):
            def stop(self):
                super().stop()
                raise OSError("the headset does not answer")

        session = FinishedSession(
            tmp_path, config_with(CONTROL), source=StuckSource(synthetic_matrix(CONTROL)),
            generator=FakeGenerator(fail_after_steps=3),
        )  # fmt: skip
        assert session.result.reason == REASON_ERROR
        (error,) = session.events_of("ERROR")
        assert (error["where"], error["message"]) == ("loop", "OSError: broken pipe")
        assert session.events[-1]["type"] == "SESSION_END"

    @pytest.mark.parametrize(
        ("chunk", "held_back"),
        [(64, 300), (64, 0), (1024, 0)],  # samples after the stop; steps left in the same chunk
    )
    def test_a_step_whose_weights_could_not_be_sent_is_recorded_once(
        self, tmp_path, chunk, held_back
    ):
        """Every step is processed exactly once: f goes on as in a session where nothing failed,
        and the record holds every step the samples allow, the failed one without t_sent."""
        matrix = synthetic_matrix(CONTROL)
        clean = FinishedSession(tmp_path / "clean", config_with(CONTROL), matrix,
                                chunk=chunk, held_back=held_back)  # fmt: skip
        failed = FinishedSession(tmp_path / "failed", config_with(CONTROL), matrix,
                                 generator=FakeGenerator(fail_after_steps=20),
                                 chunk=chunk, held_back=held_back)  # fmt: skip
        assert failed.result.reason == REASON_ERROR
        steps = [tick["k"] for tick in failed.ticks]
        assert steps == list(range(len(steps)))  # none twice, none left out
        sent = [tick for tick in failed.ticks if tick["t_sent"] is not None]
        assert len(sent) == 20 and failed.ticks[sent[-1]["k"] + 1]["t_sent"] is None
        same = {tick["k"]: tick["f"] for tick in clean.ticks}
        assert [tick["f"] for tick in failed.ticks] == [same[k] for k in steps]
        received = failed.source.delivered
        assert len(steps) == (received - CONTEXT) // H + 1

    @pytest.mark.parametrize("problem", [OSError("no space left"), ValueError("bad window")])
    def test_failure_that_repeats_after_the_stream_stopped_still_ends_the_record(
        self, tmp_path, monkeypatch, problem
    ):
        """The samples that arrive while the stream stops bring the failed step back."""
        real_tick, written = session_recorder.SessionRecorder.tick, []

        def tick_until_the_disk_is_full(self, row, filtered):
            if len(written) >= 3:
                raise problem
            written.append(row["k"])
            real_tick(self, row, filtered)

        monkeypatch.setattr(session_recorder.SessionRecorder, "tick", tick_until_the_disk_is_full)
        session = short_session(tmp_path, held_back=13)
        assert session.result.reason == REASON_ERROR
        assert session.error == f"{type(problem).__name__}: {problem}"
        assert [event["type"] for event in session.events[-4:]] == [
            "ERROR",
            "FUDI",
            "FUDI",
            "SESSION_END",
        ]
        assert len(session.events_of("ERROR")) == 1  # the repeated failure is not a second error
        assert session.source.stopped and [tick["k"] for tick in session.ticks] == [0, 1, 2]

    def test_failure_while_the_last_samples_are_recorded_is_an_error(self, tmp_path, monkeypatch):
        """A session that ran well, but whose record could not be completed, did not end well."""
        session_holder = []
        real_tick = session_recorder.SessionRecorder.tick

        def tick_while_streaming(self, row, filtered):
            if session_holder and session_holder[0].stopped:
                raise OSError("no space left")
            real_tick(self, row, filtered)

        monkeypatch.setattr(session_recorder.SessionRecorder, "tick", tick_while_streaming)
        source = FakeSource(synthetic_matrix(SHORT, tail_s=4), held_back=3 * H)
        session_holder.append(source)
        session = short_session(tmp_path, source=source)
        assert (session.result.reason, session.error) == (
            REASON_ERROR,
            "OSError: no space left",
        )
        assert session.events[-1] == {
            **session.events[-1],
            "type": "SESSION_END",
            "reason": "error",
        }
        assert session.events_of("ERROR")[0]["where"] == "loop"

    def test_generator_that_cannot_be_put_to_sleep_does_not_keep_the_stream_open(self, tmp_path):
        class DeafAtTheEnd(FakeGenerator):
            def send(self, message):
                if message.startswith("bypass 1"):
                    raise AdamusError("the generator does not answer")
                super().send(message)

        session = short_session(tmp_path, generator=DeafAtTheEnd())
        assert (session.result.reason, session.error) == (
            REASON_ERROR,
            "the generator does not answer",
        )
        assert session.source.stopped and session.generator.stderr_copies[-1] is None
        assert session.events_of("ERROR")[0]["where"] == "generator"
        assert session.events[-1]["type"] == "SESSION_END"

    def test_stream_that_cannot_be_stopped_is_an_error_with_a_closed_record(self, tmp_path):
        class Stuck(FakeSource):
            def stop(self):
                raise AdamusError("cannot stop the EEG stream")

        session = short_session(tmp_path, None, source=Stuck(synthetic_matrix(SHORT)))
        assert (session.result.reason, session.error) == (
            REASON_ERROR,
            "cannot stop the EEG stream",
        )
        assert session.events_of("ERROR")[0]["where"] == "source"
        assert session.events[-1]["type"] == "SESSION_END"

    def test_source_that_cannot_start_is_an_error_with_a_closed_folder(self, tmp_path):
        class Broken(FakeSource):
            def start(self, raw_csv):
                raise AdamusError("cannot start the EEG stream: BOARD_NOT_READY")

        session = short_session(tmp_path, None, source=Broken(synthetic_matrix(SHORT)))
        assert session.result.reason == REASON_ERROR
        assert session.error == "cannot start the EEG stream: BOARD_NOT_READY"
        assert [e["type"] for e in session.events if e["type"] != "FUDI"] == [
            "SESSION_START",
            "ERROR",
            "SESSION_END",
        ]
        assert session.events_of("ERROR")[0]["where"] == "source"
        assert session.ticks == [] and not session.source.stopped
        assert session.session_messages()[-2:] == ["bypass 1;", "rec stop;"]

    def test_failure_to_write_the_record_is_an_error_too(self, tmp_path, monkeypatch):
        def disk_full(self, meta):
            raise OSError("disk full")

        monkeypatch.setattr(session_recorder.SessionRecorder, "write_meta", disk_full)
        session = short_session(tmp_path)
        assert (session.result.reason, session.error) == (
            REASON_ERROR,
            "OSError: disk full",
        )
        assert session.events_of("ERROR")[0]["where"] == "record"
        assert session.events[-1]["type"] == "SESSION_END"

    def test_generator_that_fails_when_the_session_wakes_it_up(self, tmp_path):
        class DeafAtTheStart(FakeGenerator):
            def send(self, message):
                if message.startswith("reset"):
                    raise AdamusError("cannot send to Pd: broken pipe")
                super().send(message)

        session = short_session(tmp_path, generator=DeafAtTheStart())
        assert session.result.reason == REASON_ERROR
        (error,) = session.events_of("ERROR")
        assert (error["where"], error["message"]) == ("generator", "cannot send to Pd: broken pipe")
        assert session.ticks == [] and session.source.raw_csv is None  # the stream never started

    def test_recording_without_a_single_sample_ends_as_end_of_stream(self, tmp_path):
        session = short_session(tmp_path, synthetic_matrix(SHORT)[:, :0])
        assert (session.result.reason, session.error) == (REASON_END_OF_STREAM, None)
        assert session.ticks == [] and session.events[-1]["n"] == 0


class TestRealDevices:
    """What a session runs on outside the tests."""

    def test_recording_is_played_at_the_rate_of_the_headset_on_the_clock_of_the_session(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        write_recording(Path("rec"), synthetic_signals(SHORT), [("REST", 40)])

        def no_headset(board):
            raise AssertionError("a recording is played: the headset is not opened")

        monkeypatch.setattr(session_module, "HeadsetSource", no_headset)
        generator, keyboard = object(), object()
        devices = session_module.real_devices(config_with(), Path("rec"), generator, keyboard)
        assert devices.generator is generator and devices.keyboard is keyboard
        assert isinstance(devices.clock, session_module.Clock)
        source = devices.source
        assert isinstance(source, RecordingSource) and source._clock is devices.clock
        assert (source.is_playback, source.board_id, source.fs) == (True, 38, 256)
        # The record of the session names the file by its whole path.
        assert source.playback_file == (tmp_path / "rec" / "eeg.csv").resolve()
        assert source.playback_file.is_absolute()

    def test_without_a_recording_the_headset_of_the_configuration_is_read(self):
        muse_s = {"name": "Muse S", "id": 39}
        devices = session_module.real_devices(config_with(board=muse_s), None, None, None)
        source = devices.source
        assert isinstance(source, HeadsetSource)
        assert (source.is_playback, source.playback_file) == (False, None)
        assert (source.board_id, source.fs) == (39, 256)

    def test_clock_is_the_time_of_the_computer(self):
        clock = session_module.Clock()
        before = time.time()
        now = clock.now()
        clock.sleep(0.05)
        assert before <= now <= time.time() - 0.05
        assert clock.now() - now >= 0.05

    def test_durations_are_measured_on_the_monotonic_clock(self):
        clock = session_module.Clock()
        before = time.monotonic()
        started = clock.monotonic()
        clock.sleep(0.05)
        assert before <= started <= time.monotonic() - 0.05
        assert clock.monotonic() - started >= 0.05


class TestRunSession:
    def devices(self, matrix):
        self.source = FakeSource(matrix)
        return session_module.Devices(self.source, FakeGenerator(), StopKey(), FakeClock())

    def test_folder_is_given_before_the_session_runs_and_the_source_is_released(self, tmp_path):
        given = []

        def created(folder):
            given.append((folder, sorted(path.name for path in folder.iterdir())))

        devices = self.devices(synthetic_matrix(SHORT))
        result = session_module.run_session(
            config_with(SHORT), "S01", devices, tmp_path / "sessions", created
        )
        assert result.reason == REASON_END_OF_STREAM
        assert result.directory.parent == (tmp_path / "sessions").resolve()
        assert given == [(result.directory, [])]  # before anything was recorded
        assert self.source.closed

    def test_source_is_released_when_the_session_cannot_even_be_created(self, tmp_path):
        """E.g. two sessions of the same name started within one second."""
        given = []
        arguments = (config_with(SHORT), "S01")
        first = session_module.Session(*arguments, self.devices(synthetic_matrix(SHORT)), tmp_path)
        first._recorder.close()
        devices = self.devices(synthetic_matrix(SHORT))
        with pytest.raises(FileExistsError):
            session_module.run_session(*arguments, devices, tmp_path, given.append)
        assert self.source.closed
        assert given == []  # there is no folder of this session

    def test_a_session_that_never_ran_leaves_no_file_and_no_log_open(self, tmp_path):
        """E.g. a full disk as the console's log moves into the new folder."""

        def full_disk(folder):
            raise OSError("no space left")

        handlers = list(logging.getLogger("adamus").handlers)
        devices = self.devices(synthetic_matrix(SHORT))
        with pytest.raises(OSError, match="no space left"):
            session_module.run_session(config_with(SHORT), "S01", devices, tmp_path, full_disk)
        assert logging.getLogger("adamus").handlers == handlers
        assert self.source.closed
        (folder,) = tmp_path.iterdir()
        assert sorted(path.name for path in folder.iterdir()) == []  # nothing was recorded

    def test_folder_is_named_by_the_time_the_session_was_created(self, tmp_path):
        session = short_session(tmp_path)
        created = time.strftime("%Y%m%d-%H%M%S", time.localtime(1000.0))  # the fake clock's time
        assert session.directory.name == f"{created}_S01"


class TestInterrupt:
    def test_ctrl_c_read_as_a_key_ends_the_session_like_space(self, tmp_path):
        class InterruptingKeys:
            def __init__(self):
                self.polls = 0

            def stop_requested(self):
                self.polls += 1
                if self.polls == 120:  # six seconds into the session
                    raise KeyboardInterrupt
                return False

            def discard_pressed_keys(self):
                pass

        source, clock = FakeSource(synthetic_matrix(SHORT)), FakeClock()
        devices = session_module.Devices(source, FakeGenerator(), InterruptingKeys(), clock)
        result = session_module.Session(config_with(SHORT), "S01", devices, tmp_path).run()
        assert (result.reason, result.interrupted) == (REASON_USER, True)
        assert read_lines(result.directory / c.FILE_EVENTS)[-1]["reason"] == REASON_USER
        assert source.stopped

    def test_ctrl_c_pressed_while_the_session_was_starting_still_ends_it(self, tmp_path):
        class InterruptedAtTheStart(StopKey):
            def discard_pressed_keys(self):
                raise KeyboardInterrupt

        source = FakeSource(synthetic_matrix(SHORT))
        devices = session_module.Devices(
            source, FakeGenerator(), InterruptedAtTheStart(), FakeClock()
        )
        result = session_module.Session(config_with(SHORT), "S01", devices, tmp_path).run()
        assert (result.reason, result.interrupted) == (REASON_USER, True)
        events = read_lines(result.directory / c.FILE_EVENTS)
        assert [event["type"] for event in events][-4:] == [
            "STREAM_START",
            "FUDI",
            "FUDI",
            "SESSION_END",
        ]
        assert source.stopped

    @pytest.mark.parametrize(
        "where", ["between tick and filtered record", "after the control step"]
    )
    @pytest.mark.parametrize("signal_number", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
    def test_signal_in_the_middle_of_a_step_does_not_tear_the_record(
        self, tmp_path, monkeypatch, where, signal_number
    ):
        """A signal may arrive at any instruction; the step in progress must still be whole."""
        fired = []

        def signal_once_in_step(k):
            if k == 29 and not fired:
                fired.append(k)
                os.kill(os.getpid(), signal_number)

        if where == "between tick and filtered record":

            class SignallingJson:
                """json for the recorder: signals while the line of step 29 is being made."""

                @staticmethod
                def dumps(row, **options):
                    signal_once_in_step(row.get("k"))
                    return json.dumps(row, **options)

            monkeypatch.setattr(session_recorder, "json", SignallingJson)
        else:
            original_step = focus_control.Controller.step

            def after_the_control_step(controller, n_end, step_signal):
                control = original_step(controller, n_end, step_signal)
                signal_once_in_step((n_end - CONTEXT) // H)
                return control

            monkeypatch.setattr(focus_control.Controller, "step", after_the_control_step)

        before = signal.getsignal(signal_number)
        session = short_session(tmp_path)
        assert fired == [29]
        assert (session.result.reason, session.result.interrupted) == (REASON_USER, True)
        # The step in progress was finished, then the loop ended at its next pass.
        assert [tick["k"] for tick in session.ticks] == list(range(len(session.ticks)))
        assert 30 <= len(session.ticks) <= 32
        record_bytes = len(session.ticks) * 4 * 4 * N
        assert (session.directory / c.FILE_FILTERED).stat().st_size == record_bytes
        raw_samples = len((session.directory / c.FILE_RAW).read_text().splitlines())
        assert len(session.ticks) == (raw_samples - CONTEXT) // H + 1  # nothing lost, nothing twice
        assert session.events[-1]["type"] == "SESSION_END"
        assert signal.getsignal(signal_number) is before  # the handler is restored


class SetClock(FakeClock):
    """The time of day is set ``by`` seconds once ``at`` seconds have passed."""

    def __init__(self, at, by):
        super().__init__()
        self._at, self._by = self.time + at, by

    def sleep(self, seconds):
        super().sleep(seconds)
        if self._at is not None and self.time >= self._at:
            self.wall_offset, self._at = self._by, None


class TestTimeOfDay:
    """Durations are measured on the monotonic clock: setting the clock changes nothing."""

    def test_a_recording_plays_to_its_end_when_the_clock_is_set_back(self, tmp_path, config):
        recording = write_recording(
            tmp_path / "data" / "x" / "short", synthetic_signals(SHORT), [tuple(b) for b in SHORT]
        )
        clock = SetClock(at=20.0, by=-100.0)
        started = clock.time
        session = short_session(tmp_path, recording=recording, clock=clock)
        assert session.result.reason == REASON_END_OF_STREAM
        assert len(session.ticks) == STEPS
        # 40 s of samples and 5 s of waiting for more: it did not stand still for 100 s
        assert clock.time - started == pytest.approx(40 + 5, abs=1)

    def test_a_live_stream_is_not_taken_for_silent_when_the_clock_is_set_forward(self, tmp_path):
        class EveryOtherPoll(FakeSource):
            """Samples come with every other poll, as from a headset over Bluetooth."""

            polls = 0

            def read(self):
                self.polls += 1
                return super().read() if self.polls % 2 else self.matrix_without_samples()

            def matrix_without_samples(self):
                return self._matrix[:, :0].copy()

        source = EveryOtherPoll(synthetic_matrix(SHORT), live=True)
        session = short_session(tmp_path, source=source,
                                  clock=SetClock(at=10.0, by=+30.0))  # fmt: skip
        assert session.error == "no samples for 5 s"  # only at the true end of the samples
        assert session.source.delivered == synthetic_matrix(SHORT).shape[1]
