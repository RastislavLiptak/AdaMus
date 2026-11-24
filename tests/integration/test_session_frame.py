"""realtime.session.Session around its steps: what it records, how it ends, how it is
interrupted and the time it keeps; run with stand-in devices."""

# A test reaches into a session where nothing else shows what it checks: the clock a played
# recording keeps time by, and the record of a session that could not be run.
# pyright: reportPrivateUsage=false

import json
import os
import signal

import numpy as np
import pytest

from adamus import constants as c
from adamus.config import load_config
from adamus.detection import focus_control
from adamus.detection.focus_detector import load_models
from adamus.detection.signal_processing import Steps
from adamus.errors import AdamusError
from adamus.realtime import session_recorder
from adamus.session_format import REASON_END_OF_STREAM, REASON_ERROR, REASON_USER, Event, Meta
from tests.support.finished_sessions import (
    CONTEXT,
    FS,
    FinishedSession,
    H,
    N,
    config_with,
    synthetic_matrix,
    synthetic_signals,
)
from tests.support.short_sessions import (
    CONTROL,
    FOCUS_STYLE,
    IDLE_STYLE,
    SHORT,
    STEPS,
    short_session,
)
from tests.support.stand_in_devices import (
    StandInClock,
    StandInGenerator,
    StandInSource,
)
from tests.support.synthetic_dataset import write_recording


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
        generator, looks = StandInGenerator(), []

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
        assert "did not finish" not in (session.directory / c.FILE_LOG).read_text()

    def test_a_recording_that_is_never_finished_holds_the_session_up_only_briefly(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(session_recorder, "recording_finished", lambda path: False)
        clock = StandInClock()
        session = short_session(tmp_path,
                                  clock=clock)  # fmt: skip
        assert session.result.reason == REASON_END_OF_STREAM and session.error is None
        (end,) = session.events_of("SESSION_END")
        (stop,) = [e for e in session.events_of("FUDI") if e["msg"] == "rec stop;"]
        assert end["t"] - stop["t"] == pytest.approx(session_recorder.RECORDING_CLOSE_S, abs=0.1)
        log = (session.directory / c.FILE_LOG).read_text()
        assert "WARNING adamus.realtime.session: Pd did not finish audio.wav in time" in log

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
        assert meta["pd"] == "Pd-0.0-stand-in"
        assert (meta["board_id"], meta["fs"]) == (38, 256)
        # Where the raw file holds what: every electrode of the source, the time, the markers.
        assert meta["electrode_rows"] == {"TP9": 1, "AF7": 2, "AF8": 3, "TP10": 4}
        assert meta["detector_electrodes"] == ["AF7", "AF8", "TP9", "TP10"]
        assert (meta["timestamp_row"], meta["marker_row"]) == (6, 7)
        assert meta["playback_path"] == "stand_in.csv"
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

    def test_a_stop_requested_while_the_session_was_starting_ends_it_after_the_start(
        self, tmp_path
    ):
        session = short_session(tmp_path, stop_after_steps=0)
        assert session.result.reason == REASON_USER
        assert [event["type"] for event in session.events][-4:] == [
            "STREAM_START", "FUDI", "FUDI", "SESSION_END",
        ]  # fmt: skip
        assert session.ticks == [] and session.source.stopped


class TestEndings:
    def test_a_stop_request_ends_the_session(self, tmp_path):
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
        assert meta["playback_path"] is None

    def test_generator_that_fails_during_a_step_is_an_error(self, tmp_path):
        session = FinishedSession(
            tmp_path,
            config_with(CONTROL),
            synthetic_matrix(CONTROL),
            generator=StandInGenerator(fail_after_steps=3),
        )
        assert session.result.reason == REASON_ERROR
        (error,) = session.events_of("ERROR")
        assert (error["where"], error["message"]) == ("loop", "OSError: broken pipe")
        assert session.events[-1]["type"] == "SESSION_END" and session.source.stopped
        assert session.session_messages()[-2:] == ["bypass 1;", "rec stop;"]

    def test_samples_that_came_with_a_failing_block_start_are_still_recorded(self, tmp_path):
        """The marker of the second block cannot be put into the stream: the session fails,
        but every step of the samples it received is recorded, at its place in raw.csv."""

        class BrokenMarkers(StandInSource):
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
        source = StandInSource(synthetic_matrix(CONTROL), chunk=4 * Steps.at(FS).window)
        failing = StandInGenerator(fail_after_steps=1)
        session = FinishedSession(tmp_path, config_with(CONTROL), source=source, generator=failing)
        assert session.error == "OSError: broken pipe"
        received = np.loadtxt(session.directory / c.FILE_RAW, delimiter="\t", ndmin=2).shape[0]
        assert [t["k"] for t in session.ticks] == list(range(Steps.at(FS).count(received)))

    def test_of_several_failures_the_first_one_is_recorded(self, tmp_path):
        """The generator fails during a step, then the stream fails to stop as well."""

        class StuckSource(StandInSource):
            def stop(self):
                super().stop()
                raise OSError("the headset does not answer")

        session = FinishedSession(
            tmp_path, config_with(CONTROL), source=StuckSource(synthetic_matrix(CONTROL)),
            generator=StandInGenerator(fail_after_steps=3),
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
                                 generator=StandInGenerator(fail_after_steps=20),
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
        source = StandInSource(synthetic_matrix(SHORT, tail_s=4), held_back=3 * H)
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
        class DeafAtTheEnd(StandInGenerator):
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
        class Stuck(StandInSource):
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
        class Broken(StandInSource):
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

    def test_a_failure_that_cannot_be_recorded_still_finishes_the_session(
        self, tmp_path, monkeypatch
    ):
        """A full disk: the generator is still put to sleep and the stream stopped."""
        record = session_recorder.SessionRecorder.event

        def disk_full_for_errors(self, kind, *arguments, **fields):
            if kind == "ERROR":
                raise OSError("disk full")
            return record(self, kind, *arguments, **fields)

        monkeypatch.setattr(session_recorder.SessionRecorder, "event", disk_full_for_errors)
        session = FinishedSession(
            tmp_path, config_with(CONTROL), synthetic_matrix(CONTROL),
            generator=StandInGenerator(fail_after_steps=3),
        )  # fmt: skip
        assert session.result.reason == REASON_ERROR
        assert session.session_messages()[-2:] == ["bypass 1;", "rec stop;"]
        assert session.source.stopped and session.events[-1]["type"] == "SESSION_END"

    def test_generator_that_fails_when_the_session_wakes_it_up(self, tmp_path):
        class DeafAtTheStart(StandInGenerator):
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


class TestInterrupt:
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


class SetClock(StandInClock):
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
            tmp_path / "data" / "x" / "short", synthetic_signals(SHORT), SHORT
        )
        clock = SetClock(at=20.0, by=-100.0)
        started = clock.time
        session = short_session(tmp_path, recording=recording, clock=clock)
        assert session.result.reason == REASON_END_OF_STREAM
        assert len(session.ticks) == STEPS
        # 40 s of samples and 5 s of waiting for more: it did not stand still for 100 s
        assert clock.time - started == pytest.approx(40 + 5, abs=1)

    def test_a_live_stream_is_not_taken_for_silent_when_the_clock_is_set_forward(self, tmp_path):
        class EveryOtherPoll(StandInSource):
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
