"""One session: the main loop that turns EEG samples into generator weights.

The loop is driven by the number of samples received, not by the clock. Step k
ends where signal_processing.Steps puts it (one hop after step k - 1), so a
recording that is played back gives the same steps with the same results every time.
"""

import logging
import signal
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Unpack

import numpy as np

from adamus import constants as c
from adamus.config import Config
from adamus.detection.focus_control import Controller
from adamus.detection.focus_detector import Detector, load_models
from adamus.detection.signal_processing import Steps, design_filters, process
from adamus.errors import AdamusError
from adamus.protocol import block_spans
from adamus.realtime import generator_messages
from adamus.realtime.clock import Clock, TimeSource
from adamus.realtime.eeg_source import HeadsetSource, RecordingSource
from adamus.realtime.recent_samples import RecentSamples
from adamus.realtime.session_recorder import SessionRecorder
from adamus.session_format import (
    EVENT_BLOCK_START,
    EVENT_ERROR,
    EVENT_FUDI,
    EVENT_SESSION_END,
    EVENT_SESSION_START,
    EVENT_STREAM_START,
    REASON_END_OF_STREAM,
    REASON_ERROR,
    REASON_USER,
    STAGE_GENERATOR,
    STAGE_LOOP,
    STAGE_RECORD,
    STAGE_SOURCE,
    STAGE_STREAM,
    EventFields,
    SessionFacts,
    Tick,
)

log = logging.getLogger(__name__)
STEP_LOG = "step %d: valid=%s score=%s f=%s"  # what a step leaves in adamus.log
OUTPUT_LOST = "The audio output {output} was disconnected, so the music could not be heard."


class EegSource(Protocol):
    """Where the samples come from: the headset, or a recording played in its place."""

    @property
    def board_id(self) -> int: ...

    @property
    def fs(self) -> int: ...

    @property
    def electrode_rows(self) -> dict[str, int]: ...

    @property
    def timestamp_row(self) -> int: ...

    @property
    def marker_row(self) -> int: ...

    @property
    def playback_file(self) -> Path | None: ...

    def start(self, raw_csv: Path) -> None: ...

    def read(self) -> np.ndarray: ...

    def insert_marker(self, value: float) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class MusicOutput(Protocol):
    """Where the weights of the music go: the generator."""

    @property
    def pd_version(self) -> str | None: ...

    @property
    def output(self) -> str | None: ...  # the audio output it plays on

    @property
    def output_lost(self) -> bool: ...  # that output has gone from the Mac

    def send(self, message: str) -> None: ...

    def copy_output_to(self, session_log: Path | None) -> None: ...


class StopRequest(Protocol):
    """Set when the operator asks the running session to stop; a threading.Event is one."""

    def is_set(self) -> bool: ...


@dataclass
class Devices:
    """Everything a session talks to. Tests replace these by stand-ins."""

    source: EegSource
    generator: MusicOutput
    stop: StopRequest
    clock: TimeSource


@dataclass(frozen=True)
class SessionResult:
    directory: Path
    reason: str  # user, end_of_stream or error
    interrupted: bool  # ended by a signal such as [CTRL]+[C], not by the operator's request
    problem: str | None  # what failed, when it ended in an error


class Session:
    """One pass through the protocol over one data source, recorded in one folder."""

    def __init__(self, config: Config, name: str, devices: Devices, sessions_root: Path) -> None:
        root = Path(sessions_root).resolve()
        fs = devices.source.fs
        self._config = config
        self._devices = devices
        self._fs = fs
        self._steps = Steps.at(fs)
        self._spans = block_spans(config.protocol, fs)
        rows_of_source = devices.source.electrode_rows
        self._electrodes = tuple(name for name in c.DETECTOR_ELECTRODES if name in rows_of_source)
        if not self._electrodes:
            needed = ", ".join(c.DETECTOR_ELECTRODES)
            raise AdamusError(f"the source has none of the detector's electrodes {needed}")
        self._models = load_models()
        detector = Detector(
            self._models.by_electrodes[self._electrodes], c.PRIOR_STEPS, c.MEMORY_STEPS
        )
        self._controller = Controller(self._spans, self._steps.window, detector)
        self._filters = design_filters(fs)
        # a missing electrode is a row of zeros
        self._electrode_rows = [rows_of_source.get(name) for name in c.DETECTOR_ELECTRODES]
        rows = len(self._electrode_rows) + 1
        steps = self._steps
        self._samples = RecentSamples(rows, steps.window, steps.hop, steps.context)
        self._waiting = np.empty((rows, 0))  # received, not yet appended: a failure stopped it
        self._received = 0  # n: samples received from the source
        self._steps_done = 0  # k: the next step to process
        self._blocks_started = 0
        self._stage = STAGE_RECORD  # what the session is doing, for ERROR events
        self._streaming = False
        self._failed = False
        self._problem: str | None = None  # the first failure, as its ERROR event says it
        self._stop_signalled = False
        self._recorder = SessionRecorder(root, name, devices.clock.now())
        self.directory = self._recorder.directory

    # --- The frame of a session ----------------------------------------------

    def run(self) -> SessionResult:
        # [CTRL]+[C] must not cut a step in half (a tick line without its
        # filtered record). Until the session ends, it only sets a flag.
        stop_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        previous = {number: signal.signal(number, self._on_stop_signal) for number in stop_signals}
        try:
            self._recorder.open()
            reason = self._run_until_recorded()
            self._event(EVENT_SESSION_END, reason=reason)
            log.info("session ended: %s", reason)
        finally:
            self._recorder.close()
            for number, handler in previous.items():
                signal.signal(number, handler)
        return SessionResult(self.directory, reason, self._stop_signalled, self._problem)

    def _on_stop_signal(self, _number: int, _frame: object) -> None:
        self._stop_signalled = True

    def _run_until_recorded(self) -> str:
        """Start, loop and finish; return the reason of the end.

        Whatever fails, the session still puts the generator to sleep, stops
        the stream and ends its record with SESSION_END.
        """
        self._event(EVENT_SESSION_START)
        reason = REASON_ERROR
        try:
            self._start()
            reason = self._loop()
        except Exception as problem:  # noqa: BLE001 - any failure must end the session cleanly
            self._record_error(problem)
        finishing = (
            (STAGE_GENERATOR, self._put_generator_to_sleep),
            (STAGE_SOURCE, self._stop_stream),
            (STAGE_LOOP, self._record_last_samples),
        )
        for stage, finish in finishing:  # one failing must not skip the others
            try:
                self._stage = stage
                finish()
            except Exception as problem:  # noqa: BLE001 - the record must still get its SESSION_END
                self._record_error(problem)
        return REASON_ERROR if self._failed else reason

    def _start(self) -> None:
        """Write what the session runs on, wake the generator up, start the stream."""
        devices, recorder, source = self._devices, self._recorder, self._devices.source
        recorder.write_config(self._config.raw)
        recorder.write_model(self._models.raw)
        recorder.write_meta(self._meta())
        self._stage = STAGE_GENERATOR
        devices.generator.copy_output_to(recorder.pd_log)
        self._send(
            generator_messages.session_start_messages(self._config.generator, recorder.audio_wav)
        )
        self._stage = STAGE_SOURCE
        source.start(recorder.raw_csv)
        self._streaming = True
        self._event(EVENT_STREAM_START, board_id=source.board_id, fs=self._fs)
        self._stage = STAGE_LOOP

    def _put_generator_to_sleep(self) -> None:
        """Silence the generator and end its recording. What Pd prints until audio.wav is
        closed still belongs to this session."""
        try:
            self._send(generator_messages.session_end_messages())
            if not self._recorder.wait_for_recording(self._devices.clock.sleep):
                log.warning("Pd did not finish %s in time", self._recorder.audio_wav.name)
        finally:
            self._devices.generator.copy_output_to(None)

    def _stop_stream(self) -> None:
        if self._streaming:
            self._devices.source.stop()

    def _record_last_samples(self) -> None:
        """raw.csv holds every sample the board produced, so the steps of the samples
        that arrived while the stream was stopping are recorded too, but not sent."""
        if self._streaming:
            self._receive_and_process(sending=False)

    def _meta(self) -> SessionFacts:
        source = self._devices.source
        playback = None if source.playback_file is None else str(source.playback_file)
        return {
            "pd": self._devices.generator.pd_version,
            "audio_output": self._devices.generator.output,
            "board_id": source.board_id,
            "fs": self._fs,
            "electrode_rows": source.electrode_rows,
            "detector_electrodes": list(self._electrodes),
            "timestamp_row": source.timestamp_row,
            "marker_row": source.marker_row,
            "playback_path": playback,
            "started": self._devices.clock.now(),
        }

    # --- The main loop -------------------------------------------------------

    def _loop(self) -> str:
        """Poll the source until the session ends; return the reason."""
        devices, config = self._devices, self._config
        last_sample_at = devices.clock.monotonic()
        while True:  # until the operator, the stream or a failure ends the session
            if self._stop_signalled or devices.stop.is_set():
                return REASON_USER
            if devices.generator.output_lost:  # the music is heard nowhere: no use going on
                self._stage = STAGE_GENERATOR
                raise AdamusError(OUTPUT_LOST.format(output=devices.generator.output))
            devices.clock.sleep(config.poll_s)
            before = self._received
            self._receive_and_process(sending=True)
            if self._received > before:
                last_sample_at = devices.clock.monotonic()
            elif devices.clock.monotonic() - last_sample_at >= config.timeouts.stream_s:
                if devices.source.playback_file is not None:  # a recording has ended
                    return REASON_END_OF_STREAM
                self._stage = STAGE_STREAM
                raise AdamusError(f"no samples for {config.timeouts.stream_s:g} s")

    def _receive_and_process(self, sending: bool) -> None:
        """Take the new samples, start due blocks, process ready steps.

        A step needs its whole context, so the first one ends when that has arrived.
        What a failure left behind, steps ready or samples not yet appended, is taken up
        first, before new samples can push those steps out.
        """
        new = self._devices.source.read()
        self._received += new.shape[1]
        waiting = self._waiting = np.hstack([self._waiting, self._needed_rows(new)])
        if sending:  # a block start that fails leaves the samples above for the record
            self._start_due_blocks()
        self._process_ready_steps(sending)
        room = self._samples.room
        for start in range(0, waiting.shape[1], room):
            self._samples.append(waiting[:, start : start + room])
            self._waiting = waiting[:, start + room :]
            self._process_ready_steps(sending)

    def _process_ready_steps(self, sending: bool) -> None:
        """Every step whose samples are there and that has not been processed yet. A step
        counts as done before it is processed: one that fails is never processed twice."""
        for k in range(self._steps_done, self._steps.count(self._samples.count)):
            self._steps_done = k + 1
            self._step(k, sending)

    def _needed_rows(self, new: np.ndarray) -> np.ndarray:
        """Of all a source delivers: the electrodes of the detector, then the times."""
        missing = np.zeros(new.shape[1])
        electrodes = [missing if row is None else new[row] for row in self._electrode_rows]
        return np.vstack([*electrodes, new[self._devices.source.timestamp_row]])

    def _start_due_blocks(self) -> None:
        """Marker and event for every block whose first sample has arrived."""
        for span in self._spans[self._blocks_started :]:
            if span.start > self._received:
                break
            self._blocks_started += 1
            marker = c.BLOCK_MARKERS[span.label]
            self._devices.source.insert_marker(marker)
            self._event(EVENT_BLOCK_START, index=span.index, label=span.label, marker=marker)

    def _step(self, k: int, sending: bool) -> None:
        """Step k: process the window, update the control, send the weights, write the tick."""
        clock = self._devices.clock
        n_end = self._steps.end(k)
        samples = self._samples.window(n_end, self._steps.context)
        electrodes, timestamps = samples[:-1], samples[-1]

        t_proc_start = clock.now()
        processed = process(electrodes, self._steps.window, self._filters, self._electrodes)
        control = self._controller.step(n_end, processed)
        t_proc_end = clock.now()

        row: Tick = {
            "k": k,
            "n_end": n_end,
            "t_eeg": float(timestamps[-1]),
            "t_proc_start": t_proc_start,
            "t_proc_end": t_proc_end,
            "t_sent": None,  # until the weights have been sent
            "block_index": control.block.index,
            "block_label": control.block.label,
            "full_window": control.full_window,
            "max_abs_uv": [None if np.isnan(v) else float(v) for v in processed.max_abs_uv],
            "valid": processed.valid,
            "score": control.score,
            "f": control.f,
            "f_held": control.f_held,
            "w_idle": control.w_idle,
            "w_focus": control.w_focus,
        }
        try:
            if sending and processed.valid and control.f is not None:
                messages = generator_messages.step_messages(
                    self._config.generator, control.w_idle, control.w_focus
                )
                self._devices.generator.send(" ".join(messages))
                row["t_sent"] = clock.now()
        finally:  # a step whose weights could not be sent is recorded all the same
            self._recorder.tick(row, processed.filtered)
        log.debug(STEP_LOG, k, processed.valid, control.score, control.f)

    # --- Events --------------------------------------------------------------

    def _event(self, kind: str, **fields: Unpack[EventFields]) -> None:
        self._recorder.event(kind, self._devices.clock.now(), self._received, **fields)

    def _send(self, messages: list[str]) -> None:
        """Send session messages to the generator and log each as a FUDI event."""
        for message in messages:
            self._devices.generator.send(message)
            self._event(EVENT_FUDI, msg=message)

    def _record_error(self, problem: Exception) -> None:
        """Log the failure and note it in the record; of several failures, the first one counts."""
        log.exception("session failed in %s", self._stage)
        if self._failed:
            return
        self._failed = True
        known = isinstance(problem, AdamusError)
        message = str(problem) if known else f"{type(problem).__name__}: {problem}"
        self._problem = message
        try:
            self._event(EVENT_ERROR, where=self._stage, message=message)
        except OSError:  # a full disk: the session must still be finished, the generator quiet
            log.exception("the failure could not be recorded")


def real_devices(
    config: Config, recording: Path | None, generator: MusicOutput, stop: StopRequest
) -> Devices:
    """The devices of a real session: the headset, or ``recording`` played back in its place."""
    clock = Clock()
    source: HeadsetSource | RecordingSource
    if recording is None:
        source = HeadsetSource(config.board)
    else:
        source = RecordingSource(config.board, recording.resolve(), clock)
    return Devices(source, generator, stop, clock)


def run_session(
    config: Config,
    name: str,
    devices: Devices,
    sessions_root: Path,
    on_folder_created: Callable[[Path], None],
) -> SessionResult:
    """Run one session and close its EEG source; ``on_folder_created`` is given the folder
    of the session before the session runs."""
    try:
        session = Session(config, name, devices, sessions_root)
        on_folder_created(session.directory)
        result = session.run()
    finally:
        devices.source.close()
    return result
