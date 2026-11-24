"""Runs the sessions the app asks for: one at a time, in the main thread.

A session needs the main thread, because it takes [CTRL]+[C] and kill over while it
runs. The app asks through the thread of the server: the runner takes the request
from a queue, runs the session, stores what the headset delivered in the data folder
and computes the metrics. A session stops when the operator asks for it, or when the
app has gone.
"""

import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from adamus.activity_log import (
    ACTION_ANALYSIS_FAILED,
    ACTION_APP_ENDED,
    ACTION_PROGRAM_STOPPED,
    ACTION_RECORDING_NOT_STORED,
    ACTION_SESSION_ENDED,
    ACTION_SESSION_REQUESTED,
    ACTION_SESSION_STARTED,
    ACTION_START_FAILED,
    ACTION_STOP_REQUESTED,
    ActivityLog,
)
from adamus.errors import AdamusError
from adamus.evaluation.metrics import analyze
from adamus.machine import Machine
from adamus.own_recordings import store_recording
from adamus.realtime.session import run_session
from adamus.server.api_models import Phase, RunState
from adamus.server.new_session import PreflightResults, PreparedSession

IDLE_LOOK_S = 0.2  # how often an idle runner looks whether the app has gone
Done = TypeVar("Done")  # what work done while idle gives


class StopOrGone:
    """The stop request of a running session: the operator's, or the app having gone."""

    def __init__(self, requested: threading.Event, app_gone: Callable[[], bool]) -> None:
        self._requested = requested
        self._app_gone = app_gone

    def is_set(self) -> bool:
        return self._requested.is_set() or self._app_gone()


class SessionRunner:
    def __init__(
        self, preflight: PreflightResults, machine: Machine, activity: ActivityLog,
        app_gone: Callable[[], bool],
    ) -> None:  # fmt: skip
        self._preflight = preflight
        self._machine = machine
        self._activity = activity
        self._app_gone = app_gone
        self._requests: queue.Queue[PreparedSession] = queue.Queue()
        self._lock = threading.Lock()  # of the state; held only for a moment
        self._looking = False  # for the headset, which no session may use meanwhile
        self._state = RunState(run=0, phase="idle", session_id=None, problem=None)
        self._stop = threading.Event()

    def state(self) -> RunState:
        with self._lock:
            return self._state.model_copy()

    def look_for_headset(self, look: Callable[[], Done]) -> Done | None:
        """``look`` for the headset while no session runs; None, without looking, when one
        runs or a look is going on. Meanwhile a session from a recording may start, one with
        the headset may not."""
        with self._lock:
            if self._state.phase != "idle" or self._looking:
                return None
            self._looking = True
        try:
            return look()
        finally:
            with self._lock:
                self._looking = False

    def looking_for_headset(self) -> bool:
        with self._lock:
            return self._looking

    def request(self, session: PreparedSession) -> RunState | None:
        """Queue a session; None when one is running already, or when it is one with the
        headset, which is being looked for."""
        with self._lock:
            if self._state.phase != "idle" or (session.recording is None and self._looking):
                return None
            run = self._state.run + 1
            self._state = RunState(run=run, phase="starting", session_id=None, problem=None)
            self._stop = threading.Event()
            self._requests.put(session)
            return self._state.model_copy()

    def stop(self) -> None:
        """Ask the running session to stop; the request is noted in its activity."""
        with self._lock:
            if self._state.phase in ("starting", "running") and not self._stop.is_set():
                self._stop.set()
                self._activity.write(ACTION_STOP_REQUESTED)

    def serve(self) -> None:
        """Run the sessions that are asked for until the app has gone."""
        while True:  # until the app has gone
            if self._app_gone():
                return
            try:
                session = self._requests.get(timeout=IDLE_LOOK_S)
            except queue.Empty:
                continue
            self._run(session)

    def _set(self, phase: Phase, **changes: str | None) -> None:
        with self._lock:
            self._state = self._state.model_copy(update={"phase": phase, **changes})

    def _run(self, session: PreparedSession) -> None:
        """Run one session and finish it: its own recording, then its metrics."""
        assert self._preflight.generator is not None, "the preflight passed"
        recording = None if session.recording is None else session.recording.name
        self._activity.write(
            ACTION_SESSION_REQUESTED, kind=session.source.kind, label=session.source.label,
            recording=recording, name=session.name,
        )  # fmt: skip
        stop = StopOrGone(self._stop, self._app_gone)
        try:
            self._play_on(session.output)
            devices = self._machine.devices(
                session.config, session.recording, self._preflight.generator, stop
            )
            sessions = self._machine.sessions_root
            result = run_session(session.config, session.name, devices, sessions, self._started)
        except (AdamusError, OSError) as problem:
            # Noted in its folder when it was made, so that its dashboard says why; otherwise
            # the attempt goes with the next session, as what happened before it.
            self._activity.write(ACTION_START_FAILED, message=str(problem))
            self._activity.wait_for_the_next_session()
            self._set("idle", problem=str(problem))
            return
        if result.problem is not None:  # it ran, but not to its end: the operator is told why
            with self._lock:
                self._state = self._state.model_copy(update={"problem": result.problem})
        if result.interrupted:
            self._activity.write(ACTION_PROGRAM_STOPPED)
        elif self._app_gone() and not self._stop.is_set():
            self._activity.write(ACTION_APP_ENDED)
        self._activity.write(ACTION_SESSION_ENDED, reason=result.reason)
        if self._app_gone() or result.interrupted:  # nothing more to play: Pd goes at once,
            self._preflight.generator.stop()  # not after the storing and the analysis
        self._set("finishing")
        try:
            if session.recording is None:
                self._store_recording(result.directory)
            if result.interrupted:  # the program quits, but what the headset recorded is kept
                raise KeyboardInterrupt
            self._analyze(result.directory)
        finally:
            self._activity.wait_for_the_next_session()  # what follows belongs to the next one
            self._set("idle")

    def _play_on(self, output: str) -> None:
        """Have the generator play on ``output``: Pd plays on one output from its start to its
        end, so it starts again on another, or on one that was disconnected. The start returns
        once the music of the prompts is ready, so that the session starts quiet."""
        generator = self._preflight.generator
        assert generator is not None, "the preflight passed"
        if generator.running and generator.output == output and not generator.output_lost:
            return
        generator.stop()
        generator.start(output)

    def _started(self, folder: Path) -> None:
        """What the program does for the session goes into its folder from now on."""
        self._activity.store_in(folder)
        self._activity.write(ACTION_SESSION_STARTED)
        self._set("running", session_id=folder.name)

    def _store_recording(self, directory: Path) -> None:
        """What the headset delivered, as a recording of the data folder."""
        try:
            store_recording(directory, self._machine.datasets_root)
        except (AdamusError, OSError) as problem:
            self._activity.write(ACTION_RECORDING_NOT_STORED, message=str(problem))

    def _analyze(self, directory: Path) -> None:
        """The metrics of the session, next to its record. Whatever keeps them from being
        computed is noted for the dashboard: the session itself was recorded, and the program
        goes on to the next one."""
        try:
            analyze(directory)
        except Exception as problem:  # noqa: BLE001 - any problem of the analysis, see above
            self._activity.write(ACTION_ANALYSIS_FAILED, message=str(problem))
