"""The interactive console.

Welcome, preflight checklist, then a loop: select a source, name a session,
run it, show its statistics. The console is a shell around a session and adds
no computation. It is operated with single keys, shown in square brackets; the
only typed text is the session name.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

from rich.console import Console, RenderableType
from rich.live import Live

from adamus import constants as c
from adamus import datasets
from adamus.config import Config, is_session_name
from adamus.console import data_sources, keyboard, printed_texts
from adamus.console.console_log import (
    ACTION_KEY,
    ACTION_QUIT,
    ACTION_SESSION_CANCELLED,
    ACTION_SESSION_ENDED,
    ACTION_SESSION_NAMED,
    ACTION_SESSION_STARTED,
    ACTION_SOURCE_SELECTED,
    ConsoleLog,
)
from adamus.console.data_sources import Source
from adamus.console.keyboard import Keyboard
from adamus.console.machine import Machine
from adamus.console.preflight_checks import HOW_TO_FIX, Preflight
from adamus.console.session_statistics import statistics_of
from adamus.errors import AdamusError
from adamus.evaluation.metrics import analyze
from adamus.own_recordings import store_recording
from adamus.realtime.session import run_session
from adamus.session_files import REASON_USER

SOURCE, RECORDING, NEW_SESSION = "SOURCE", "RECORDING", "NEW_SESSION"  # states of the console
READY, RUNNING = "READY", "RUNNING"
Choice = TypeVar("Choice")  # a row of a list the operator chooses from
ERASE_LAST_CHARACTER = chr(8) + " " + chr(8)  # step back, overwrite with a space, step back


class ConsoleApp:
    def __init__(
        self,
        config_path: Path,
        machine: Machine | None = None,
        keys: Keyboard | None = None,
        output: Console | None = None,
    ) -> None:
        self._config_path = Path(config_path)
        self._machine = machine or Machine()
        self._keys = keys or Keyboard()
        self._output = output or Console(highlight=False)
        self.log = ConsoleLog()

    def run(self) -> int:
        """Run until the operator quits; return the exit code of the program."""
        printed_texts.welcome(self._output, self._config_path)
        ready = interrupted = False
        with Live(console=self._output) as display:
            preflight = Preflight(self._config_path, self._machine, display, self.log)
            try:
                ready = preflight.run()
            except KeyboardInterrupt:
                interrupted = True
        try:
            if interrupted:  # [CTRL]+[C] quits the program normally in any state
                return c.EXIT_OK
            if not ready:
                printed_texts.errors(self._output, preflight.failed_checks(), HOW_TO_FIX)
                return c.EXIT_ERROR
            printed_texts.all_ready(self._output)
            self._keys.start()
            while True:  # until the operator quits
                if not self._session_cycle(preflight):
                    return c.EXIT_OK
        except KeyboardInterrupt:  # [CTRL]+[C] in any state quits the program normally
            return c.EXIT_OK
        finally:
            self._keys.restore_terminal()
            if preflight.generator is not None:
                preflight.generator.stop()
            self.log.write(ACTION_QUIT)

    def _session_cycle(self, preflight: Preflight) -> bool:
        """Source, question, name, READY, RUNNING, statistics. False = quit the program."""
        assert preflight.config is not None and preflight.generator is not None, "it passed"
        self.log.wait_for_the_next_session()
        source, recording = self._select_source(preflight)
        chosen = None if recording is None else recording.name
        self.log.write(
            ACTION_SOURCE_SELECTED, kind=source.kind, label=source.label, recording=chosen
        )
        missing = self._missing_electrodes(preflight.config, source, recording)
        if missing is None:
            return True
        printed_texts.new_session_question(self._output)
        if self._wait_for_key(NEW_SESSION, ["Y", "N"]) == "N":
            return False
        name = self._ask_name()
        self.log.write(ACTION_SESSION_NAMED, name=name)
        try:
            config = data_sources.session_config(preflight.config, recording)
            printed_texts.ready(self._output, name, source, recording, missing)
            if self._wait_for_key(READY, [keyboard.SPACE, "Q"]) == "Q":
                self.log.write(ACTION_SESSION_CANCELLED, name=name)
                printed_texts.cancelled(self._output)
                return True
            printed_texts.running(self._output)
            devices = self._machine.devices(config, recording, preflight.generator, self._keys)
            sessions = self._machine.sessions_root
            result = run_session(config, name, devices, sessions, self._session_started)
        except (AdamusError, OSError) as problem:
            printed_texts.could_not_start(self._output, problem)
            return True
        self.log.write(ACTION_SESSION_ENDED, reason=result.reason)
        stored = None if recording is not None else self._store_recording(result.directory)
        if result.interrupted:  # the program quits, but what the headset recorded is kept
            raise KeyboardInterrupt
        if result.reason == REASON_USER:  # the session read this key itself
            self.log.write(ACTION_KEY, state=RUNNING, key=keyboard.key_name(keyboard.SPACE))
        preflight.rescan_datasets()
        self._report(result.directory, stored)
        return True

    def _store_recording(self, directory: Path) -> Path | None:
        """What the headset delivered, as a recording of the data folder; None when there is
        nothing, or when it cannot be stored, which is reported."""
        try:
            return store_recording(directory, self._machine.datasets_root)
        except (AdamusError, OSError) as problem:
            printed_texts.not_read(self._output, printed_texts.NOT_STORED, problem)
            return None

    def _session_started(self, folder: Path) -> None:
        """What the console did for the session goes into its folder from now on."""
        self.log.store_in(folder)
        self.log.write(ACTION_SESSION_STARTED)

    def _report(self, directory: Path, stored: Path | None) -> None:
        """The metrics of the session next to its record, then its statistics. A folder that
        cannot be read is reported, and the operator goes on with the next session."""
        try:
            analyze(directory)
        except AdamusError as problem:
            printed_texts.not_read(self._output, printed_texts.NO_METRICS, problem)
        try:
            shown = replace(statistics_of(directory), recording=stored)
            printed_texts.statistics(self._output, shown)
        except AdamusError as problem:
            printed_texts.not_read(self._output, printed_texts.NO_STATISTICS, problem)

    def _missing_electrodes(
        self, config: Config, source: Source, recording: Path | None
    ) -> list[str] | None:
        """The detector's electrodes the source lacks; None (and why) if it has none of them."""
        try:
            missing = data_sources.missing_electrodes(config, recording)
        except (AdamusError, OSError) as problem:
            printed_texts.could_not_start(self._output, problem)
            return None
        if len(missing) == len(c.DETECTOR_ELECTRODES):
            printed_texts.lacks_electrodes(self._output, source, recording)
            return None
        return missing

    def _wait_for_key(self, state: str, accepted: list[str]) -> str:
        """Wait for one of the accepted keys; every other key is ignored."""
        while True:  # until one of them is pressed
            key = self._keys.next_key()
            key = key.upper() if len(key) == 1 else key  # letters are not case sensitive
            if key in accepted:
                self.log.write(ACTION_KEY, state=state, key=keyboard.key_name(key))
                return key

    def _select_source(self, preflight: Preflight) -> tuple[Source, Path | None]:
        """The headset, or a dataset and the recording of it that is to be played."""
        while True:  # until the headset or a recording is chosen
            source = self._pick(
                SOURCE,
                preflight.sources,
                printed_texts.source_list,
                on_key={"R": preflight.rescan_headset},
            )
            assert source is not None, "the list of sources cannot be left"
            if source.dataset is None:
                return source, None
            recording = self._select_recording(source.dataset)
            if recording is not None:
                return source, recording

    def _select_recording(self, dataset: Path) -> Path | None:
        """The recording of a dataset to play; None when the operator goes back with [Q]."""
        recordings = datasets.recordings_of(dataset)
        return self._pick(
            RECORDING,
            lambda: recordings,
            printed_texts.recording_list,
            leave_on=["Q"],
        )

    def _pick(
        self,
        state: str,
        rows_now: Callable[[], list[Choice]],
        draw: Callable[[list[Choice], int], RenderableType],
        on_key: dict[str, Callable[[], None]] | None = None,
        leave_on: Sequence[str] = (),
    ) -> Choice | None:
        """Let the operator move through a list and choose a row of it with [ENTER].

        A key of ``on_key`` runs its action and the list is drawn again; a key
        of ``leave_on`` leaves the list without a choice (None).
        """
        actions = on_key or {}
        keys = [keyboard.UP, keyboard.DOWN, keyboard.ENTER, *actions, *leave_on]
        selected = 0
        with Live(console=self._output, auto_refresh=False) as display:
            while True:  # until a row is chosen or the list is left
                rows = rows_now()
                selected = max(0, min(selected, len(rows) - 1))
                display.update(draw(rows, selected), refresh=True)
                key = self._wait_for_key(state, keys)
                if key == keyboard.UP:
                    selected -= 1
                elif key == keyboard.DOWN:
                    selected += 1
                elif key in actions:
                    actions[key]()
                elif key in leave_on:
                    return None
                elif rows:
                    return rows[selected]

    def _ask_name(self) -> str:
        """Read the session name: typed characters, [BACKSPACE] to correct, [ENTER] to finish."""
        typed = ""
        printed_texts.session_name_question(self._output)
        while True:  # until a valid name is entered
            key = self._keys.next_key()
            if key == keyboard.ENTER:
                self._output.print()
                if is_session_name(typed):
                    return typed
                printed_texts.invalid_name(self._output)
                printed_texts.session_name_question(self._output)
                typed = ""
            elif key == keyboard.BACKSPACE and typed:
                typed = typed[:-1]
                self._output.file.write(ERASE_LAST_CHARACTER)
                self._output.file.flush()
            elif len(key) == 1 and key.isprintable():
                typed += key
                self._output.print(key, end="", markup=False)
