"""Running the program in tests: on the stand-in machine, in a temporary folder."""

import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from adamus import constants as c
from adamus.app_link import APP_INPUT_THREAD, READY, AppMessages
from adamus.config import write_yaml
from adamus.main import Program
from adamus.session_format import read_jsonl
from tests.support.finished_sessions import synthetic_signals
from tests.support.stand_in_machine import LIVE_PROTOCOL, StandInMachine, place_focus_model
from tests.support.stand_in_songs import place_song
from tests.support.synthetic_dataset import Protocol, write_recording

READY_LOOK_S = 0.01  # how often the app looks whether the program is ready
ENDS_S = 10  # the longest the program may take to hear that the app has ended


class AppInput:
    """The input of the program, which the app holds open while it runs: ``close`` is the
    app ending."""

    def __init__(self) -> None:
        reading, writing = os.pipe()
        self.reader = os.fdopen(reading, encoding="utf-8")
        self._writer = os.fdopen(writing, "w", encoding="utf-8")

    def close(self) -> None:
        """The app ends."""
        if not self._writer.closed:
            self._writer.close()

    def release(self) -> None:
        """Both ends let go, once the program, which watches the input, has ended."""
        self.close()
        for thread in threading.enumerate():
            if thread.name == APP_INPUT_THREAD:
                thread.join(ENDS_S)  # it ends at the end of the input, which the app closed
                assert not thread.is_alive(), "the program did not hear the app end"
        self.reader.close()


class ProgramRun:
    """One run of the program in a temporary folder, from start to exit."""

    def __init__(self, folder: Path, raw_config: dict[str, Any]) -> None:
        self.folder = folder
        self.raw = raw_config
        self.machine = StandInMachine(Path("sessions"), Path("data"))
        self.output = io.StringIO()  # the messages to the app
        self.app = AppInput()
        self.code: int | None = None
        self.program: Program | None = None
        place_song(self.folder / c.LIBRARY_DIR)
        place_focus_model(self.folder)

    def add_recording(
        self, name: str, protocol: Protocol | None = None, dataset: str = "simulated"
    ) -> Path:
        """Put a recording of synthetic signal into a dataset of the data folder."""
        protocol = protocol or LIVE_PROTOCOL
        folder = self.folder / "data" / dataset / name
        return write_recording(folder, synthetic_signals(protocol), protocol)

    def start(self, write_config: bool = True, app_ends_when_ready: bool = True) -> int:
        """Run the program until it ends; return its exit code. Unless the test ends the app
        itself, the app ends as soon as the program is ready, as when it is quit at once."""
        if write_config:
            write_yaml(self.raw, self.folder / "config.yaml")
        if app_ends_when_ready:
            threading.Thread(target=self._end_app_when_ready, daemon=True).start()
        messages = AppMessages(self.output)
        self.program = Program(Path("config.yaml"), self.machine, messages, self.app.reader)
        try:
            self.code = self.program.run()
        finally:
            self.app.close()
        return self.code

    def _end_app_when_ready(self) -> None:
        while True:  # until the program is ready, or has ended without being so
            if self.code is not None or self.ready() is not None:
                break
            time.sleep(READY_LOOK_S)
        self.app.close()

    def messages(self) -> list[dict[str, Any]]:
        """What the program told the app so far, in order."""
        return [json.loads(line) for line in self.output.getvalue().splitlines()]

    def checklists(self) -> list[list[dict[str, Any]]]:
        """Each checklist the app was shown, in order."""
        return [message["checks"] for message in self.messages() if "checks" in message]

    def ready(self) -> dict[str, str] | None:
        """Where the API answers and the token it wants, once the program has said it."""
        return next((message[READY] for message in self.messages() if READY in message), None)

    def log(self) -> list[dict[str, Any]]:
        """Everything the program logged: what the sessions hold, then what no session took."""
        stored = [
            entry
            for folder in self.session_folders()
            for entry in read_jsonl(folder / c.FILE_ACTIVITY)
        ]
        assert self.program is not None, "the program was started"
        return sorted(stored, key=lambda entry: entry["t"]) + self.program.activity.waiting

    def logged(self, kind: str) -> list[dict[str, Any]]:
        return [entry for entry in self.log() if entry["type"] == kind]

    def session_folders(self) -> list[Path]:
        return sorted((self.folder / "sessions").glob("*"))
