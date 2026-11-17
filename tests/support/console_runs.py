"""Driving the interactive console in tests: a stand-in machine and scripted keys."""

import io
import shutil
import time
from pathlib import Path

from rich.console import Console

from adamus import constants as c
from adamus.config import write_yaml
from adamus.console import keyboard
from adamus.console.console_app import ConsoleApp
from adamus.console.keyboard import Keyboard
from adamus.console.machine import Machine
from adamus.errors import AdamusError
from adamus.realtime.eeg_source import RecordingSource
from adamus.realtime.session import Devices
from tests.support.fake_generator_files import install_generator_files
from tests.support.finished_sessions import (
    read_lines,
    synthetic_matrix,
    synthetic_signals,
)
from tests.support.project_paths import PROJECT_MODEL
from tests.support.stand_in_devices import (
    FakeClock,
    FakeGenerator,
    FakeSource,
    ScriptedKeys,
)
from tests.support.synthetic_dataset import write_recording
from tests.support.waiting import wait_until

# What the stand-in headset delivers: a long rest, then two FOCUS blocks.
LIVE_PROTOCOL = [["REST", 64], ["FOCUS", 20], ["REST", 20], ["FOCUS", 20]]
POLL_TAKES_S = 0.001  # of real time: long enough for a key to arrive between two polls
KEY_WAIT_S = 20  # the longest a scripted key waits for what it is pressed after
SAMPLES_PAUSE_S = 0.0002  # between two looks for samples: a key must come early in a session
CONSOLE_WIDTH = 120  # characters


class PacedClock(FakeClock):
    """Fake time that still takes a moment per poll, so that a key can arrive mid-session."""

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        time.sleep(POLL_TAKES_S)


class WatchedRecording(RecordingSource):
    """The real player of recordings; the tests can see how far it got and that it was closed."""

    @property
    def delivered(self) -> int:
        return self._delivered

    @property
    def closed(self) -> bool:
        return self._raw is None


class FakeMachine(Machine):
    """A computer whose audio device, headsets and generator the test decides.

    Sessions run for real: a recording is played by the real player on a clock
    that does not wait; a live headset is a stand-in that delivers 124 s of
    synthetic signal and then falls silent.
    """

    def __init__(self, sessions_root: Path, datasets_root: Path) -> None:
        self.sessions_root = sessions_root
        self.datasets_root = datasets_root
        self.audio = ("MacBook Pro Speakers", float(c.AUDIO_RATE))
        self.headsets_on = {"Muse 2"}
        self.generator = FakeGenerator()
        self.probed: list[int] = []
        self.sources: list[FakeSource] = []
        self.source_failure = ""
        self.live_samples: int | None = None  # the headset delivers only this many, if set
        self.installation = "fails"  # or "installs" the missing files, or "does nothing"
        self.installed: list[str] = []  # the generator folders an installation ran for
        self.training = "fails"  # or "trains" the models of the project, or "does nothing"
        self.trainings = 0
        self.paced = False  # a key comes while a session runs: each poll takes real time

    def train_focus_detector(self):
        self.trainings += 1
        log_file = c.FOCUS_DETECTOR_DIR / c.SETUP_LOG
        if self.training == "fails":
            raise AdamusError(f"the training of the focus detector failed, see {log_file}")
        if self.training == "trains":
            place_focus_model(Path())
        return log_file

    def install_generator(self, generator):
        self.installed.append(generator.folder)
        log_file = generator.directory / c.SETUP_LOG
        if self.installation == "fails":
            raise AdamusError(f"the installation of the generator failed, see {log_file}")
        if self.installation == "installs":
            install_generator_files(generator.directory, generator.model)
        return log_file

    def audio_output(self):
        return self.audio

    def headset_is_available(self, board):
        self.probed.append(board.id)
        return board.name in self.headsets_on

    def make_generator(self, config, report):
        return self.generator

    def devices(self, config, recording, generator, keyboard):
        if self.source_failure:
            raise OSError(self.source_failure)
        clock = PacedClock() if self.paced else FakeClock()
        if recording is None:
            matrix = synthetic_matrix(LIVE_PROTOCOL)[:, : self.live_samples]
            self.sources.append(FakeSource(matrix, live=True))
        else:
            self.sources.append(WatchedRecording(config.board, recording, clock))
        return Devices(self.sources[-1], generator, keyboard, clock)


def place_focus_model(folder: Path) -> None:
    """The project's trained focus models, as the program keeps them below ``folder``."""
    (folder / c.FOCUS_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROJECT_MODEL, folder / c.FOCUS_MODEL_PATH)


class Screen(io.StringIO):
    """What the console printed; ``shown`` is what had reached the screen at the last flush."""

    shown = ""

    def flush(self) -> None:
        super().flush()
        self.shown = self.getvalue()


class ConsoleRun:
    """One run of the console in a temporary folder, from start to exit."""

    def __init__(self, folder: Path, raw_config: dict) -> None:
        self.folder = folder
        self.raw = raw_config
        self.machine = FakeMachine(Path("sessions"), Path("data"))
        self.output = Screen()
        self.code: int | None = None
        self.app: ConsoleApp | None = None
        self.raw["generator"]["folder"] = "generator"
        install_generator_files(self.folder / "generator", self.raw["generator"]["model"])
        place_focus_model(self.folder)

    def add_recording(self, name: str, protocol: list | None = None, dataset="simulated") -> Path:
        """Put a recording of synthetic signal into a dataset of the data folder."""
        protocol = protocol or LIVE_PROTOCOL
        blocks = [tuple(block) for block in protocol]
        folder = self.folder / "data" / dataset / name
        return write_recording(folder, synthetic_signals(protocol), blocks)

    def play_recording(self, protocol=None, finished="finished", while_running=()):
        """With the headset off: play a recording of a dataset as session S01, wait until
        ``finished`` is printed, go back to the sources and quit. The exit code."""
        self.machine.headsets_on = set()
        self.machine.paced = self.machine.paced or bool(while_running)
        self.add_recording("short", protocol)
        after = self.shown(finished)
        name = ("S", "0", "1", keyboard.ENTER)
        return self.start(
            keyboard.ENTER, keyboard.ENTER, "y", *name, keyboard.SPACE, *while_running,
            after(keyboard.ENTER), keyboard.ENTER, "n",
        )  # fmt: skip

    def shown(self, text: str):
        """A scripted key that is "pressed" only once ``text`` has been printed."""

        def wait_then(key: str):
            def press() -> str:
                printed = f"{text!r} was never printed"
                wait_until(lambda: text in self.output.getvalue(), printed, KEY_WAIT_S)
                return key

            return press

        return wait_then

    def streaming(self, key: str):
        """A scripted key that is "pressed" once the running session has received samples.

        Keys pressed earlier, while the session is still starting, are discarded.
        """

        self.machine.paced = True

        def received() -> bool:
            return bool(self.machine.sources and self.machine.sources[-1].delivered)

        def press() -> str:
            wait_until(received, "no session received any sample", KEY_WAIT_S, SAMPLES_PAUSE_S)
            return key

        return press

    def start(self, *keys, write_config: bool = True, on: str = "file") -> int:
        """Run the console with the given keys; after them, [CTRL]+[C] is pressed.

        It prints ``on`` a "file", on a "terminal" (that is redrawn in place), or on
        the "standard output", through the console the program makes itself.
        """
        if write_config:
            write_yaml(self.raw, self.folder / "config.yaml")
        terminal = on == "terminal"
        output = Console(
            file=self.output, width=CONSOLE_WIDTH, force_terminal=terminal, color_system=None,
            highlight=False,
        )  # fmt: skip
        self.app = ConsoleApp(
            Path("config.yaml"),
            machine=self.machine,
            keys=Keyboard(ScriptedKeys(*keys)),
            output=None if on == "standard output" else output,
        )
        self.code = self.app.run()
        return self.code

    @property
    def text(self) -> str:
        return self.output.getvalue()

    def log(self) -> list[dict]:
        """Everything the console logged: what the sessions hold, then what no session took."""
        stored = [
            entry
            for folder in self.session_folders()
            for entry in read_lines(folder / c.FILE_CONSOLE)
        ]
        return sorted(stored, key=lambda entry: entry["t"]) + self.app.log.waiting

    def logged(self, kind: str) -> list[dict]:
        return [entry for entry in self.log() if entry["type"] == kind]

    def session_folders(self) -> list[Path]:
        return sorted((self.folder / "sessions").glob("*"))
