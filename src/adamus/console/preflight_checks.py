"""The checklist run when the console starts.

The checks run top to bottom. A required check that fails ends the program;
an optional one (the datasets, the headset) may find nothing. What the program obtains
itself (the focus detector, the music generator) is made when it is missing,
always the same way: its output goes to a log in its folder, and then it is
looked at again.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from rich.console import RenderableType

from adamus import constants as c
from adamus import datasets
from adamus.config import Config, Generator, load_config
from adamus.console import data_sources, printed_texts
from adamus.console.console_log import ACTION_PREFLIGHT, ConsoleLog
from adamus.console.data_sources import Source
from adamus.console.machine import Machine
from adamus.detection.focus_detector import load_models
from adamus.detection.focus_training import DATASET_URL
from adamus.errors import AdamusError
from adamus.realtime.music_generator import PATCH_TEMPLATE, MusicGenerator

CONFIGURATION = "Configuration"
STORAGE = "Storage"
FOCUS_DETECTOR = "Focus detector"
AUDIO_OUTPUT = "Audio output"
GENERATOR_FILES = "Generator files"
GENERATOR_START = "Generator start"
DATASETS = "Datasets"
LIVE_HEADSET = "Live headset"
DATA_SOURCE = "Data source available"

TAKES_A_WHILE = " (this takes a while) ..."
SKIPPED, NONE_FOUND = "skipped", "none found"  # what such a check shows

HOW_TO_FIX = {
    CONFIGURATION: f"Fix the configuration file ({c.FILE_CONFIG} in the repository lists"
    " every key).",
    STORAGE: "Start the program in a folder you can write to.",
    FOCUS_DETECTOR: "Read the log named above; the first training downloads its data from"
    f" {urlparse(DATASET_URL).hostname}, so it needs the internet.",
    AUDIO_OUTPUT: f"Set the output device to {c.AUDIO_RATE} Hz in Audio MIDI Setup.",
    GENERATOR_FILES: "Read the installation log named above; the installation needs Xcode with"
    " its Metal toolchain, cmake, git and curl. Check generator.folder and model.",
    DATA_SOURCE: f"Switch the headset on, or put a dataset into {c.DATASETS_DIR}/.",
}


class Display(Protocol):
    """Where the checklist is shown as it changes."""

    def update(self, renderable: RenderableType) -> None: ...


@dataclass
class Check:
    name: str
    status: str = c.CHECK_PENDING
    detail: str = ""


def _focus_model_problem() -> str:
    """Why the trained focus models cannot be used; '' when they can."""
    try:
        load_models()
    except AdamusError as problem:
        return str(problem)
    return ""


class Preflight:
    """Runs the checks and keeps what they found: configuration, generator, sources."""

    def __init__(
        self, config_path: Path, machine: Machine, display: Display, console_log: ConsoleLog
    ) -> None:
        self._config_path = config_path
        self._machine = machine
        self._display = display  # shows the checklist as it changes
        self._console_log = console_log
        self.checks = [
            Check(name)
            for name in (
                CONFIGURATION, STORAGE, FOCUS_DETECTOR, AUDIO_OUTPUT, GENERATOR_FILES,
                GENERATOR_START, DATASETS, LIVE_HEADSET, DATA_SOURCE,
            )
        ]  # fmt: skip
        self.config: Config | None = None
        self.generator: MusicGenerator | None = None
        self.headset_found = False
        self.datasets: list[Path] = []

    def run(self) -> bool:
        """Run all checks; True when every required one passed."""
        self._display.update(printed_texts.checklist(self.checks))
        self._require(CONFIGURATION, self._configuration, [])
        self._require(STORAGE, self._storage, [])
        self._require(FOCUS_DETECTOR, self._focus_detector, [])
        self._require(AUDIO_OUTPUT, self._audio_output, [])
        self._require(GENERATOR_FILES, self._generator_files, [CONFIGURATION])
        self._require(
            GENERATOR_START, self._generator_start, [AUDIO_OUTPUT, GENERATOR_FILES],
            running="loading model ...",
        )  # fmt: skip
        self._look_for_datasets()
        self._look_for_headset()
        self._require(DATA_SOURCE, self._data_source, [CONFIGURATION])
        return all(check.status in (c.CHECK_PASSED, c.CHECK_NONE_FOUND) for check in self.checks)

    def rescan_headset(self) -> None:
        """Repeat the headset check for the list of sources; the checklist is no longer shown."""
        assert self.config is not None
        check, started = Check(LIVE_HEADSET), time.monotonic()
        self._probe(check)
        self._log_result(check, started, headset=self.config.board.name)

    def rescan_datasets(self) -> None:
        """The datasets again, for the list of sources: a session may have added one."""
        self.datasets = datasets.datasets_in(self._machine.datasets_root)

    def sources(self) -> list[Source]:
        """What a session can run from: the headset, when it was found, and the datasets."""
        found = self.config.board if self.config is not None and self.headset_found else None
        return data_sources.available_sources(found, self.datasets)

    def failed_checks(self) -> list[Check]:
        return [check for check in self.checks if check.status == c.CHECK_FAILED]

    def _check(self, name: str) -> Check:
        return next(check for check in self.checks if check.name == name)

    def _require(
        self, name: str, check_method: Callable[[], str], needs: list[str], running: str = ""
    ) -> None:
        """Run one required check, unless a check it needs did not pass.

        ``check_method`` returns what it found, or raises with the cause; ``running`` is
        what the check shows meanwhile.
        """
        check = self._check(name)
        if any(self._check(needed).status != c.CHECK_PASSED for needed in needs):
            self._finish(check, c.CHECK_SKIPPED, SKIPPED, time.monotonic())
            return
        started = self._start(check, running)
        try:
            detail = check_method()
        except (AdamusError, OSError) as problem:
            self._finish(check, c.CHECK_FAILED, str(problem), started)
            return
        self._finish(check, c.CHECK_PASSED, detail, started)

    def _start(self, check: Check, detail: str = "") -> float:
        check.status, check.detail = c.CHECK_RUNNING, detail
        self._display.update(printed_texts.checklist(self.checks))
        return time.monotonic()

    def _finish(self, check: Check, status: str, detail: str, started: float, **more: Any) -> None:
        check.status, check.detail = status, detail
        self._display.update(printed_texts.checklist(self.checks))
        self._log_result(check, started, **more)

    def _log_result(self, check: Check, started: float, **more: Any) -> None:
        self._console_log.write(
            ACTION_PREFLIGHT,
            check=check.name,
            result=check.status,
            detail=check.detail,
            duration=time.monotonic() - started,
            **more,
        )

    # --- The checks ----------------------------------------------------------

    def _configuration(self) -> str:
        self.config = load_config(self._config_path)
        return f"{self._config_path.name}, {len(self.config.protocol)} protocol blocks"

    def _storage(self) -> str:
        root = self._machine.sessions_root
        root.mkdir(parents=True, exist_ok=True)
        trial = root / ".write-test"
        trial.write_text("ok", encoding="utf-8")
        trial.unlink()
        return f"{root}/ writable"

    def _audio_output(self) -> str:
        name, rate = self._machine.audio_output()
        if rate != c.AUDIO_RATE:
            raise AdamusError(f"{name}, {rate:g} Hz (need {c.AUDIO_RATE} Hz)")
        return f"{name}, {c.AUDIO_RATE} Hz"

    def _obtained(
        self, name: str, doing: str, missing: Callable[[], str], make: Callable[[], Path]
    ) -> bool:
        """Make what the check ``name`` looks at when ``missing`` names a problem, and look
        at it again. True when it had to be made."""
        if not missing():
            return False
        self._check(name).detail = doing + TAKES_A_WHILE
        self._display.update(printed_texts.checklist(self.checks))
        log_file = make()
        problem = missing()
        if problem:
            raise AdamusError(f"still missing after the setup ({log_file}): {problem}")
        return True

    def _focus_detector(self) -> str:
        """The trained focus models; trained first when they are missing or do not fit."""
        made = self._obtained(
            FOCUS_DETECTOR, "training the focus detector", _focus_model_problem,
            self._machine.train_focus_detector,
        )  # fmt: skip
        people = len(load_models().raw["trained_on"]["people"])
        return ("trained: " if made else "") + f"{c.FOCUS_MODEL_PATH}, {people} people"

    def _generator_files(self) -> str:
        """The files of the generator; missing ones are installed first."""
        assert self.config is not None
        generator = self.config.generator
        made = self._obtained(
            GENERATOR_FILES, "installing the music generator",
            lambda: self._missing_generator_files(generator),
            lambda: self._machine.install_generator(generator),
        )  # fmt: skip
        return ("installed: " if made else "") + ", ".join(self._generator_paths(generator))

    @staticmethod
    def _generator_paths(generator: Generator) -> dict[str, Path]:
        return {
            "pd": generator.pd_binary,
            "mrt2~": generator.external_file,
            "patch": PATCH_TEMPLATE,
            generator.model: generator.model_file,
        }

    def _missing_generator_files(self, generator: Generator) -> str:
        """The files of the generator that are not there, as a list; '' when all are."""
        files = [*self._generator_paths(generator).values(), *generator.asset_files]
        return ", ".join(str(path) for path in files if not path.is_file())

    def _generator_start(self) -> str:
        assert self.config is not None
        generator = self._machine.make_generator(self.config, self._console_log.write)
        generator.start()
        self.generator = generator
        return "model loaded"

    def _look_for_datasets(self) -> None:
        check = self._check(DATASETS)
        started = self._start(check)
        root = self._machine.datasets_root
        self.datasets = datasets.datasets_in(root)
        if self.datasets:
            self._finish(check, c.CHECK_PASSED, f"{len(self.datasets)} in {root}/", started)
        else:
            self._finish(check, c.CHECK_NONE_FOUND, NONE_FOUND, started)

    def _look_for_headset(self) -> None:
        """Whether the headset of the configuration can be reached."""
        check = self._check(LIVE_HEADSET)
        if self.config is None:
            self._finish(check, c.CHECK_SKIPPED, SKIPPED, time.monotonic())
            return
        started = self._start(check)
        self._probe(check)
        self._finish(check, check.status, check.detail, started, headset=self.config.board.name)

    def _probe(self, check: Check) -> None:
        """Try to reach the headset; note the answer in ``check`` and in ``headset_found``."""
        assert self.config is not None
        headset = self.config.board
        self.headset_found = self._machine.headset_is_available(headset)
        check.status = c.CHECK_PASSED if self.headset_found else c.CHECK_NONE_FOUND
        check.detail = headset.name if self.headset_found else NONE_FOUND

    def _data_source(self) -> str:
        count = len(self.sources())
        if count == 0:
            raise AdamusError("no headset and no dataset")
        return f"{count} source" + ("s" if count != 1 else "")
