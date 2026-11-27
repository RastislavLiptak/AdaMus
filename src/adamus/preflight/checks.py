"""The checklist run when the program starts, which the app shows as it goes.

The checks run top to bottom; one that fails ends the program. What the program obtains
itself (the focus detector) is made when it is missing: its output goes to a log in its
folder, and then it is looked at again. The sources of a session (the headset, the datasets),
its song and the audio output it plays on are not checked here: the form of a new session
offers them as they are when it is filled in (server/new_session.py).
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from adamus import constants as c
from adamus.activity_log import ACTION_PREFLIGHT, ActivityLog
from adamus.config import Config, load_config
from adamus.detection.focus_detector import load_models
from adamus.detection.focus_training import DATASET_URL
from adamus.errors import AdamusError
from adamus.machine import Machine

CONFIGURATION = "Configuration"
STORAGE = "Storage"
FOCUS_DETECTOR = "Focus detector"

# The states of a check.
PENDING, RUNNING, PASSED, FAILED = "pending", "running", "passed", "failed"
SKIPPED = "skipped"  # a check it needs did not pass; it is also what such a check shows

TAKES_A_WHILE = " (this takes a while) ..."

HOW_TO_FIX = {
    CONFIGURATION: f"Fix the configuration file ({c.FILE_CONFIG} in the repository lists"
    " every key).",
    STORAGE: "Start the program in a folder you can write to.",
    FOCUS_DETECTOR: "Read the log named above; the first training downloads its data from"
    f" {urlparse(DATASET_URL).hostname}, so it needs the internet.",
}


@dataclass
class Check:
    name: str
    status: str = PENDING
    detail: str = ""


#: Told the checklist whenever it changes.
Shown = Callable[[Sequence[Check]], None]


def _focus_model_problem() -> str:
    """Why the trained focus models cannot be used; '' when they can."""
    try:
        load_models()
    except AdamusError as problem:
        return str(problem)
    return ""


class Preflight:
    """Runs the checks and keeps what they found: the configuration."""

    def __init__(
        self, config_path: Path, machine: Machine, shown: Shown, activity_log: ActivityLog
    ) -> None:
        self._config_path = config_path
        self._machine = machine
        self._shown = shown
        self._activity_log = activity_log
        self.checks = [Check(name) for name in (CONFIGURATION, STORAGE, FOCUS_DETECTOR)]
        self.config: Config | None = None

    def run(self) -> bool:
        """Run all checks; True when every required one passed."""
        self._show()
        self._require(CONFIGURATION, self._configuration, [])
        self._require(STORAGE, self._storage, [])
        self._require(FOCUS_DETECTOR, self._focus_detector, [CONFIGURATION])
        return all(check.status == PASSED for check in self.checks)

    def _check(self, name: str) -> Check:
        return next(check for check in self.checks if check.name == name)

    def _require(self, name: str, check_method: Callable[[], str], needs: list[str]) -> None:
        """Run one required check, unless a check it needs did not pass.

        ``check_method`` returns what it found, or raises with the cause.
        """
        check = self._check(name)
        if any(self._check(needed).status != PASSED for needed in needs):
            self._finish(check, SKIPPED, SKIPPED, time.monotonic())
            return
        check.status, check.detail = RUNNING, ""
        self._show()
        started = time.monotonic()
        try:
            detail = check_method()
        except (AdamusError, OSError) as problem:
            self._finish(check, FAILED, str(problem), started)
            return
        self._finish(check, PASSED, detail, started)

    def _show(self) -> None:
        self._shown(self.checks)

    def _finish(self, check: Check, status: str, detail: str, started: float) -> None:
        check.status, check.detail = status, detail
        self._show()
        self._log_result(check, started)

    def _log_result(self, check: Check, started: float) -> None:
        self._activity_log.write(
            ACTION_PREFLIGHT,
            check=check.name,
            result=check.status,
            detail=check.detail,
            duration=time.monotonic() - started,
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

    def _obtained(
        self, name: str, doing: str, missing: Callable[[], str], make: Callable[[], Path]
    ) -> bool:
        """Make what the check ``name`` looks at when ``missing`` names a problem, and look
        at it again. True when it had to be made."""
        if not missing():
            return False
        self._check(name).detail = doing + TAKES_A_WHILE
        self._show()
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
