"""The log of what the console did: one line per action.

The actions belong to the session they led to. Until that session has its
folder they wait in memory; from then on they are written to its console.jsonl.
"""

import json
import time
from pathlib import Path
from typing import Any

from adamus import constants as c

# The actions of the console. The generator reports its own (music_generator).
ACTION_PREFLIGHT = "PREFLIGHT"
ACTION_SOURCE_SELECTED = "SOURCE_SELECTED"
ACTION_SESSION_NAMED = "SESSION_NAMED"
ACTION_KEY = "KEY"
ACTION_SESSION_STARTED = "SESSION_STARTED"
ACTION_SESSION_ENDED = "SESSION_ENDED"
ACTION_SESSION_CANCELLED = "SESSION_CANCELLED"
ACTION_QUIT = "QUIT"


class ConsoleLog:
    def __init__(self) -> None:
        self.waiting: list[dict[str, Any]] = []  # actions no session has taken yet
        self._file: Path | None = None

    def write(self, kind: str, /, **fields: Any) -> None:
        """Log one action: the time, the kind of action and its fields."""
        action = {"t": time.time(), "type": kind, **fields}
        if self._file is None:
            self.waiting.append(action)
        else:
            self._append([action])

    def store_in(self, session_directory: Path) -> None:
        """The waiting actions and those that follow go to the folder of this session."""
        self._file = Path(session_directory) / c.FILE_CONSOLE
        self._append(self.waiting)
        self.waiting = []

    def wait_for_the_next_session(self) -> None:
        """The actions that follow belong to the session that comes next."""
        self._file = None

    def _append(self, actions: list[dict[str, Any]]) -> None:
        assert self._file is not None
        with self._file.open("a", encoding="utf-8") as handle:
            for action in actions:
                print(json.dumps(action), file=handle)
