"""The log of what the program did around its sessions: one line per action.

The actions belong to the session they led to. Until that session has its
folder they wait in memory; from then on they are written to its activity.jsonl.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from adamus import constants as c

log = logging.getLogger(__name__)

# The actions of the program. The generator reports its own (music_generator).
ACTION_PREFLIGHT = "PREFLIGHT"
ACTION_HEADSET_LOOKED_FOR = "HEADSET_LOOKED_FOR"  # for the form of a new session: was it found
ACTION_SESSION_REQUESTED = "SESSION_REQUESTED"  # the source and the name the operator chose
ACTION_SESSION_STARTED = "SESSION_STARTED"
ACTION_START_FAILED = "START_FAILED"  # why the session requested did not run
ACTION_STOP_REQUESTED = "STOP_REQUESTED"
ACTION_APP_ENDED = "APP_ENDED"  # the app was closed while the session ran: it ended it
ACTION_PROGRAM_STOPPED = "PROGRAM_STOPPED"  # a signal ([CTRL]+[C]) ended it: not analysed
ACTION_SESSION_ENDED = "SESSION_ENDED"
ACTION_RECORDING_NOT_STORED = "RECORDING_NOT_STORED"  # what the headset delivered stays only here
ACTION_ANALYSIS_FAILED = "ANALYSIS_FAILED"  # the metrics of the session could not be computed


class ActivityLog:
    """Written by the session in the main thread and by the server of the API in its own."""

    def __init__(self) -> None:
        self.waiting: list[dict[str, Any]] = []  # actions no session has taken yet
        self._file: Path | None = None
        self._lock = threading.Lock()

    def write(self, kind: str, /, **fields: Any) -> None:
        """Log one action: the time, the kind of action and its fields."""
        action = {"t": time.time(), "type": kind, **fields}
        with self._lock:
            if self._file is None:
                self.waiting.append(action)
            else:
                self._append([action])

    def store_in(self, session_directory: Path) -> None:
        """The waiting actions and those that follow go to the folder of this session."""
        with self._lock:
            self._file = Path(session_directory) / c.FILE_ACTIVITY
            self._append(self.waiting)
            self.waiting = []

    def wait_for_the_next_session(self) -> None:
        """The actions that follow belong to the session that comes next."""
        with self._lock:
            self._file = None

    def _append(self, actions: list[dict[str, Any]]) -> None:
        """Into the file of the session; one that cannot be written (a full disk) is said in
        the log and lost, as the session and the program go on."""
        assert self._file is not None
        try:
            with self._file.open("a", encoding="utf-8") as handle:
                for action in actions:
                    print(json.dumps(action), file=handle)
        except OSError:
            log.exception("the activity of the session could not be written")
