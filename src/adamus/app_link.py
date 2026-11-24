"""What the program and the app it runs under say to each other.

The app starts the program and reads its output: a line of JSON for each message. What else
would be printed there (by the program, a library or a C extension) goes to the error output,
which the app keeps to show when the program ends before it is ready. The program reads
nothing from its input: it ends when the input closes, which happens when the app ends in any
way, even when the app is killed.
"""

import json
import os
import sys
import threading
from collections.abc import Callable, Sequence
from typing import Any, TextIO

from adamus.preflight.checks import FAILED, HOW_TO_FIX, Check

CHECKS, READY = "checks", "ready"  # the kinds of message
APP_INPUT_THREAD = "app-input"  # the thread that watches the input


class AppMessages:
    """The messages to the app, each a line of JSON on ``output``."""

    def __init__(self, output: TextIO) -> None:
        self._output = output
        self._lock = threading.Lock()

    def checks(self, checks: Sequence[Check]) -> None:
        """The checklist as it is now; a check that failed says how to fix it."""
        rows = [
            {
                "name": check.name,
                "status": check.status,
                "detail": check.detail,
                "fix": HOW_TO_FIX.get(check.name) if check.status == FAILED else None,
            }
            for check in checks
        ]
        self._say({CHECKS: rows})

    def ready(self, address: str, token: str) -> None:
        """The API answers at ``address``, to calls that carry ``token``."""
        self._say({READY: {"address": address, "token": token}})

    def _say(self, message: dict[str, Any]) -> None:
        with self._lock:
            print(json.dumps(message), file=self._output, flush=True)


def messages_only_on_output() -> TextIO:
    """The output of the process, kept for the messages; everything else written to it from
    now on, also by C code, goes to the error output instead."""
    sys.stdout.flush()
    kept = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return kept


def watch_input(source: TextIO, on_end: Callable[[], None]) -> threading.Thread:
    """Call ``on_end`` once ``source`` closes; what comes in is not read as anything."""

    def watch() -> None:
        try:
            for _ in source:  # whatever the app sends, until it closes the input
                pass
        finally:  # also when the input cannot be read any more: the app is not heard from again
            on_end()

    thread = threading.Thread(target=watch, name=APP_INPUT_THREAD, daemon=True)
    thread.start()
    return thread
