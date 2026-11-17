"""Single keys without Enter.

``readchar.readkey()`` blocks until a key arrives, while the main loop of a
session must keep running. A background thread therefore reads the keys and
passes each one on through a queue.
"""

import contextlib
import logging
import queue
import sys
import termios
import threading
from collections.abc import Callable
from typing import Any

import readchar

KEY_WAIT_S = 0.2  # how often a wait for a key wakes up
log = logging.getLogger(__name__)

NO_TERMINAL = (termios.error, ValueError, OSError)  # input that is not, or no longer, a terminal
UP = readchar.key.UP
DOWN = readchar.key.DOWN
ENTER = readchar.key.ENTER
SPACE = readchar.key.SPACE
BACKSPACE = readchar.key.BACKSPACE
INTERRUPT = "\x03"  # [CTRL]+[C] read as a character; also stands for a keyboard that is gone

# Terminals send either of two characters for these keys; readchar names one of each.
CARRIAGE_RETURN, BACKSPACE_CHARACTER = chr(13), chr(8)
_SAME_KEY = {CARRIAGE_RETURN: ENTER, BACKSPACE_CHARACTER: BACKSPACE}
_NAMES = {UP: "UP", DOWN: "DOWN", ENTER: "ENTER", SPACE: "SPACE", BACKSPACE: "BACKSPACE"}


def key_name(key: str) -> str:
    """How a key is written in the log, e.g. SPACE or Y."""
    return _NAMES.get(key, key)


class Keyboard:
    def __init__(self, read_key: Callable[[], str] = readchar.readkey) -> None:
        self._read_key = read_key
        self._keys: queue.Queue[str] = queue.Queue()
        self._terminal_settings: list[Any] | None = None

    def start(self) -> None:
        """Remember the terminal settings and start the reading thread."""
        with contextlib.suppress(*NO_TERMINAL):  # input is not a terminal
            self._terminal_settings = termios.tcgetattr(sys.stdin.fileno())
        threading.Thread(target=self._read_keys, name="keyboard", daemon=True).start()

    def restore_terminal(self) -> None:
        """Undo what the reading thread, still waiting for a key, did to the terminal.

        Without this the shell would not echo what is typed after the program ends.
        A terminal that went away meanwhile has nothing left to restore.
        """
        if self._terminal_settings is not None:
            with contextlib.suppress(*NO_TERMINAL):
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._terminal_settings)

    def next_key(self) -> str:
        """Wait for a key; [CTRL]+[C] raises KeyboardInterrupt."""
        while True:  # until a key arrives
            try:
                # Waking up regularly keeps the main thread responsive to signals.
                return self._delivered(self._keys.get(timeout=KEY_WAIT_S))
            except queue.Empty:
                continue

    def stop_requested(self) -> bool:
        """Has [SPACE] been pressed? Polled by the main loop of a running session."""
        for _ in range(self._keys.qsize()):
            if self._delivered(self._keys.get_nowait()) == SPACE:
                return True
        return False

    def discard_pressed_keys(self) -> None:
        """Forget the keys pressed so far; [CTRL]+[C] among them still raises KeyboardInterrupt."""
        for _ in range(self._keys.qsize()):
            self._delivered(self._keys.get_nowait())

    def _read_keys(self) -> None:
        while True:  # for as long as there is a keyboard
            try:
                self._keys.put(self._read_key())
            except KeyboardInterrupt:  # readkey raises it for the character \x03
                self._keys.put(INTERRUPT)
            except Exception:  # noqa: BLE001 - no keyboard: input closed or not a terminal
                log.warning("the keyboard can no longer be read", exc_info=True)
                self._keys.put(INTERRUPT)  # nobody can press a key any more: quit
                return

    @staticmethod
    def _delivered(key: str) -> str:
        if key == INTERRUPT:
            raise KeyboardInterrupt
        return _SAME_KEY.get(key, key)
