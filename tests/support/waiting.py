"""Waiting in a test for what another thread or process does, for a limited time."""

import time

PAUSE_S = 0.005  # between two looks


def wait_until(condition, message: str, seconds: float, pause_s: float = PAUSE_S) -> None:
    """Return as soon as ``condition()`` holds; fail with ``message`` after ``seconds``."""
    deadline = time.monotonic() + seconds
    while True:  # until the condition holds, or the time is up
        if condition():
            return
        assert time.monotonic() < deadline, message
        time.sleep(pause_s)
