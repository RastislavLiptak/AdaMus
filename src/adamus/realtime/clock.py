"""The time a session runs on: the real one, and what a stand-in must offer in its place."""

import time
from typing import Protocol


class TimeSource(Protocol):
    """The time of day for the record, and a monotonic time for durations."""

    def now(self) -> float: ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class Clock:
    """The time of day for the record, and a monotonic time for durations: the time of day
    jumps when the system clock is set."""

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)
