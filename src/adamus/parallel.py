"""Work spread over the processors of the computer the program runs on, whatever it has.

As many processes work at once as leave two processors to the person using the computer and
fit, each with the memory it needs, into half of its memory; the other half stays theirs. The
tests take the computer alone: all its processors, and all its memory the system does not
need, so that they end as soon as that computer can end them.
The processes are spawned, never forked: a forked Python that has loaded the libraries of
macOS crashes. What every process needs and is large (``shared``) is handed to each once,
when it starts, not with every item.
"""

import multiprocessing
import os
from collections.abc import Callable, Sequence
from typing import Any

FREE_PROCESSORS = 2  # left to the person using the computer
FREE_MEMORY_SHARE = 0.5  # likewise
SYSTEM_MEMORY_SHARE = 0.25  # left by the tests too: macOS and the apps open on it
GIGABYTE = 1024**3

_shared: list[Any] = []  # in a working process: what every process was handed when it started


def memory_of_the_computer() -> int:
    """Bytes of memory of the computer."""
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def processes_for(memory_each: int, processors: int, memory: int, *, alone: bool = False) -> int:
    """How many processes that take ``memory_each`` bytes each may work at once on a computer
    of ``processors`` and ``memory`` bytes: at least one. ``alone``: as the tests, not beside
    the person using it."""
    free_processors, free_share = (
        (0, SYSTEM_MEMORY_SHARE) if alone else (FREE_PROCESSORS, FREE_MEMORY_SHARE)
    )
    fit = int(memory * (1 - free_share)) // memory_each
    return max(1, min(processors - free_processors, fit))


def processes_here(memory_each: int, *, alone: bool = False) -> int:
    """processes_for on this computer."""
    return processes_for(memory_each, os.cpu_count() or 1, memory_of_the_computer(), alone=alone)


def _receive(shared: Any) -> None:
    _shared.append(shared)


def shared() -> Any:
    """In a working process: what spread() handed every process."""
    return _shared[0]


def spread[T, R](
    work: Callable[[T], R], items: Sequence[T], memory_each: int, shared: Any = None
) -> list[R]:
    """``work`` done on every item in processes of their own, the results in the order of the
    items. ``work`` lies at the top of a module, as a spawned process imports it from there."""
    count = min(len(items), processes_here(memory_each))
    # A pool that ends, its processes too, at once when the work is interrupted: the end of the
    # app does not wait for a training of minutes.
    with multiprocessing.get_context("spawn").Pool(
        max(1, count), initializer=_receive, initargs=(shared,)
    ) as pool:
        return pool.map(work, items)
