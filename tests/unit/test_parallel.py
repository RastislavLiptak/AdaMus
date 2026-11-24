"""parallel: work spread over the processors, as many processes as the computer has room for."""

import multiprocessing
import os
import signal
import time

import pytest

from adamus import parallel

INTERRUPT_AFTER_S = 1.0  # after the processes have started
LONG_WORK_S = 30  # more than a test may wait

GB = parallel.GIGABYTE


@pytest.mark.parametrize(
    ("processors", "gigabytes", "each", "processes"),
    [(10, 16, 2, 4), (10, 64, 2, 8), (4, 64, 2, 2), (2, 16, 2, 1), (10, 2, 2, 1), (8, 16, 1, 6)],
)
def test_processes_leave_two_processors_and_half_the_memory_free(
    processors, gigabytes, each, processes
):
    """A 16 GB Mac ran out of memory with eight processes of 2 GB; at least one always works."""
    assert parallel.processes_for(each * GB, processors, gigabytes * GB) == processes


@pytest.mark.parametrize(
    ("processors", "gigabytes", "each", "processes"),
    [(8, 16, 1, 8), (14, 48, 1, 14), (10, 16, 2, 6), (10, 64, 2, 10), (2, 1, 2, 1)],
)
def test_the_tests_take_every_processor_and_the_memory_the_system_does_not_need(
    processors, gigabytes, each, processes
):
    """They are to end as soon as the computer they run on can end them, whatever it is: a
    quarter of its memory stays to the system."""
    assert parallel.processes_for(each * GB, processors, gigabytes * GB, alone=True) == processes


def test_this_computer_is_measured():
    assert parallel.memory_of_the_computer() > GB


def added_to_the_shared(item):
    """Runs in a spawned process."""
    return parallel.shared() + item, os.getpid()


def test_every_item_is_done_in_processes_of_their_own_in_the_order_of_the_items():
    done = parallel.spread(added_to_the_shared, [3, 1, 2], GB, shared=10)
    assert [result for result, _ in done] == [13, 11, 12]
    assert os.getpid() not in {process for _, process in done}


def sleeps(seconds):
    time.sleep(seconds)
    return os.getpid()


def interrupted_after(seconds):
    """[CTRL]+[C], or the end of the app, ``seconds`` from now."""

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, seconds)


def test_an_interruption_ends_the_work_and_its_processes_at_once():
    """A training may take minutes: the end of the app must not wait for it."""
    before = signal.getsignal(signal.SIGALRM)
    began = time.monotonic()
    try:
        interrupted_after(INTERRUPT_AFTER_S)
        with pytest.raises(KeyboardInterrupt):
            parallel.spread(sleeps, [LONG_WORK_S] * 2, memory_each=1)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, before)
    assert time.monotonic() - began < LONG_WORK_S / 2
    assert multiprocessing.active_children() == []
