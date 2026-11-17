"""recent_samples.RecentSamples: the buffer the main loop cuts the signal of a step from."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from adamus.realtime.recent_samples import RecentSamples

N, H = 16, 4
CONTEXT = 2 * N  # a step is computed from its window and the window before it


def stream(count, rows=2):
    """Row 0 counts samples, row 1 counts them negatively."""
    index = np.arange(count, dtype=np.float64)
    return np.vstack([index, -index])[:rows]


def buffer():
    return RecentSamples(2, N, H, CONTEXT)


def test_window_returns_the_requested_samples_of_the_stream():
    samples = buffer()
    samples.append(stream(N))
    samples.append(stream(2 * N)[:, N:])
    assert samples.count == 2 * N
    assert np.array_equal(samples.window(2 * N, CONTEXT), stream(2 * N))
    assert np.array_equal(samples.window(2 * N, N), stream(2 * N)[:, N:])


def test_older_contexts_stay_available_for_steps_not_yet_processed():
    samples = buffer()
    data = stream(3 * N + H)
    for start in (0, N, 2 * N, 3 * N):
        samples.append(data[:, start : start + N])
    # The oldest step that may still wait ends less than one append (N) before the end,
    # and its context reaches CONTEXT further back: N + H + CONTEXT samples in all.
    oldest_end = 3 * N + H - N
    assert np.array_equal(
        samples.window(oldest_end, CONTEXT), data[:, oldest_end - CONTEXT : oldest_end]
    )
    assert np.array_equal(samples.window(3 * N + H, CONTEXT), data[:, N + H :])


def test_empty_append_changes_nothing():
    samples = buffer()
    samples.append(stream(N))
    samples.append(np.empty((2, 0)))
    assert samples.count == N and np.array_equal(samples.window(N, N), stream(N))


def test_at_most_one_window_may_be_appended_at_a_time():
    samples = buffer()
    assert samples.room == N
    with pytest.raises(AssertionError, match="at most one window"):
        samples.append(stream(N + 1))


def test_context_older_than_the_buffer_is_refused():
    samples = buffer()
    for _ in range(6):
        samples.append(stream(N))  # 6 N samples; the buffer holds the last N + CONTEXT + H
    held = N + CONTEXT + H
    samples.window(6 * N - held + CONTEXT, CONTEXT)  # the oldest context still held
    with pytest.raises(AssertionError, match="no longer"):
        samples.window(6 * N - held + CONTEXT - 1, CONTEXT)


def test_window_beyond_the_received_samples_is_refused():
    samples = buffer()
    samples.append(stream(N))
    with pytest.raises(AssertionError, match="not yet"):
        samples.window(N + 1, N)


def test_context_must_hold_at_least_a_window():
    with pytest.raises(AssertionError):
        RecentSamples(2, N, H, N - 1)


@given(st.lists(st.integers(0, N), min_size=1, max_size=30))
def test_every_step_sees_its_slice_of_the_whole_stream(sizes):
    """However the samples arrive, step k sees samples [k*H, k*H + CONTEXT) of the stream."""
    data = stream(sum(sizes))
    samples = buffer()
    done, position = 0, 0
    for size in sizes:
        samples.append(data[:, position : position + size])
        position += size
        ready = (samples.count - CONTEXT) // H + 1
        for k in range(done, ready):
            seen = samples.window(CONTEXT + k * H, CONTEXT)
            assert np.array_equal(seen, data[:, k * H : k * H + CONTEXT])
        done = max(done, ready)
    assert done == max(0, (position - CONTEXT) // H + 1)
