"""The latest samples of a session, in an array of fixed size, as the main loop needs them."""

import numpy as np


class RecentSamples:
    """The latest samples of the rows the loop needs, in an array of fixed size.

    Samples are appended at most one window at a time, and every step that
    became ready is processed before the next append. The oldest step still
    waiting therefore ends after the samples held before the append: less than
    one window back, with the signal it is computed from (its context) reaching
    less than one window plus that context back. The array holds them, and one
    hop to spare.
    """

    def __init__(self, rows: int, window: int, hop: int, context: int) -> None:
        assert window <= context
        self.room = window  # the most samples one append may bring
        self.count = 0  # samples appended since the stream started
        self._data = np.zeros((rows, window + context + hop))

    def append(self, chunk: np.ndarray) -> None:
        size = chunk.shape[1]
        assert size <= self.room, "append at most one window at a time"
        if size > 0:
            self._data[:, :-size] = self._data[:, size:]
            self._data[:, -size:] = chunk
            self.count += size

    def window(self, n_end: int, length: int) -> np.ndarray:
        """Samples [n_end - length, n_end) of the stream."""
        behind = self.count - n_end
        capacity = self._data.shape[1]
        assert 0 <= behind <= capacity - length, "the window is no longer (or not yet) held"
        return self._data[:, capacity - behind - length : capacity - behind]
