"""Stand-ins for what a session talks to: EEG source, generator, clock, stop request."""

from pathlib import Path

import numpy as np
from brainflow.board_shim import BoardShim

from adamus.config import Board
from adamus.errors import AdamusError
from adamus.realtime.eeg_source import electrodes_of_headset

MUSE_2 = 38  # BrainFlow board id
NUDGE_S = 1e-6  # every reading of the stand-in clock moves it on by this much
CHUNK = 64  # samples a poll of a stand-in source delivers at most


class StandInClock:
    """Deterministic time: ``sleep`` advances it, every reading nudges it forward.

    ``wall_offset`` moves the time of day only, as when the system clock is set:
    the monotonic time goes on as before.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        self.time = start
        self.wall_offset = 0.0

    def now(self) -> float:
        self.time += NUDGE_S
        return self.time + self.wall_offset

    def monotonic(self) -> float:
        self.time += NUDGE_S
        return self.time

    def sleep(self, seconds: float) -> None:
        self.time += seconds


class StandInSource:
    """Stands in for the headset: delivers a prepared board matrix in chunks, without waiting.

    When the stream is stopped, ``held_back`` more samples are still handed
    over, like samples that reach a real board between the last poll of the
    main loop and the end of the stream.
    """

    def __init__(self, matrix, chunk=CHUNK, live=False, held_back=0, board_id=MUSE_2):
        board = BoardShim.get_board_descr(board_id)
        self.board_id = board_id
        self.fs = board["sampling_rate"]
        names = electrodes_of_headset(Board("stand-in", board_id, ""))
        self.electrode_rows = dict(zip(names, board["eeg_channels"], strict=False))
        self.timestamp_row = board["timestamp_channel"]
        self.marker_row = board["marker_channel"]
        self.playback_file = None if live else Path("stand_in.csv")
        self.markers: list[tuple[int, float]] = []  # (samples delivered, value)
        self.raw_csv: Path | None = None
        self.stopped = False
        self.closed = False
        self._matrix = np.array(matrix, dtype=np.float64)
        self._chunk = chunk
        self._held_back = held_back
        self._produced = self._matrix.shape[1] - held_back  # what the board produces while polled
        self._delivered = 0
        self._pending_marker: float | None = None

    def start(self, raw_csv: Path) -> None:
        self.raw_csv = Path(raw_csv)

    def read(self) -> np.ndarray:
        end = self._produced if self.stopped else min(self._delivered + self._chunk, self._produced)
        block = self._matrix[:, self._delivered : end]
        if block.shape[1] and self._pending_marker is not None:
            block[self.marker_row, 0] = self._pending_marker
            self._pending_marker = None
        self._delivered = end
        return block.copy()

    @property
    def delivered(self) -> int:
        """Samples handed over so far."""
        return self._delivered

    def insert_marker(self, value: float) -> None:
        self.markers.append((self._delivered, value))
        self._pending_marker = value

    def close(self) -> None:
        self.closed = True

    def stop(self) -> None:
        self.stopped = True
        self._produced = min(self._delivered + self._held_back, self._matrix.shape[1])
        # Like the BrainFlow streamer: every sample the board produced is in the
        # file, one per line, tab-separated, with six decimals.
        assert self.raw_csv is not None, "stopped after it started"
        np.savetxt(self.raw_csv, self._matrix[:, : self._produced].T, delimiter="\t", fmt="%.6f")


class StandInGenerator:
    """Records every message a session sends."""

    pd_version = "Pd-0.0-stand-in"

    def __init__(
        self, fail_after_steps: int | None = None, fail_to_start: str | tuple[str, ...] = ""
    ) -> None:
        self.started = self.stopped = 0
        self._running = False
        self._fail_to_start = fail_to_start
        self.messages: list[str] = []
        self.stderr_copies: list[Path | None] = []
        self._fail_after_steps = fail_after_steps

    def start(self) -> None:
        if self._fail_to_start:
            lines = self._fail_to_start
            raise AdamusError(*([lines] if isinstance(lines, str) else lines))
        self.started += 1
        self._running = True

    def stop(self) -> None:
        """Stopped once, as Pd ends once: a second stop does nothing."""
        if self._running:
            self.stopped += 1
        self._running = False

    def send(self, message: str) -> None:
        is_step = message.count("prompt") == 2
        limit = self._fail_after_steps
        if is_step and limit is not None and len(self.weights) >= limit:
            raise OSError("broken pipe")
        self.messages.append(message)

    def copy_stderr_to(self, session_log: Path | None) -> None:
        self.stderr_copies.append(session_log)

    @property
    def weights(self) -> list[tuple[float, float]]:
        """(w_idle, w_focus) of every step message, as sent (three decimals)."""
        steps = [message for message in self.messages if message.count("prompt") == 2]
        pairs = [step.split(";")[:2] for step in steps]
        return [(float(idle.split()[-1]), float(focus.split()[-1])) for idle, focus in pairs]


class StopWhen:
    """A stop request that is set as soon as ``condition()`` says so."""

    def __init__(self, condition=lambda: False) -> None:
        self._condition = condition

    def is_set(self) -> bool:
        return self._condition()
