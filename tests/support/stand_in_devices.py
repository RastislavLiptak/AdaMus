"""Stand-ins for the devices a session talks to: EEG source, generator, clock, keys."""

import json
from pathlib import Path

import numpy as np
from brainflow.board_shim import BoardShim

from adamus.config import Board
from adamus.errors import AdamusError
from adamus.realtime.eeg_source import electrodes_of_headset

MUSE_2 = 38  # BrainFlow board id
NUDGE_S = 1e-6  # every reading of the fake clock moves it on by this much
CHUNK = 64  # samples a poll of a stand-in source delivers at most


class FakeClock:
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


class FakeSource:
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
        self.is_playback = not live
        self.playback_file = None if live else Path("fake.csv")
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
        np.savetxt(self.raw_csv, self._matrix[:, : self._produced].T, delimiter="\t", fmt="%.6f")


class FakeGenerator:
    """Records every message a session sends."""

    pd_version = "Pd-0.0-fake"

    def __init__(
        self, fail_after_steps: int | None = None, fail_to_start: str | tuple[str, ...] = ""
    ) -> None:
        self.started = self.stopped = 0
        self._fail_to_start = fail_to_start
        self.messages: list[str] = []
        self.stderr_copies: list[Path | None] = []
        self._fail_after_steps = fail_after_steps

    def start(self) -> None:
        if self._fail_to_start:
            lines = self._fail_to_start
            raise AdamusError(*([lines] if isinstance(lines, str) else lines))
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def send(self, message: str) -> None:
        is_step = message.count("prompt") == 2
        limit = self._fail_after_steps
        if is_step and limit is not None and len(self.weights) >= limit:
            raise OSError("broken pipe")
        self.messages.append(message)

    def copy_stderr_to(self, path: Path | None) -> None:
        self.stderr_copies.append(path)

    @property
    def weights(self) -> list[tuple[float, float]]:
        """(w_idle, w_focus) of every step message, as sent (three decimals)."""
        steps = [message for message in self.messages if message.count("prompt") == 2]
        return [
            tuple(float(prompt.split()[-1]) for prompt in step.split(";")[:2]) for step in steps
        ]


class StopKey:
    """A keyboard on which [SPACE] is pressed once ``pressed()`` says so."""

    def __init__(self, pressed=lambda: False, events_file=None) -> None:
        self._pressed = pressed
        self._events_file = events_file
        self.discarded_at: list[str] = []  # the last event written when keys were discarded

    def stop_requested(self) -> bool:
        return self._pressed()

    def discard_pressed_keys(self) -> None:
        last_event = ""
        if self._events_file is not None:
            last_event = json.loads(self._events_file().read_text().splitlines()[-1])["type"]
        self.discarded_at.append(last_event)


class ScriptedKeys:
    """Stands in for ``readchar.readkey``: plays the given keys, then presses [CTRL]+[C].

    An item of the script may also be a function; it is called (to wait for
    something, for example) and its result is the key.
    """

    def __init__(self, *script, interrupt_at_end: bool = True) -> None:
        self._script = iter(script)
        self._interrupt_at_end = interrupt_at_end

    def __call__(self) -> str:
        for item in self._script:
            return item() if callable(item) else item
        if self._interrupt_at_end:
            self._interrupt_at_end = False
            raise KeyboardInterrupt
        import threading

        threading.Event().wait()  # the keyboard stays silent from now on
        return ""
