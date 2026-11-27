"""An audio output that takes the music in real time, as a device would, and plays nothing:
in place of sounddevice's, so that a test runs the real player and makes no sound."""

import threading
import time
from collections.abc import Mapping, Sequence

import numpy as np
import sounddevice

from tests.support.stand_in_devices import StandInClock

OUTPUT = "Stand-in Speakers"
RATE = 48000
BLOCK = 512  # frames a call asks for, as the speakers of a Mac do
FAST = 50.0  # times as fast as the music plays, for a session that runs faster than time
TURN_S = 0.001  # of real time, between two looks at how far the music has played
WAIT_S = 10.0  # the longest a poll waits for the music: a stuck output fails the test, not hangs it


class PacedOutput:
    """An output stream that asks for a block of audio each time a block would have played,
    ``speed`` times as often as a real one."""

    speed = 1.0

    def __init__(self, samplerate, device, channels, dtype, latency, callback):
        self._callback = callback
        self._ended = threading.Event()
        self._thread = threading.Thread(target=self._play, name="stand-in-output", daemon=True)
        self.blocks = 0

    @property
    def played_s(self) -> float:
        return self.blocks * BLOCK / RATE

    def start(self):
        self._thread.start()

    def stop(self):
        self._ended.set()
        self._thread.join()

    def close(self):
        self._ended.set()

    def _play(self):
        """Ask for a block, then wait until it would have played; until stopped."""
        due = time.monotonic()
        while True:  # until stop()
            if self._ended.is_set():
                return
            out = np.zeros((BLOCK, 2), dtype=np.float32)
            try:
                self._callback(out, BLOCK, None, None)
            except sounddevice.CallbackAbort:
                return
            self.blocks += 1
            due += BLOCK / RATE / self.speed
            time.sleep(max(0.0, due - time.monotonic()))


class FastOutput(PacedOutput):
    """One that takes the music faster than it plays: a session of the tests runs faster than
    time, and its music should go on as far."""

    speed = FAST
    made: list["FastOutput"] = []

    def __init__(self, *arguments, **options):
        super().__init__(*arguments, **options)
        FastOutput.made.append(self)


ONE_OUTPUT = [{"name": OUTPUT, "max_output_channels": 2, "default_samplerate": float(RATE)}]


class MusicClock(StandInClock):
    """Stand-in time that, at each poll, waits until the last stand-in output made has played as
    far as the time has gone since it was made: the music of a session that runs faster than
    time keeps up with it however busy the computer is, as it does in real time."""

    def __init__(self) -> None:
        super().__init__()
        self._output: PacedOutput | None = None
        self._from = 0.0

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        if FastOutput.made and FastOutput.made[-1] is not self._output:
            self._output, self._from = FastOutput.made[-1], self.time
        output = self._output
        for _ in range(round(WAIT_S / TURN_S)):
            if output is None or output.played_s >= self.time - self._from:
                return
            time.sleep(TURN_S)


def stand_in_output(
    patcher, output: type = PacedOutput, devices: Sequence[Mapping[str, object]] = ONE_OUTPUT
) -> None:
    """Have sounddevice offer ``devices``, and play on them with ``output``."""
    patcher.setattr(sounddevice, "OutputStream", output)
    patcher.setattr(sounddevice, "_terminate", lambda: None)
    patcher.setattr(sounddevice, "_initialize", lambda: None)
    patcher.setattr(
        sounddevice,
        "query_devices",
        lambda index=None: devices if index is None else devices[index],
    )
