"""Plays a song of the library on an audio output, as the sequencer chooses its parts, and
records what it plays. It plays its own copy of the song, which the session keeps: the library
may change the song, or delete it, while it plays and later.

The output asks for audio in its own thread (sounddevice's callback): the sequencer renders it
there, and a thread of the player writes the same audio into the session's recording, so that
the output never waits for the disk. A failure in the output's thread is kept and told by
``problem``, and so is an output that has gone from the Mac, as Core Audio tells it. Each block
the output had to play before it was ready (a dropout) is counted.
"""

import contextlib
import queue
import threading
import time
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice
import soundfile

from adamus.errors import AdamusError
from adamus.realtime.song_sequencer import Decision, Sequencer
from adamus.song_library import CHANNELS, Library, analysis_in, audio_in, decode, song_in

RECORDING_FORMAT = "PCM_16"
# The output's buffer. Its own lowest (28 ms here) would leave a 10 ms turn for each block, and
# the session's own work may hold Python longer than that; a jump waits for a beat anyway.
LATENCY_S = 0.1
WATCH_S = 1.0  # how often the output is looked for while the music plays


def output_index(name: str) -> int:
    """PortAudio's number of the output ``name``. PortAudio lists the devices once, when it
    starts, so it is started again first: headphones plugged in since are then there too. The
    player plays one song at a time, so no stream is open meanwhile."""
    # how sounddevice's own documentation reads the devices again
    sounddevice._terminate()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    sounddevice._initialize()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    for index, device in enumerate(sounddevice.query_devices()):
        if device["name"] == name and device["max_output_channels"] >= CHANNELS:
            return index
    raise AdamusError(f"{name} is not connected, or does not play in stereo")


class SongPlayer:
    """One song played once, on one output, into one recording."""

    def __init__(
        self, library: Library, song_id: str, output: str, outputs: Callable[[], Collection[str]]
    ) -> None:
        self.output = output
        self._library = library
        self._song_id = song_id
        self._outputs = outputs  # the names of the outputs the Mac has now
        self._sequencer: Sequencer | None = None
        self._stream: sounddevice.OutputStream | None = None
        self._recording: queue.SimpleQueue[np.ndarray | None] = queue.SimpleQueue()
        self._writer: threading.Thread | None = None
        self._failure: str | None = None  # of the output's thread
        self.dropouts = 0  # blocks the output wanted before they were ready
        self._looked_at = 0.0  # on the monotonic clock
        self._lost = False

    def start(self, recording: Path, song: Path) -> float:
        """Copy the song into the new folder ``song`` and begin it at rest, recorded into
        ``recording``; when the first frame was handed to the output (time of day)."""
        index = output_index(self.output)
        rate = int(sounddevice.query_devices(index)["default_samplerate"])
        self._library.copy(self._song_id, song)
        audio = decode(audio_in(song), rate)
        self._sequencer = Sequencer(audio, rate, analysis_in(song), song_in(song).marks)
        sound = soundfile.SoundFile(recording, "w", rate, CHANNELS, RECORDING_FORMAT)
        self._writer = threading.Thread(target=self._write, args=(sound,), name="music-recording")
        self._writer.start()
        self._stream = sounddevice.OutputStream(
            samplerate=rate, device=index, channels=CHANNELS, dtype="float32", latency=LATENCY_S,
            callback=self._play,
        )  # fmt: skip
        self._stream.start()
        return time.time()

    def want(self, mark: str) -> None:
        assert self._sequencer is not None, "the music plays"
        self._sequencer.want(mark)

    def decisions(self) -> list[Decision]:
        """Where the music went since the last call."""
        found: list[Decision] = []
        if self._sequencer is None:
            return found
        for _ in range(self._sequencer.decisions.qsize() + 1):  # what came meanwhile waits
            try:
                found.append(self._sequencer.decisions.get_nowait())
            except queue.Empty:
                break
        return found

    def problem(self) -> str | None:
        """Why the music is not heard, if it is not: its output failed or has gone."""
        if self._failure is not None:
            return self._failure
        now = time.monotonic()
        if not self._lost and now - self._looked_at >= WATCH_S:
            self._looked_at = now
            self._lost = self.output not in self._outputs()
        if self._lost:
            return (
                f"The audio output {self.output} was disconnected, so the music could not be heard."
            )
        return None

    def stop(self) -> None:
        """End the music and its recording; whatever of them was started."""
        if self._stream is not None:
            with contextlib.suppress(sounddevice.PortAudioError):  # an output gone stops anyway
                self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._writer is not None:
            self._recording.put(None)
            self._writer.join()
            self._writer = None

    def _play(self, out: np.ndarray, frames: int, _time: Any, status: Any) -> None:
        """The output's thread: the next ``frames`` of the music, and a copy for the recording."""
        assert self._sequencer is not None
        if status and status.output_underflow:  # the block before came too late
            self.dropouts += 1
        try:
            block = self._sequencer.render(frames)
        except Exception as failure:  # noqa: BLE001 - kept for the session, which ends then
            self._failure = f"The music failed: {type(failure).__name__}: {failure}"
            out.fill(0)
            raise sounddevice.CallbackAbort from failure
        out[:] = block
        if self._failure is None:  # nothing writes it any more once the recording failed
            self._recording.put(block)

    def _write(self, sound: soundfile.SoundFile) -> None:
        """The player's thread: what the output was given, into the recording, until the end.
        A recording that cannot be written (a full disk) is kept as the failure, which ends the
        session: the music is not recorded."""
        try:
            with sound:
                while True:  # until stop() sends None
                    block = self._recording.get()
                    if block is None:
                        return
                    sound.write(block)
        except Exception as failure:  # noqa: BLE001 - kept for the session, which ends then
            self._failure = f"The music could not be recorded: {type(failure).__name__}: {failure}"
