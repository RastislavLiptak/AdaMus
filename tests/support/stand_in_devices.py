"""Stand-ins for what a session talks to: EEG source, music, clock, stop request."""

import shutil
from pathlib import Path

import numpy as np
import soundfile
from brainflow.board_shim import BoardShim

from adamus import constants as c
from adamus.config import Board
from adamus.errors import AdamusError
from adamus.realtime.eeg_source import electrodes_of_headset
from adamus.realtime.song_sequencer import Decision
from adamus.session_format import START, SWITCH
from tests.support.stand_in_songs import ARTIST, SONG, place_song

MUSE_2 = 38  # BrainFlow board id
NUDGE_S = 1e-6  # every reading of the stand-in clock moves it on by this much
CHUNK = 64  # samples a poll of a stand-in source delivers at most
MUSIC_RATE = 8000  # Hz of the silence the stand-in music records


TV, SPEAKERS = "TV", "MacBook Pro Speakers"  # the audio outputs of the stand-in Mac


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


class StandInMusic:
    """Stands in for the song player: notes the state it is told at every step and plays
    nothing. It switches as soon as it is told another state, where the player waits for the
    end of a beat. Started, it copies the song in the folder ``song`` into the session, as the
    player does, or places the stand-in song there; stopped, it records silence as long as it
    played, as the player records what it played."""

    def __init__(
        self, clock: StandInClock, fail_after_steps: int | None = None,
        fail_to_start: str = "", song: Path | None = None,
    ) -> None:  # fmt: skip
        self.output = SPEAKERS
        self.song = song
        self.states: list[str] = []  # what it was told, step by step
        self.recording: Path | None = None
        self.started = self.stopped = 0
        self.problem_said: str | None = None  # as the test sets it
        self.dropouts = 0
        self._clock = clock
        self._began = 0.0
        self._went_on = 0.0  # monotonic: the player counts what it played, not the time of day
        self._fail_after_steps = fail_after_steps
        self._fail_to_start = fail_to_start
        self._made: list[Decision] = []

    def start(self, recording: Path, song: Path) -> float:
        if self._fail_to_start:
            raise AdamusError(self._fail_to_start)
        self.started += 1
        if self.song is None:
            place_song(song.parent, song.name, title=SONG, artist=ARTIST)
        else:
            shutil.copytree(self.song, song)
        self.recording = recording
        self._began = self._clock.now()
        self._went_on = self._clock.monotonic()
        self._made.append(Decision(0.0, START, c.REST, None, 0.0))
        self._playing = c.REST
        return self._began

    def want(self, mark: str) -> None:
        limit = self._fail_after_steps
        if limit is not None and len(self.states) >= limit:
            raise OSError("the output is gone")
        self.states.append(mark)
        if mark != self._playing:
            out_s = self._clock.now() - self._began
            self._made.append(Decision(out_s, SWITCH, mark, out_s, out_s))
            self._playing = mark

    def decisions(self) -> list[Decision]:
        made, self._made = self._made, []
        return made

    def problem(self) -> str | None:
        return self.problem_said

    def stop(self) -> None:
        self.stopped += 1
        if self.recording is not None and self.stopped == 1:
            played = round((self._clock.monotonic() - self._went_on) * MUSIC_RATE)
            soundfile.write(self.recording, np.zeros((played, 2)), MUSIC_RATE, subtype="PCM_16")


class StopWhen:
    """A stop request that is set as soon as ``condition()`` says so."""

    def __init__(self, condition=lambda: False) -> None:
        self._condition = condition

    def is_set(self) -> bool:
        return self._condition()
