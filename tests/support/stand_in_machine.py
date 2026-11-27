"""A stand-in for the computer the program runs on: its audio outputs, headsets, music and
training are what the test decides."""

import shutil
import time
from pathlib import Path

from adamus import constants as c
from adamus.errors import AdamusError
from adamus.machine import Machine
from adamus.realtime.eeg_source import RecordingSource
from adamus.realtime.session import Devices
from tests.support.finished_sessions import synthetic_matrix
from tests.support.project_paths import PROJECT_MODEL
from tests.support.stand_in_devices import (
    SPEAKERS,
    TV,
    StandInClock,
    StandInMusic,
    StandInSource,
)

# What the stand-in headset delivers: a long rest, then two FOCUS blocks.
LIVE_PROTOCOL = [["REST", 64], ["FOCUS", 20], ["REST", 20], ["FOCUS", 20]]
POLL_TAKES_S = 0.001  # of real time: long enough for a stop request to arrive between two polls


class PacedClock(StandInClock):
    """Stand-in time that still takes a moment per poll, so that a stop request can arrive
    mid-session."""

    def __init__(self, poll_takes_s: float) -> None:
        super().__init__()
        self._poll_takes_s = poll_takes_s

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        time.sleep(self._poll_takes_s)


class WatchedRecording(RecordingSource):
    """The real player of recordings; the tests can see how far it got and that it was closed."""

    @property
    def delivered(self) -> int:
        return self._delivered

    @property
    def closed(self) -> bool:
        return self._raw is None


class StandInMachine(Machine):
    """A computer whose audio outputs, headsets and music the test decides.

    Sessions run for real: a recording is played by the real player on a clock
    that does not wait; a live headset is a stand-in that delivers 124 s of
    synthetic signal and then falls silent. The music is a stand-in that plays nothing; it
    plays the song the session asks for when the library has it.
    """

    def __init__(self, sessions_root: Path, datasets_root: Path) -> None:
        self.sessions_root = sessions_root
        self.datasets_root = datasets_root
        self.library_root = sessions_root.parent / c.LIBRARY_DIR
        self.audio: str | None = SPEAKERS  # the system's
        self.outputs = [TV, SPEAKERS]
        self.headsets_on = {"Muse 2"}
        self.musics: list[StandInMusic] = []  # of the sessions, in turn
        self.music_failure = ""  # what the music says when it cannot start, if set
        self.probed: list[int] = []
        self.sources: list[StandInSource | WatchedRecording] = []
        self.source_failure = ""
        self.live_samples: int | None = None  # the headset delivers only this many, if set
        self.training = "fails"  # or "trains" the models of the project, or "does nothing"
        self.trainings = 0
        # a stop request comes while a session with the headset runs: each poll takes real time;
        # one over a recording ends by itself, as fast as it can
        self.paced = False
        self.poll_takes_s = POLL_TAKES_S  # when paced

    def train_focus_detector(self):
        self.trainings += 1
        log_file = c.FOCUS_DETECTOR_DIR / c.SETUP_LOG
        if self.training == "fails":
            raise AdamusError(f"the training of the focus detector failed, see {log_file}")
        if self.training == "trains":
            place_focus_model(Path())
        return log_file

    def audio_output(self):
        if self.audio is None:
            raise AdamusError("the Mac has no audio output")
        return self.audio

    def audio_outputs(self):
        return list(self.outputs)

    def headset_is_available(self, board):
        self.probed.append(board.id)
        return board.name in self.headsets_on

    def devices(self, config, recording, song, output, stop):
        if self.source_failure:
            raise OSError(self.source_failure)
        clock = (
            PacedClock(self.poll_takes_s) if self.paced and recording is None else StandInClock()
        )
        if recording is None:
            matrix = synthetic_matrix(LIVE_PROTOCOL)[:, : self.live_samples]
            self.sources.append(StandInSource(matrix, live=True))
        else:
            self.sources.append(WatchedRecording(config.board, recording, clock))
        played = self.library.folder(song)  # a song the test did not place fails, as it should
        self.musics.append(StandInMusic(clock, fail_to_start=self.music_failure, song=played))
        self.musics[-1].output = output
        return Devices(self.sources[-1], self.musics[-1], stop, clock)


def place_focus_model(folder: Path) -> None:
    """The project's trained focus models, as the program keeps them below ``folder``."""
    (folder / c.FOCUS_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROJECT_MODEL, folder / c.FOCUS_MODEL_PATH)
