"""The parts of the computer the program works with: what the checks look at, what they
obtain when it is missing, and what a session talks to. Tests substitute them."""

import traceback
from functools import cached_property
from pathlib import Path

from adamus import audio_outputs
from adamus import constants as c
from adamus.config import Board, Config
from adamus.detection.focus_training import train
from adamus.errors import AdamusError
from adamus.realtime import eeg_source, session
from adamus.realtime.session import StopRequest
from adamus.realtime.song_player import SongPlayer
from adamus.song_library import Library


def setup_log(folder: Path) -> Path:
    """The log of what is trained into ``folder``, which is created."""
    folder.mkdir(parents=True, exist_ok=True)
    return folder / c.SETUP_LOG


class Machine:
    """The real computer; the tests put a stand-in in its place."""

    sessions_root: Path = c.SESSIONS_DIR
    datasets_root: Path = c.DATASETS_DIR
    library_root: Path = c.LIBRARY_DIR

    @cached_property
    def library(self) -> Library:
        """The one library of songs: the form, the API and the sessions share it, so that its
        changes go one at a time."""
        return Library(self.library_root)

    def train_focus_detector(self) -> Path:
        """Train the focus models into their folder; the output goes to the returned log."""
        log_file = setup_log(c.FOCUS_DETECTOR_DIR)
        with log_file.open("w", encoding="utf-8") as log:
            try:
                train(c.FOCUS_DETECTOR_DIR, log)
            except Exception as problem:  # noqa: BLE001 - whatever stops the training is logged
                traceback.print_exc(file=log)
                failed = f"the training of the focus detector failed, see {log_file}"
                raise AdamusError(failed) from problem
        return log_file

    def audio_output(self) -> str:
        """The output the system plays on now."""
        output = audio_outputs.system_output()
        if output is None:
            raise AdamusError("the Mac has no audio output")
        return output

    def audio_outputs(self) -> list[str]:
        """Every output the Mac has now."""
        return audio_outputs.outputs()

    def headset_is_available(self, board: Board) -> bool:
        return eeg_source.headset_is_available(board)

    def devices(
        self, config: Config, recording: Path | None, song: str, output: str, stop: StopRequest
    ) -> session.Devices:
        """What a session talks to: the headset, or the file ``recording`` played in its place;
        and the song ``song`` of the library, played on ``output``."""
        music = SongPlayer(self.library, song, output, audio_outputs.outputs)
        return session.real_devices(config, recording, music, stop)
