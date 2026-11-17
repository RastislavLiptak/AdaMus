"""The parts of the computer the console works with: what the checks look at, what they
obtain when it is missing, and what a session talks to. Tests substitute them."""

import os
import subprocess
import sys
import traceback
from pathlib import Path

import sounddevice

from adamus import constants as c
from adamus.config import Board, Config, Generator
from adamus.detection.focus_training import train
from adamus.errors import AdamusError
from adamus.realtime import eeg_source, session
from adamus.realtime.music_generator import MusicGenerator, Report
from adamus.realtime.session import MusicOutput, StopKeys

INSTALL_SCRIPT = Path("scripts/install_generator.sh")  # relative to the project folder


def setup_log(folder: Path) -> Path:
    """The log of what is installed or trained into ``folder``, which is created."""
    folder.mkdir(parents=True, exist_ok=True)
    return folder / c.SETUP_LOG


class Machine:
    """The real computer; the tests put a stand-in in its place."""

    sessions_root: Path = c.SESSIONS_DIR
    datasets_root: Path = c.DATASETS_DIR
    install_script: Path = INSTALL_SCRIPT

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

    def install_generator(self, generator: Generator) -> Path:
        """Run the installation script of the generator; its output goes to the returned log."""
        if not self.install_script.is_file():
            raise AdamusError(f"{self.install_script} not found: start the program in its folder")
        log_file = setup_log(generator.directory)
        tools = str(Path(sys.executable).parent)  # the venv's mrt, also when it is not activated
        environment = {**os.environ, "PATH": tools + os.pathsep + os.environ.get("PATH", "")}
        with log_file.open("w", encoding="utf-8") as log:
            done = subprocess.run(
                ["sh", str(self.install_script), str(generator.directory), generator.model],
                stdout=log, stderr=subprocess.STDOUT, env=environment, check=False,
            )  # fmt: skip
        if done.returncode != 0:
            raise AdamusError(f"the installation of the generator failed, see {log_file}")
        return log_file

    def audio_output(self) -> tuple[str, float]:
        """Name and default sample rate of the default output device."""
        try:
            device = sounddevice.query_devices(kind="output")
        except (sounddevice.PortAudioError, ValueError) as problem:
            raise AdamusError(f"no audio output device: {problem}") from problem
        return device["name"], device["default_samplerate"]

    def headset_is_available(self, board: Board) -> bool:
        return eeg_source.headset_is_available(board)

    def make_generator(self, config: Config, report: Report) -> MusicGenerator:
        return MusicGenerator(config.generator, config.timeouts, report)

    def devices(
        self, config: Config, recording: Path | None, generator: MusicOutput, keyboard: StopKeys
    ) -> session.Devices:
        """What a session talks to: the headset, or the file ``recording`` played in its place."""
        return session.real_devices(config, recording, generator, keyboard)
