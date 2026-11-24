"""The parts of the computer the program works with: what the checks look at, what they
obtain when it is missing, and what a session talks to. Tests substitute them."""

import os
import platform
import shutil
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Protocol

from adamus import audio_outputs
from adamus import constants as c
from adamus.audio_outputs import AudioOutput
from adamus.config import Board, Config, Generator
from adamus.detection.focus_training import train
from adamus.errors import AdamusError
from adamus.realtime import eeg_source, session
from adamus.realtime.music_generator import MusicGenerator, Report
from adamus.realtime.session import MusicOutput, StopRequest

INSTALL_SCRIPT = Path("scripts/install_generator.sh")  # relative to the project folder
# What the installation script runs besides the tools of the project's Python, and what the
# music model needs to run in real time.
INSTALL_TOOLS = ("curl", "unzip", "hdiutil", "git", "cmake")
HOMEBREW_TOOLS = "/opt/homebrew/bin"  # on Apple Silicon
APPLE_SILICON = "arm64"
NO_APPLE_SILICON = "a Mac with Apple Silicon"
NO_METAL = "the Metal toolchain of Xcode (xcodebuild -downloadComponent MetalToolchain)"


def missing_install_tools(path: str) -> list[str]:
    """What the installation of the generator needs and this computer lacks, looking for the
    tools on the search ``path``."""
    missing = [] if platform.machine() == APPLE_SILICON else [NO_APPLE_SILICON]
    missing += [tool for tool in INSTALL_TOOLS if shutil.which(tool, path=path) is None]
    xcrun = shutil.which("xcrun", path=path)
    found = xcrun is not None and subprocess.run(
        [xcrun, "--find", "metal"], capture_output=True, check=False
    ).returncode == 0  # fmt: skip
    return missing if found else [*missing, NO_METAL]


class RunningGenerator(MusicOutput, Protocol):
    """The music generator, as the program uses it: started when the checks pass, stopped
    when the program ends, and between the two what a session sends to it. Started again on
    another audio output, when the operator chooses one."""

    @property
    def running(self) -> bool: ...

    def start(self, output: str) -> None: ...

    def stop(self) -> None: ...


def setup_log(folder: Path) -> Path:
    """The log of what is installed or trained into ``folder``, which is created."""
    folder.mkdir(parents=True, exist_ok=True)
    return folder / c.SETUP_LOG


def with_project_tools(environment: dict[str, str]) -> dict[str, str]:
    """The environment with the tools of the program's Python first on the PATH, also when
    its virtual environment is not activated; and last, where Homebrew puts the tools it
    installs (cmake), which an app opened from the Finder is not given."""
    tools = str(Path(sys.executable).parent)
    given = environment.get("PATH", "").split(os.pathsep)
    places = [tools, *given, *([HOMEBREW_TOOLS] if HOMEBREW_TOOLS not in given else [])]
    return {**environment, "PATH": os.pathsep.join(place for place in places if place)}


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
        environment = with_project_tools(dict(os.environ))  # the venv's mrt
        missing = missing_install_tools(environment["PATH"])
        if missing:
            raise AdamusError(f"the installation of the generator needs {', '.join(missing)}")
        log_file = setup_log(generator.directory)
        with log_file.open("w", encoding="utf-8") as log:
            # In a session of its own, so that an interruption (the end of the app) ends the
            # script together with every tool it started: a download or a build goes no further.
            script = subprocess.Popen(
                ["sh", str(self.install_script), str(generator.directory), generator.model],
                stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True,
            )  # fmt: skip
            try:
                script.wait()
            except BaseException:
                os.killpg(script.pid, signal.SIGTERM)
                script.wait()
                raise
        if script.returncode != 0:
            raise AdamusError(f"the installation of the generator failed, see {log_file}")
        return log_file

    def audio_output(self) -> AudioOutput:
        """The output the system plays on now."""
        output = audio_outputs.system_output()
        if output is None:
            raise AdamusError("the Mac has no audio output")
        return output

    def audio_outputs(self) -> list[AudioOutput]:
        """Every output the Mac has now."""
        return audio_outputs.outputs()

    def headset_is_available(self, board: Board) -> bool:
        return eeg_source.headset_is_available(board)

    def make_generator(self, config: Config, report: Report) -> RunningGenerator:
        def present() -> list[str]:
            return [output.name for output in audio_outputs.outputs()]

        return MusicGenerator(config.generator, config.timeouts, report, present)

    def devices(
        self, config: Config, recording: Path | None, generator: MusicOutput, stop: StopRequest
    ) -> session.Devices:
        """What a session talks to: the headset, or the file ``recording`` played in its place."""
        return session.real_devices(config, recording, generator, stop)
