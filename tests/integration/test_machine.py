"""machine: the real computer, which the stand-in machine of the other tests replaces."""

import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from adamus import constants as c
from adamus import machine as the_computer
from adamus.activity_log import ActivityLog
from adamus.audio_outputs import AudioOutput
from adamus.errors import AdamusError
from tests.support.project_paths import ROOT

LONG_S = 30  # what a tool of the installation would take: more than a test waits
INTERRUPT_AFTER_S = 1.0  # once the script has started its tool
POLL_S = 0.05


class TestTheRealMachine:
    """What the stand-in machine of the other tests replaces."""

    def test_audio_output_is_the_one_the_system_plays_on(self, monkeypatch):
        speakers = AudioOutput("MacBook Pro Speakers", 48000.0)
        monkeypatch.setattr(the_computer.audio_outputs, "system_output", lambda: speakers)
        assert the_computer.Machine().audio_output() == speakers

    def test_computer_without_an_audio_output(self, monkeypatch):
        monkeypatch.setattr(the_computer.audio_outputs, "system_output", lambda: None)
        with pytest.raises(AdamusError) as error:
            the_computer.Machine().audio_output()
        assert str(error.value) == "the Mac has no audio output"

    def test_audio_outputs_are_those_of_the_mac_now(self, monkeypatch):
        found = [AudioOutput("TV", 48000.0), AudioOutput("Headphones", 44100.0)]
        monkeypatch.setattr(the_computer.audio_outputs, "outputs", lambda: found)
        assert the_computer.Machine().audio_outputs() == found

    def test_headsets_generator_and_devices_are_the_real_ones(self, program, monkeypatch):
        from adamus.config import parse_config
        from adamus.realtime.music_generator import MusicGenerator

        config = parse_config(program.raw)
        machine = the_computer.Machine()
        monkeypatch.setattr(the_computer.eeg_source, "headset_is_available", lambda b: b.id)
        assert machine.headset_is_available(config.board) == 38
        assert machine.sessions_root == c.SESSIONS_DIR
        assert machine.datasets_root == c.DATASETS_DIR

        log = ActivityLog()
        generator = machine.make_generator(config, log.write)
        assert isinstance(generator, MusicGenerator)
        assert (generator.config, generator.timeouts) == (config.generator, config.timeouts)
        found = [AudioOutput("TV", 48000.0), AudioOutput("Headphones", 44100.0)]
        monkeypatch.setattr(the_computer.audio_outputs, "outputs", lambda: found)
        assert generator.present_outputs() == ["TV", "Headphones"]  # what it watches for

        recording = program.add_recording("rec")
        stop = threading.Event()
        devices = machine.devices(config, recording, generator, stop)
        assert devices.generator is generator and devices.stop is stop
        assert devices.source.playback_file == (recording / "eeg.csv").resolve()


class TestInstallationOfTheGenerator:
    """The real machine runs the installation script, its output going to a log."""

    @pytest.fixture(autouse=True)
    def lacking_nothing(self, monkeypatch):
        """A computer with every tool the installation needs (looked for in the tests below)."""
        monkeypatch.setattr(the_computer, "missing_install_tools", lambda path: [])

    def generator(self, folder="generator"):
        from adamus.config import Generator

        return Generator(folder, "mrt2_small", "quiet piano", "loud orchestra")

    def script(self, tmp_path, body):
        script = tmp_path / the_computer.INSTALL_SCRIPT
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(body, encoding="utf-8")
        return script

    def test_an_interruption_ends_the_script_and_every_tool_it_started(self, tmp_path, monkeypatch):
        """As when the app ends during a first start: no download or build goes on after it."""
        monkeypatch.chdir(tmp_path)
        machine = the_computer.Machine()
        started = tmp_path / "tool.pid"
        machine.install_script = self.script(
            tmp_path, f"sleep {LONG_S} &\necho $! > {started}\nwait\n"
        )

        def interrupt(signum, frame):
            raise KeyboardInterrupt

        before = signal.signal(signal.SIGALRM, interrupt)
        signal.setitimer(signal.ITIMER_REAL, INTERRUPT_AFTER_S)
        began = time.monotonic()
        try:
            with pytest.raises(KeyboardInterrupt):
                machine.install_generator(self.generator())
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, before)
        assert time.monotonic() - began < LONG_S / 2  # told to end, not waited for
        tool = int(started.read_text())
        with pytest.raises(ProcessLookupError):
            for _ in range(round(LONG_S / POLL_S)):  # it ends a moment after it is told to
                os.kill(tool, 0)
                time.sleep(POLL_S)

    def test_the_tools_of_homebrew_are_found_also_without_a_terminal(self):
        """An app opened from the Finder runs the program with the PATH of the system only."""
        path = the_computer.with_project_tools({"PATH": "/usr/bin:/bin"})["PATH"].split(":")
        assert path == [
            str(Path(sys.executable).parent), "/usr/bin", "/bin",
            "/opt/homebrew/bin",
        ]  # fmt: skip
        mine = the_computer.with_project_tools({"PATH": "/opt/homebrew/bin:/usr/bin"})["PATH"]
        assert mine.split(":")[1:] == ["/opt/homebrew/bin", "/usr/bin"]  # not twice
        none = the_computer.with_project_tools({})["PATH"]  # given no PATH at all
        assert none.split(":") == [str(Path(sys.executable).parent), "/opt/homebrew/bin"]

    def test_runs_the_script_with_the_folder_and_the_model(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        machine = the_computer.Machine()
        machine.install_script = self.script(
            tmp_path, 'echo "folder $1 model $2"' + "\ncommand -v python\n"
        )
        log_file = machine.install_generator(self.generator("tools/gen"))
        assert log_file == Path("tools/gen") / "setup.log"
        lines = log_file.read_text().splitlines()
        assert lines[0] == "folder tools/gen model mrt2_small"
        # the tools of the program's own environment come first, activated or not
        assert Path(lines[1]).parent == Path(sys.executable).parent

    def test_an_earlier_log_in_an_existing_folder_is_replaced(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "generator").mkdir()
        (tmp_path / "generator" / "setup.log").write_text("old run", encoding="utf-8")
        machine = the_computer.Machine()
        machine.install_script = self.script(tmp_path, "echo new run")
        log_file = machine.install_generator(self.generator())
        assert log_file.read_text() == "new run\n"

    def test_errors_go_into_the_log_too_and_a_failure_names_it(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        machine = the_computer.Machine()
        machine.install_script = self.script(tmp_path, "echo no cmake >&2\nexit 3")
        with pytest.raises(AdamusError) as error:
            machine.install_generator(self.generator())
        log_file = Path("generator") / "setup.log"
        assert str(error.value) == f"the installation of the generator failed, see {log_file}"
        assert log_file.read_text() == "no cmake\n"

    def test_without_the_script_the_program_was_started_elsewhere(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(AdamusError) as error:
            the_computer.Machine().install_generator(self.generator())
        assert str(error.value) == (
            "scripts/install_generator.sh not found: start the program in its folder"
        )

    def test_the_script_is_the_one_of_the_project(self):
        assert (ROOT / the_computer.Machine.install_script).is_file()


def tools_in(folder, names, metal=True):
    """A search path that holds only ``names``, and xcrun, which finds Metal or not."""
    folder.mkdir()
    for name in names:
        tool = folder / name
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(0o755)
    xcrun = folder / "xcrun"  # finds the compiler of Metal when asked for it by its name
    found = '[ "$1" = "--find" ] && [ "$2" = "metal" ]' if metal else "exit 1"
    xcrun.write_text(f"#!/bin/sh\n{found}\n")
    xcrun.chmod(0o755)
    return str(folder)


class TestToolsOfTheInstallation:
    """What the installation needs is named before it runs, not found in its log after."""

    def test_a_mac_with_every_tool_lacks_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(the_computer.platform, "machine", lambda: "arm64")
        path = tools_in(tmp_path / "bin", the_computer.INSTALL_TOOLS)
        assert the_computer.missing_install_tools(path) == []

    def test_every_missing_tool_is_named(self, tmp_path, monkeypatch):
        monkeypatch.setattr(the_computer.platform, "machine", lambda: "x86_64")
        path = tools_in(tmp_path / "bin", ["curl", "unzip", "hdiutil", "git"], metal=False)
        assert the_computer.missing_install_tools(path) == [
            the_computer.NO_APPLE_SILICON, "cmake", the_computer.NO_METAL,
        ]  # fmt: skip
        assert the_computer.missing_install_tools(str(tmp_path / "nothing"))[-1] == (
            the_computer.NO_METAL  # without xcrun, too
        )

    def test_the_script_does_not_run_without_them(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(the_computer, "missing_install_tools", lambda path: ["cmake"])
        machine = the_computer.Machine()
        machine.install_script = tmp_path / "install.sh"
        machine.install_script.write_text("touch ran\n")
        generator = TestInstallationOfTheGenerator().generator()
        with pytest.raises(AdamusError, match="^the installation of the generator needs cmake$"):
            machine.install_generator(generator)
        assert not (tmp_path / "ran").exists()


class TestTrainingOfTheFocusDetector:
    """The real machine trains the models into their folder, its output going to a log."""

    def test_trains_into_the_folder_of_the_models_with_a_log(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = []

        def train(folder, log):
            calls.append(folder)
            log.write("Experiment_1/Person01 read")
            return folder / c.FOCUS_MODEL_FILE

        monkeypatch.setattr(the_computer, "train", train)
        log_file = the_computer.Machine().train_focus_detector()
        assert calls == [c.FOCUS_DETECTOR_DIR] == [Path("models/focus_detector")]
        assert log_file == c.FOCUS_DETECTOR_DIR / "setup.log"
        assert log_file.read_text() == "Experiment_1/Person01 read"

    def test_whatever_stops_the_training_goes_into_the_log(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        def train(folder, log):
            log.write("downloading ...\n")
            raise ConnectionResetError("the server hung up")

        monkeypatch.setattr(the_computer, "train", train)
        with pytest.raises(AdamusError) as error:
            the_computer.Machine().train_focus_detector()
        log_file = c.FOCUS_DETECTOR_DIR / "setup.log"
        assert str(error.value) == f"the training of the focus detector failed, see {log_file}"
        assert isinstance(error.value.__cause__, ConnectionResetError)
        lines = log_file.read_text().splitlines()
        assert lines[:2] == ["downloading ...", "Traceback (most recent call last):"]
        assert lines[-1] == "ConnectionResetError: the server hung up"

    def test_the_training_is_the_one_of_the_program(self):
        from adamus.detection import focus_training

        assert the_computer.train is focus_training.train
