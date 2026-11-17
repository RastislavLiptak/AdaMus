"""console.machine: the real computer, which the stand-in machine of the other tests replaces."""

import sys
from pathlib import Path

import pytest

from adamus import constants as c
from adamus.console import machine as the_computer
from adamus.console.console_log import ConsoleLog
from adamus.errors import AdamusError


class TestTheRealMachine:
    """What the stand-in machine of the other tests replaces."""

    def test_audio_output_is_the_default_output_device(self, monkeypatch):
        asked = []

        def query_devices(kind):
            asked.append(kind)
            return {"name": "MacBook Pro Speakers", "default_samplerate": 48000.0, "index": 1}

        monkeypatch.setattr(the_computer.sounddevice, "query_devices", query_devices)
        assert the_computer.Machine().audio_output() == ("MacBook Pro Speakers", 48000.0)
        assert asked == ["output"]

    @pytest.mark.parametrize(
        "problem", [the_computer.sounddevice.PortAudioError("no device"), ValueError("none")]
    )
    def test_computer_without_an_audio_output(self, monkeypatch, problem):
        def query_devices(kind):
            raise problem

        monkeypatch.setattr(the_computer.sounddevice, "query_devices", query_devices)
        with pytest.raises(AdamusError) as error:
            the_computer.Machine().audio_output()
        assert str(error.value) == f"no audio output device: {problem}"

    def test_headsets_generator_and_devices_are_the_real_ones(self, console, monkeypatch):
        from adamus.config import parse_config
        from adamus.realtime.music_generator import MusicGenerator

        config = parse_config(console.raw)
        machine = the_computer.Machine()
        monkeypatch.setattr(the_computer.eeg_source, "headset_is_available", lambda b: b.id)
        assert machine.headset_is_available(config.board) == 38
        assert machine.sessions_root == c.SESSIONS_DIR
        assert machine.datasets_root == c.DATASETS_DIR

        log = ConsoleLog()
        generator = machine.make_generator(config, log.write)
        assert isinstance(generator, MusicGenerator)
        assert (generator._config, generator._timeouts) == (config.generator, config.timeouts)
        assert generator._report == log.write

        recording = console.add_recording("rec")
        devices = machine.devices(config, recording, generator, "the keyboard")
        assert devices.generator is generator and devices.keyboard == "the keyboard"
        assert devices.source.playback_file == (recording / "eeg.csv").resolve()


class TestInstallationOfTheGenerator:
    """The real machine runs the installation script, its output going to a log."""

    def generator(self, folder="generator"):
        from adamus.config import Generator

        return Generator(folder, "mrt2_small", 3000, "quiet piano", "loud orchestra")

    def script(self, tmp_path, body):
        script = tmp_path / "scripts" / "install_generator.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(body, encoding="utf-8")
        return script

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
        assert the_computer.Machine.install_script == Path("scripts/install_generator.sh")
        assert (Path(__file__).resolve().parents[2] / "scripts" / "install_generator.sh").is_file()


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
