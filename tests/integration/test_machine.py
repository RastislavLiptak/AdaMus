"""machine: the real computer, which the stand-in machine of the other tests replaces."""

# A test looks at what the real machine gives the music, which nothing else shows.
# pyright: reportPrivateUsage=false

import threading
from pathlib import Path

import pytest

from adamus import constants as c
from adamus import machine as the_computer
from adamus.errors import AdamusError


class TestTheRealMachine:
    """What the stand-in machine of the other tests replaces."""

    def test_audio_output_is_the_one_the_system_plays_on(self, monkeypatch):
        speakers = "MacBook Pro Speakers"
        monkeypatch.setattr(the_computer.audio_outputs, "system_output", lambda: speakers)
        assert the_computer.Machine().audio_output() == speakers

    def test_computer_without_an_audio_output(self, monkeypatch):
        monkeypatch.setattr(the_computer.audio_outputs, "system_output", lambda: None)
        with pytest.raises(AdamusError) as error:
            the_computer.Machine().audio_output()
        assert str(error.value) == "the Mac has no audio output"

    def test_audio_outputs_are_those_of_the_mac_now(self, monkeypatch):
        found = ["TV", "Headphones"]
        monkeypatch.setattr(the_computer.audio_outputs, "outputs", lambda: found)
        assert the_computer.Machine().audio_outputs() == found

    def test_headsets_music_and_devices_are_the_real_ones(self, program, monkeypatch):
        from adamus.config import parse_config
        from adamus.realtime.song_player import SongPlayer

        config = parse_config(program.raw)
        machine = the_computer.Machine()
        monkeypatch.setattr(the_computer.eeg_source, "headset_is_available", lambda b: b.id)
        assert machine.headset_is_available(config.board) == 38
        assert machine.sessions_root == c.SESSIONS_DIR
        assert machine.datasets_root == c.DATASETS_DIR
        assert machine.library_root == c.LIBRARY_DIR

        recording = program.add_recording("rec")
        stop = threading.Event()
        devices = machine.devices(config, recording, "A Song", "TV", stop)
        music = devices.music
        assert isinstance(music, SongPlayer) and devices.stop is stop
        assert (music.output, music._song_id, music._library) == ("TV", "A Song", machine.library)
        assert machine.library.root == c.LIBRARY_DIR
        assert music._outputs is the_computer.audio_outputs.outputs  # what it watches for
        assert devices.source.playback_file == (recording / "eeg.csv").resolve()


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
