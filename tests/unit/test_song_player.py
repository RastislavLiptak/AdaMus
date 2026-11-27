"""song_player: a song played on an output and recorded, without a sound: a stand-in takes the
output's place and asks for the audio as an output does."""

import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import sounddevice
import soundfile

from adamus import constants as c
from adamus.errors import AdamusError
from adamus.realtime import song_player
from adamus.realtime.song_player import SongPlayer, output_index
from adamus.session_format import START, SWITCH
from adamus.song_library import Library
from tests.support.stand_in_outputs import stand_in_output
from tests.support.synthetic_songs import SECONDS, add_marked, write_song

OUTPUT, OTHER = "Speakers", "Microphone"
DEVICE_RATE = 48000
BLOCK = 1024  # frames the stand-in output asks for at a time


class StandInOutput:
    """An output stream that asks for audio only when the test says so."""

    made: list["StandInOutput"] = []

    def __init__(self, samplerate, device, channels, dtype, latency, callback):
        self.rate, self.device, self.callback = samplerate, device, callback
        self.given: list[np.ndarray] = []
        self.started = self.stopped = self.closed = False
        StandInOutput.made.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def ask(self, blocks: int) -> None:
        for _ in range(blocks):
            out = np.zeros((BLOCK, 2), dtype=np.float32)
            self.callback(out, BLOCK, None, None)
            self.given.append(out)


DEVICES = [
    {"name": OTHER, "max_output_channels": 0, "default_samplerate": 44100.0},
    {"name": OUTPUT, "max_output_channels": 2, "default_samplerate": float(DEVICE_RATE)},
]


@pytest.fixture(autouse=True)
def asked_by_the_test(monkeypatch):
    StandInOutput.made = []
    stand_in_output(monkeypatch, StandInOutput, DEVICES)


@pytest.fixture(scope="module")
def marked(tmp_path_factory):
    """A library with a song whose first half is marked rest, its second focus."""
    root = tmp_path_factory.mktemp("player")
    library = Library(root / "library")
    return library.root, add_marked(library, write_song(root / "Beat.wav"))


@pytest.fixture
def player(marked, tmp_path):
    root, song_id = marked
    shutil.copytree(root, tmp_path / "library")
    return SongPlayer(Library(tmp_path / "library"), song_id, OUTPUT, lambda: [OUTPUT])


def test_the_output_is_found_by_its_name_among_those_that_play_stereo():
    assert output_index(OUTPUT) == 1
    for name in (OTHER, "Headphones"):
        with pytest.raises(AdamusError, match=f"{name} is not connected"):
            output_index(name)


def test_what_the_output_is_given_is_recorded_as_it_was_given(player, tmp_path):
    wav = tmp_path / "audio.wav"
    player.start(wav, tmp_path / "song")
    stream = StandInOutput.made[-1]
    assert (stream.rate, stream.device, stream.started) == (DEVICE_RATE, 1, True)
    stream.ask(40)
    player.stop()
    assert stream.stopped and stream.closed
    recorded, rate = soundfile.read(wav, dtype="float32")
    given = np.vstack(stream.given)
    assert rate == DEVICE_RATE and recorded.shape == given.shape
    assert np.abs(recorded - given).max() < 1 / 2**14  # stored in 16 bits
    assert np.abs(given).max() > 0  # the song, at its rest, was played
    assert [d.kind for d in player.decisions()] == [START]


def test_the_music_follows_the_state_it_is_told(player, tmp_path):
    player.start(tmp_path / "audio.wav", tmp_path / "song")
    stream = StandInOutput.made[-1]
    stream.ask(10)
    player.want(c.FOCUS)
    stream.ask(40)
    player.stop()
    switches = [d for d in player.decisions() if d.kind == SWITCH]
    assert switches and switches[0].mark == c.FOCUS and switches[0].to_s >= SECONDS / 2


def test_a_block_the_output_played_before_it_was_ready_is_counted(player, tmp_path):
    player.start(tmp_path / "audio.wav", tmp_path / "song")
    stream = StandInOutput.made[-1]
    late, on_time = SimpleNamespace(output_underflow=True), SimpleNamespace(output_underflow=False)
    for status in (late, on_time, late, None):
        stream.callback(np.zeros((BLOCK, 2), dtype=np.float32), BLOCK, None, status)
    player.stop()
    assert player.dropouts == 2


def test_a_failure_of_the_music_is_told_and_the_output_gets_silence(player, tmp_path, monkeypatch):
    player.start(tmp_path / "audio.wav", tmp_path / "song")
    stream = StandInOutput.made[-1]

    def broken(frames):
        raise ValueError("broken")

    monkeypatch.setattr(player._sequencer, "render", broken)
    with pytest.raises(sounddevice.CallbackAbort):
        stream.ask(1)
    assert player.problem() == "The music failed: ValueError: broken"
    player.stop()


def test_an_output_that_has_gone_is_told(marked, tmp_path, monkeypatch):
    root, song_id = marked
    shutil.copytree(root, tmp_path / "library")
    present = [OUTPUT]
    player = SongPlayer(Library(tmp_path / "library"), song_id, OUTPUT, lambda: present)
    monkeypatch.setattr(song_player, "WATCH_S", 0.0)
    player.start(tmp_path / "audio.wav", tmp_path / "song")
    assert player.problem() is None
    present.clear()
    assert player.problem() == (
        f"The audio output {OUTPUT} was disconnected, so the music could not be heard."
    )
    player.stop()


def test_stopping_is_safe_whatever_was_started(player, tmp_path, monkeypatch):
    player.stop()  # nothing started
    monkeypatch.setattr(StandInOutput, "start", lambda self: (_ for _ in ()).throw(
        sounddevice.PortAudioError("no output")))  # fmt: skip
    with pytest.raises(sounddevice.PortAudioError):
        player.start(tmp_path / "audio.wav", tmp_path / "song")
    player.stop()  # the recording that was opened is closed, its thread ended
    assert player._writer is None and (tmp_path / "audio.wav").is_file()
    player.stop()


def test_a_recording_that_cannot_be_written_is_told_and_takes_no_more(
    player, tmp_path, monkeypatch
):
    def full(self, block):
        raise OSError("No space left on device")

    monkeypatch.setattr(soundfile.SoundFile, "write", full)
    player.start(tmp_path / "audio.wav", tmp_path / "song")
    stream = StandInOutput.made[-1]
    stream.ask(1)
    player._writer.join(timeout=5)  # it ended, having kept why
    assert player.problem() == "The music could not be recorded: OSError: No space left on device"
    stream.ask(40)
    assert player._recording.empty()  # the music played on, but nothing keeps it waiting
    player.stop()
