"""A session with its real music: the player of a song of the library, on an output that takes
the music faster than it plays, as the session of the tests runs faster than time."""

import shutil

import pytest
import soundfile

from adamus import constants as c
from adamus.realtime.song_player import SongPlayer
from adamus.server.dashboard import dashboard_of
from adamus.song_library import Library, analysis_in, audio_in, song_in
from tests.support.finished_sessions import FinishedSession, config_with, synthetic_matrix
from tests.support.short_sessions import CONTROL
from tests.support.stand_in_outputs import (
    BLOCK,
    OUTPUT,
    RATE,
    FastOutput,
    MusicClock,
    stand_in_output,
)
from tests.support.synthetic_songs import BEAT_S, SECONDS, add_marked, write_song


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """A song whose first half is marked rest and its second focus."""
    root = tmp_path_factory.mktemp("songs") / "library"
    library = Library(root)
    add_marked(library, write_song(root.parent / "Beat.flac", title="A Beat"))
    return library


@pytest.fixture
def played(library, tmp_path, monkeypatch):
    stand_in_output(monkeypatch, FastOutput)
    FastOutput.made = []
    shutil.copytree(library.root, tmp_path / "library")
    (song,) = Library(tmp_path / "library").songs()
    music = SongPlayer(Library(tmp_path / "library"), song.id, OUTPUT, lambda: [OUTPUT])
    clock = MusicClock()
    return FinishedSession(tmp_path, config_with(CONTROL), synthetic_matrix(CONTROL),
                           music=music, clock=clock)  # fmt: skip


def test_the_music_given_to_the_output_is_the_recording_of_the_session(played):
    (output,) = FastOutput.made
    info = soundfile.info(played.directory / c.FILE_AUDIO)
    assert (info.samplerate, info.channels, info.subtype) == (RATE, 2, "PCM_16")
    assert info.frames == output.blocks * BLOCK > 0  # every block, none since it stopped
    assert played.events_of("MUSIC_END")[0]["dropouts"] == 0


def test_where_the_music_went_follows_the_states_it_was_told(played):
    went = played.music_lines
    assert (went[0]["kind"], went[0]["mark"], went[0]["from_s"]) == ("start", c.REST, None)
    switches = [line for line in went if line["kind"] == "switch"]
    assert switches and switches[0]["mark"] == c.FOCUS  # the first focus block was found
    # a beat belongs to the part its middle lies in: the first of focus may begin before it
    assert all(line["to_s"] >= SECONDS / 2 - BEAT_S / 2 for line in went if line["mark"] == c.FOCUS)
    assert all(line["to_s"] < SECONDS / 2 for line in went if line["mark"] == c.REST)


def test_the_session_keeps_the_song_as_the_library_had_it(played, tmp_path):
    """Whatever the library does with the song later: the session can be played again."""
    library = Library(tmp_path / "library")
    (song,) = library.songs()
    draft = library.draft_of(song.id)
    library.mark(draft.id, 0.0, SECONDS, c.FOCUS)
    library.mark(draft.id, 0.0, 1.0, c.REST)
    library.keep(draft.id, "Another", None)
    played_from = audio_in(library.folder(song.id)).read_bytes()
    library.delete(song.id)
    kept = played.directory / c.SONG_DIR
    assert song_in(kept).title == "A Beat"
    assert [m.mark for m in song_in(kept).marks] == [c.REST, c.FOCUS]
    assert song_in(kept).marks[0].end_s == pytest.approx(SECONDS / 2)
    assert audio_in(kept).read_bytes() == played_from
    assert analysis_in(kept).beats


def test_the_overview_shows_the_song_the_session_kept(played):
    song = dashboard_of(played.directory, None, False).song
    assert song is not None and song.title == "A Beat"
    assert [part.mark for part in song.parts] == [c.REST, c.FOCUS]
    assert song.parts[-1].end_s == pytest.approx(song.length_s)
