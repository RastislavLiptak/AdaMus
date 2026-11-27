"""song_library: songs analysed into drafts, marked there and kept in the library; described,
copied and deleted."""

import json
import shutil
import threading
import unicodedata

import numpy as np
import pytest

from adamus import constants as c
from adamus.errors import AdamusError
from adamus.song_library import (
    DRAFT_OF,
    LOUDNESS_BARS,
    TITLE_HINT,
    WRITING,
    Library,
    Mark,
    analysis_in,
    audio_in,
    painted,
    song_in,
    write_whole,
)
from tests.support.synthetic_songs import (
    BEAT_S,
    BEATS_PER_BAR,
    FIRST_BEAT_S,
    PICTURE,
    SECONDS,
    beat_times,
    with_picture,
    write_song,
)

TOLERANCE_S = 0.05  # how far a found beat may lie from where it is: two frames of the tracker
THREADS = 8
HALF = SECONDS / 2


@pytest.fixture(scope="module")
def analysed(tmp_path_factory):
    """A draft of a FLAC with tags and a picture, and the file. Analysing takes seconds, so each
    test gets a copy of the draft."""
    root = tmp_path_factory.mktemp("songs")
    source = with_picture(write_song(root / "Beat.flac", title="A Beat", artist="Nobody"))
    draft = Library(root / "library").draft(source)
    kept = shutil.copytree(root / "library" / ".drafts" / draft.id, root / "analysed")
    return kept, source


@pytest.fixture
def library(tmp_path):
    return Library(tmp_path / "library")


def drafted(library, analysed, title=None):
    """A copy of the analysed draft in ``library``, called ``title`` if given: its id."""
    folder = shutil.copytree(analysed[0], library.drafts / f"draft {len(list_of(library))}")
    if title is not None:
        kept = json.loads((folder / c.FILE_SONG).read_text())
        (folder / c.FILE_SONG).write_text(json.dumps({**kept, "title": title}))
    return folder.name


def list_of(library):
    return list(library.drafts.iterdir()) if library.drafts.is_dir() else []


def kept(library, analysed, title=None):
    """A song of the library: the analysed draft, its first half marked rest, its second focus;
    called ``title``, if given."""
    draft = drafted(library, analysed)
    library.mark(draft, 0.0, HALF, c.REST)
    library.mark(draft, HALF, SECONDS, c.FOCUS)
    return library.keep(draft, title or "A Beat", None).id


@pytest.fixture
def draft(library, analysed):
    return drafted(library, analysed)


@pytest.fixture
def song_id(library, analysed):
    return kept(library, analysed)


def test_a_draft_of_a_file_keeps_it_with_the_title_artist_and_picture_it_carries(
    analysed, library, draft
):
    song = song_in(library.draft_folder(draft))
    assert (song.title, song.artist, song.marks) == ("A Beat", "Nobody", [])
    assert song.duration_s == pytest.approx(SECONDS, abs=0.01)
    assert audio_in(library.draft_folder(draft)).read_bytes() == analysed[1].read_bytes()
    assert song.cover is not None and song.cover.read_bytes() == PICTURE
    assert len(song.loudness) == LOUDNESS_BARS and max(song.loudness) == 1.0
    assert np.mean(song.loudness[LOUDNESS_BARS // 2 :]) > 2 * np.mean(
        song.loudness[: LOUDNESS_BARS // 2]
    )
    assert library.songs() == []  # not in the library before it is marked and kept


def test_the_beats_and_the_first_beat_of_the_bars_are_found(analysed):
    analysis = analysis_in(analysed[0])
    found = np.array(analysis.beats)
    expected = beat_times()
    near = [np.min(np.abs(expected - t)) for t in found]
    assert max(near) < TOLERANCE_S and len(found) >= len(expected) - 2
    downbeats = found[analysis.first_downbeat :: BEATS_PER_BAR]
    bar_s = BEAT_S * BEATS_PER_BAR
    phase = ((downbeats - FIRST_BEAT_S) / bar_s) % 1.0
    assert np.all(np.minimum(phase, 1.0 - phase) * bar_s < TOLERANCE_S)
    assert len(analysis.sound) == len(found)
    lengths = np.linalg.norm(np.array(analysis.sound), axis=1)
    assert np.allclose(lengths, np.sqrt(2), atol=0.01)  # harmony and timbre, each of length one


def test_a_file_without_tags_is_called_by_its_name_and_has_no_picture(tmp_path):
    song = Library(tmp_path / "library").draft(write_song(tmp_path / "My Song.wav", seconds=8.0))
    assert (song.title, song.artist, song.cover) == ("My Song", None, None)


def test_a_kept_draft_is_a_song_of_the_library_called_as_its_title(library, analysed, draft):
    library.mark(draft, 0.0, HALF, c.REST)
    library.mark(draft, HALF, SECONDS, c.FOCUS)
    song = library.keep(draft, "A Beat", "Nobody")
    assert (song.id, song.title, song.cover is not None) == ("A Beat", "A Beat", True)
    assert song.marks == [Mark(0.0, HALF, c.REST), Mark(HALF, SECONDS, c.FOCUS)]
    assert library.songs() == [song] and list_of(library) == []
    assert DRAFT_OF not in json.loads((library.folder(song.id) / c.FILE_SONG).read_text())
    assert audio_in(library.folder(song.id)).read_bytes() == analysed[1].read_bytes()


def test_a_draft_goes_into_the_library_only_once_a_beat_is_marked_each_state(library, draft):
    with pytest.raises(AdamusError, match="No beat of the song is marked rest and focus"):
        library.keep(draft, "A", None)
    library.mark(draft, 0.0, HALF, c.REST)
    with pytest.raises(AdamusError, match="No beat of the song is marked focus: mark a part so"):
        library.keep(draft, "A", None)
    beats = analysis_in(library.draft_folder(draft)).beats
    library.mark(draft, beats[20], beats[20] + 0.1, c.FOCUS)  # no beat's middle lies in it
    with pytest.raises(AdamusError, match="marked focus"):
        library.keep(draft, "A", None)
    library.mark(draft, beats[20], beats[21], c.FOCUS)
    assert library.keep(draft, "A", None).id in [song.id for song in library.songs()]


def test_a_second_song_of_the_same_title_gets_a_number(library, analysed):
    assert [kept(library, analysed), kept(library, analysed)] == ["A Beat", "A Beat 2"]


def test_a_name_the_mac_counts_as_taken_gets_a_number_too(library, analysed):
    """The file system of a Mac ignores case and how an accent is written: Café written as one
    letter and as e with an accent is one name, as are Song and song."""
    first, second = (unicodedata.normalize(form, "Café") for form in ("NFC", "NFD"))
    for title in (first, second, "Song", "song", "a/b: c"):
        kept(library, analysed, title)
    named = sorted(unicodedata.normalize("NFC", song.id) for song in library.songs())
    assert named == ["Café", "Café 2", "Song", "a-b- c", "song 2"]


def test_a_file_without_a_suffix_is_refused(tmp_path):
    bare = write_song(tmp_path / "Song.wav", seconds=8.0).rename(tmp_path / "Song")
    with pytest.raises(AdamusError, match="has a suffix that tells its kind"):
        Library(tmp_path / "library").draft(bare)


def test_what_is_not_a_song_is_refused_and_leaves_no_draft(tmp_path, library):
    text = tmp_path / "notes.mp3"
    text.write_text("not a song")
    with pytest.raises(AdamusError, match="not a song macOS can read"):
        library.draft(text)
    silent = tmp_path / "silence.wav"
    write_song(silent, seconds=1.0)
    with pytest.raises(AdamusError, match="no beat"):
        library.draft(silent)
    assert list_of(library) == [] and library.songs() == []


def test_songs_go_by_title_whatever_the_case(library, analysed):
    assert library.songs() == []  # no folder yet
    for title in ("beta", "Gamma", "Alpha"):
        kept(library, analysed, title)
    assert [song.title for song in library.songs()] == ["Alpha", "beta", "Gamma"]


def test_a_mark_is_where_it_was_put_as_far_as_the_song_reaches(library, draft):
    song = song_in(library.draft_folder(draft))
    assert library.mark(draft, 1.23, 4.56, c.REST).marks == [Mark(1.23, 4.56, c.REST)]
    marked = library.mark(draft, 7.5, 3.25, c.FOCUS).marks  # dragged backwards
    assert marked == [Mark(1.23, 3.25, c.REST), Mark(3.25, 7.5, c.FOCUS)]
    marked = library.mark(draft, -2.0, SECONDS + 5.0, c.REST).marks
    assert marked == [Mark(0.0, song.duration_s, c.REST)]
    assert library.mark(draft, 5.0, 5.0, c.FOCUS).marks == marked  # nothing between


def test_unmarking_cuts_a_mark_in_two(library, draft):
    library.mark(draft, 2.0, 14.0, c.REST)
    song = library.mark(draft, 6.0, 8.0, None)
    assert song.marks == [Mark(2.0, 6.0, c.REST), Mark(8.0, 14.0, c.REST)]


@pytest.mark.parametrize("start, end", [(float("nan"), 1.0), (0.0, float("inf"))])
def test_a_mark_needs_times_in_the_song(library, draft, start, end):
    with pytest.raises(AdamusError, match="time in the song"):
        library.mark(draft, start, end, c.REST)


def test_a_mark_is_rest_or_focus(library, draft):
    with pytest.raises(AdamusError, match="rest or focus"):
        library.mark(draft, 1.0, 4.0, "calm")


@pytest.mark.parametrize("name", ["missing", "..", ".", "", "a/b", ".hidden", ".drafts"])
def test_only_a_song_of_the_library_or_a_draft_is_found(library, name):
    with pytest.raises(AdamusError, match="no such song"):
        library.song(name)
    with pytest.raises(AdamusError, match="no such draft"):
        library.mark(name, 0.0, 1.0, c.REST)


def test_a_song_is_never_marked_where_the_library_has_it(library, song_id):
    with pytest.raises(AdamusError, match="no such draft"):
        library.mark(song_id, 0.0, 1.0, c.REST)


def test_a_draft_of_a_song_changes_it_only_once_kept_and_whole(library, song_id, tmp_path):
    before = library.song(song_id)
    draft = library.draft_of(song_id).id
    library.mark(draft, 0.0, SECONDS, c.FOCUS)
    library.mark(draft, 0.0, 2.0, c.REST)
    assert library.song(song_id) == before
    picture = tmp_path / "Front.PNG"
    picture.write_bytes(b"a chosen picture")
    song = library.keep(draft, "  Renamed ", " Someone ", picture)
    assert (song.id, song.title, song.artist) == (song_id, "Renamed", "Someone")
    assert song.marks == [Mark(0.0, 2.0, c.REST), Mark(2.0, SECONDS, c.FOCUS)]
    assert song.cover is not None and song.cover.name == "cover.png"
    assert song.cover.read_bytes() == b"a chosen picture"  # the one it carried is gone
    assert list_of(library) == [] and len(library.songs()) == 1


def test_a_song_kept_without_a_picture_chosen_keeps_its_own_and_an_unknown_artist(library, song_id):
    before = library.song(song_id)
    song = library.keep(library.draft_of(song_id).id, before.title, "  ")
    assert song.artist is None and song.cover == before.cover
    assert song.cover is not None and song.cover.read_bytes() == PICTURE


@pytest.mark.parametrize(
    ("title", "picture", "problem"),
    [("   ", None, TITLE_HINT), ("A", "cover.gif", "cover.gif: a picture is a JPEG or PNG file")],
)
def test_a_draft_without_a_title_or_with_a_picture_of_another_kind_is_not_kept(
    library, song_id, tmp_path, title, picture, problem
):
    draft = library.draft_of(song_id).id
    with pytest.raises(AdamusError, match=problem):
        library.keep(draft, title, None, None if picture is None else tmp_path / picture)
    assert library.song(song_id).title == "A Beat" and len(list_of(library)) == 1


def test_a_discarded_draft_leaves_the_song_as_it_was(library, song_id):
    before = library.song(song_id)
    draft = library.draft_of(song_id).id
    library.mark(draft, 0.0, SECONDS, c.FOCUS)
    library.discard(draft)
    assert library.song(song_id) == before and list_of(library) == []


def test_a_draft_of_a_song_deleted_meanwhile_cannot_be_kept(library, song_id):
    draft = library.draft_of(song_id).id
    library.delete(song_id)
    with pytest.raises(AdamusError, match="no such song"):
        library.keep(draft, "A", None)


def test_the_drafts_left_behind_are_put_aside_as_the_library_is_opened(library, draft):
    again = Library(library.root)
    assert list_of(again) == []
    with pytest.raises(AdamusError, match="no such draft"):
        again.keep(draft, "A", None)


def test_a_copy_of_a_song_stays_as_the_library_had_it(library, song_id, tmp_path):
    copy = library.copy(song_id, tmp_path / "session" / "song")
    library.keep(library.draft_of(song_id).id, "Later", None)
    library.delete(song_id)
    assert song_in(copy).title == "A Beat" and song_in(copy).cover is not None
    assert len(song_in(copy).marks) == 2 and analysis_in(copy).beats


def test_marks_made_at_once_from_many_threads_are_all_kept(library, draft):
    stretches = [(2.0 * i, 2.0 * i + 1) for i in range(THREADS)]
    threads = [threading.Thread(target=library.mark, args=(draft, a, b, c.FOCUS))
               for a, b in stretches]  # fmt: skip
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    marks = song_in(library.draft_folder(draft)).marks
    assert marks == [Mark(a, b, c.FOCUS) for a, b in stretches]


@pytest.mark.parametrize("text", ["{", "{}", '{"title": "A", "marks": [{"start": 1}]}'])
def test_a_song_that_cannot_be_read_says_which(library, song_id, text):
    (library.root / song_id / c.FILE_SONG).write_text(text)
    with pytest.raises(AdamusError, match=f"{song_id}: song.json cannot be read"):
        library.song(song_id)


def test_a_deleted_song_is_gone(library, song_id):
    library.delete(song_id)
    assert library.songs() == []
    with pytest.raises(AdamusError):
        library.delete(song_id)


def test_a_file_is_written_whole(tmp_path):
    path = tmp_path / "kept.json"
    write_whole(path, json.dumps({"a": 1}))
    write_whole(path, json.dumps({"a": 2}))
    assert json.loads(path.read_text()) == {"a": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["kept.json"]
    assert not any(p.name.endswith(WRITING) for p in tmp_path.iterdir())


R, F = c.REST, c.FOCUS


@pytest.mark.parametrize(
    "marks, new, expected",
    [
        ([], Mark(2, 4, R), [Mark(2, 4, R)]),
        ([Mark(0, 10, R)], Mark(4, 6, F), [Mark(0, 4, R), Mark(4, 6, F), Mark(6, 10, R)]),
        ([Mark(0, 4, R)], Mark(4, 8, R), [Mark(0, 8, R)]),  # touching, the same kind: one
        ([Mark(0, 4, R), Mark(6, 9, R)], Mark(3, 7, R), [Mark(0, 9, R)]),
        ([Mark(0, 4, F), Mark(4, 8, R)], Mark(2, 6, F), [Mark(0, 6, F), Mark(6, 8, R)]),
        ([Mark(2, 4, R)], Mark(0, 10, F), [Mark(0, 10, F)]),  # covered whole
    ],
)
def test_a_mark_paints_over_the_marks_it_overlaps(marks, new, expected):
    assert painted(marks, new, new.start_s, new.end_s) == expected


def test_unmarking_removes_only_the_stretch():
    assert painted([Mark(0, 10, R)], None, 3, 5) == [Mark(0, 3, R), Mark(5, 10, R)]
    assert painted([Mark(0, 10, R)], None, 0, 10) == []
