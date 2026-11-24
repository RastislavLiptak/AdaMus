"""server.session_catalogue: the stored sessions, newest first, a page at a time; deletion."""

import pytest

from adamus import constants as c
from adamus.server.session_catalogue import SessionCatalogue

FOLDERS = [
    "20261110-150412_first",
    "20261111-090000_second_take",
    "20261112-120000_third",
]


@pytest.fixture
def catalogue(tmp_path):
    for name in FOLDERS:
        (tmp_path / "sessions" / name).mkdir(parents=True)
        (tmp_path / "sessions" / name / "meta.json").write_text("{}")
    (tmp_path / "sessions" / "notes.txt").write_text("not a session")
    (tmp_path / "sessions" / "unsorted").mkdir()
    (tmp_path / "sessions" / "20261399-000000_bad_date").mkdir()
    return SessionCatalogue(tmp_path / "sessions", tmp_path / "data")


def test_sessions_are_listed_newest_first_by_the_name_they_were_given(catalogue):
    page = catalogue.page(0, 10)
    assert page.total == 3
    assert [(s.id, s.name) for s in page.sessions] == [
        ("20261112-120000_third", "third"),
        ("20261111-090000_second_take", "second_take"),
        ("20261110-150412_first", "first"),
    ]


@pytest.mark.parametrize(
    ("offset", "limit", "names"),
    [(0, 2, ["third", "second_take"]), (2, 2, ["first"]), (3, 2, []), (1, 1, ["second_take"])],
)
def test_a_page_is_limit_sessions_from_offset_on(catalogue, offset, limit, names):
    page = catalogue.page(offset, limit)
    assert page.total == 3
    assert [s.name for s in page.sessions] == names


def test_no_sessions_folder_means_no_sessions(tmp_path):
    assert SessionCatalogue(tmp_path / "none", tmp_path).page(0, 5).total == 0


@pytest.mark.parametrize(
    "session_id", ["notes.txt", "unsorted", "../sessions", "20261110-150412_x"]
)
def test_only_a_stored_session_folder_is_found(catalogue, session_id):
    assert catalogue.folder(session_id) is None


def test_a_session_is_found_by_its_folder(catalogue, tmp_path):
    assert catalogue.folder(FOLDERS[0]) == tmp_path / "sessions" / FOLDERS[0]


def test_deleting_takes_the_session_and_its_own_recording(catalogue, tmp_path):
    own = tmp_path / "data" / c.OWN_RECORDINGS / FOLDERS[0]
    own.mkdir(parents=True)
    (own / "eeg.csv").write_text("AF7")
    other = tmp_path / "data" / c.OWN_RECORDINGS / FOLDERS[1]
    other.mkdir()
    assert catalogue.own_recording(FOLDERS[0]) == own
    catalogue.delete(FOLDERS[0])
    assert not (tmp_path / "sessions" / FOLDERS[0]).exists() and not own.exists()
    assert other.is_dir() and (tmp_path / "sessions" / FOLDERS[1]).is_dir()
    assert catalogue.page(0, 10).total == 2


def test_a_session_without_its_own_recording_is_deleted_all_the_same(catalogue, tmp_path):
    catalogue.delete(FOLDERS[2])
    assert catalogue.own_recording(FOLDERS[2]) is None
    assert not (tmp_path / "sessions" / FOLDERS[2]).exists()


def test_deleting_what_is_not_a_session_deletes_nothing(catalogue, tmp_path):
    catalogue.delete("unsorted")
    catalogue.delete("..")
    assert (tmp_path / "sessions" / "unsorted").is_dir()
    assert catalogue.page(0, 10).total == 3
