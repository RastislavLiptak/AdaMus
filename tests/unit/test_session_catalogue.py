"""server.session_catalogue: the stored sessions, newest first, a page at a time; renaming,
deletion."""

from pathlib import Path

import pytest

from adamus import constants as c
from adamus.errors import AdamusError
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


def test_renaming_renames_the_session_and_its_own_recording(catalogue, tmp_path):
    own = tmp_path / "data" / c.OWN_RECORDINGS / FOLDERS[0]
    own.mkdir(parents=True)
    (own / "eeg.csv").write_text("AF7")
    summary = catalogue.rename(FOLDERS[0], "Karel 8")
    assert (summary.id, summary.name) == ("20261110-150412_Karel 8", "Karel 8")
    assert (tmp_path / "sessions" / summary.id / "meta.json").read_text() == "{}"
    assert (catalogue.own_recording(summary.id) / "eeg.csv").read_text() == "AF7"
    assert catalogue.folder(FOLDERS[0]) is None and catalogue.own_recording(FOLDERS[0]) is None
    assert [s.name for s in catalogue.page(0, 10).sessions] == ["third", "second_take", "Karel 8"]


def test_a_session_without_its_own_recording_is_renamed_all_the_same(catalogue, tmp_path):
    catalogue.rename(FOLDERS[2], "renamed")
    assert (
        catalogue.folder("20261112-120000_renamed")
        == tmp_path / "sessions" / "20261112-120000_renamed"
    )
    assert catalogue.own_recording("20261112-120000_renamed") is None


def test_only_a_letter_of_another_size_may_be_changed(catalogue):
    assert catalogue.rename(FOLDERS[2], "Third").id == "20261112-120000_Third"
    assert [s.name for s in catalogue.page(0, 10).sessions][0] == "Third"


def test_a_name_another_session_of_the_same_start_has_is_refused(catalogue, tmp_path):
    (tmp_path / "sessions" / "20261110-150412_taken").mkdir()
    with pytest.raises(AdamusError, match="started at the same time is called taken"):
        catalogue.rename(FOLDERS[0], "taken")
    assert catalogue.folder(FOLDERS[0]) is not None


def test_an_own_recording_of_that_name_also_refuses_it(catalogue, tmp_path):
    (tmp_path / "data" / c.OWN_RECORDINGS / "20261110-150412_taken").mkdir(parents=True)
    with pytest.raises(AdamusError):
        catalogue.rename(FOLDERS[0], "taken")
    assert catalogue.folder(FOLDERS[0]) is not None


def test_the_session_keeps_its_name_when_its_own_recording_cannot_move(
    catalogue, tmp_path, monkeypatch
):
    own = tmp_path / "data" / c.OWN_RECORDINGS / FOLDERS[0]
    own.mkdir(parents=True)
    rename = Path.rename

    def refused_for_the_recording(self, target):
        if self == own:
            raise PermissionError(13, "Permission denied")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", refused_for_the_recording)
    with pytest.raises(PermissionError):
        catalogue.rename(FOLDERS[0], "new")
    assert catalogue.folder(FOLDERS[0]) is not None and own.is_dir()
    assert catalogue.folder("20261110-150412_new") is None
