"""session_files: the name of a session folder, made and read back; the format of the
hand-made folders of the tests."""

import datetime
import json
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from adamus import constants as c
from adamus.config import SESSION_NAME, is_session_name
from adamus.session_files import Event, Meta, Tick, folder_name, name_of
from tests.support.handmade_session_folders import event, tick, write_session_folder

STARTED = datetime.datetime(2026, 11, 10, 15, 4, 12).timestamp()


def test_folder_is_named_by_the_local_start_time_then_the_session_name():
    assert folder_name(STARTED, "S01") == "20261110-150412_S01"
    assert is_session_name("S01-second_2")


@given(st.from_regex(SESSION_NAME, fullmatch=True))
def test_the_name_of_a_session_comes_back_from_its_folder(name):
    """Names may hold the separator themselves, e.g. ``subject_01``."""
    assert name_of(Path("sessions") / folder_name(STARTED, name)) == name


def test_hand_made_session_folders_are_in_the_format_a_session_writes(tmp_path):
    """The exact-value tests of metrics and statistics read such folders; a session that
    writes another format would leave them testing something else."""
    assert list(tick(0)) == list(Tick.__annotations__)
    assert set(event("SESSION_END", 1.0, reason="user")) <= set(Event.__annotations__)
    folder = write_session_folder(tmp_path / "S", {"protocol": []}, [tick(0)], [])
    meta = json.loads((folder / c.FILE_META).read_text())
    assert set(meta) <= set(Meta.__annotations__)
