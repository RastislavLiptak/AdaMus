"""session_format: the name of a session folder, made and read back; the format of the
hand-made folders of the tests."""

import datetime
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from adamus import constants as c
from adamus.errors import AdamusError
from adamus.session_format import (
    SESSION_NAME,
    Event,
    Meta,
    Tick,
    folder_name,
    is_session_name,
    log_time,
    name_of,
    read_jsonl,
    read_log_line,
)
from tests.support.handmade_session_folders import event, tick, write_session_folder

STARTED = datetime.datetime(2026, 11, 10, 15, 4, 12).timestamp()


def test_folder_is_named_by_the_local_start_time_then_the_session_name():
    assert folder_name(STARTED, "Karel Jedlička 8") == "20261110-150412_Karel Jedlička 8"


@pytest.mark.parametrize(
    "name",
    ["S01", "Karel Jedlička 8", "měření; ráno", "a.b", "日本語", "x" * 50, " ok ", "a~b"],
)
def test_a_session_may_be_called_by_any_text_a_folder_can_be_named_by(name):
    assert is_session_name(name)


@pytest.mark.parametrize("name", ["", "   ", "a/b", "a:b", "a\nb", "a\tb", "x" * 51, "a\ud800"])
def test_a_name_that_no_folder_could_have_is_refused(name):
    """Empty, a separator of paths, a control character, half of a character (which JSON can
    carry), or longer than a folder's name."""
    assert not is_session_name(name)


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


@given(st.floats(min_value=1e9, max_value=4e9))
def test_a_line_of_a_log_gives_back_the_moment_it_was_written_to_the_millisecond(t):
    read = read_log_line(log_time(t) + " INFO adamus: it began")
    assert read is not None
    written, said = read
    # cut to the millisecond, after the microseconds were rounded
    assert -0.000001 <= t - written < 0.001 + 0.000001
    assert said == "INFO adamus: it began"


def test_the_time_of_a_line_of_a_log_is_the_local_one_with_its_offset():
    t = datetime.datetime(2026, 11, 19, 7, 30, 15, 123000).timestamp()
    assert log_time(t).startswith("2026-11-19T07:30:15.123")
    assert datetime.datetime.fromisoformat(log_time(t)).utcoffset() is not None


def test_a_line_without_a_moment_goes_on_with_the_one_before():
    assert read_log_line('  File "session.py", line 3, in run') is None
    assert read_log_line("2026-11-19 07:30:15 a time without its offset") is None
    assert read_log_line("") is None


def test_a_last_line_cut_short_is_left_out_and_a_whole_one_kept(tmp_path):
    """The program may end while it writes a line: what came before is the record."""
    path = tmp_path / c.FILE_TICKS
    path.write_text('{"k": 0}\n{"k": 1}\n{"k": 2, "va')
    assert read_jsonl(path) == [{"k": 0}, {"k": 1}]
    path.write_text('{"k": 0}\n{"k": 1}')  # whole, though without its line feed
    assert read_jsonl(path) == [{"k": 0}, {"k": 1}]
    path.write_text('{"k": 0}\nbroken\n{"k": 2}\n')
    with pytest.raises(AdamusError) as damaged:  # not at the end: damage, not an end
        read_jsonl(path)
    said = "Expecting value: line 1 column 1 (char 0)"  # as json says it of the broken line
    assert str(damaged.value) == f"{c.FILE_TICKS} is damaged ({said})"
