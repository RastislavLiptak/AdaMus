"""console_log.ConsoleLog: what the console did, stored in the session it led to."""

import time

from adamus.console.console_log import ConsoleLog
from tests.support.finished_sessions import read_lines


def stored(folder):
    return read_lines(folder / "console.jsonl")


def test_every_action_has_its_time_kind_and_fields():
    log = ConsoleLog()
    before = time.time()
    log.write("PD_STARTED", pid=4242)
    log.write("SESSION_NAMED", name="měření")
    log.write("QUIT")
    first, second, third = log.waiting
    assert list(first) == ["t", "type", "pid"]
    assert (first["type"], first["pid"]) == ("PD_STARTED", 4242)
    assert (second["type"], second["name"]) == ("SESSION_NAMED", "měření")
    assert third == {"t": third["t"], "type": "QUIT"}
    assert before <= first["t"] <= second["t"] <= third["t"] <= time.time()


def test_a_field_may_not_take_the_place_of_the_kind():
    log = ConsoleLog()
    log.write("KEY", kind="x", state="READY")
    assert log.waiting[0]["type"] == "KEY"


def test_actions_wait_until_a_session_takes_them(tmp_path):
    log = ConsoleLog()
    log.write("PREFLIGHT", check="Storage")
    log.write("SESSION_NAMED", name="měření")
    assert list(tmp_path.iterdir()) == []  # nothing is written without a session
    log.store_in(tmp_path)
    assert [entry["type"] for entry in stored(tmp_path)] == ["PREFLIGHT", "SESSION_NAMED"]
    assert stored(tmp_path)[1]["name"] == "měření"
    assert log.waiting == []


def test_actions_of_a_running_session_are_written_as_they_happen(tmp_path):
    log = ConsoleLog()
    log.store_in(tmp_path)
    assert stored(tmp_path) == []  # the file is there from the start of the session
    log.write("SESSION_STARTED")
    assert [entry["type"] for entry in stored(tmp_path)] == ["SESSION_STARTED"]
    log.write("SESSION_ENDED", reason="user")
    assert [entry["type"] for entry in stored(tmp_path)] == ["SESSION_STARTED", "SESSION_ENDED"]
    assert log.waiting == []


def test_what_follows_a_session_belongs_to_the_next_one(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    log = ConsoleLog()
    log.write("PREFLIGHT")
    log.store_in(first)
    log.write("SESSION_ENDED")
    log.wait_for_the_next_session()
    log.write("SOURCE_SELECTED")
    assert [entry["type"] for entry in log.waiting] == ["SOURCE_SELECTED"]
    log.store_in(second)
    log.write("SESSION_STARTED")
    assert [entry["type"] for entry in stored(first)] == ["PREFLIGHT", "SESSION_ENDED"]
    assert [entry["type"] for entry in stored(second)] == ["SOURCE_SELECTED", "SESSION_STARTED"]
