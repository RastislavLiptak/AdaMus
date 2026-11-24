"""activity_log.ActivityLog: what the program did, stored in the session it led to."""

import time

from adamus.activity_log import ActivityLog
from adamus.session_format import read_jsonl


def stored(folder):
    return read_jsonl(folder / "activity.jsonl")


def test_every_action_has_its_time_kind_and_fields():
    log = ActivityLog()
    before = time.time()
    log.write("PD_STARTED", pid=4242)
    log.write("SESSION_REQUESTED", name="měření")
    log.write("STOP_REQUESTED")
    first, second, third = log.waiting
    assert list(first) == ["t", "type", "pid"]
    assert (first["type"], first["pid"]) == ("PD_STARTED", 4242)
    assert (second["type"], second["name"]) == ("SESSION_REQUESTED", "měření")
    assert third == {"t": third["t"], "type": "STOP_REQUESTED"}
    assert before <= first["t"] <= second["t"] <= third["t"] <= time.time()


def test_a_field_may_not_take_the_place_of_the_kind():
    log = ActivityLog()
    log.write("SESSION_ENDED", kind="x", reason="user")
    assert log.waiting[0]["type"] == "SESSION_ENDED"


def test_actions_wait_until_a_session_takes_them(tmp_path):
    log = ActivityLog()
    log.write("PREFLIGHT", check="Storage")
    log.write("SESSION_REQUESTED", name="měření")
    assert list(tmp_path.iterdir()) == []  # nothing is written without a session
    log.store_in(tmp_path)
    assert [entry["type"] for entry in stored(tmp_path)] == ["PREFLIGHT", "SESSION_REQUESTED"]
    assert stored(tmp_path)[1]["name"] == "měření"
    assert log.waiting == []


def test_actions_of_a_running_session_are_written_as_they_happen(tmp_path):
    log = ActivityLog()
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
    log = ActivityLog()
    log.write("PREFLIGHT")
    log.store_in(first)
    log.write("SESSION_ENDED")
    log.wait_for_the_next_session()
    log.write("SESSION_REQUESTED")
    assert [entry["type"] for entry in log.waiting] == ["SESSION_REQUESTED"]
    log.store_in(second)
    log.write("SESSION_STARTED")
    assert [entry["type"] for entry in stored(first)] == ["PREFLIGHT", "SESSION_ENDED"]
    assert [entry["type"] for entry in stored(second)] == ["SESSION_REQUESTED", "SESSION_STARTED"]


def test_an_action_that_cannot_be_written_is_lost_but_stops_nothing(tmp_path, caplog):
    activity = ActivityLog()
    activity.store_in(tmp_path / "gone")  # no such folder: like a full disk, it is not written
    activity.write("SESSION_ENDED", reason="user")
    assert "could not be written" in caplog.text
