"""main.Program as the app runs it: what it tells the app from its start to its end, what the
operator does through its API, and how the app's end ends it. The test does what the app would:
it reads the messages, calls the API and closes the program's input."""

import json
import os
import signal
import threading
import time
import urllib.error
import urllib.request

import pytest

from adamus import constants as c
from adamus.preflight import checks
from adamus.server.api import TOKEN_HEADER
from adamus.session_format import read_jsonl
from tests.support.stand_in_devices import SPEAKERS
from tests.support.stand_in_songs import SONG
from tests.support.waiting import wait_until

WAIT_S = 30  # the longest the operator waits for the program
POLL_S = 0.05
CHECKS = 3  # of the preflight


class Operator:
    """Uses the API of a running program, as the app would."""

    def __init__(self, program):
        self.program = program
        wait_until(program.ready, "the program did not say where its API answers", WAIT_S)
        ready = program.ready()
        self.address, self.token = ready["address"], ready["token"]

    def call(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        headers = {TOKEN_HEADER: self.token, "Content-Type": "application/json"}
        request = urllib.request.Request(self.address + path, data, headers, method=method)
        with urllib.request.urlopen(request) as response:
            text = response.read().decode()
        return json.loads(text) if text else None

    def finished(self, run):
        """The state of the run once its session has ended and its metrics are computed."""

        def done():
            state = self.call("GET", "/api/run")
            return state is not None and state["run"] == run and state["phase"] == "idle"

        wait_until(done, "the session did not finish", WAIT_S, POLL_S)
        return self.call("GET", "/api/run")


def operate(program, actions):
    """Start the program; ``actions`` uses it as the app would, then the app ends."""
    failures = []

    def use():
        try:
            actions(Operator(program))
        except Exception as problem:  # noqa: BLE001 - reported by the test below
            failures.append(problem)
        finally:
            program.app.close()

    threading.Thread(target=use, daemon=True).start()
    code = program.start(app_ends_when_ready=False)
    assert failures == []
    return code


def test_a_session_over_a_recording_from_the_form_to_the_dashboard(program):
    program.add_recording("short", [["REST", 20], ["FOCUS", 20]])
    seen = {}

    def actions(operator):
        form = operator.call("POST", "/api/new-session/refresh")  # as the form does as it opens
        seen["sources"] = [source["id"] for source in form["sources"]]
        request = {
            "song_id": SONG,
            "source_id": "dataset:simulated",
            "recording": "short",
            "name": "S01",
            "output": SPEAKERS,
        }
        started = operator.call("POST", "/api/run", request)
        state = operator.finished(started["run"])
        seen["dashboard"] = operator.call("GET", f"/api/sessions/{state['session_id']}/dashboard")
        seen["page"] = operator.call("GET", "/api/sessions?offset=0&limit=10")

    assert operate(program, actions) == 0
    assert seen["sources"] == ["headset:Muse 2", "dataset:simulated"]
    assert seen["page"]["sessions"][0]["name"] == "S01"
    assert seen["dashboard"]["problem"] is None
    (folder,) = program.session_folders()
    assert (folder / c.FILE_METRICS).is_file()
    logged = [entry["type"] for entry in read_jsonl(folder / c.FILE_ACTIVITY)]
    assert logged[:CHECKS] == ["PREFLIGHT"] * CHECKS  # what the program found at its start
    assert logged[CHECKS:] == [
        "HEADSET_LOOKED_FOR", "SESSION_REQUESTED", "SESSION_STARTED", "SESSION_ENDED"
    ]  # fmt: skip
    assert program.machine.musics[-1].stopped == 1


def test_the_app_sees_the_checklist_as_it_changes_then_where_the_api_answers(program):
    seen = {}

    def actions(operator):
        seen["address"] = operator.address

    assert operate(program, actions) == 0
    lists = program.checklists()
    assert [row["name"] for row in lists[0]] == [
        checks.CONFIGURATION, checks.STORAGE, checks.FOCUS_DETECTOR,
    ]  # fmt: skip
    assert {row["status"] for row in lists[0]} == {checks.PENDING}
    assert any(row["status"] == checks.RUNNING for checklist in lists for row in checklist)
    assert {row["status"] for row in lists[-1]} == {checks.PASSED}
    assert all(row["fix"] is None for row in lists[-1])
    assert list(program.messages()[-1]) == ["ready"]  # once every check has passed
    assert seen["address"].startswith(f"http://{c.LOCAL_HOST}:")


def test_the_api_answers_only_calls_with_the_token(program):
    seen = {}

    def actions(operator):
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(operator.address + "/api/run", timeout=WAIT_S)
        seen["refused"] = refused.value.code
        refused.value.close()  # the answer it holds
        seen["answered"] = operator.call("GET", "/api/run")["phase"]

    assert operate(program, actions) == 0
    assert seen == {"refused": 404, "answered": "idle"}


def test_the_server_stops_with_the_program(program):
    seen = {}

    def actions(operator):
        seen["address"] = operator.address
        operator.call("GET", "/api/run")

    assert operate(program, actions) == 0
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(seen["address"] + "/api/run", timeout=WAIT_S)


def test_what_failed_is_told_with_how_to_fix_it_and_the_program_ends(program):
    (program.folder / "sessions").write_text("a file, where the sessions would be")
    assert program.start() == 1
    last = {row["name"]: row for row in program.checklists()[-1]}
    assert last[checks.STORAGE]["status"] == checks.FAILED
    assert last[checks.STORAGE]["fix"] == "Start the program in a folder you can write to."
    assert last[checks.FOCUS_DETECTOR]["status"] == checks.PASSED  # it does not need storage
    assert program.ready() is None


def test_the_form_looks_for_the_headset_not_the_start_of_the_program(program):
    program.machine.headsets_on = set()
    program.add_recording("short")
    seen = {}

    def actions(operator):
        seen["before"] = [s["id"] for s in operator.call("GET", "/api/new-session")["sources"]]
        seen["probed"] = list(program.machine.probed)
        operator.call("POST", "/api/new-session/refresh")  # switched off
        program.machine.headsets_on = {"Muse 2"}
        refreshed = operator.call("POST", "/api/new-session/refresh")
        seen["after"] = [source["id"] for source in refreshed["sources"]]

    assert operate(program, actions) == 0
    assert seen == {
        "before": ["dataset:simulated"], "probed": [],
        "after": ["headset:Muse 2", "dataset:simulated"],
    }  # fmt: skip
    looks = program.logged("HEADSET_LOOKED_FOR")
    assert [(entry["headset"], entry["found"]) for entry in looks] == [
        ("Muse 2", False),
        ("Muse 2", True),
    ]


def test_deleting_a_session_with_the_headset_takes_its_recording_from_the_sources(program):
    program.machine.live_samples = 30 * 256
    seen = {}

    def actions(operator):
        operator.call("POST", "/api/new-session/refresh")  # as the form does as it opens
        request = {
            "song_id": SONG,
            "source_id": "headset:Muse 2",
            "recording": None,
            "name": "L",
            "output": SPEAKERS,
        }
        started = operator.call("POST", "/api/run", request)
        state = operator.finished(started["run"])
        sources = operator.call("GET", "/api/new-session")["sources"]
        seen["with"] = [source["id"] for source in sources]
        operator.call("DELETE", f"/api/sessions/{state['session_id']}")
        seen["without"] = [s["id"] for s in operator.call("GET", "/api/new-session")["sources"]]

    assert operate(program, actions) == 0
    assert seen["with"] == ["headset:Muse 2", f"dataset:{c.OWN_RECORDINGS}"]
    assert seen["without"] == ["headset:Muse 2"]
    assert program.session_folders() == []


def test_the_app_ending_mid_session_ends_the_session_as_a_stop_and_then_the_program(program):
    """The session keeps what it recorded, and its music is stopped: quitting the app, or its
    being killed, loses nothing."""
    program.machine.paced = True
    seen = {}

    def actions(operator):
        operator.call("POST", "/api/new-session/refresh")
        request = {
            "song_id": SONG,
            "source_id": "headset:Muse 2",
            "recording": None,
            "name": "live",
            "output": SPEAKERS,
        }
        operator.call("POST", "/api/run", request)
        wait_until(lambda: operator.call("GET", "/api/run")["phase"] == "running",
                   "the session did not start", WAIT_S, POLL_S)  # fmt: skip
        seen["delivered"] = True

    assert operate(program, actions) == 0
    assert seen == {"delivered": True}
    (folder,) = program.session_folders()
    assert read_jsonl(folder / c.FILE_EVENTS)[-1]["reason"] == "user"
    assert (folder / c.FILE_METRICS).is_file()
    assert program.machine.musics[-1].stopped == 1


def test_the_app_ending_during_the_checks_ends_the_program_at_once(program, monkeypatch):
    """The first training of the focus detector takes minutes."""
    (program.folder / c.FOCUS_MODEL_PATH).unlink()

    def app_ends_meanwhile():
        program.app.close()
        time.sleep(WAIT_S)  # the program is interrupted before this ends

    monkeypatch.setattr(program.machine, "train_focus_detector", app_ends_meanwhile)
    began = time.monotonic()
    assert program.start() == 0
    assert time.monotonic() - began < WAIT_S
    assert program.ready() is None


def test_ctrl_c_during_the_checks_ends_the_program_normally(program, monkeypatch):
    (program.folder / c.FOCUS_MODEL_PATH).unlink()

    def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr(program.machine, "train_focus_detector", interrupted)
    assert program.start() == 0
    assert program.ready() is None


def test_ctrl_c_while_the_api_answers_ends_the_program(program):
    def press_ctrl_c():
        wait_until(program.ready, "the program did not get ready", WAIT_S)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=press_ctrl_c, daemon=True).start()
    assert program.start(app_ends_when_ready=False) == 0
