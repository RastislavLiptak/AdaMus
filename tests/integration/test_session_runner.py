"""server.session_runner: the sessions the app asks for, run in the main thread."""

import json
import threading

import pytest

from adamus import constants as c
from adamus.activity_log import ActivityLog
from adamus.errors import AdamusError
from adamus.server import session_runner
from adamus.server.api_models import SessionRequest
from adamus.server.dashboard import DID_NOT_START, dashboard_of
from adamus.server.new_session import NewSession
from adamus.server.session_runner import SessionRunner
from adamus.session_format import read_jsonl
from tests.support.finished_sessions import synthetic_signals
from tests.support.stand_in_devices import SPEAKERS, TV
from tests.support.stand_in_machine import StandInMachine
from tests.support.stand_in_preflight import Checked
from tests.support.synthetic_dataset import write_recording
from tests.support.waiting import wait_until

SHORT = [("REST", 10), ("FOCUS", 10)]
WAIT_S = 20  # the longest a test waits for the runner


class Run:
    """A runner over a stand-in machine in ``folder``, serving until the test closes it."""

    def __init__(self, folder):
        self.machine = StandInMachine(folder / "sessions", folder / "data")
        write_recording(folder / "data" / "synthetic" / "short", synthetic_signals(SHORT), SHORT)
        self.checked = Checked()
        self.activity = ActivityLog()
        self.app_gone = threading.Event()
        self.runner = SessionRunner(self.checked, self.machine, self.activity, self.app_gone.is_set)
        # The form logs its looks for the headset apart: tests/unit/test_new_session.py.
        self.form = NewSession(self.checked, self.machine, ActivityLog())
        self.form.look_for_headset()

    def ask(self, source="dataset:synthetic", recording="short", name="S01", output=SPEAKERS):
        request = SessionRequest(source_id=source, recording=recording, name=name, output=output)
        return self.runner.request(self.form.prepare(request))

    def serve_one(self, while_running=None):
        """Serve until the asked-for session is finished; ``while_running`` is done meanwhile."""
        asked = self.runner.state().run

        def finished():
            state = self.runner.state()
            return state.run == asked and state.phase == "idle"

        def watch():
            if while_running is not None:
                wait_until(lambda: self.runner.state().phase == "running" and self.delivered(),
                           "the session did not start", WAIT_S)  # fmt: skip
                while_running()
            wait_until(finished, "the session did not finish", WAIT_S)
            self.app_gone.set()

        threading.Thread(target=watch, daemon=True).start()
        self.runner.serve()
        return self.runner.state()

    def delivered(self):
        return bool(self.machine.sources and self.machine.sources[-1].delivered)

    def actions(self, folder):
        return [entry["type"] for entry in read_jsonl(folder / c.FILE_ACTIVITY)]


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path)


def test_a_session_over_a_recording_runs_to_its_metrics(run, tmp_path, monkeypatch):
    phases = []

    def analyze(directory):
        phases.append(run.runner.state().phase)  # while its metrics are computed
        return real_analyze(directory)

    real_analyze = session_runner.analyze
    monkeypatch.setattr(session_runner, "analyze", analyze)
    assert run.ask().phase == "starting"
    state = run.serve_one()
    assert phases == ["finishing"]
    (folder,) = (tmp_path / "sessions").iterdir()
    assert (state.run, state.phase, state.session_id, state.problem) == (
        1,
        "idle",
        folder.name,
        None,
    )
    assert (folder / c.FILE_METRICS).is_file()
    assert run.actions(folder) == ["SESSION_REQUESTED", "SESSION_STARTED", "SESSION_ENDED"]
    requested = read_jsonl(folder / c.FILE_ACTIVITY)[0]
    assert (requested["kind"], requested["label"], requested["recording"], requested["name"]) == (
        "dataset", "synthetic", "short", "S01",
    )  # fmt: skip
    assert read_jsonl(folder / c.FILE_ACTIVITY)[-1]["reason"] == "end_of_stream"
    assert not (tmp_path / c.DATASETS_DIR / c.OWN_RECORDINGS).exists()


def test_a_session_with_the_headset_stops_when_asked_and_keeps_its_recording(run, tmp_path):
    run.machine.paced = True
    run.ask(source="headset:Muse 2", recording=None, name="live")
    state = run.serve_one(while_running=run.runner.stop)
    folder = tmp_path / "sessions" / state.session_id
    assert run.actions(folder) == [
        "SESSION_REQUESTED", "SESSION_STARTED", "STOP_REQUESTED", "SESSION_ENDED",
    ]  # fmt: skip
    assert read_jsonl(folder / c.FILE_EVENTS)[-1]["reason"] == "user"
    assert (
        tmp_path / c.DATASETS_DIR / c.OWN_RECORDINGS / folder.name / c.FILE_DATASET_EEG
    ).is_file()
    assert run.machine.sources[-1].delivered < run.machine.sources[-1]._matrix.shape[1]


def test_a_session_that_cannot_start_leaves_its_reason_and_no_folder(run, tmp_path):
    run.machine.source_failure = "the headset went away"
    run.ask()
    state = run.serve_one()
    assert (state.phase, state.session_id, state.problem) == ("idle", None, "the headset went away")
    assert not (tmp_path / "sessions").exists()
    waiting = run.activity.waiting  # goes with the next session, as what happened before it
    assert [entry["type"] for entry in waiting] == ["SESSION_REQUESTED", "START_FAILED"]
    assert waiting[-1]["message"] == "the headset went away"

    run.machine.source_failure = ""
    run.app_gone.clear()
    run.ask(name="next")
    folder = tmp_path / "sessions" / run.serve_one().session_id
    assert run.actions(folder)[:3] == ["SESSION_REQUESTED", "START_FAILED", "SESSION_REQUESTED"]
    assert dashboard_of(folder, None, False).problem is None  # it did start


def test_a_session_that_fails_after_its_folder_was_made_lets_the_next_one_log_alone(
    run, tmp_path, monkeypatch
):
    real = session_runner.run_session

    def fails_once_started(config, name, devices, root, on_folder_created):
        if name == "first":
            folder = root / f"20260101-000000_{name}"
            folder.mkdir(parents=True)
            on_folder_created(folder)
            raise OSError("disk full")
        return real(config, name, devices, root, on_folder_created)

    monkeypatch.setattr(session_runner, "run_session", fails_once_started)
    run.ask(name="first")
    assert run.serve_one().problem == "disk full"
    run.app_gone.clear()
    run.ask(name="second")
    second = run.serve_one()
    first = tmp_path / "sessions" / "20260101-000000_first"
    assert run.actions(first) == ["SESSION_REQUESTED", "SESSION_STARTED", "START_FAILED"]
    assert read_jsonl(first / c.FILE_ACTIVITY)[-1]["message"] == "disk full"
    assert dashboard_of(first, None, False).problem == f"{DID_NOT_START}: disk full"
    assert run.actions(tmp_path / "sessions" / second.session_id) == [
        "SESSION_REQUESTED", "SESSION_STARTED", "SESSION_ENDED",
    ]  # fmt: skip


def test_while_the_headset_is_looked_for_only_a_session_from_a_recording_starts(run):
    looking, done = threading.Event(), threading.Event()
    looked = []

    def look_for_the_headset():
        looking.set()
        done.wait(WAIT_S)  # a look for the headset takes seconds
        return "found"

    def look():
        looked.append(run.runner.look_for_headset(look_for_the_headset))

    thread = threading.Thread(target=look)
    thread.start()
    looking.wait(WAIT_S)
    assert run.runner.looking_for_headset()
    assert run.ask(source="headset:Muse 2", recording=None) is None  # it is being looked for
    assert run.runner.look_for_headset(look_for_the_headset) is None  # one look at a time
    assert run.ask() is not None  # a recording needs no headset
    done.set()
    thread.join(WAIT_S)
    assert looked == ["found"] and not run.runner.looking_for_headset()
    assert run.runner.look_for_headset(look_for_the_headset) is None  # a session is starting now


def test_a_look_that_fails_is_over_all_the_same(run):
    def fails():
        raise OSError("Bluetooth is off")

    with pytest.raises(OSError):
        run.runner.look_for_headset(fails)
    assert not run.runner.looking_for_headset()


def test_a_second_request_while_a_session_runs_is_refused(run):
    assert run.ask() is not None
    assert run.ask(name="S02") is None
    assert run.runner.state().run == 1


def test_a_stop_without_a_session_is_not_logged(run):
    run.runner.stop()
    assert run.activity.waiting == []


def test_the_app_ending_stops_the_running_session_says_so_and_stops_pd_first(
    run, tmp_path, monkeypatch
):
    """Pd goes at once, not after the storing and the analysis, which may take a while."""
    generator = run.checked.generator  # started by the checks, as the program starts it
    real_analyze = session_runner.analyze
    stopped_when_analysed = []

    def analyze(directory):
        stopped_when_analysed.append(generator.stopped)
        real_analyze(directory)

    monkeypatch.setattr(session_runner, "analyze", analyze)
    run.machine.paced = True
    run.ask(source="headset:Muse 2", recording=None, name="live")
    state = run.serve_one(while_running=run.app_gone.set)
    folder = tmp_path / "sessions" / state.session_id
    assert read_jsonl(folder / c.FILE_EVENTS)[-1]["reason"] == "user"
    assert "STOP_REQUESTED" not in run.actions(folder) and "APP_ENDED" in run.actions(folder)
    assert stopped_when_analysed == [1]
    (run_section,) = [s for s in dashboard_of(folder, None, False).sections if s.title == "Run"]
    ended = [i.value.value for i in run_section.items if i.kind == "fact" and i.label == "Ended By"]
    assert ended == ["The app was closed"]


def test_ctrl_c_during_a_session_quits_after_keeping_the_recording(run, tmp_path, monkeypatch):
    from adamus.realtime import session as session_module

    real_run = session_module.Session.run

    def interrupted(self):
        result = real_run(self)
        return session_module.SessionResult(result.directory, result.reason, True, None)

    monkeypatch.setattr(session_module.Session, "run", interrupted)
    run.ask(source="headset:Muse 2", recording=None, name="live")
    run.machine.live_samples = 30 * 256
    with pytest.raises(KeyboardInterrupt):
        run.runner.serve()
    (folder,) = (tmp_path / "sessions").iterdir()
    assert (tmp_path / c.DATASETS_DIR / c.OWN_RECORDINGS / folder.name).is_dir()
    assert not (folder / c.FILE_METRICS).exists() and "PROGRAM_STOPPED" in run.actions(folder)
    assert run.runner.state().phase == "idle"


@pytest.mark.parametrize(
    "problem",
    [AdamusError("no filtered.f32"), ValueError("an odd session")],  # foreseen, or not
    ids=["foreseen", "unforeseen"],
)
def test_metrics_that_cannot_be_computed_are_noted_and_the_program_goes_on(
    run, tmp_path, monkeypatch, problem
):
    def failing(directory):
        raise problem

    monkeypatch.setattr(session_runner, "analyze", failing)
    run.ask()
    state = run.serve_one()
    lines = read_jsonl(tmp_path / "sessions" / state.session_id / c.FILE_ACTIVITY)
    assert (lines[-1]["type"], lines[-1]["message"]) == ("ANALYSIS_FAILED", str(problem))
    assert run.runner.state().phase == "idle"  # ready for the next session


def test_a_recording_that_cannot_be_stored_is_noted_in_the_activity(run, tmp_path, monkeypatch):
    from adamus.server import session_runner

    def failing(directory, root):
        raise OSError("disk full")

    monkeypatch.setattr(session_runner, "store_recording", failing)
    run.machine.live_samples = 30 * 256
    run.ask(source="headset:Muse 2", recording=None, name="live")
    state = run.serve_one()
    lines = read_jsonl(tmp_path / "sessions" / state.session_id / c.FILE_ACTIVITY)
    stored = [line for line in lines if line["type"] == "RECORDING_NOT_STORED"]
    assert stored == [{**stored[0], "message": "disk full"}]
    assert (tmp_path / "sessions" / state.session_id / c.FILE_METRICS).is_file()


def test_the_runner_ends_at_once_when_the_app_has_gone(run):
    run.app_gone.set()
    run.runner.serve()
    assert run.runner.state().phase == "idle"


def test_what_follows_a_finished_session_belongs_to_the_next_one(run, tmp_path):
    run.ask()
    state = run.serve_one()
    run.activity.write("PREFLIGHT")
    assert run.actions(tmp_path / "sessions" / state.session_id)[-1] != "PREFLIGHT"
    assert [entry["type"] for entry in run.activity.waiting] == ["PREFLIGHT"]


def test_a_session_on_another_output_starts_the_generator_again_there_first(run, tmp_path):
    generator = run.checked.generator
    run.ask(output=TV)
    folder = tmp_path / "sessions" / run.serve_one().session_id
    assert (generator.outputs, generator.stopped, generator.output) == ([SPEAKERS, TV], 1, TV)
    assert json.loads((folder / c.FILE_META).read_text())["audio_output"] == TV


def test_a_session_on_the_output_the_music_plays_on_starts_nothing_again(run):
    run.ask()
    run.serve_one()
    assert run.checked.generator.outputs == [SPEAKERS]


def test_a_session_on_an_output_that_was_disconnected_starts_the_generator_again(run):
    run.checked.generator.output_lost = True  # and is back, as the form offers it again
    run.ask()
    run.serve_one()
    assert run.checked.generator.outputs == [SPEAKERS, SPEAKERS]


def test_a_generator_that_cannot_start_on_the_output_starts_no_session(run, tmp_path):
    run.checked.generator._fail_to_start = ("Pd could not play on TV", "Pure Data printed:")
    run.ask(output=TV)
    state = run.serve_one()
    assert (state.phase, state.session_id) == ("idle", None)
    assert state.problem == "Pd could not play on TV\nPure Data printed:"
    assert not (tmp_path / "sessions").exists()
    assert [entry["type"] for entry in run.activity.waiting] == [
        "SESSION_REQUESTED",
        "START_FAILED",
    ]


def test_a_session_whose_output_is_disconnected_ends_with_an_error_the_operator_is_told(
    run, tmp_path
):
    """The music heard nowhere, the session is of no use: it ends, its recording kept."""
    generator = run.checked.generator
    run.machine.paced = True
    run.ask(source="headset:Muse 2", recording=None, name="live")

    def disconnected():
        generator.output_lost = True

    state = run.serve_one(while_running=disconnected)
    expected = f"The audio output {SPEAKERS} was disconnected, so the music could not be heard."
    assert (state.phase, state.problem) == ("idle", expected)
    folder = tmp_path / "sessions" / state.session_id
    events = read_jsonl(folder / c.FILE_EVENTS)
    assert [(e["where"], e["message"]) for e in events if e["type"] == "ERROR"] == [
        ("generator", expected)
    ]
    assert events[-1]["reason"] == "error"
    assert generator.messages[-1] == "rec stop;"  # the recording of the music is closed
    assert (
        tmp_path / c.DATASETS_DIR / c.OWN_RECORDINGS / folder.name / c.FILE_DATASET_EEG
    ).is_file()
