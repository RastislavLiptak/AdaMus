"""server.new_session: the form of a new session: its sources, its checks, the session made."""

import shutil

import pytest

from adamus.activity_log import ActivityLog
from adamus.errors import AdamusError
from adamus.server.api_models import SessionRequest
from adamus.server.new_session import NAME_HINT, NewSession, missing_warning
from adamus.session_format import SESSION_NAME
from tests.support.finished_sessions import synthetic_signals
from tests.support.stand_in_machine import StandInMachine
from tests.support.stand_in_preflight import Checked
from tests.support.synthetic_dataset import write_recording

SHORT = [("REST", 5), ("FOCUS", 5)]


@pytest.fixture
def data(tmp_path):
    signals = synthetic_signals([list(block) for block in SHORT])
    write_recording(tmp_path / "zeta" / "one", signals, SHORT)
    write_recording(tmp_path / "alpha" / "S01", signals, SHORT)
    write_recording(tmp_path / "alpha" / "S00", signals, SHORT)
    eeg = tmp_path / "alpha" / "S00" / "eeg.csv"
    eeg.write_text(eeg.read_text().replace("AF7", "Fp1", 1))
    write_recording(tmp_path / "frontal" / "only_fp", signals, SHORT)
    lacking = tmp_path / "frontal" / "only_fp" / "eeg.csv"
    header, rest = lacking.read_text().split("\n", 1)
    lacking.write_text(",".join(f"X{i}" for i in range(len(header.split(",")))) + "\n" + rest)
    return tmp_path


@pytest.fixture
def machine(data):
    """A computer with the headset switched on."""
    return StandInMachine(data.parent / "sessions", data)


@pytest.fixture
def activity():
    return ActivityLog()


@pytest.fixture
def form(machine, activity):
    """The form once the headset has been looked for, and found."""
    form = NewSession(Checked(), machine, activity)
    form.look_for_headset()
    return form


def test_the_form_offers_the_headset_first_then_the_datasets_and_the_rule_of_names(form):
    offered = form.form()
    assert [(s.id, s.title, s.live) for s in offered.sources] == [
        ("headset:Muse 2", "Muse 2 (headset)", True), ("dataset:alpha", "alpha", False),
        ("dataset:frontal", "frontal", False), ("dataset:zeta", "zeta", False),
    ]  # fmt: skip
    assert (offered.name_pattern, offered.name_hint) == (SESSION_NAME, NAME_HINT)
    assert NAME_HINT == "Use at most 50 characters, without “/” or “:”."  # as the form says it
    assert offered.sources_hint == (
        "Switch the headset on and look again, or put a dataset into data/."
    )


def test_the_headset_is_offered_once_a_look_for_it_found_it(machine, activity):
    form = NewSession(Checked(), machine, activity)
    assert [s.id for s in form.form().sources][0] == "dataset:alpha"  # not looked for yet
    machine.headsets_on = set()
    assert [s.id.split(":")[0] for s in form.look_for_headset().sources] == ["dataset"] * 3
    machine.headsets_on = {"Muse 2"}
    assert form.look_for_headset().sources[0].id == "headset:Muse 2"
    looks = [(e["type"], e["headset"], e["found"]) for e in activity.waiting]
    assert looks == [
        ("HEADSET_LOOKED_FOR", "Muse 2", False),
        ("HEADSET_LOOKED_FOR", "Muse 2", True),
    ]
    assert all(0 <= entry["duration"] < 1 for entry in activity.waiting)  # the stand-in at once


def test_the_datasets_are_those_in_the_data_folder_whenever_the_form_is_read(form, data):
    write_recording(data / "beta" / "one", synthetic_signals([list(b) for b in SHORT]), SHORT)
    assert "dataset:beta" in [source.id for source in form.form().sources]
    shutil.rmtree(data / "zeta")
    assert "dataset:zeta" not in [source.id for source in form.form().sources]


def test_the_recordings_of_a_dataset_and_none_of_the_headset(form):
    offered = {source.id: source.recordings for source in form.form().sources}
    assert offered["dataset:alpha"] == ["S00", "S01"]
    assert offered["headset:Muse 2"] == []
    assert "dataset:beta: no such source" in form.check("dataset:beta", None).problem


def test_a_recording_with_every_electrode_has_nothing_missing(form):
    checked = form.check("dataset:alpha", "S01")
    assert (checked.warning, checked.problem) == (None, None)


def test_the_electrodes_a_recording_lacks_are_named(form):
    checked = form.check("dataset:alpha", "S00")
    assert checked.problem is None
    assert checked.warning == "No AF7: recorded as zeros, the detector uses the others."


def test_every_electrode_a_recording_lacks_is_named_in_one_warning():
    said = missing_warning(["AF7", "TP10"])
    assert said == "No AF7, TP10: recorded as zeros, the detector uses the others."
    assert missing_warning([]) is None


def test_a_recording_without_any_electrode_of_the_detector_cannot_run(form):
    checked = form.check("dataset:frontal", "only_fp")
    assert checked.warning is None  # the problem says it all
    assert checked.problem == (
        "The source has none of AF7, AF8, TP9, TP10. The detector needs at least one."
    )


@pytest.mark.parametrize(
    ("source", "recording", "problem"),
    [("dataset:alpha", "S07", "alpha: no recording S07"),
     ("dataset:alpha", None, "alpha: no recording None"),
     ("dataset:beta", "S01", "dataset:beta: no such source")],
)  # fmt: skip
def test_a_source_or_recording_that_is_not_there(form, source, recording, problem):
    assert form.check(source, recording).problem == problem


def test_a_recording_whose_protocol_cannot_be_played_says_why(form, data):
    (data / "zeta" / "one" / "protocol.yaml").write_text("protocol: [[NAP, 5]]\n")
    problem = form.check("dataset:zeta", "one").problem
    assert problem.startswith(f"{data / 'zeta' / 'one' / 'protocol.yaml'}: the recording cannot")


def test_a_session_over_a_recording_takes_its_protocol(form, data):
    prepared = form.prepare(SessionRequest(source_id="dataset:zeta", recording="one", name="S_1"))
    assert prepared.name == "S_1" and prepared.recording == data / "zeta" / "one"
    assert [(b.label, b.duration_s) for b in prepared.config.protocol] == SHORT
    assert prepared.source.label == "zeta"


def test_a_session_with_the_headset_takes_the_configuration(form):
    prepared = form.prepare(SessionRequest(source_id="headset:Muse 2", recording=None, name="L"))
    assert prepared.recording is None and len(prepared.config.protocol) == 9


@pytest.mark.parametrize("name", ["", "  ", "a/b"])
def test_a_name_outside_the_rule_is_refused(form, name):
    with pytest.raises(AdamusError, match=NAME_HINT):
        form.prepare(SessionRequest(source_id="headset:Muse 2", recording=None, name=name))


def test_a_name_is_taken_without_the_spaces_at_its_ends(form):
    request = SessionRequest(source_id="headset:Muse 2", recording=None, name=" Karel Jedlička 8 ")
    assert form.prepare(request).name == "Karel Jedlička 8"


def test_a_source_that_cannot_run_is_refused(form):
    request = SessionRequest(source_id="dataset:frontal", recording="only_fp", name="S")
    with pytest.raises(AdamusError, match="The detector needs at least one"):
        form.prepare(request)
