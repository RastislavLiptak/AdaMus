"""preflight.checks: the six checks run when the program starts, as the app is told them."""

import json

import pytest

from adamus import constants as c
from adamus.activity_log import ActivityLog
from adamus.audio_outputs import AudioOutput
from adamus.config import write_yaml
from adamus.preflight import checks
from adamus.preflight.checks import FAILED, PASSED, PENDING, RUNNING, SKIPPED, Preflight
from tests.support.project_paths import PROJECT_MODEL
from tests.support.stand_in_devices import StandInGenerator


def people():
    """How many people the project's focus models were trained on."""
    return len(json.loads(PROJECT_MODEL.read_text())["trained_on"]["people"])


def told(program):
    """The checklist the app was told last, by the name of the check: (status, detail)."""
    return {row["name"]: (row["status"], row["detail"]) for row in program.checklists()[-1]}


def fixes(program):
    """How to fix each check that failed, as the app was told last."""
    return {row["name"]: row["fix"] for row in program.checklists()[-1] if row["fix"]}


class TestAllPass:
    def test_every_check_says_what_it_found(self, program):
        program.add_recording("first")
        assert program.start() == 0
        assert told(program) == {
            "Configuration": (PASSED, "config.yaml, 9 protocol blocks"),
            "Storage": (PASSED, "sessions/ writable"),
            "Focus detector": (PASSED, f"{c.FOCUS_MODEL_PATH}, {people()} people"),
            "Audio output": (PASSED, "MacBook Pro Speakers, 48000 Hz"),
            "Generator files": (PASSED, "pd, mrt2~, patch, mrt2_small"),
            "Generator start": (PASSED, "model loaded, playing on MacBook Pro Speakers"),
        }
        assert fixes(program) == {}
        assert program.machine.generator.outputs == ["MacBook Pro Speakers"]  # started once

    def test_results_are_logged_in_order_with_their_duration(self, program):
        program.start()
        entries = program.logged("PREFLIGHT")
        assert [(e["check"], e["result"]) for e in entries] == [
            ("Configuration", "passed"), ("Storage", "passed"), ("Focus detector", "passed"),
            ("Audio output", "passed"), ("Generator files", "passed"),
            ("Generator start", "passed"),
        ]  # fmt: skip
        assert all(e["duration"] >= 0 and "detail" in e and e["t"] > 0 for e in entries)


class TestRequiredChecks:
    def test_missing_configuration_skips_what_needs_it(self, program):
        assert program.start(write_config=False) == 1
        rows = told(program)
        assert rows["Configuration"][0] == FAILED
        assert rows["Configuration"][1].startswith("cannot read config.yaml")
        # Storage and the audio output do not depend on the configuration.
        assert {rows[name][0] for name in ("Storage", "Audio output")} == {PASSED}
        # The focus detector waits for it: a typo must not start a long training first.
        skipped = ("Focus detector", "Generator files", "Generator start")
        assert {rows[name] for name in skipped} == {(SKIPPED, SKIPPED)}
        assert program.machine.generator.started == 0
        assert program.ready() is None

    def test_only_the_checks_that_failed_themselves_say_how_to_fix_them(self, program):
        program.start(write_config=False)
        assert fixes(program) == {"Configuration": checks.HOW_TO_FIX[checks.CONFIGURATION]}
        assert "config.yaml in the repository lists every key" in fixes(program)["Configuration"]

    def test_invalid_configuration_is_reported_with_its_problems(self, program):
        program.raw["styles"][0]["focus"] = "disco 80"
        assert program.start() == 1
        assert "styles[0].focus: not a valid style text" in told(program)["Configuration"][1]

    @pytest.mark.parametrize("model", ["missing", "{}", "not json"])
    def test_a_focus_model_that_cannot_be_used_is_trained(self, program, model):
        path = program.folder / c.FOCUS_MODEL_PATH
        path.unlink() if model == "missing" else path.write_text(model)
        program.machine.training = "trains"
        assert program.start() == 0
        trained = f"trained: {c.FOCUS_MODEL_PATH}, {people()} people"
        assert told(program)["Focus detector"] == (PASSED, trained)
        assert program.machine.trainings == 1

    def test_a_focus_model_of_other_features_is_trained_again(self, program):
        path = program.folder / c.FOCUS_MODEL_PATH
        written = json.loads(path.read_text())
        written["features"]["artifact_uv"] = 1.0
        path.write_text(json.dumps(written))
        program.machine.training = "trains"
        assert program.start() == 0
        assert program.machine.trainings == 1

    def test_a_usable_focus_model_is_not_trained_again(self, program):
        assert program.start() == 0
        assert program.machine.trainings == 0

    def test_a_training_that_fails_is_reported_with_its_log(self, program):
        (program.folder / c.FOCUS_MODEL_PATH).unlink()
        assert program.start() == 1
        log_file = c.FOCUS_DETECTOR_DIR / c.SETUP_LOG
        assert told(program)["Focus detector"] == (
            FAILED, f"the training of the focus detector failed, see {log_file}",
        )  # fmt: skip
        assert (
            "the first training downloads its data from physionet.org"
            in (fixes(program)["Focus detector"])
        )

    def test_a_model_still_unusable_after_the_training_is_named_with_the_problem(self, program):
        (program.folder / c.FOCUS_MODEL_PATH).unlink()
        program.machine.training = "does nothing"
        assert program.start() == 1
        log_file = c.FOCUS_DETECTOR_DIR / c.SETUP_LOG
        assert told(program)["Focus detector"][1].startswith(
            f"still missing after the setup ({log_file}): cannot read the focus model "
            f"{c.FOCUS_MODEL_PATH}: [Errno 2]"
        )

    def test_storage_that_is_not_a_folder(self, program):
        (program.folder / "sessions").write_text("a file")
        assert program.start() == 1
        rows = told(program)
        assert rows["Storage"][0] == FAILED
        assert rows["Generator files"][0] == rows["Generator start"][0] == PASSED  # no storage
        assert (program.machine.generator.started, program.machine.generator.stopped) == (1, 1)

    def test_storage_may_lie_on_any_path(self, program, tmp_path):
        spaced = tmp_path / "my folder; 2026"
        spaced.mkdir()
        program.machine.sessions_root = spaced / "sessions"
        assert program.start() == 0
        assert (spaced / "sessions").is_dir()

    @pytest.mark.parametrize("rate", [44100.0, 96000.0])
    def test_audio_output_must_run_at_48_khz(self, program, rate):
        program.machine.audio = AudioOutput("USB Headphones", rate)
        assert program.start() == 1
        rows = told(program)
        assert rows["Audio output"] == (FAILED, f"USB Headphones, {rate:g} Hz (need 48000 Hz)")
        assert rows["Generator start"] == (SKIPPED, SKIPPED)
        assert fixes(program)["Audio output"] == (
            "Set the output device to 48000 Hz in Audio MIDI Setup."
        )

    @pytest.mark.parametrize(
        "missing",
        [c.GENERATOR_PD_BINARY, c.GENERATOR_EXTERNAL_FILE,
         c.GENERATOR_MODEL_FILE.format(model="mrt2_small"),
         *(f"{c.GENERATOR_ASSETS}/{file}" for file in c.GENERATOR_ASSET_FILES)],
    )  # fmt: skip
    def test_a_missing_file_of_the_generator_starts_its_installation(self, program, missing):
        """Also a single file of the assets: what an interrupted download leaves behind."""
        (program.folder / "generator" / missing).unlink()
        program.machine.installation = "installs"
        assert program.start() == 0
        rows = told(program)
        assert rows["Generator files"] == (PASSED, "installed: pd, mrt2~, patch, mrt2_small")
        assert rows["Generator start"][0] == PASSED
        assert program.machine.installed == ["generator"]

    def test_an_installation_that_fails_is_reported_with_its_log(self, program):
        (program.folder / "generator" / c.GENERATOR_PD_BINARY).unlink()
        assert program.start() == 1
        rows = told(program)
        assert rows["Generator files"] == (
            FAILED, "the installation of the generator failed, see generator/setup.log",
        )  # fmt: skip
        assert rows["Generator start"] == (SKIPPED, SKIPPED)
        assert "or read the installation log named there" in fixes(program)["Generator files"]

    def test_files_still_missing_after_the_installation_are_named(self, program):
        (program.folder / "generator" / c.GENERATOR_EXTERNAL_FILE).unlink()
        program.machine.installation = "does nothing"
        assert program.start() == 1
        assert told(program)["Generator files"][1] == (
            "still missing after the setup (generator/setup.log): "
            f"generator/{c.GENERATOR_EXTERNAL_FILE}"
        )

    def test_installed_generator_is_not_installed_again(self, program):
        assert program.start() == 0
        assert program.machine.installed == []

    def test_a_generator_that_fails_to_start_says_why_itself(self, program):
        """With what Pure Data printed: nothing is stored without a session, so the app is
        where it can be read."""
        program.machine.generator = StandInGenerator(
            fail_to_start=(
                "mrt2~ did not load the model within 120 s",
                "Pure Data printed:",
                "  mrt2~: loading model /somewhere/model.mlxfn",
            )
        )
        assert program.start() == 1
        assert told(program)["Generator start"] == (
            FAILED,
            "mrt2~ did not load the model within 120 s\nPure Data printed:\n"
            "  mrt2~: loading model /somewhere/model.mlxfn",
        )
        assert fixes(program) == {}  # no advice follows: the generator says what went wrong


def test_remedies_exist_for_every_check_but_the_start_of_the_generator():
    """A generator that does not start says why itself, in the words of Pure Data."""
    assert sorted(checks.HOW_TO_FIX) == sorted([
        checks.CONFIGURATION, checks.STORAGE, checks.FOCUS_DETECTOR, checks.AUDIO_OUTPUT,
        checks.GENERATOR_FILES,
    ])  # fmt: skip


def run_preflight(program, shown):
    """The checks of the configuration of the run; ``shown`` is told each checklist."""
    write_yaml(program.raw, program.folder / "config.yaml")
    preflight = Preflight(program.folder / "config.yaml", program.machine, shown, ActivityLog())
    return preflight.run()


class Shown:
    """What the app is told as the checks go: (name, status, detail) of each check, each time."""

    def __init__(self):
        self.lists = []

    def __call__(self, checklist):
        self.lists.append([(check.name, check.status, check.detail) for check in checklist])


class TestWhatTheOperatorSees:
    @pytest.fixture
    def shown(self, program):
        shown = Shown()
        assert run_preflight(program, shown) is True
        return shown

    def test_the_whole_list_is_shown_before_the_first_check_runs(self, shown):
        assert len(shown.lists[0]) == 6
        assert {status for _, status, _ in shown.lists[0]} == {PENDING}

    def test_every_check_is_shown_running_and_then_with_its_result(self, shown):
        for position, name in enumerate(
            ["Configuration", "Storage", "Focus detector", "Audio output", "Generator files",
             "Generator start"]
        ):  # fmt: skip
            states = [checklist[position][:2] for checklist in shown.lists]
            first_running = states.index((name, RUNNING))
            assert states[first_running + 1] == (name, PASSED)
            assert set(states[:first_running]) == {(name, PENDING)}

    def test_only_the_generator_start_says_what_it_waits_for(self, shown):
        details = {
            (name, detail)
            for checklist in shown.lists
            for name, status, detail in checklist
            if status == RUNNING
        }
        assert ("Generator start", "loading model ...") in details
        assert {detail for name, detail in details if name != "Generator start"} == {""}

    def announced_while_making(self, program, method):
        """The running checks shown while the machine's ``method`` runs."""
        shown = Shown()
        seen = []
        make = getattr(program.machine, method)

        def watched(*arguments):
            seen.append([(name, detail) for name, status, detail in shown.lists[-1]
                         if status == RUNNING])  # fmt: skip
            return make(*arguments)

        setattr(program.machine, method, watched)
        assert run_preflight(program, shown) is True
        return seen

    def test_an_installation_is_announced_before_it_runs(self, program):
        (program.folder / "generator" / c.GENERATOR_PD_BINARY).unlink()
        program.machine.installation = "installs"
        announced = ("Generator files", "installing the music generator (this takes a while) ...")
        assert self.announced_while_making(program, "install_generator") == [[announced]]

    def test_a_training_is_announced_before_it_runs(self, program):
        (program.folder / c.FOCUS_MODEL_PATH).unlink()
        program.machine.training = "trains"
        announced = ("Focus detector", "training the focus detector (this takes a while) ...")
        assert self.announced_while_making(program, "train_focus_detector") == [[announced]]


class TestLogAndDetails:
    def test_failed_and_skipped_checks_are_logged_as_such(self, program):
        program.machine.audio = AudioOutput("USB Headphones", 44100.0)
        program.start()
        results = {entry["check"]: entry["result"] for entry in program.logged("PREFLIGHT")}
        assert (results["Audio output"], results["Generator start"]) == ("failed", "skipped")
        assert results["Storage"] == "passed"

    def test_results_of_a_run_that_ends_in_the_preflight_are_stored_nowhere(self, program):
        program.machine.audio = AudioOutput("USB Headphones", 44100.0)
        assert program.start() == 1
        assert list((program.folder / "sessions").iterdir()) == []

    def test_duration_of_a_check_is_the_time_it_took(self, program):
        program.start()
        durations = [entry["duration"] for entry in program.logged("PREFLIGHT")]
        assert all(0 <= duration < 30 for duration in durations)

    def test_every_missing_file_of_the_generator_is_named(self, program):
        folder = program.folder / "generator"
        (folder / c.GENERATOR_PD_BINARY).unlink()
        (folder / c.GENERATOR_EXTERNAL_FILE).unlink()
        program.machine.installation = "does nothing"
        assert program.start() == 1
        missing = f"generator/{c.GENERATOR_PD_BINARY}, generator/{c.GENERATOR_EXTERNAL_FILE}"
        assert missing in told(program)["Generator files"][1]

    def test_sessions_folder_is_created_with_the_folders_above_it(self, program, tmp_path):
        program.machine.sessions_root = tmp_path / "new" / "deeper" / "sessions"
        assert program.start() == 0
        assert (tmp_path / "new" / "deeper" / "sessions").is_dir()

    def test_sessions_folder_that_exists_is_used(self, program):
        (program.folder / "sessions").mkdir()
        (program.folder / "sessions" / "earlier.txt").write_text("kept")
        assert program.start() == 0
        assert (program.folder / "sessions" / "earlier.txt").read_text() == "kept"
