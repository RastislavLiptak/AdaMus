"""console.preflight_checks: the nine checks run when the console starts."""

import json
import re

import pytest

from adamus import constants as c
from adamus.config import write_yaml
from adamus.console import preflight_checks
from adamus.console.console_log import ConsoleLog
from adamus.console.preflight_checks import Preflight
from tests.support.project_paths import PROJECT_MODEL
from tests.support.stand_in_devices import FakeGenerator


def people():
    """How many people the project's focus models were trained on."""
    return len(json.loads(PROJECT_MODEL.read_text())["trained_on"]["people"])


def checklist(text):
    """The printed checklist as (mark, name, detail) rows."""
    rows = []
    for line in text[text.index("Preflight") :].splitlines()[1:]:
        match = re.fullmatch(r"  (.{2})(.{23})(.*)", line)
        if match is None:
            break  # the checklist has ended
        assert match.group(1) in ("✓ ", "✗ ", "– ", "  ")
        rows.append(tuple(part.strip() for part in match.groups()))
    return rows


class TestAllPass:
    def test_every_check_shows_what_it_found(self, console):
        console.add_recording("first")
        assert console.start() == 0  # the keys end with [CTRL]+[C], which quits normally
        assert console.text.splitlines()[:2] == ["AdaMus · config: config.yaml", ""]
        assert checklist(console.text) == [
            ("✓", "Configuration", "config.yaml, 9 protocol blocks"),
            ("✓", "Storage", "sessions/ writable"),
            ("✓", "Focus detector", f"{c.FOCUS_MODEL_PATH}, {people()} people"),
            ("✓", "Audio output", "MacBook Pro Speakers, 48000 Hz"),
            ("✓", "Generator files", "pd, mrt2~, patch, mrt2_small"),
            ("✓", "Generator start", "model loaded"),
            ("✓", "Datasets", "1 in data/"),
            ("✓", "Live headset", "Muse 2"),
            ("✓", "Data source available", "2 sources"),
        ]
        assert "All systems ready." in console.text and "Errors" not in console.text
        assert console.machine.generator.started == 1

    def test_results_are_logged_in_order_with_their_duration(self, console):
        console.start()
        entries = console.logged("PREFLIGHT")
        assert [(e["check"], e["result"]) for e in entries] == [
            ("Configuration", "ok"), ("Storage", "ok"), ("Focus detector", "ok"),
            ("Audio output", "ok"), ("Generator files", "ok"), ("Generator start", "ok"),
            ("Datasets", "unavailable"), ("Live headset", "ok"),
            ("Data source available", "ok"),
        ]  # fmt: skip
        assert all(e["duration"] >= 0 and "detail" in e and e["t"] > 0 for e in entries)
        assert entries[7]["headset"] == "Muse 2"


def checklist_rows(text):
    """The printed checklist by the name of the check: (mark, detail)."""
    return {name: (mark, detail) for mark, name, detail in checklist(text)}


def unwrapped(text):
    """The text without the line breaks of the width of the screen."""
    return " ".join(text.split())


def run_preflight(console, screen):
    """The checks of the console's configuration, shown on ``screen``."""
    write_yaml(console.raw, console.folder / "config.yaml")
    screen.preflight = Preflight(
        console.folder / "config.yaml", console.machine, screen, ConsoleLog()
    )
    return screen.preflight.run()


class TestRequiredChecks:
    def test_missing_configuration_skips_what_needs_it(self, console):
        assert console.start(write_config=False) == 1
        rows = checklist(console.text)
        assert rows[0][:2] == ("✗", "Configuration") and rows[0][2].startswith(
            "cannot read config.yaml"
        )
        # Storage, the focus detector and the audio output do not depend on the configuration.
        assert [row[:2] for row in rows[1:4]] == [
            ("✓", "Storage"), ("✓", "Focus detector"), ("✓", "Audio output"),
        ]  # fmt: skip
        assert rows[4:] == [
            ("✗", "Generator files", "skipped"), ("✗", "Generator start", "skipped"),
            ("–", "Datasets", "none found"), ("✗", "Live headset", "skipped"),
            ("✗", "Data source available", "skipped"),
        ]  # fmt: skip
        assert console.machine.generator.started == 0

    def test_errors_name_the_check_the_cause_and_the_remedy(self, console):
        console.start(write_config=False)
        errors = console.text[console.text.index("Errors") :]
        assert "\nConfiguration\n  cannot read config.yaml" in errors
        assert "config.yaml in the repository lists every key" in errors
        assert "skipped" not in errors  # only the checks that failed themselves

    def test_invalid_configuration_is_reported_with_its_problems(self, console):
        console.raw["generator"]["prompt_focus"] = "disco 80"
        assert console.start() == 1
        assert "generator.prompt_focus: not a valid style text" in console.text

    @pytest.mark.parametrize("model", ["missing", "{}", "not json"])
    def test_a_focus_model_that_cannot_be_used_is_trained(self, console, model):
        path = console.folder / c.FOCUS_MODEL_PATH
        path.unlink() if model == "missing" else path.write_text(model)
        console.machine.training = "trains"
        assert console.start() == 0
        rows = checklist_rows(console.text)
        assert rows["Focus detector"] == ("✓", f"trained: {c.FOCUS_MODEL_PATH}, {people()} people")
        assert console.machine.trainings == 1

    def test_a_focus_model_of_other_features_is_trained_again(self, console):
        path = console.folder / c.FOCUS_MODEL_PATH
        written = json.loads(path.read_text())
        written["features"]["artifact_uv"] = 1.0
        path.write_text(json.dumps(written))
        console.machine.training = "trains"
        assert console.start() == 0
        assert console.machine.trainings == 1

    def test_a_usable_focus_model_is_not_trained_again(self, console):
        assert console.start() == 0
        assert console.machine.trainings == 0

    def test_a_training_that_fails_is_reported_with_its_log(self, console):
        (console.folder / c.FOCUS_MODEL_PATH).unlink()
        assert console.start() == 1
        rows = checklist_rows(console.text)
        assert rows["Focus detector"][0] == "✗"
        flat = unwrapped(console.text)
        log_file = c.FOCUS_DETECTOR_DIR / c.SETUP_LOG
        assert f"the training of the focus detector failed, see {log_file}" in flat
        assert "the first training downloads its data from physionet.org" in flat

    def test_a_model_still_unusable_after_the_training_is_named_with_the_problem(self, console):
        (console.folder / c.FOCUS_MODEL_PATH).unlink()
        console.machine.training = "does nothing"
        assert console.start() == 1
        flat = unwrapped(console.text)
        log_file = c.FOCUS_DETECTOR_DIR / c.SETUP_LOG
        assert (
            f"still missing after the setup ({log_file}): cannot read the focus model "
            f"{c.FOCUS_MODEL_PATH}: [Errno 2]"
        ) in flat

    def test_storage_that_is_not_a_folder(self, console):
        (console.folder / "sessions").write_text("a file")
        assert console.start() == 1
        rows = checklist_rows(console.text)
        assert rows["Storage"][0] == "✗"
        assert rows["Generator files"][0] == "✓"
        assert rows["Generator start"][0] == "✓"  # it needs no storage
        assert (console.machine.generator.started, console.machine.generator.stopped) == (1, 1)

    def test_storage_may_lie_on_any_path(self, console, tmp_path):
        spaced = tmp_path / "my folder; 2026"
        spaced.mkdir()
        console.machine.sessions_root = spaced / "sessions"
        assert console.start() == 0
        assert (spaced / "sessions").is_dir()

    @pytest.mark.parametrize("rate", [44100.0, 96000.0])
    def test_audio_output_must_run_at_48_khz(self, console, rate):
        console.machine.audio = ("USB Headphones", rate)
        assert console.start() == 1
        rows = checklist_rows(console.text)
        assert rows["Audio output"] == ("✗", f"USB Headphones, {rate:g} Hz (need 48000 Hz)")
        assert rows["Generator start"] == ("✗", "skipped")
        assert "Set the output device to 48000 Hz" in console.text

    @pytest.mark.parametrize(
        "missing",
        [c.GENERATOR_PD_BINARY, c.GENERATOR_EXTERNAL_FILE,
         c.GENERATOR_MODEL_FILE.format(model="mrt2_small"),
         *(f"{c.GENERATOR_ASSETS}/{file}" for file in c.GENERATOR_ASSET_FILES)],
    )  # fmt: skip
    def test_a_missing_file_of_the_generator_starts_its_installation(self, console, missing):
        """Also a single file of the assets: what an interrupted download leaves behind."""
        (console.folder / "generator" / missing).unlink()
        console.machine.installation = "installs"
        assert console.start() == 0
        rows = checklist_rows(console.text)
        assert rows["Generator files"] == ("✓", "installed: pd, mrt2~, patch, mrt2_small")
        assert rows["Generator start"][0] == "✓"
        assert console.machine.installed == ["generator"]

    def test_an_installation_that_fails_is_reported_with_its_log(self, console):
        (console.folder / "generator" / c.GENERATOR_PD_BINARY).unlink()
        assert console.start() == 1
        rows = checklist_rows(console.text)
        assert rows["Generator files"][0] == "✗"
        flat = unwrapped(console.text)
        assert "the installation of the generator failed, see generator/setup.log" in flat
        assert rows["Generator start"] == ("✗", "skipped")
        assert "Read the installation log named above" in flat

    def test_files_still_missing_after_the_installation_are_named(self, console):
        (console.folder / "generator" / c.GENERATOR_EXTERNAL_FILE).unlink()
        console.machine.installation = "does nothing"
        assert console.start() == 1
        flat = unwrapped(console.text)
        assert (
            "still missing after the setup (generator/setup.log): "
            f"generator/{c.GENERATOR_EXTERNAL_FILE}"
        ) in flat

    def test_installed_generator_is_not_installed_again(self, console):
        assert console.start() == 0
        assert console.machine.installed == []

    def test_generator_that_fails_to_start(self, console):
        console.machine.generator = FakeGenerator(
            fail_to_start="the generator reported: mrt2~: failed to load model"
        )
        assert console.start() == 1
        rows = checklist_rows(console.text)
        assert rows["Generator start"] == (
            "✗",
            "the generator reported: mrt2~: failed to load model",
        )
        errors = console.text[console.text.index("Errors") :].splitlines()
        assert [line.rstrip() for line in errors if line.strip()] == [
            "Errors",
            "Generator start",
            "  the generator reported: mrt2~: failed to load model",
        ]  # no advice follows: the generator says itself what went wrong

    def test_generator_that_fails_to_start_shows_what_pure_data_printed(self, console):
        """Nothing is stored without a session, so the screen is where it can be read."""
        console.machine.generator = FakeGenerator(
            fail_to_start=(
                "mrt2~ did not load the model within 120 s",
                "Pure Data printed:",
                "  mrt2~: loading model /somewhere/model.mlxfn",
            )
        )
        assert console.start() == 1
        errors = console.text[console.text.index("Errors") :].splitlines()
        assert [line.rstrip() for line in errors if line.strip()] == [
            "Errors",
            "Generator start",
            "  mrt2~ did not load the model within 120 s",
            "  Pure Data printed:",
            "    mrt2~: loading model /somewhere/model.mlxfn",
        ]

    def test_no_source_at_all(self, console):
        console.machine.headsets_on = set()
        assert console.start() == 1
        rows = checklist_rows(console.text)
        assert rows["Datasets"] == ("–", "none found")
        assert rows["Live headset"] == ("–", "none found")
        assert rows["Data source available"] == ("✗", "no headset and no dataset")
        assert "Switch the headset on, or put a dataset into data/." in console.text

    def test_generator_started_by_the_checks_is_stopped_when_a_later_check_fails(self, console):
        console.machine.headsets_on = set()
        console.start()
        assert (console.machine.generator.started, console.machine.generator.stopped) == (1, 1)


class TestOptionalChecks:
    def test_missing_headset_alone_does_not_stop_the_console(self, console):
        console.machine.headsets_on = set()
        console.add_recording("a")
        assert console.start() == 0
        assert ("–", "Live headset", "none found") in checklist(console.text)
        assert "All systems ready." in console.text

    def test_the_headset_has_its_row_and_its_log_entry(self, console):
        console.start()
        assert checklist(console.text)[7] == ("✓", "Live headset", "Muse 2")
        assert console.machine.probed == [38]
        (logged,) = [e for e in console.logged("PREFLIGHT") if e["check"] == "Live headset"]
        assert (logged["headset"], logged["result"]) == ("Muse 2", "ok")

    def test_datasets_that_hold_a_recording_are_counted(self, console):
        console.add_recording("one", dataset="first")
        console.add_recording("one", dataset="second")
        console.add_recording("two", dataset="second")
        (console.folder / "data" / "empty").mkdir()
        (console.folder / "data" / "notes.txt").write_text("")
        console.start()
        assert ("✓", "Datasets", "2 in data/") in checklist(console.text)


def test_remedies_exist_for_every_required_check():
    required = [
        preflight_checks.CONFIGURATION, preflight_checks.STORAGE,
        preflight_checks.FOCUS_DETECTOR, preflight_checks.AUDIO_OUTPUT,
        preflight_checks.GENERATOR_FILES, preflight_checks.DATA_SOURCE,
    ]  # fmt: skip
    # A generator that does not start says why itself, in the words of Pure Data.
    assert sorted(preflight_checks.HOW_TO_FIX) == sorted(required)


class Screen:
    """Stands in for the live display: remembers the state of the checks at every update."""

    def __init__(self):
        self.preflight = None
        self.shown = []

    def update(self, renderable):
        self.shown.append([(check.name, check.status) for check in self.preflight.checks])


class TestWhatTheOperatorSees:
    @pytest.fixture
    def run(self, console):
        screen = Screen()
        assert run_preflight(console, screen) is True
        return screen

    def test_whole_list_is_shown_before_the_first_check_runs(self, run):
        assert len(run.shown[0]) == 9
        assert {status for _, status in run.shown[0]} == {"pending"}

    def test_every_check_is_shown_running_and_then_with_its_result(self, run):
        for position, name in enumerate(
            [
                "Configuration",
                "Storage",
                "Focus detector",
                "Audio output",
                "Generator files",
                "Generator start",
            ]
        ):
            states = [shown[position] for shown in run.shown]
            first_running = states.index((name, "running"))
            assert states[first_running + 1] == (name, "ok")
            assert set(states[:first_running]) == {(name, "pending")}

    def test_only_the_generator_start_says_what_it_waits_for(self, console):
        details = []

        class Watching(Screen):
            def update(self, renderable):
                details.extend(
                    (check.name, check.detail)
                    for check in self.preflight.checks
                    if check.status == "running"
                )

        run_preflight(console, Watching())
        assert ("Generator start", "loading model ...") in details
        assert {detail for name, detail in details if name != "Generator start"} == {""}

    def announced_while_making(self, console, method):
        """The running checks shown while the machine's ``method`` runs."""
        shown, seen = [], []

        class Watching(Screen):
            def update(self, renderable):
                shown.append(
                    [(check.name, check.detail) for check in self.preflight.checks
                     if check.status == "running"]
                )  # fmt: skip

        make = getattr(console.machine, method)

        def watched(*arguments):
            seen.append(shown[-1])
            return make(*arguments)

        setattr(console.machine, method, watched)
        assert run_preflight(console, Watching()) is True
        return seen

    def test_an_installation_is_announced_before_it_runs(self, console):
        (console.folder / "generator" / c.GENERATOR_PD_BINARY).unlink()
        console.machine.installation = "installs"
        announced = ("Generator files", "installing the music generator (this takes a while) ...")
        assert self.announced_while_making(console, "install_generator") == [[announced]]

    def test_a_training_is_announced_before_it_runs(self, console):
        (console.folder / c.FOCUS_MODEL_PATH).unlink()
        console.machine.training = "trains"
        announced = ("Focus detector", "training the focus detector (this takes a while) ...")
        assert self.announced_while_making(console, "train_focus_detector") == [[announced]]


class TestLogAndDetails:
    def test_failed_and_skipped_checks_are_logged_as_such(self, console):
        console.machine.audio = ("USB Headphones", 44100.0)
        console.start()
        results = {entry["check"]: entry["result"] for entry in console.logged("PREFLIGHT")}
        assert (results["Audio output"], results["Generator start"]) == ("failed", "skipped")
        assert results["Storage"] == "ok"

    def test_results_of_a_run_that_ends_in_the_preflight_are_stored_nowhere(self, console):
        console.machine.audio = ("USB Headphones", 44100.0)
        assert console.start() == 1
        assert list((console.folder / "sessions").iterdir()) == []

    def test_duration_of_a_check_is_the_time_it_took(self, console):
        console.start()
        durations = [entry["duration"] for entry in console.logged("PREFLIGHT")]
        assert all(0 <= duration < 30 for duration in durations)

    def test_one_source_is_counted_in_the_singular(self, console):
        console.start()  # the headset alone
        assert checklist(console.text)[-1] == ("✓", "Data source available", "1 source")

    def test_every_missing_file_of_the_generator_is_named(self, console):
        folder = console.folder / "generator"
        (folder / c.GENERATOR_PD_BINARY).unlink()
        (folder / c.GENERATOR_EXTERNAL_FILE).unlink()
        console.machine.installation = "does nothing"
        assert console.start() == 1
        missing = f"generator/{c.GENERATOR_PD_BINARY}, generator/{c.GENERATOR_EXTERNAL_FILE}"
        assert missing in " ".join(console.text.split())

    def test_sessions_folder_is_created_with_the_folders_above_it(self, console, tmp_path):
        console.machine.sessions_root = tmp_path / "new" / "deeper" / "sessions"
        assert console.start() == 0
        assert (tmp_path / "new" / "deeper" / "sessions").is_dir()

    def test_sessions_folder_that_exists_is_used(self, console):
        (console.folder / "sessions").mkdir()
        (console.folder / "sessions" / "earlier.txt").write_text("kept")
        assert console.start() == 0
        assert (console.folder / "sessions" / "earlier.txt").read_text() == "kept"
