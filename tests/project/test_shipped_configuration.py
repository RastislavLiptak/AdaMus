"""config.yaml of the repository: the values must fit together and fit the design.

parse_config refuses what cannot work. These tests go further and guard what
the shipped values are meant to achieve, so that a careless change of one
number is noticed here and not in the laboratory.
"""

import re
import tomllib
from pathlib import Path

import pytest
import soundfile
from brainflow.board_shim import BoardShim

import adamus
from adamus import constants as c
from adamus import machine
from adamus.config import load_config
from adamus.datasets import datasets_in
from adamus.detection.signal_processing import Steps
from tests.support.project_paths import CONFIG_FILE, ROOT

CONFIG = load_config(CONFIG_FILE)
FS = BoardShim.get_sampling_rate(CONFIG.board.id)
GIT_IGNORED = (ROOT / ".gitignore").read_text().splitlines()
INSTALL_SCRIPT = (ROOT / machine.INSTALL_SCRIPT).read_text()
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
README = (ROOT / "README.md").read_text()


class TestDetector:
    def test_detector_runs_on_the_four_electrodes_of_the_muse_2_at_its_rate(self):
        board = BoardShim.get_board_descr(CONFIG.board.id)
        assert board["name"] == "Muse2"
        assert set(c.DETECTOR_ELECTRODES) == set(board["eeg_names"].split(","))
        assert FS == c.DETECTOR_FS

    def test_window_is_a_whole_number_of_hops(self):
        """Then block boundaries on whole seconds fall on step boundaries."""
        window, hop = round(c.WINDOW_S * FS), round(c.HOP_S * FS)
        assert (window, hop) == (512, 128)
        assert window % hop == 0

    def test_the_focus_model_is_not_shipped_the_program_trains_it(self):
        assert PROJECT["tool"]["setuptools"]["package-data"] == {
            "adamus.realtime": ["pure_data_patch.pd"],
            "adamus.server": ["head_model.json", "styles/*/*"],  # the head; the styles' files
        }
        assert not list((ROOT / "src").rglob(c.FOCUS_MODEL_FILE))


def png_size(path):
    """Width and height of a PNG, from its header."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR"
    return int.from_bytes(data[16:20]), int.from_bytes(data[20:24])


class TestStyles:
    def test_every_style_has_a_folder_of_a_sample_and_a_cover_and_nothing_else_lies_there(self):
        shipped = sorted(path.relative_to(c.STYLE_FILES) for path in c.STYLE_FILES.rglob("*"))
        expected = sorted(
            path
            for style in CONFIG.styles
            for path in (
                Path(style.id),
                Path(style.id, c.FILE_SAMPLE),
                Path(style.id, c.FILE_COVER),
            )
        )
        assert shipped == expected

    def test_a_sample_is_the_rest_then_the_focus_and_a_cover_a_square_picture(self):
        for style in CONFIG.styles:
            sample = soundfile.info(c.STYLE_FILES / style.id / c.FILE_SAMPLE)
            # its rest, then its focus: the samples were recorded so, with these settings
            assert sample.duration == pytest.approx(2 * c.SAMPLE_FOCUS_FROM_S, abs=0.1), style.id
            assert png_size(c.STYLE_FILES / style.id / c.FILE_COVER) == (512, 512), style.id

    def test_the_focus_is_never_quieter_than_the_rest(self):
        for style in CONFIG.styles:
            assert style.rest_volume_db <= style.focus_volume_db, style.id


class TestTiming:
    def test_several_polls_fit_into_one_frame_of_the_model(self):
        """Otherwise a dropout of one frame could hide between two polls."""
        assert CONFIG.poll_s <= c.MODEL_FRAME_S * 2

    def test_polling_adds_at_most_a_tenth_of_a_step_of_latency(self):
        assert CONFIG.poll_s <= c.HOP_S / 10

    def test_a_dropout_shorter_than_the_signal_of_a_step_does_not_end_the_session(self):
        assert CONFIG.timeouts.stream_s > Steps.at(c.DETECTOR_FS).context / c.DETECTOR_FS

    def test_model_gets_more_time_to_load_than_pd_to_connect(self):
        assert CONFIG.timeouts.model_load_s > CONFIG.timeouts.pd_connect_s > 0


class TestProtocol:
    BLOCKS = CONFIG.protocol

    def test_starts_with_a_minute_of_rest(self):
        assert (self.BLOCKS[0].label, self.BLOCKS[0].duration_s) == ("REST", 60)

    def test_blocks_alternate_rest_and_focus_in_equal_numbers(self):
        timed = [b.label for b in self.BLOCKS if b.label != c.FREE]
        assert timed == ["REST", "FOCUS"] * (len(timed) // 2)
        assert len(timed) >= 2

    def test_every_timed_block_holds_several_full_steps(self):
        for block in self.BLOCKS:
            if block.duration_s:
                assert block.duration_s >= 2 * c.WINDOW_S

    def test_session_ends_with_free_play_until_stopped(self):
        assert (self.BLOCKS[-1].label, self.BLOCKS[-1].duration_s) == ("FREE", 0)


class TestGenerator:
    def test_every_model_is_made_in_the_models_folder_kept_out_of_git(self):
        folder = CONFIG.generator.directory
        assert folder == c.MODELS_DIR / "music_generator"
        assert c.FOCUS_DETECTOR_DIR == c.MODELS_DIR / "focus_detector"
        assert f"/{c.MODELS_DIR}/" in GIT_IGNORED
        assert not any(line.startswith(f"/{folder}") for line in GIT_IGNORED)

    def test_install_script_puts_files_where_the_program_looks_for_them(self):
        """The layout below generator.folder is written in two places; they must agree."""
        assert "$ROOT/" + c.GENERATOR_PD_BINARY in INSTALL_SCRIPT
        assert "$ROOT/" + c.GENERATOR_EXTERNAL_FILE in INSTALL_SCRIPT
        assert f'"$ROOT/{c.GENERATOR_ASSETS}/$file"' in INSTALL_SCRIPT
        (listed,) = re.findall(r'^ASSET_FILES="([^"]*)"$', INSTALL_SCRIPT, re.MULTILINE)
        assert tuple(listed.split()) == c.GENERATOR_ASSET_FILES
        model_file = c.GENERATOR_MODEL_FILE.format(model="$MODEL")
        assert "$ROOT/" + model_file in INSTALL_SCRIPT

    def test_model_package_and_the_built_object_are_the_same_release(self):
        """The model files come from the Python package, mrt2~ is built from the git tag."""
        (requirement,) = PROJECT["project"]["optional-dependencies"]["generator"]
        package, pinned = requirement.split("==")
        assert package == "magenta-rt[mlx]"
        assert f'MRT_VERSION="v{pinned}"' in INSTALL_SCRIPT


class TestProject:
    def test_the_program_carries_no_release_number_of_its_own(self):
        """Releases are git tags; a number kept here as well would sooner or later differ."""
        assert PROJECT["project"]["version"] == "0"
        assert not hasattr(adamus, "__version__")

    def test_every_library_is_pinned_to_one_release(self):
        """Another numpy or scipy trains another model and gives other metrics."""
        project = PROJECT["project"]
        every = [*project["dependencies"], *project["optional-dependencies"]["generator"],
                 *project["optional-dependencies"]["dev"]]  # fmt: skip
        assert every and all(re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[0-9][0-9.]*", d) for d in every)

    def test_the_pins_are_the_releases_installed_here(self):
        from importlib.metadata import version

        for dependency in [
            *PROJECT["project"]["dependencies"],
            *PROJECT["project"]["optional-dependencies"]["dev"],
        ]:
            name, pinned = dependency.split("==")
            assert version(name) == pinned, dependency

    def test_the_app_starts_the_command_of_the_project_and_the_readme_how_to_build_it(self):
        assert PROJECT["project"]["scripts"] == {"adamus": "adamus.main:main"}
        program = (ROOT / "src" / "ui" / "AdaMus" / "Program" / "Program.swift").read_text()
        assert 'folder.appending(path: ".venv/bin/adamus")' in program
        assert "sh scripts/build_app.sh" in README and "`AdaMus.app`" in README
        assert (ROOT / "scripts" / "build_app.sh").is_file()
        assert f"`{c.DATASETS_DIR}/`" in README

    def test_readme_names_every_dataset_and_its_source(self):
        """Also the one the sessions with the headset fill, before there is a first one."""
        folders = {folder.name for folder in datasets_in(ROOT / c.DATASETS_DIR)}
        for dataset in folders | {c.OWN_RECORDINGS}:
            assert f"| `{dataset}` |" in README, dataset

    def test_readme_says_the_datasets_keep_their_own_licenses(self):
        assert "The datasets in `data/` keep their own licenses" in README

    def test_readme_names_the_test_commands_that_exist(self):
        """Each group of tests by its command; the marks are those pytest knows."""
        commands = [
            ".venv/bin/pytest",
            *(f".venv/bin/pytest -m {mark}" for mark in ("fast", "slow", "window")),
            ".venv/bin/python scripts/mutation_testing.py",
        ]
        assert "```sh\n" + "\n".join(commands) + "\n```" in README
        known = PROJECT["tool"]["pytest"]["ini_options"]["markers"]
        assert {marker.split(":")[0] for marker in known} == {"fast", "slow", "window"}
        assert (ROOT / "scripts" / "mutation_testing.py").is_file()


def test_sessions_are_kept_out_of_git():
    assert f"/{c.SESSIONS_DIR}/" in GIT_IGNORED


def test_the_recordings_of_people_with_the_headset_are_kept_out_of_git():
    """They are the biodata of the people who wore it."""
    assert f"/{c.DATASETS_DIR}/{c.OWN_RECORDINGS}/" in GIT_IGNORED


def test_the_program_has_one_name_everywhere():
    """The app (its product and the name macOS shows) and the README."""
    project = (ROOT / "src" / "ui" / "AdaMus.xcodeproj" / "project.pbxproj").read_text()
    shown = set(re.findall(r"INFOPLIST_KEY_CFBundleDisplayName = (\w+);", project))
    assert shown == {c.PROGRAM_NAME}
    assert f"path = {c.PROGRAM_NAME}.app;" in project
    assert README.startswith(f"# {c.PROGRAM_NAME}\n")
