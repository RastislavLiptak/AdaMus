"""config.yaml of the repository: the values must fit together and fit the design.

parse_config refuses what cannot work. These tests go further and guard what
the shipped values are meant to achieve, so that a careless change of one
number is noticed here and not in the laboratory.
"""

import re
import tomllib

from brainflow.board_shim import BoardShim

import adamus
from adamus import constants as c
from adamus.config import load_config
from adamus.datasets import datasets_in
from adamus.detection.signal_processing import Steps
from adamus.own_recordings import OWN_RECORDINGS
from tests.support.project_paths import CONFIG_FILE

REPO = CONFIG_FILE.parent
CONFIG = load_config(CONFIG_FILE)
FS = BoardShim.get_sampling_rate(CONFIG.board.id)
GIT_IGNORED = (REPO / ".gitignore").read_text().splitlines()
INSTALL_SCRIPT = (REPO / "scripts" / "install_generator.sh").read_text()
PROJECT = tomllib.loads((REPO / "pyproject.toml").read_text())
README = (REPO / "README.md").read_text()


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
            "adamus.realtime": ["pure_data_patch.pd"]
        }
        assert not list((REPO / "src").rglob(c.FOCUS_MODEL_FILE))


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

    def test_install_script_defaults_are_the_configured_folder_and_model(self):
        assert f'FOLDER="${{1:-{CONFIG.generator.folder}}}"' in INSTALL_SCRIPT
        assert f'MODEL="${{2:-{CONFIG.generator.model}}}"' in INSTALL_SCRIPT

    def test_install_script_builds_with_one_compiler_per_four_gigabytes_of_memory(self):
        """Eight compilers at once ran a Mac with 16 GB out of memory."""
        assert "--parallel 8" not in INSTALL_SCRIPT
        assert "JOBS=$(( $(sysctl -n hw.memsize) / 4294967296 ))" in INSTALL_SCRIPT
        assert '--parallel "$(( JOBS > 0 ? JOBS : 1 ))"' in INSTALL_SCRIPT

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
        for pinned in ("brainflow==5.24.0", "rich==15.0.0", "readchar==4.2.2"):
            assert pinned in project["dependencies"]

    def test_the_pins_are_the_releases_installed_here(self):
        from importlib.metadata import version

        for dependency in [
            *PROJECT["project"]["dependencies"],
            *PROJECT["project"]["optional-dependencies"]["dev"],
        ]:
            name, pinned = dependency.split("==")
            assert version(name) == pinned, dependency

    def test_readme_names_the_command_and_the_folder_of_the_datasets(self):
        assert PROJECT["project"]["scripts"] == {"adamus": "adamus.cli:main"}
        assert "```sh\nadamus\n```" in README
        assert f"`{c.DATASETS_DIR}/`" in README

    def test_readme_names_every_dataset_and_its_source(self):
        """Also the one the sessions with the headset fill, before there is a first one."""
        folders = {folder.name for folder in datasets_in(REPO / c.DATASETS_DIR)}
        for dataset in folders | {OWN_RECORDINGS}:
            assert f"| `{dataset}` |" in README, dataset

    def test_readme_says_the_datasets_keep_their_own_licenses(self):
        assert "The datasets in `data/` keep their own licenses" in README

    def test_readme_names_the_test_commands_that_exist(self):
        markers = PROJECT["tool"]["pytest"]["ini_options"]["markers"]
        quick = " and ".join(f"not {marker.split(':')[0]}" for marker in markers)
        assert f'pytest -m "{quick}"' in README
        assert (REPO / "scripts" / "mutation_testing.py").is_file()
        assert "python scripts/mutation_testing.py src/adamus/config.py" in README


def test_sessions_are_kept_out_of_git():
    assert f"/{c.SESSIONS_DIR}/" in GIT_IGNORED
