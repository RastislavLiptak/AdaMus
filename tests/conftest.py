"""Fixtures shared by all tests."""

import copy
import os
import shutil
from pathlib import Path

import pytest
from hypothesis import settings

from adamus import parallel
from adamus.config import parse_config, read_yaml
from tests.support.finished_sessions import synthetic_signals
from tests.support.played_sessions import play_session
from tests.support.project_paths import (
    CONFIG_FILE,
    ROOT,
    require_trained_models,
    what_the_program_keeps,
)

TEST_MEMORY = parallel.GIGABYTE  # a process of the tests at the most: a session


def pytest_collection_modifyitems(items):
    """A test that is neither slow nor uses the window is fast: -m fast runs it."""
    for item in items:
        if not any(item.get_closest_marker(mark) for mark in ("slow", "window")):
            item.add_marker(pytest.mark.fast)


def pytest_xdist_auto_num_workers(config):
    """As many processes of the tests as the computer has room for, all of it theirs."""
    return parallel.processes_here(TEST_MEMORY, alone=True)


# Property tests: no time limit per example (the first call loads BrainFlow), and no database
# of examples: one found against a mutant would be played again in every later run.
settings.register_profile("adamus", deadline=None, database=None)
settings.load_profile("adamus")


@pytest.fixture(scope="session", autouse=True)
def in_the_project_folder():
    """The tests run in the project folder, as the program does, wherever pytest was started,
    and with the focus models the program trained there."""
    started_in = Path.cwd()
    os.chdir(ROOT)
    require_trained_models()
    kept = what_the_program_keeps()
    yield
    os.chdir(started_in)
    assert what_the_program_keeps() == kept, (
        "a test wrote into the project's sessions, data or models"
    )


@pytest.fixture
def raw_config():
    """The repository configuration as a mapping that a test may modify."""
    return copy.deepcopy(read_yaml(CONFIG_FILE))


@pytest.fixture
def config(raw_config):
    return parse_config(raw_config)


@pytest.fixture(scope="module")
def default_protocol():
    return read_yaml(CONFIG_FILE)["protocol"]


@pytest.fixture(scope="module")
def synthetic_default_signals(default_protocol):
    """The synthetic recording of the default protocol: the detector's electrodes, 480 s."""
    return synthetic_signals(default_protocol)


PLAYED_PROTOCOL = [["REST", 20], ["FOCUS", 20], ["REST", 10], ["FREE", 0]]


@pytest.fixture(scope="session")
def played(tmp_path_factory):
    """A session over the recording data/synthetic/played, with music and metrics, as the
    program leaves it: for the tests of what the program reads from a session folder."""
    root = tmp_path_factory.mktemp("played")
    return play_session(root, "S01", PLAYED_PROTOCOL, recording="played")


@pytest.fixture
def copied(played, tmp_path):
    """A copy of the played session that a test may change."""
    folder = tmp_path / "sessions" / played.directory.name
    shutil.copytree(played.directory, folder)
    return folder
