"""Fixtures shared by all tests."""

import copy
import os
from pathlib import Path

import pytest
from hypothesis import settings

from adamus.config import parse_config, read_yaml
from tests.support.fake_generator_files import install_fake_generator
from tests.support.finished_sessions import synthetic_signals
from tests.support.project_paths import (
    CONFIG_FILE,
    ROOT,
    require_trained_models,
    what_the_program_keeps,
)

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


@pytest.fixture
def fake_generator(tmp_path):
    """The ``generator`` section of a configuration that points at a fake installation."""
    return install_fake_generator(tmp_path / "generator")
