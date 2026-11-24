"""What the integration tests share: a run of the program in a temporary folder, and two
finished sessions over synthetic signal."""

import pytest

from tests.support.finished_sessions import FinishedSession, config_with, synthetic_matrix
from tests.support.program_runs import ProgramRun
from tests.support.short_sessions import CONTROL, SHORT


@pytest.fixture
def program(tmp_path, raw_config, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = ProgramRun(tmp_path, raw_config)
    yield run
    run.app.release()


@pytest.fixture(scope="module")
def short(tmp_path_factory):
    """A finished session over 40 s of synthetic signal."""
    return FinishedSession(
        tmp_path_factory.mktemp("short"), config_with(SHORT), synthetic_matrix(SHORT)
    )


@pytest.fixture(scope="module")
def control(tmp_path_factory):
    """A finished session over 120 s of synthetic signal, in blocks of 30 s."""
    return FinishedSession(
        tmp_path_factory.mktemp("control"), config_with(CONTROL), synthetic_matrix(CONTROL)
    )
