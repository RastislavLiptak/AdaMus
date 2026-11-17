"""What the integration tests share: a run of the console in a temporary folder."""

import pytest

from tests.support.console_runs import ConsoleRun


@pytest.fixture
def console(tmp_path, raw_config, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return ConsoleRun(tmp_path, raw_config)
