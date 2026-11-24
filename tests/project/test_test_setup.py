"""How the test run itself is set up: the folder it runs in and the models it needs."""

from pathlib import Path

import numpy as np
import pytest

from adamus import constants as c
from tests.support.project_paths import (
    MISSING_MODEL,
    OTHER_MODEL,
    PROJECT_MODEL,
    ROOT,
    require_reference_model,
    require_trained_models,
    shipped_datasets,
    what_the_program_keeps,
)
from tests.support.synthetic_dataset import write_recording


def test_the_reference_tests_see_only_the_datasets_shipped_with_the_project(tmp_path):
    """The sessions of the operator are stored in the data folder too; they are no reference."""
    for dataset in ("synthetic", c.OWN_RECORDINGS):
        write_recording(tmp_path / dataset / "one", np.zeros((4, 1)), [("REST", 1 / 256)])
    assert shipped_datasets(tmp_path) == [tmp_path / "synthetic"]


def test_a_file_a_test_would_write_into_the_project_is_noticed(tmp_path):
    (tmp_path / "sessions").mkdir()
    before = what_the_program_keeps(tmp_path)
    (tmp_path / "sessions" / "20261117-090000_S01").mkdir()
    (tmp_path / "sessions" / "20261117-090000_S01" / "ticks.jsonl").write_text("")
    assert what_the_program_keeps(tmp_path) != before


def test_tests_run_in_the_project_folder():
    assert Path.cwd() == ROOT
    assert (ROOT / "pyproject.toml").is_file()


def test_missing_models_stop_the_run_with_how_to_get_them(tmp_path):
    missing = tmp_path / "focus_model.json"
    with pytest.raises(pytest.exit.Exception) as stopped:
        require_trained_models(missing)
    assert str(stopped.value) == f"{missing}: {MISSING_MODEL}"
    assert not list(tmp_path.iterdir())


def test_reference_values_need_the_model_they_were_computed_with(tmp_path):
    require_reference_model()  # the project's model is that one
    other = tmp_path / "focus_model.json"
    other.write_bytes(PROJECT_MODEL.read_bytes().replace(b"1", b"2", 1))
    with pytest.raises(pytest.fail.Exception) as failed:
        require_reference_model(other)
    assert str(failed.value) == f"{other} {OTHER_MODEL}"
