"""What the unit tests of the datasets and of the data sources share."""

import pytest

from tests.support.finished_sessions import synthetic_signals
from tests.support.synthetic_dataset import write_recording

SHORT = [("REST", 5), ("FOCUS", 5)]


@pytest.fixture
def data(tmp_path):
    """A data folder with two datasets, a dataset without a recording and stray files."""
    signals = synthetic_signals([list(block) for block in SHORT])
    for dataset, recording in (("zeta", "one"), ("alpha", "Subject01"), ("alpha", "Subject00")):
        write_recording(tmp_path / dataset / recording, signals, SHORT)
    (tmp_path / "empty").mkdir()
    (tmp_path / "notes.txt").write_text("")
    (tmp_path / "alpha" / "notes.txt").write_text("")
    (tmp_path / "alpha" / "half").mkdir()
    (tmp_path / "alpha" / "half" / "eeg.csv").write_text("")
    return tmp_path
