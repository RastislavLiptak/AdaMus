"""datasets: the datasets of the data folder, and how their recordings are written and read."""

from pathlib import Path

import numpy as np
import pytest

from adamus import constants as c
from adamus.config import Block, read_yaml
from adamus.datasets import (
    datasets_in,
    electrodes_of_recording,
    read_recording,
    recordings_of,
    write_recording,
)
from adamus.errors import AdamusError
from tests.support.finished_sessions import (
    FS,
)

SHORT = [("REST", 5), ("FOCUS", 5)]


def test_datasets_are_the_folders_that_hold_a_recording_in_alphabetical_order(data):
    assert datasets_in(data) == [data / "alpha", data / "zeta"]
    assert datasets_in(data / "missing") == []


def test_recordings_are_the_folders_with_both_files_in_alphabetical_order(data):
    assert recordings_of(data / "alpha") == [
        data / "alpha" / "Subject00",
        data / "alpha" / "Subject01",
    ]
    assert recordings_of(data / "empty") == []


def test_the_order_is_alphabetical_whatever_order_the_folder_lists_them_in(data, monkeypatch):
    """APFS lists names sorted already; another file system need not."""
    listed = Path.glob
    monkeypatch.setattr(
        Path, "glob", lambda folder, pattern: reversed(list(listed(folder, pattern)))
    )
    assert datasets_in(data) == [data / "alpha", data / "zeta"]
    assert recordings_of(data / "alpha") == [
        data / "alpha" / "Subject00",
        data / "alpha" / "Subject01",
    ]


class TestElectrodesOfARecording:
    def test_its_first_line(self, data):
        names = electrodes_of_recording(data / "zeta" / "one" / c.FILE_DATASET_EEG)
        assert names == list(c.DETECTOR_ELECTRODES)

    @pytest.mark.parametrize("first_line", ["AF7,,AF8", "AF7,AF8,AF7", ""])
    def test_every_column_is_named_once(self, tmp_path, first_line):
        path = tmp_path / "eeg.csv"
        path.write_text(first_line + "\n1,2,3\n")
        with pytest.raises(AdamusError) as error:
            electrodes_of_recording(path)
        assert str(error.value) == f"{path}: the first line must name every column, and none twice"

    def test_a_file_saved_by_a_spreadsheet_keeps_its_first_electrode(self, tmp_path):
        """It may begin with a byte order mark, and put a space after each comma."""
        path = tmp_path / "eeg.csv"
        path.write_bytes(b"\xef\xbb\xbfTP9, AF7, AF8\n1,2,3\n")
        assert electrodes_of_recording(path) == ["TP9", "AF7", "AF8"]

    def test_a_file_that_is_not_text(self, tmp_path):
        path = tmp_path / "eeg.csv"
        path.write_bytes(b"\xff\xfe\x00A")
        with pytest.raises(AdamusError) as error:
            electrodes_of_recording(path)
        assert str(error.value) == f"cannot read {path}: {error.value.__cause__}"

    def test_a_missing_file(self, tmp_path):
        with pytest.raises(AdamusError, match="^cannot read "):
            electrodes_of_recording(tmp_path / "missing.csv")


class TestReadRecording:
    def test_the_names_and_the_samples_of_every_electrode(self, tmp_path):
        path = tmp_path / "eeg.csv"
        path.write_text("AF7,AF8\n1,2\n3.5,-4\n")
        names, samples = read_recording(path)
        assert names == ["AF7", "AF8"]
        assert np.array_equal(samples, [[1.0, 3.5], [2.0, -4.0]])

    def test_a_recording_without_a_sample(self, tmp_path):
        path = tmp_path / "eeg.csv"
        path.write_text("AF7,AF8\n")
        with pytest.raises(AdamusError) as error:
            read_recording(path)
        assert str(error.value) == f"{path}: no samples after the names of the electrodes"

    @pytest.mark.parametrize(("text", "problem"), [
        ("AF7,AF8\n1,2\n3\n", "^cannot read "), ("AF7,AF8\n1,x\n", "^cannot read "),
        ("AF7,AF8,TP9\n1,2\n3,4\n", ": every line must hold 3 values$"),
    ])  # fmt: skip
    def test_lines_that_do_not_fit_the_names(self, tmp_path, text, problem):
        path = tmp_path / "eeg.csv"
        path.write_text(text)
        with pytest.raises(AdamusError, match=problem) as error:
            read_recording(path)
        if error.value.__cause__ is not None:  # numpy could not read the lines
            assert str(error.value) == f"cannot read {path}: {error.value.__cause__}"


class TestWriteRecording:
    def test_it_is_read_back_as_every_recording_of_a_dataset(self, tmp_path):
        samples = np.array([[1.0, 2.0, 3.0], [-4.5, 5.25, 6.125], [7.0, 8.0, 9.0]])
        folder = write_recording(
            tmp_path / "own" / "S01", ["TP9", "AF7", "AF8"], samples, [Block("REST", 3 / FS)]
        )
        assert recordings_of(tmp_path / "own") == [folder]
        assert electrodes_of_recording(folder / c.FILE_DATASET_EEG) == ["TP9", "AF7", "AF8"]
        stored = np.loadtxt(folder / c.FILE_DATASET_EEG, delimiter=",", skiprows=1, ndmin=2)
        assert np.array_equal(stored.T, samples)
        assert read_yaml(folder / c.FILE_DATASET_PROTOCOL) == {"protocol": [["REST", 3 / FS]]}

    def test_microvolts_are_kept_to_three_decimals(self, tmp_path):
        folder = write_recording(tmp_path / "S01", ["AF7"], np.array([[1.23456]]), [])
        assert (folder / c.FILE_DATASET_EEG).read_text().splitlines() == ["AF7", "1.235"]

    def test_a_recording_is_never_written_over(self, tmp_path):
        write_recording(tmp_path / "S01", ["AF7"], np.zeros((1, 1)), [])
        with pytest.raises(FileExistsError):
            write_recording(tmp_path / "S01", ["AF7"], np.ones((1, 1)), [])
