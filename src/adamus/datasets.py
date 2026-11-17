"""The datasets of the data folder, and how a recording of one is written and read.

A dataset is a folder of recordings. A recording is a folder with the samples
of all its electrodes, at the sampling rate of the headset, and the protocol
they were recorded under.
"""

import itertools
from pathlib import Path
from typing import Any

import numpy as np

from adamus import constants as c
from adamus.config import Block, read_yaml, write_yaml
from adamus.errors import AdamusError

SEPARATOR = ","  # between the values of a line of a recording
SAMPLE_FORMAT = "%.3f"  # microvolts
ENCODING = "utf-8-sig"  # a file saved by a spreadsheet may begin with a byte order mark


def datasets_in(root: Path) -> list[Path]:
    """The datasets of the data folder that hold a recording, in alphabetical order."""
    folders = [folder for folder in Path(root).glob("*") if recordings_of(folder)]
    return sorted(folders, key=lambda folder: folder.name)


def recordings_of(dataset: Path) -> list[Path]:
    """The recordings of a dataset: its folders that hold both files, in alphabetical order."""
    folders = [
        folder
        for folder in Path(dataset).glob("*")
        if eeg_of(folder).is_file() and (folder / c.FILE_DATASET_PROTOCOL).is_file()
    ]
    return sorted(folders, key=lambda folder: folder.name)


def eeg_of(recording: Path) -> Path:
    """The file of the samples of a recording."""
    return Path(recording) / c.FILE_DATASET_EEG


def electrodes_of_recording(eeg_csv: Path) -> list[str]:
    """The names in the first line of a recording: one per column."""
    try:
        with Path(eeg_csv).open(encoding=ENCODING) as lines:
            names = [name.strip() for name in lines.readline().split(SEPARATOR)]
    except (OSError, ValueError) as problem:
        raise AdamusError(f"cannot read {eeg_csv}: {problem}") from problem
    if "" in names or len(set(names)) != len(names):
        raise AdamusError(f"{eeg_csv}: the first line must name every column, and none twice")
    return names


def read_recording(eeg_csv: Path) -> tuple[list[str], np.ndarray]:
    """The electrodes of a recording and its samples [electrodes x samples]."""
    names = electrodes_of_recording(eeg_csv)
    try:
        with Path(eeg_csv).open(encoding=ENCODING) as lines:
            lines.readline()  # the names
            first = lines.readline()
            if not first.strip():
                raise AdamusError(f"{eeg_csv}: no samples after the names of the electrodes")
            rows = itertools.chain([first], lines)
            samples = np.loadtxt(rows, delimiter=SEPARATOR, ndmin=2)
    except (OSError, ValueError) as problem:
        raise AdamusError(f"cannot read {eeg_csv}: {problem}") from problem
    if samples.shape[1] != len(names):
        raise AdamusError(f"{eeg_csv}: every line must hold {len(names)} values")
    return names, samples.T


def protocol_of(recording: Path) -> Any:
    """The protocol a recording was recorded under, as its protocol.yaml has it."""
    return read_yaml(recording / c.FILE_DATASET_PROTOCOL).get("protocol")


def write_recording(
    folder: Path, names: list[str], samples: np.ndarray, protocol: list[Block]
) -> Path:
    """A new recording: a line of the electrode names, then one line per sample of
    ``samples`` [electrodes x n]; and the protocol of those samples."""
    folder = Path(folder)
    folder.mkdir(parents=True)
    np.savetxt(
        eeg_of(folder), samples.T, fmt=SAMPLE_FORMAT, delimiter=SEPARATOR,
        header=SEPARATOR.join(names), comments="",
    )  # fmt: skip
    blocks = [[block.label, block.duration_s] for block in protocol]
    write_yaml({"protocol": blocks}, folder / c.FILE_DATASET_PROTOCOL)
    return folder
