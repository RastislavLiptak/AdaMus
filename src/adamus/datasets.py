"""The datasets of the data folder, and how a recording of one is written and read.

A dataset is a folder of recordings. A recording is a folder with the samples
of all its electrodes, at the sampling rate of the headset, and the protocol
they were recorded under.
"""

import itertools
import re
from pathlib import Path
from typing import Any

import numpy as np

from adamus import constants as c
from adamus.config import Block, read_yaml, write_yaml
from adamus.errors import AdamusError

PARTIAL = ".partial"  # the end of the name of a recording being written

SEPARATOR = ","  # between the values of a line of a recording
SAMPLE_FORMAT = "%.3f"  # microvolts
DIGITS = re.compile(r"(\d+)")


def in_name_order(folder: Path) -> tuple[list[str | int], str]:
    """The order of names that Finder keeps: numbers in them by their value, so that S2 comes
    before S10, and letters whatever their case; names equal so (S1, S01) by the name itself.
    Splitting at the numbers leaves each number at an odd place."""
    parts = DIGITS.split(folder.name)
    key = [int(part) if place % 2 else part.lower() for place, part in enumerate(parts)]
    return key, folder.name


def datasets_in(root: Path) -> list[Path]:
    """The datasets of the data folder that hold a recording, in the order of their names."""
    folders = [folder for folder in Path(root).glob("*") if recordings_of(folder)]
    return sorted(folders, key=in_name_order)


def recordings_of(dataset: Path) -> list[Path]:
    """The recordings of a dataset, the folders holding both files, in the order of their names."""
    folders = [
        folder
        for folder in Path(dataset).glob("*")
        if eeg_of(folder).is_file() and (folder / c.FILE_DATASET_PROTOCOL).is_file()
    ]
    return sorted(folders, key=in_name_order)


def eeg_of(recording: Path) -> Path:
    """The file of the samples of a recording."""
    return Path(recording) / c.FILE_DATASET_EEG


def electrodes_of_recording(eeg_csv: Path) -> list[str]:
    """The names in the first line of a recording: one per column."""
    try:
        with Path(eeg_csv).open(encoding=c.TEXT_ENCODING) as lines:
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
        with Path(eeg_csv).open(encoding=c.TEXT_ENCODING) as lines:
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


def person_of(recording: Path) -> str | None:
    """Who was recorded, as the dataset of the recording names them (so the focus model names
    the people it was trained on); None where its protocol.yaml does not say."""
    person = read_yaml(recording / c.FILE_DATASET_PROTOCOL).get("person")
    return person if isinstance(person, str) else None


def write_recording(
    folder: Path, names: list[str], samples: np.ndarray, protocol: list[Block]
) -> Path:
    """A new recording: a line of the electrode names, then one line per sample of
    ``samples`` [electrodes x n]; and the protocol of those samples. Written into a hidden
    folder first, and named when whole: one cut short is no recording."""
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"{folder} is a recording already")
    being_written = folder.with_name(f".{folder.name}{PARTIAL}")
    being_written.mkdir(parents=True)
    np.savetxt(
        eeg_of(being_written), samples.T, fmt=SAMPLE_FORMAT, delimiter=SEPARATOR,
        header=SEPARATOR.join(names), comments="",
    )  # fmt: skip
    blocks = [[block.label, block.duration_s] for block in protocol]
    write_yaml({"protocol": blocks}, being_written / c.FILE_DATASET_PROTOCOL)
    being_written.rename(folder)
    return folder
