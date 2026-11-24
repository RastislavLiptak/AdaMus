"""The data sources a session can run from: the headset, and the datasets of the data folder."""

from dataclasses import dataclass
from pathlib import Path

from adamus import constants as c
from adamus import datasets
from adamus.config import INVALID, Board, Config, parse_config
from adamus.errors import AdamusError
from adamus.realtime.eeg_source import electrodes_of_headset

HEADSET, DATASET = "headset", "dataset"


@dataclass(frozen=True)
class Source:
    kind: str  # HEADSET or DATASET
    label: str  # the name of the headset, or of the dataset
    dataset: Path | None = None  # the folder of a dataset


def available_sources(headset: Board | None, folders: list[Path]) -> list[Source]:
    """The headset first, when it was found, then the datasets in ``folders``."""
    found = [Source(HEADSET, headset.name)] if headset is not None else []
    return found + [Source(DATASET, folder.name, dataset=folder) for folder in folders]


def session_config(config: Config, recording: Path | None) -> Config:
    """The configuration of a session: over a recording, the protocol is the recording's own."""
    if recording is None:
        return config
    try:
        return parse_config({**config.raw, "protocol": datasets.protocol_of(recording)})
    except AdamusError as problem:  # only the protocol differs from a configuration that loaded
        heading = f"{recording / c.FILE_DATASET_PROTOCOL}: the recording cannot be played:"
        lines = [line for line in problem.args if line != INVALID]
        raise AdamusError(heading, *lines) from problem


def missing_electrodes(config: Config, recording: Path | None) -> list[str]:
    """The electrodes of the detector that the source does not have."""
    if recording is None:
        present = electrodes_of_headset(config.board)
    else:
        present = datasets.electrodes_of_recording(datasets.eeg_of(recording))
    return [name for name in c.DETECTOR_ELECTRODES if name not in present]
