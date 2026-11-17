"""The format of a session folder: its name, the events and what its lines hold. Written
by a session, read back by the evaluation and the console; the names of the files are in
constants.

Old session folders are read again later, so the writer and the readers must
agree on every key; typed here, a key renamed on one side only is a type error.
"""

import time
from pathlib import Path
from typing import Literal, TypedDict

FOLDER_TIME = "%Y%m%d-%H%M%S"  # the local time the session started
RAW_SEPARATOR = chr(9)  # a tab, as in the raw file BrainFlow writes for a headset
NAME_SEPARATOR = "_"  # between the time the session started and its name

# The types of the lines of events.jsonl.
EVENT_SESSION_START = "SESSION_START"
EVENT_SESSION_END = "SESSION_END"
EVENT_STREAM_START = "STREAM_START"
EVENT_FUDI = "FUDI"
EVENT_BLOCK_START = "BLOCK_START"
EVENT_ERROR = "ERROR"

# Where a session was when it failed, in its ERROR event.
STAGE_RECORD, STAGE_GENERATOR, STAGE_SOURCE = "record", "generator", "source"
STAGE_LOOP, STAGE_STREAM = "loop", "stream"

# Why a session ended, in its SESSION_END event.
REASON_USER = "user"
REASON_END_OF_STREAM = "end_of_stream"
REASON_ERROR = "error"


class Tick(TypedDict):
    """One line of ticks.jsonl: step k."""

    k: int
    n_end: int  # the sample after the last of the step
    t_eeg: float  # the time of that last sample, from the source
    t_proc_start: float
    t_proc_end: float
    t_sent: float | None  # None when the weights were not sent
    block_index: int
    block_label: str
    full_window: bool  # the whole window lies inside the block
    max_abs_uv: list[float | None]  # per electrode of the detector; None where no number came
    valid: bool
    score: float | None
    f: float | None
    f_held: bool
    w_idle: float
    w_focus: float


class EventFields(TypedDict, total=False):
    """What an event holds besides its time and type; which of these depends on the type."""

    reason: str  # SESSION_END
    board_id: int  # STREAM_START
    fs: int  # STREAM_START
    index: int  # BLOCK_START
    label: str  # BLOCK_START
    marker: int  # BLOCK_START
    msg: str  # FUDI
    where: str  # ERROR
    message: str  # ERROR


class Event(EventFields):
    """One line of events.jsonl."""

    t: float
    n: int  # samples received so far
    type: str


class SessionFacts(TypedDict):
    """What a session ran on: its source, its detector and its generator."""

    pd: str | None
    board_id: int
    fs: int
    electrode_rows: dict[str, int]
    detector_electrodes: list[str]
    timestamp_row: int
    marker_row: int
    mode: Literal["live", "playback"]
    playback_path: str | None
    started: float


class Meta(SessionFacts):
    """meta.json: the facts of the session, the versions it ran with, the foreign files."""

    adamus_commit: str | None
    python: str
    versions: dict[str, str | None]  # of the libraries whose release changes the results
    files: dict[str, str]  # written by the source and by Pd, not by the recorder


def folder_name(started: float, name: str) -> str:
    """The folder of a session: the local time it started, then the name it was given."""
    return time.strftime(FOLDER_TIME, time.localtime(started)) + NAME_SEPARATOR + name


def name_of(folder: Path) -> str:
    """The name the session was given, from its folder."""
    return folder.name.split(NAME_SEPARATOR, 1)[-1]
