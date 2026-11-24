"""The format of a session folder: its name, the events and what its lines hold. Written
by a session, read back by the evaluation and the user interface; the names of the files are in
constants.

Old session folders are read again later, so the writer and the readers must
agree on every key; typed here, a key renamed on one side only is a type error.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from adamus.constants import LINE_END
from adamus.errors import AdamusError

FOLDER_TIME = "%Y%m%d-%H%M%S"  # the local time the session started
RAW_SEPARATOR = chr(9)  # a tab, as in the raw file BrainFlow writes for a headset
NAME_SEPARATOR = "_"  # between the time the session started and its name
LOG_TIME_PRECISION = "milliseconds"  # of the time each line of adamus.log and pd.log begins with
LOG_SEPARATOR = " "  # between the time of a line of a log and what it says

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
    audio_output: NotRequired[str | None]  # the music was played on; not known of older sessions
    board_id: int
    fs: int
    electrode_rows: dict[str, int]
    detector_electrodes: list[str]
    timestamp_row: int
    marker_row: int
    playback_path: str | None
    started: float


class Meta(SessionFacts):
    """meta.json: the facts of the session, the versions it ran with, the foreign files."""

    adamus_commit: str | None
    python: str
    versions: dict[str, str | None]  # of the libraries whose release changes the results


#: What a session may be called: any text a folder can be named by, as "Karel Jedlička 8": at
#: least one character that is not a space, none that ends a folder's name (/ :), is a
#: control character or half of a character (a lone surrogate, which no file name can hold),
#: and at most NAME_MAX characters, so that the folder's name stays within the
#: 255 bytes macOS allows also in scripts of 4 bytes a character. Spaces at its ends are
#: trimmed. The app checks a name by it as the operator types (as a regular expression of
#: ICU, so no syntax of Python's alone).
NAME_MAX = 50  # characters
_NOT_IN_NAMES = r"/:\x00-\x1f\x7f\ud800-\udfff"
SESSION_NAME = rf"(?=.{{1,{NAME_MAX}}}$)[^{_NOT_IN_NAMES}]*[^\s{_NOT_IN_NAMES}][^{_NOT_IN_NAMES}]*"


def is_session_name(name: str) -> bool:
    return re.fullmatch(SESSION_NAME, name) is not None


def folder_name(started: float, name: str) -> str:
    """The folder of a session: the local time it started, then the name it was given."""
    return time.strftime(FOLDER_TIME, time.localtime(started)) + NAME_SEPARATOR + name


def renamed(folder: str, name: str) -> str:
    """The folder of a session given another name: the time it started stays."""
    return folder.partition(NAME_SEPARATOR)[0] + NAME_SEPARATOR + name


def name_of(folder: Path) -> str:
    """The name the session was given, from its folder."""
    return folder.name.split(NAME_SEPARATOR, 1)[-1]


def is_session_folder(name: str) -> bool:
    """Is ``name`` one that a session gives its folder: the time it started, then its name?"""
    started, _, session = name.partition(NAME_SEPARATOR)
    try:
        time.strptime(started, FOLDER_TIME)
    except ValueError:
        return False
    return is_session_name(session)


def log_time(t: float) -> str:
    """The time a line of a log begins with: the local time to the millisecond, with its
    offset from UTC, so that it is read back as the very moment (ISO 8601)."""
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec=LOG_TIME_PRECISION)


def read_log_line(line: str) -> tuple[float, str] | None:
    """When a line of a log was written, and what it says; None for a line without a time,
    which goes on with the line before it (as a traceback does)."""
    stamp, _, said = line.partition(LOG_SEPARATOR)
    try:
        written = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return (written.timestamp(), said) if written.tzinfo is not None else None


def _parsed(path: Path, text: str) -> Any:
    """``text`` of the file ``path`` as JSON; what cannot be read is named by its file."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as problem:
        raise AdamusError(f"{path.name} is damaged ({problem})") from problem


def read_json(path: Path) -> Any:
    """A .json file of a session."""
    return _parsed(path, path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    """The lines of a .jsonl file of a session, each one JSON. A last line cut short, as when
    the program ended while writing it, is left out: what came before is the record."""
    *lines, last = path.read_text(encoding="utf-8").split(LINE_END)
    try:
        ended = [json.loads(last)] if last else []
    except json.JSONDecodeError:
        ended = []
    return [_parsed(path, line) for line in lines if line] + ended
