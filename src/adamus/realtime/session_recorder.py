"""Writes the session folder: everything a session received, computed and sent."""

import contextlib
import json
import logging
import platform
import subprocess
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any, BinaryIO, TextIO, Unpack

import numpy as np

import adamus
from adamus import constants as c
from adamus.config import write_yaml
from adamus.session_format import (
    EventFields,
    Meta,
    SessionFacts,
    Tick,
    folder_name,
    log_time,
)

RECORDING_CLOSE_S = 2.0  # the longest wait for Pd to finish audio.wav after "rec stop"
RECORDING_LOOK_S = 0.05  # between two looks at audio.wav meanwhile
RIFF = b"RIFF"
RIFF_HEADER = len(RIFF) + 4  # bytes: "RIFF", then the length of all that follows

# The libraries whose release changes what a session computes or plays.
VERSIONED = ("brainflow", "numpy", "scipy", "librosa", "magenta-rt")


def _version_of(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _commit() -> str | None:
    """Commit of the adamus checkout, if it runs from a git repository."""
    folder = str(Path(adamus.__file__).parent)
    try:
        git = subprocess.run(
            ["git", "-C", folder, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return git.stdout.strip() if git.returncode == 0 else None


def recording_finished(path: Path) -> bool:
    """Has Pd closed the WAV file of the session? Then its header counts every byte that follows it.

    No file means nothing to wait for.
    """
    try:
        with path.open("rb") as wav:
            header = wav.read(RIFF_HEADER)
        size = path.stat().st_size
    except FileNotFoundError:
        return True
    counted = header[len(RIFF) :]
    whole = len(header) == RIFF_HEADER and header.startswith(RIFF)
    return whole and int.from_bytes(counted, "little") == size - RIFF_HEADER


class LogFormat(logging.Formatter):
    """A line of adamus.log: its time as the app reads it back, then the level,
    the part of the program and the message."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # The name and the arguments are those logging calls; a format of the date is not used.
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802, ARG002
        return log_time(record.created)


class SessionRecorder:
    """The files of one session folder that adamus itself writes.

    The folder is made at once; the files are opened by ``open`` when the session
    runs, so that a session that never ran leaves nothing open.
    """

    _ticks: TextIO
    _events: TextIO
    _filtered: BinaryIO

    def __init__(self, sessions_root: Path, name: str, started: float) -> None:
        self.directory = Path(sessions_root) / folder_name(started, name)
        self.directory.mkdir(parents=True)
        self.raw_csv = self.directory / c.FILE_RAW  # written by the source (BrainFlow, player)
        self.audio_wav = self.directory / c.FILE_AUDIO  # written by Pd
        self.pd_log = self.directory / c.FILE_PD_LOG  # written by the generator's stderr thread
        self._open = contextlib.ExitStack()  # what close() closes, whatever was opened

    def open(self) -> None:
        """Open the files of the session and send the log of the program to its adamus.log."""
        opened = self._open.enter_context
        self._ticks = opened((self.directory / c.FILE_TICKS).open("w", encoding="utf-8"))
        self._events = opened((self.directory / c.FILE_EVENTS).open("w", encoding="utf-8"))
        self._filtered = opened((self.directory / c.FILE_FILTERED).open("wb"))
        handler = logging.FileHandler(self.directory / c.FILE_LOG, encoding="utf-8")
        handler.setFormatter(LogFormat())
        logger = logging.getLogger(adamus.__name__)
        self._open.callback(handler.close)
        self._open.callback(logger.setLevel, logger.level)
        self._open.callback(logger.removeHandler, handler)  # done first: the last one in
        logger.setLevel(logging.DEBUG)  # everything goes into adamus.log
        logger.addHandler(handler)

    def write_config(self, raw_config: dict[str, Any]) -> None:
        write_yaml(raw_config, self.directory / c.FILE_CONFIG)

    def write_model(self, raw_model: dict[str, Any]) -> None:
        """The focus model the session ran with, so that its steps can be computed again."""
        text = json.dumps(raw_model, indent=1)
        (self.directory / c.FOCUS_MODEL_FILE).write_text(text, encoding="utf-8")

    def write_meta(self, session: SessionFacts) -> None:
        """meta.json: what the session ran on, plus the ``session`` facts given."""
        meta: Meta = {
            **session,
            "adamus_commit": _commit(),
            "python": platform.python_version(),
            "versions": {name: _version_of(name) for name in VERSIONED},
        }
        with (self.directory / c.FILE_META).open("w", encoding="utf-8") as handle:
            print(json.dumps(meta, indent=2, ensure_ascii=False), file=handle)

    def event(self, kind: str, t: float, n: int, **fields: Unpack[EventFields]) -> None:
        """One line of events.jsonl; ``n`` is the number of samples received so far."""
        print(json.dumps({"t": t, "n": n, "type": kind, **fields}), file=self._events)
        self._events.flush()

    def tick(self, row: Tick, filtered: np.ndarray) -> None:
        """Step k: line k of ticks.jsonl and record k of filtered.f32, both on disk at once."""
        print(json.dumps(row, allow_nan=False), file=self._ticks)
        self._ticks.flush()
        with np.errstate(over="ignore"):  # a value too large for float32 is stored as inf
            record = np.ascontiguousarray(filtered, dtype=c.FILTERED_DTYPE)
        record.tofile(self._filtered)  # unbuffered

    def wait_for_recording(self, sleep: Callable[[float], None]) -> bool:
        """Wait until Pd has finished audio.wav, at most RECORDING_CLOSE_S; whether it did."""
        for _ in range(round(RECORDING_CLOSE_S / RECORDING_LOOK_S)):
            if recording_finished(self.audio_wav):
                return True
            sleep(RECORDING_LOOK_S)
        return False

    def close(self) -> None:
        """Close what ``open`` opened; nothing when it was never called."""
        self._open.close()
