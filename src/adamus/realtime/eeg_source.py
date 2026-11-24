"""The EEG source: a live headset, or a recording of a dataset played at the same pace.

Both deliver everything they have: one row per electrode, a row of times and
a row of markers. The session takes the electrodes it needs by their names.
The two sources differ in where the samples come from, and in that a recording
comes to an end where a headset that falls silent is a fault.
"""

import logging
import sys
from pathlib import Path
from typing import IO, Any

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowError, BrainFlowInputParams

from adamus import datasets
from adamus.config import Board
from adamus.errors import AdamusError
from adamus.realtime.clock import TimeSource
from adamus.session_format import RAW_SEPARATOR

log = logging.getLogger(__name__)

NAMES_SEPARATOR = ","  # between the names BrainFlow has for the rows of a board
UNNAMED_ROW = "EEG "  # then the place of a row of EEG BrainFlow has no name for
# How the raw file of a recording is written: the samples, the times, the markers.
RAW_TIME, RAW_MARKER = "%.6f", "%g"  # the samples as a recording of a dataset has them


def _description(board: Board) -> dict[str, Any]:
    try:
        description: dict[str, Any] = BoardShim.get_board_descr(board.id)
    except BrainFlowError as problem:
        raise AdamusError(f"unknown board id {board.id}: {problem}") from problem
    return description


def _open_parameters(board: Board) -> BrainFlowInputParams:
    parameters = BrainFlowInputParams()
    parameters.serial_number = board.serial_number
    return parameters


def _shim(board: Board) -> BoardShim:
    """BrainFlow's handle of a headset; BrainFlow's own log stays quiet."""
    BoardShim.disable_board_logger()
    return BoardShim(board.id, _open_parameters(board))


def said_before_bluetooth(what: str) -> None:
    """Said on the error output before the headset is reached over Bluetooth: macOS ends a
    program at once that may not use it, and the app shows the last lines the program said."""
    print(what, file=sys.stderr, flush=True)


def headset_is_available(board: Board) -> bool:
    """Try to open a live headset and release it again."""
    shim = _shim(board)
    said_before_bluetooth(f"Looking for {board.name} over Bluetooth.")
    try:
        shim.prepare_session()
        shim.release_session()
    except BrainFlowError:
        return False
    return True


def electrodes_of_headset(board: Board) -> list[str]:
    """The names BrainFlow has for the EEG rows of a board, in their order; for some it has none."""
    names = str(_description(board).get("eeg_names", ""))
    return names.split(NAMES_SEPARATOR) if names else []


class HeadsetSource:
    """Delivers the samples of a live headset, as BrainFlow hands them over."""

    playback_file = None

    def __init__(self, board: Board) -> None:
        description = _description(board)
        self.board_id = board.id
        self.fs = int(description["sampling_rate"])
        names = electrodes_of_headset(board)
        self.electrode_rows = {  # a row BrainFlow has no name for is named by its place
            names[index] if index < len(names) else f"{UNNAMED_ROW}{index + 1}": row
            for index, row in enumerate(description["eeg_channels"])
        }
        self.timestamp_row = int(description["timestamp_channel"])
        self.marker_row = int(description["marker_channel"])
        self._name = board.name
        self._shim = _shim(board)

    def start(self, raw_csv: Path) -> None:
        """Open the board and start streaming; BrainFlow itself writes ``raw_csv``."""
        said_before_bluetooth(f"Opening {self._name} over Bluetooth.")
        try:
            self._shim.prepare_session()
            self._shim.add_streamer(f"file://{raw_csv}:w")
            self._shim.start_stream()
        except BrainFlowError as problem:
            raise AdamusError(f"cannot start the EEG stream: {problem}") from problem

    def read(self) -> np.ndarray:
        """All samples received since the previous call, as [board rows x samples]."""
        samples: np.ndarray = self._shim.get_board_data()
        return samples

    def insert_marker(self, value: float) -> None:
        self._shim.insert_marker(value)

    def stop(self) -> None:
        """Stop streaming; samples already received can still be read."""
        try:
            self._shim.stop_stream()
        except BrainFlowError as problem:
            log.warning("stopping the stream failed: %s", problem)

    def close(self) -> None:
        """Let go of the headset; a link already lost is only noted, the session is over."""
        try:
            if self._shim.is_prepared():
                self._shim.release_session()
        except BrainFlowError as problem:
            log.warning("releasing the headset failed: %s", problem)


class RecordingSource:
    """Delivers the samples of a recording at the sampling rate of the headset.

    A recording is a folder of a dataset; its table of samples is played (datasets).
    Sample k is due k sampling periods after the start; it gets that time and is
    written to the raw file of the session like the samples of a headset.
    """

    def __init__(self, board: Board, recording: Path, clock: TimeSource) -> None:
        played = datasets.eeg_of(recording)
        names, samples = datasets.read_recording(played)
        self.board_id = board.id
        self.fs = int(_description(board)["sampling_rate"])
        self.playback_file = played
        self.electrode_rows = {name: row for row, name in enumerate(names)}
        self.timestamp_row = len(names)
        self.marker_row = len(names) + 1
        self._samples = samples  # [electrodes x samples]
        self._clock = clock
        self._started_at = 0.0  # time of day of the first sample, for the times of the samples
        self._started_monotonic = 0.0  # for which samples are due
        self._delivered = 0
        self._marker = 0.0  # waits for the next sample
        self._raw: IO[str] | None = None

    def start(self, raw_csv: Path) -> None:
        self._raw = Path(raw_csv).open("w", encoding="utf-8")  # noqa: SIM115 - closed by close()
        self._started_at = self._clock.now()
        self._started_monotonic = self._clock.monotonic()

    def read(self) -> np.ndarray:
        """The samples that became due since the previous call, with their times and markers."""
        assert self._raw is not None, "the source was not started"
        due = int((self._clock.monotonic() - self._started_monotonic) * self.fs)
        index = np.arange(self._delivered, min(due, self._samples.shape[1]))
        self._delivered += len(index)
        times = self._started_at + index / self.fs
        markers = np.zeros(len(index))
        if len(index) > 0:
            markers[0], self._marker = self._marker, 0.0
        block = np.vstack([self._samples[:, index], times, markers])
        formats = [datasets.SAMPLE_FORMAT] * len(self.electrode_rows) + [RAW_TIME, RAW_MARKER]
        np.savetxt(self._raw, block.T, fmt=formats, delimiter=RAW_SEPARATOR)
        return block

    def insert_marker(self, value: float) -> None:
        self._marker = value

    def stop(self) -> None:
        """Nothing to stop: what is due can still be read."""

    def close(self) -> None:
        if self._raw is not None:
            self._raw.close()
            self._raw = None
