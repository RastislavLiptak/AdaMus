"""Reading a session folder back, for metrics and statistics."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adamus import constants as c
from adamus.config import Config, load_config
from adamus.detection.signal_processing import Steps
from adamus.errors import AdamusError
from adamus.protocol import BlockSpan, block_spans
from adamus.session_files import EVENT_FUDI, RAW_SEPARATOR, Event, Meta, Tick

NOT_COMPLETE = "is not a complete session folder"


@dataclass(frozen=True)
class SessionFolder:
    directory: Path
    config: Config
    meta: Meta
    ticks: list[Tick]
    events: list[Event]

    @property
    def fs(self) -> int:
        return int(self.meta["fs"])

    @property
    def steps(self) -> Steps:
        return Steps.at(self.fs)

    @property
    def spans(self) -> list[BlockSpan]:
        return block_spans(self.config.protocol, self.fs)

    @property
    def evaluation_spans(self) -> list[BlockSpan]:
        """The REST and FOCUS blocks."""
        return [s for s in self.spans if s.label != c.FREE]

    def event(self, kind: str) -> Event | None:
        """The first event of a kind, or None."""
        return next((event for event in self.events if event["type"] == kind), None)

    def fudi_time(self, message: str) -> float | None:
        """When a session message (e.g. ``rec start;``) was sent."""
        for event in self.events:
            if event["type"] == EVENT_FUDI and event["msg"] == message.strip():
                return float(event["t"])
        return None

    def controlled_ticks(self) -> list[Tick]:
        """Valid steps: the steps that set a new f."""
        return [tick for tick in self.ticks if tick["valid"] and tick["f"] is not None]

    def block_f(self, span: BlockSpan) -> list[float]:
        """f of the valid full steps of one block."""
        return [
            f
            for tick in self.controlled_ticks()
            if (f := tick["f"]) is not None
            and tick["block_index"] == span.index
            and tick["full_window"]
        ]

    def raw_samples(self) -> np.ndarray:
        """Every row of raw.csv [rows x samples]: none when the session ended before its first
        sample, as BrainFlow then writes an empty raw.csv or none at all."""
        path = self.directory / c.FILE_RAW
        if not path.is_file() or path.stat().st_size == 0:
            return np.empty((0, 0))
        try:  # numpy reads the nan and inf that a recording may hold; BrainFlow does not
            samples: np.ndarray = np.loadtxt(path, delimiter=RAW_SEPARATOR, ndmin=2).T
        except (OSError, ValueError) as problem:
            raise AdamusError(f"{self.directory} {NOT_COMPLETE}: {problem}") from problem
        return samples

    def control_latencies(self) -> list[float]:
        """Control latency: t_sent - t_eeg of every step that was sent, in seconds."""
        return [t["t_sent"] - t["t_eeg"] for t in self.ticks if t["t_sent"] is not None]


def _read_lines(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_session_folder(directory: Path) -> SessionFolder:
    directory = Path(directory)
    try:
        return SessionFolder(
            directory,
            load_config(directory / c.FILE_CONFIG),
            json.loads((directory / c.FILE_META).read_text(encoding="utf-8")),
            _read_lines(directory / c.FILE_TICKS),
            _read_lines(directory / c.FILE_EVENTS),
        )
    except (AdamusError, OSError, ValueError) as problem:
        raise AdamusError(f"{directory} {NOT_COMPLETE}: {problem}") from problem
