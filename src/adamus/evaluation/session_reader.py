"""Reading a session folder back, for metrics and statistics.

A session keeps a copy of the configuration it ran with. It is read as it was written, not
against the configuration the program takes today, which may have changed since; only its
protocol is checked, as the evaluation is made of it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adamus import constants as c
from adamus.config import Block, parse_protocol, read_yaml
from adamus.detection.signal_processing import Steps
from adamus.errors import AdamusError
from adamus.protocol import BlockSpan, block_spans
from adamus.session_format import (
    EVENT_FUDI,
    RAW_SEPARATOR,
    Event,
    Meta,
    Tick,
    read_json,
    read_jsonl,
)

NOT_COMPLETE = "is not a complete session folder"


@dataclass(frozen=True)
class SessionFolder:
    directory: Path
    settings: dict[str, Any]  # the configuration the session ran with, as written
    protocol: list[Block]
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
        return block_spans(self.protocol, self.fs)

    @property
    def begun_spans(self) -> list[BlockSpan]:
        """The blocks the session got to: those begun by its last step."""
        end = self.ticks[-1]["n_end"] if self.ticks else 0
        return [span for span in self.spans if span.start < end]

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
            if event["type"] == EVENT_FUDI and event.get("msg") == message.strip():
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
        """Every whole row of raw.csv [rows x samples]: none when the session ended before its
        first sample, as BrainFlow then writes an empty raw.csv or none at all."""
        path = self.directory / c.FILE_RAW
        if not path.is_file() or path.stat().st_size == 0:
            return np.empty((0, 0))
        whole_rows = path.read_bytes().count(c.LINE_END.encode())  # not a last row cut short
        try:  # numpy reads the nan and inf that a recording may hold; BrainFlow does not
            samples: np.ndarray = np.loadtxt(
                path, delimiter=RAW_SEPARATOR, ndmin=2, max_rows=whole_rows
            ).T
        except (OSError, ValueError) as problem:
            raise AdamusError(f"{self.directory} {NOT_COMPLETE}: {problem}") from problem
        return samples

    def control_latencies(self) -> list[float]:
        """Control latency: t_sent - t_eeg of every step that was sent, in seconds."""
        return [t["t_sent"] - t["t_eeg"] for t in self.ticks if t["t_sent"] is not None]


def load_session_folder(directory: Path) -> SessionFolder:
    directory = Path(directory)
    try:
        settings = read_yaml(directory / c.FILE_CONFIG)
        return SessionFolder(
            directory,
            settings,
            parse_protocol(settings.get("protocol")),
            read_json(directory / c.FILE_META),
            read_jsonl(directory / c.FILE_TICKS),
            read_jsonl(directory / c.FILE_EVENTS),
        )
    except (AdamusError, OSError, ValueError) as problem:
        raise AdamusError(f"{directory} {NOT_COMPLETE}: {problem}") from problem
