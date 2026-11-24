"""Running a whole session over prepared data, with stand-ins for all devices."""

from pathlib import Path
from typing import Any

import numpy as np

from adamus import constants as c
from adamus.config import parse_config, read_yaml
from adamus.detection.signal_processing import Steps
from adamus.realtime.eeg_source import RecordingSource
from adamus.realtime.session import Devices, Session
from adamus.session_format import EVENT_ERROR, EVENT_FUDI, read_jsonl
from tests.support import synthetic_dataset
from tests.support.project_paths import CONFIG_FILE
from tests.support.stand_in_devices import (
    SPEAKERS,
    StandInClock,
    StandInGenerator,
    StandInSource,
    StopWhen,
)
from tests.support.synthetic_dataset import board_matrix

FS = c.DETECTOR_FS  # of Muse 2, the headset of the configuration
N, H, CONTEXT = Steps.at(FS).window, Steps.at(FS).hop, Steps.at(FS).context


def config_with(protocol=None, **sections):
    """The repository configuration with a protocol and/or parts of sections replaced."""
    raw = read_yaml(CONFIG_FILE)
    if protocol is not None:
        raw["protocol"] = protocol
    for name, values in sections.items():
        raw[name] = {**raw[name], **values}
    return parse_config(raw)


def synthetic_signals(protocol):
    """The synthetic signal of the detector's electrodes for the timed blocks of a protocol."""
    return synthetic_dataset.signals([tuple(block) for block in protocol], FS)


def synthetic_matrix(protocol, tail_s=0):
    """A board matrix with the synthetic signal of a protocol, plus ``tail_s`` of REST."""
    signals = synthetic_signals(protocol)
    if tail_s:
        signals = np.hstack([signals, synthetic_signals([[c.REST, tail_s]])])
    return board_matrix(signals, config_with())


class FinishedSession:
    """One session run to its end, and everything it left behind."""

    def __init__(self, root, config, matrix=None, name="S01", stop_after_steps=None,
                 generator=None, source=None, recording=None, clock=None,
                 **source_options):  # fmt: skip
        """Over a board matrix of the stand-in headset, or over the ``recording`` of a dataset."""
        self.root = Path(root)
        self.clock = clock or StandInClock()
        if recording is not None:
            source = RecordingSource(config.board, recording, self.clock)
        # the stand-in headset, whose notes the tests read, or the source of a recording
        self.source: Any = source or StandInSource(matrix, **source_options)
        self.generator = generator or StandInGenerator()
        if not self.generator.running:  # as the program has it when a session starts
            self.generator.start(SPEAKERS)
        self._stop_after_steps = stop_after_steps
        devices = Devices(self.source, self.generator, StopWhen(self._enough_steps), self.clock)
        self._session = Session(config, name, devices, Path(root) / "sessions")
        self.directory = self._session.directory
        self.result = self._session.run()
        if recording is not None:
            self.source.close()  # as run_session does; it holds the raw file open
        self.ticks = read_jsonl(self.directory / c.FILE_TICKS)
        self.events = read_jsonl(self.directory / c.FILE_EVENTS)

    def _enough_steps(self):
        if self._stop_after_steps is None:
            return False
        written = len((self.directory / c.FILE_TICKS).read_text().splitlines())
        return written >= self._stop_after_steps

    @property
    def error(self):
        """The message of the ERROR event, or None when the session did not fail."""
        messages = [event["message"] for event in self.events_of(EVENT_ERROR)]
        return messages[0] if messages else None

    def events_of(self, kind):
        return [event for event in self.events if event["type"] == kind]

    def session_messages(self):
        """The FUDI messages logged as events (everything but the step weights)."""
        return [event["msg"] for event in self.events_of(EVENT_FUDI)]
