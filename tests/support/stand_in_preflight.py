"""Stands in for the preflight that passed, as the user interface asks it: its configuration
and the generator it started, on the output the system played on."""

from tests.support.finished_sessions import config_with
from tests.support.stand_in_devices import SPEAKERS, StandInGenerator


class Checked:
    def __init__(self) -> None:
        self.config = config_with()
        self.generator = StandInGenerator()
        self.generator.start(SPEAKERS)
