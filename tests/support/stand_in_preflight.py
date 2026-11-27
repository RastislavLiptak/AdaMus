"""Stands in for the preflight that passed, as the user interface asks it: its configuration."""

from tests.support.finished_sessions import config_with


class Checked:
    def __init__(self) -> None:
        self.config = config_with()
