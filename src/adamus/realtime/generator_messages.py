"""The messages the program sends to the generator, as Pd reads them (FUDI).

A message is atoms separated by spaces and ended by a semicolon.
"""

import re
from pathlib import Path

from adamus.config import Generator

WEIGHT_DECIMALS = 3
MEANS_SOMETHING_TO_PD = r"[\s;,$\\]"  # inside a message; a backslash in front takes it away


def fudi(*atoms: object) -> str:
    return " ".join(str(atom) for atom in atoms) + ";"


def path_atom(path: Path) -> str:
    """A path as one atom of a message, wherever the path lies and whatever it is called."""
    return re.sub(MEANS_SOMETHING_TO_PD, lambda found: "\\" + found.group(), str(path.absolute()))


REC_START = fudi("rec", "start")
REC_STOP = fudi("rec", "stop")


def step_messages(config: Generator, w_idle: float, w_focus: float) -> list[str]:
    """What one step tells the generator: how much of each form of the music.

    Slot 0 is the quiet form of the music, slot 1 its full form.
    """
    assert 0.0 <= w_idle <= 1.0 and 0.0 <= w_focus <= 1.0
    return [
        fudi("prompt", 0, config.prompt_idle, f"{w_idle:.{WEIGHT_DECIMALS}f}"),
        fudi("prompt", 1, config.prompt_focus, f"{w_focus:.{WEIGHT_DECIMALS}f}"),
    ]


def session_start_messages(config: Generator, audio_wav: Path) -> list[str]:
    return [
        fudi("reset"),
        *step_messages(config, 1.0, 0.0),
        fudi("rec", "open", path_atom(audio_wav)),
        REC_START,
        fudi("bypass", 0),
    ]


def session_end_messages() -> list[str]:
    return [fudi("bypass", 1), REC_STOP]
