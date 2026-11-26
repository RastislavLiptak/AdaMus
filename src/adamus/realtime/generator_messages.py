"""The messages the program sends to the generator, as Pd reads them (FUDI).

A message is atoms separated by spaces and ended by a semicolon.
"""

import re
from pathlib import Path

from adamus.config import Generator, Style

WEIGHT_DECIMALS = 3
VOLUME_DECIMALS = 2
MEANS_SOMETHING_TO_PD = r"[\s;,$\\]"  # inside a message; a backslash in front takes it away


def fudi(*atoms: object) -> str:
    return " ".join(str(atom) for atom in atoms) + ";"


def path_atom(path: Path) -> str:
    """A path as one atom of a message, wherever the path lies and whatever it is called."""
    return re.sub(MEANS_SOMETHING_TO_PD, lambda found: "\\" + found.group(), str(path.absolute()))


REC_START = fudi("rec", "start")
START_W_IDLE, START_W_FOCUS = 1.0, 0.0  # a session starts the music in its quiet form
REC_STOP = fudi("rec", "stop")


def style_messages(generator: Generator, style: Style) -> list[str]:
    """How the generator plays ``style``: what it keeps until it is told otherwise, as a reset
    does not undo it."""
    return [
        fudi("cfgmusiccoca", generator.style_guidance),
        fudi("cfgnotes", generator.note_guidance),
        fudi("temperature", style.temperature),
        fudi("topk", style.top_k),
    ]


def step_messages(style: Style, w_idle: float, w_focus: float, focused: bool) -> list[str]:
    """What one step tells the generator: how much of each form of the music, how loud, and
    whether drums may play (when ``focused``, if the style has them).

    Slot 0 is the quiet form of the music, slot 1 its full form; the volume lies between the
    two forms' as the weights do.
    """
    assert 0.0 <= w_idle <= 1.0 and 0.0 <= w_focus <= 1.0
    volume = style.rest_volume_db + w_focus * (style.focus_volume_db - style.rest_volume_db)
    drums = style.focus_drums and focused
    return [
        fudi("prompt", 0, style.rest, f"{w_idle:.{WEIGHT_DECIMALS}f}"),
        fudi("prompt", 1, style.focus, f"{w_focus:.{WEIGHT_DECIMALS}f}"),
        fudi("volume", f"{volume:.{VOLUME_DECIMALS}f}"),
        fudi("drumless", 0 if drums else 1),
    ]


def prompt_messages(styles: list[Style]) -> list[str]:
    """Every style text of ``styles`` once: the generator keeps the music of a text it has been
    given, so that a session finds it ready."""
    texts = dict.fromkeys(text for style in styles for text in (style.rest, style.focus))
    return [fudi("prompt", 0, text, f"{1.0:.{WEIGHT_DECIMALS}f}") for text in texts]


def session_start_messages(generator: Generator, style: Style, audio_wav: Path) -> list[str]:
    return [
        fudi("reset"),
        *style_messages(generator, style),
        *step_messages(style, START_W_IDLE, START_W_FOCUS, focused=False),
        fudi("rec", "open", path_atom(audio_wav)),
        REC_START,
        fudi("bypass", 0),
    ]


def session_end_messages() -> list[str]:
    return [fudi("bypass", 1), REC_STOP]
