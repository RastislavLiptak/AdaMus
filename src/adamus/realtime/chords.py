"""The loop of chords under the music: which notes the generator is given, and when.

The generator follows the notes it is held in every frame of its music, so the same loop of
chords under both forms of a style keeps them one piece, and keeps its harmony and its bars
repeating as a song does. A chord lasts a bar of the style's tempo; in focus a style may play
it as a power chord. A chord is let go, then after the style's gap the next is struck; with no
gap both go in one message. A chord with nothing to let go before it is struck at once.
"""

import math

from adamus.config import Style
from adamus.realtime.generator_messages import fudi

NOTES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}  # semitones above C
INTERVALS = {"": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "m7": (0, 3, 7, 10),
             "maj7": (0, 4, 7, 11), "5": (0, 7)}  # fmt: skip  # semitones above the root
POWER = "5"  # the root and the fifth over the root in the bass: root, fifth and octave
OCTAVE = 12
BASS_FROM = 40  # MIDI, the E of a guitar's lowest string: the root of a chord in the octave from it
UPPER_FROM = 57  # MIDI, the A below middle C: the other notes of a chord in the octave from it
ROUNDING = 1e-9  # of a bar: a moment computed as the start plus whole bars lies in the last of them


def voicing(chord: str, power: bool = False) -> tuple[int, ...]:
    """The MIDI notes of ``chord`` (as config checks it: a root, then its kind): its root in
    the bass, then its notes in the octave from UPPER_FROM; ``power``: root, fifth and octave
    only."""
    root = NOTES[chord[0]]
    bass = BASS_FROM + (root - BASS_FROM) % OCTAVE
    intervals = INTERVALS[POWER if power else chord[1:]]
    upper = {UPPER_FROM + (root + i - UPPER_FROM) % OCTAVE for i in intervals}
    return (bass, *sorted(upper))


def messages(kind: str, notes: tuple[int, ...]) -> list[str]:
    """``noteon`` or ``noteoff`` for every one of ``notes``."""
    return [fudi(kind, note) for note in notes]


class ChordLoop:
    """The chords of one style, struck one a bar from ``start`` on, in their order and again."""

    def __init__(self, style: Style, start: float) -> None:
        self._style = style
        self._start = start  # on the monotonic clock
        self._sounding: tuple[int, ...] = ()  # the notes the generator holds now
        self._wanted: tuple[int, ...] = ()  # what it is to hold
        self._strike_at: float | None = None  # when the wanted notes are struck, once let go

    def due(self, now: float, focused: bool) -> list[str]:
        """What the generator is to be told at ``now``: the chord of the bar, as a power chord
        when the style plays one in focus, let go and struck if it is not what it holds."""
        bar = self._bar(now)
        chords = self._style.chords
        wanted = voicing(chords[bar % len(chords)], focused and self._style.focus_power_chords)
        told: list[str] = []
        if wanted != self._wanted:
            if self._sounding:  # let go, and struck after the gap
                told += messages("noteoff", self._sounding)
                self._strike_at = now + self._style.chord_gap_s
            elif self._strike_at is None:  # nothing to let go: struck at once
                self._strike_at = now
            self._sounding, self._wanted = (), wanted
        if self._strike_at is not None and now >= self._strike_at:
            told += messages("noteon", self._wanted)
            self._sounding, self._strike_at = self._wanted, None
        return told

    def wait(self, now: float) -> float:
        """How long until something is due: the next notes struck, or the next bar."""
        next_bar = self._start + (self._bar(now) + 1) * self._style.bar_s
        moments = [next_bar] + ([self._strike_at] if self._strike_at is not None else [])
        return max(0.0, min(moments) - now)

    def _bar(self, now: float) -> int:
        """The bar ``now`` lies in, 0 from the start."""
        return math.floor((now - self._start) / self._style.bar_s + ROUNDING)

    def release(self) -> list[str]:
        """Let go of every note held: the generator keeps them when it is reset."""
        told = messages("noteoff", self._sounding)
        self._sounding, self._wanted, self._strike_at = (), (), None
        return told
