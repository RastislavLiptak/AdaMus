"""Turns the detector's scores into f and the generator weights.

There is no calibration: the first clean step already controls the music. The
scores of the steps so far are combined by a hidden Markov model of two states,
rest and focus, run forward only: before a step the person may have changed
state with the chance CHANGE_CHANCE, and the step's score then adds its evidence
(the score is the log odds of focus against rest). f is the chance of focus,
w_focus = f and w_idle = 1 - f. A step that is not clean keeps the last f.
"""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from adamus.detection.signal_processing import StepSignal
from adamus.protocol import BlockSpan, locate_step

CHANGE_CHANCE = 0.05  # that the state changes from one step to the next; validated with the models


@dataclass(frozen=True)
class StepControl:
    block: BlockSpan
    full_window: bool
    score: float | None  # the detector's score; None when the step was not clean
    f: float | None  # None until the first clean step
    f_held: bool
    w_idle: float
    w_focus: float


def focus_of(level: float) -> float:
    """The chance of focus at the log odds ``level``; no exponential overflows."""
    if level >= 0:
        return 1.0 / (1.0 + math.exp(-level))
    rising = math.exp(level)
    return rising / (1.0 + rising)


def next_level(level: float, score: float) -> float:
    """Log odds of focus after one more step: a possible change of state, then the score."""
    focus = focus_of(level)
    before = focus * (1.0 - CHANGE_CHANCE) + (1.0 - focus) * CHANGE_CHANCE
    return math.log(before / (1.0 - before)) + score


class Scorer(Protocol):
    """What the control needs of the detector: the score of a clean step."""

    def step(self, covariances: list[np.ndarray]) -> float: ...


class Controller:
    """Control state of one session: the detector and the log odds of focus."""

    def __init__(self, spans: list[BlockSpan], window: int, detector: Scorer) -> None:
        self._spans = spans
        self._window = window
        self._detector = detector
        self._level = 0.0  # even odds before the first step
        self._f: float | None = None

    def step(self, n_end: int, signal: StepSignal) -> StepControl:
        """Advance by the step of the main loop that ends at sample ``n_end``."""
        block, full = locate_step(self._spans, n_end, self._window)
        score = None
        if signal.valid:
            score = self._detector.step(signal.covariances)
            self._level = next_level(self._level, score)
            self._f = focus_of(self._level)
        held = not signal.valid and self._f is not None
        w_focus = self._f if self._f is not None else 0.0
        return StepControl(block, full, score, self._f, held, 1.0 - w_focus, w_focus)
