"""focus_control: from the detector's scores to f and the generator weights; holding."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from adamus.config import Block
from adamus.detection.focus_control import CHANGE_CHANCE, Controller, focus_of, next_level
from adamus.detection.signal_processing import StepSignal
from adamus.protocol import block_spans
from tests.support.finished_sessions import CONTEXT, FS, H, N

BLOCK = 10 * FS
PROTOCOL = [Block("REST", 10), Block("FOCUS", 10), Block("FREE", 0)]


class ScriptedDetector:
    """Stands in for the detector: gives the scores one after another."""

    def __init__(self, scores):
        self._scores, self.taken = list(scores), []

    def step(self, covariances):
        self.taken.append(covariances)
        return self._scores.pop(0)


def signal(valid=True, tag=0):
    four = np.zeros(4)
    return StepSignal(valid, four, four, four > -1, [np.eye(4) * (tag + 1)], np.zeros((4, N)))


def run(validities, scores=(0.0, 1.0, -1.0, 2.0, 0.5, -3.0, 4.0, 0.0)):
    detector = ScriptedDetector(scores)
    control = Controller(block_spans(PROTOCOL, FS), N, detector)
    steps = [control.step(CONTEXT + k * H, signal(v, k)) for k, v in enumerate(validities)]
    return steps, detector


def logit(p):
    return math.log(p / (1 - p))


class TestFocusOfLevel:
    def test_even_odds_are_the_middle(self):
        assert focus_of(0.0) == 0.5

    @pytest.mark.parametrize("level", [0.3, 2.0, 7.5])
    def test_logistic_and_symmetric(self, level):
        assert focus_of(level) == pytest.approx(1 / (1 + math.exp(-level)))
        assert focus_of(-level) == pytest.approx(1 - focus_of(level))

    @pytest.mark.parametrize(("level", "f"), [(1000.0, 1.0), (-1000.0, 0.0), (800.0, 1.0)])
    def test_no_overflow_however_large_the_level(self, level, f):
        assert focus_of(level) == pytest.approx(f)

    @given(st.floats(-50, 50), st.floats(-50, 50))
    def test_a_higher_level_never_gives_a_lower_f(self, a, b):
        low, high = sorted((a, b))
        assert 0.0 <= focus_of(low) <= focus_of(high) <= 1.0


class TestHiddenMarkovModel:
    def test_one_step_adds_its_score_to_the_odds_after_a_possible_change(self):
        """From odds of focus p, the chance before the step is p(1 - q) + (1 - p)q."""
        p = focus_of(1.5)
        before = p * (1 - CHANGE_CHANCE) + (1 - p) * CHANGE_CHANCE
        assert next_level(1.5, 0.7) == pytest.approx(logit(before) + 0.7)

    def test_from_even_odds_the_first_score_is_the_level(self):
        assert next_level(0.0, 1.25) == pytest.approx(1.25)

    def test_the_state_can_change_however_sure_the_model_was(self):
        """Even at certainty the chance of a change keeps the odds within reach."""
        assert next_level(1000.0, 0.0) == pytest.approx(logit(1 - CHANGE_CHANCE))
        assert next_level(-1000.0, 0.0) == pytest.approx(logit(CHANGE_CHANCE))

    def test_steady_evidence_settles_where_change_and_score_balance(self):
        level = 0.0
        for _ in range(200):
            level = next_level(level, 1.0)
        assert next_level(level, 1.0) == pytest.approx(level)
        assert 1.0 < level < 1.0 + abs(logit(CHANGE_CHANCE))

    def test_a_single_contrary_step_does_not_flip_a_settled_state(self):
        level = 0.0
        for _ in range(20):
            level = next_level(level, 1.5)
        assert next_level(level, -1.0) > 0

    def test_three_contrary_steps_do(self):
        level = 0.0
        for _ in range(20):
            level = next_level(level, 1.5)
        for _ in range(3):
            level = next_level(level, -1.5)
        assert level < 0

    def test_the_chance_of_a_change_is_that_of_the_validation(self):
        assert CHANGE_CHANCE == 0.05


class TestFirstStep:
    def test_nothing_is_controlled_before_the_first_clean_step(self):
        steps, detector = run([False, False])
        assert all(s.f is None and not s.f_held and s.w_focus == 0.0 for s in steps)
        assert (steps[0].w_idle, steps[0].score) == (1.0, None)
        assert detector.taken == []

    def test_the_first_clean_step_controls_at_once(self):
        steps, _ = run([True], scores=[2.0])
        assert steps[0].f == pytest.approx(focus_of(2.0))
        assert steps[0].w_focus == steps[0].f


class TestControl:
    def test_f_follows_the_odds_of_the_scores_so_far(self):
        scores = [0.0, 2.0, -1.0, 0.5]
        steps, _ = run([True] * 4, scores=scores)
        level = 0.0
        for step, score in zip(steps, scores, strict=True):
            level = next_level(level, score)
            assert step.f == pytest.approx(focus_of(level))
            assert step.score == score
        assert all(s.w_focus == s.f and s.w_idle == pytest.approx(1 - s.f) for s in steps)

    def test_an_invalid_step_keeps_the_last_f_and_tells_the_detector_nothing(self):
        steps, detector = run([True, False, False, True], scores=[2.0, -2.0])
        held = focus_of(2.0)
        assert [s.f for s in steps[:3]] == pytest.approx([held] * 3)
        assert steps[3].f == pytest.approx(focus_of(next_level(2.0, -2.0)))
        assert [s.f_held for s in steps] == [False, True, True, False]
        assert [s.score for s in steps] == [2.0, None, None, -2.0]
        assert len(detector.taken) == 2

    def test_detector_gets_the_covariances_of_the_step(self):
        _, detector = run([True, True], scores=[0.0, 0.0])
        assert [d[0][0, 0] for d in detector.taken] == [1.0, 2.0]

    def test_block_and_full_window_are_reported(self):
        last_of_first = (BLOCK - CONTEXT) // H
        steps, _ = run([True] * (last_of_first + 6), scores=[0.0] * 100)
        assert (steps[0].block.index, steps[0].full_window) == (0, True)
        first_of_second = last_of_first + 1
        assert steps[first_of_second].block.index == 1
        assert not steps[first_of_second].full_window
        assert steps[first_of_second + 3].full_window  # its window lies in block 1 alone
