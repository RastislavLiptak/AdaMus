"""evaluation.session_statistics: the short facts of a session."""

import pytest

from adamus.evaluation.session_reader import load_session_folder
from adamus.evaluation.session_statistics import statistics_of
from tests.support.handmade_session_folders import (
    CONTEXT,
    FS,
    H,
    event,
    finished_session_folder,
    protocol_ticks,
    tick,
    write_session_folder,
)

PROTOCOL = [["REST", 10], ["FOCUS", 10], ["REST", 10], ["FOCUS", 10], ["REST", 10], ["FREE", 0]]
BLOCK = 10 * FS
ALL_STEPS = (5 * BLOCK - CONTEXT) // H + 1
NO_CLEAN_SIGNAL = "not one clean step for the detector"


def ticks_with(f_of_block, steps=ALL_STEPS, valid=lambda k: True):
    return protocol_ticks(PROTOCOL, {5: 0.5, **f_of_block}, steps=steps, valid=valid)


def free_play(f_values):
    return [tick(k, block=5, label="FREE", f=f) for k, f in enumerate(f_values)]


@pytest.fixture
def statistics(tmp_path, raw_config):
    """Statistics of a session folder built from ticks and events."""
    raw_config["protocol"] = PROTOCOL

    def build(ticks, events=(), reason="user", duration=582.0):
        folder = tmp_path / "sessions" / "20261110-150412_S01"
        session = finished_session_folder(folder, raw_config, ticks, events, reason, duration)
        return statistics_of(load_session_folder(session))

    return build


class TestHeader:
    def test_duration_and_reason(self, statistics):
        result = statistics(ticks_with({b: 0.2 for b in range(5)}), reason="end_of_stream")
        assert result.duration_s == 582.0
        assert result.reason == "end_of_stream"

    @pytest.mark.parametrize(
        "events",
        [[], [event("SESSION_START", 100.0)], [event("SESSION_END", 130.0, reason="user")]],
        ids=["no events", "no end", "no start"],
    )
    def test_record_that_was_cut_off_has_no_duration(self, tmp_path, raw_config, events):
        folder = tmp_path / "20261110-150412_S01"
        written = write_session_folder(folder, raw_config, [], events)
        result = statistics_of(load_session_folder(written))
        assert result.duration_s == 0.0

    def test_record_without_its_end_is_an_error(self, tmp_path, raw_config):
        folder = tmp_path / "20261110-150412_S01"
        written = write_session_folder(folder, raw_config, [], [event("SESSION_START", 100.0)])
        result = statistics_of(load_session_folder(written))
        assert (result.reason, result.verdict, result.detail) == (
            "error",
            "ERROR",
            "the session did not finish",
        )

    def test_usable_share_is_valid_steps_over_all_steps(self, statistics):
        result = statistics(
            ticks_with({b: 0.2 for b in range(5)}, steps=40, valid=lambda k: k % 4 != 0)
        )
        assert result.usable_share == 30 / 40

    def test_session_without_steps(self, statistics):
        result = statistics([], [])
        assert (result.usable_share, result.focus) == (None, None)


class TestVerdict:
    def test_ok_when_a_step_was_clean_and_not_ended_by_an_error(self, statistics):
        result = statistics(ticks_with({b: 0.2 for b in range(5)}))
        assert (result.verdict, result.detail) == ("OK", None)

    def test_one_clean_step_is_enough(self, statistics):
        ticks = ticks_with({b: 0.3 for b in range(6)}, steps=33, valid=lambda k: k == 20)
        assert statistics(ticks).verdict == "OK"

    def test_not_one_clean_step_means_no_clean_signal(self, statistics):
        ticks = ticks_with({b: None for b in range(6)}, steps=33, valid=lambda k: False)
        result = statistics(ticks, [], reason="end_of_stream")
        assert (result.verdict, result.detail) == ("NO CLEAN SIGNAL", NO_CLEAN_SIGNAL)
        assert result.focus is None and result.usable_share == 0.0

    def test_session_without_steps_says_it_ended_too_early(self, statistics):
        result = statistics([], [])
        assert (result.verdict, result.detail, result.focus) == (
            "NO CLEAN SIGNAL", "ended before the signal of the first step had arrived", None
        )  # fmt: skip

    def test_error_keeps_the_focus_statistics(self, statistics):
        error = event("ERROR", 150.0, where="stream", message="no samples for 5 s")
        result = statistics(
            ticks_with({b: 0.2 for b in range(5)}, steps=60), [error], reason="error"
        )
        assert (result.verdict, result.detail) == ("ERROR", "no samples for 5 s")
        assert result.focus is not None

    def test_session_that_ended_with_reason_error_but_logged_none(self, statistics):
        result = statistics([], [], reason="error")
        assert (result.verdict, result.detail) == ("ERROR", "the session did not finish")


class TestFocus:
    def test_switching_on_is_counted_between_successive_valid_steps(self, statistics):
        focus = statistics(free_play([0.2, 0.6, 0.7, 0.4, 0.5, 0.1, 0.49, 0.9])).focus
        assert focus.switched_on == 3  # at 0.6, 0.5 and 0.9
        assert focus.on_share == 4 / 8

    def test_f_of_exactly_one_half_is_on(self, statistics):
        focus = statistics(free_play([0.49, 0.5])).focus
        assert (focus.switched_on, focus.on_share) == (1, 0.5)

    def test_starting_on_is_not_a_switch(self, statistics):
        assert statistics(free_play([0.9, 0.9, 0.9])).focus.switched_on == 0

    def test_invalid_steps_count_nowhere(self, statistics):
        ticks = [
            tick(0, block=5, label="FREE", f=0.2),
            tick(1, block=5, label="FREE", f=0.2, valid=False),
            tick(2, block=5, label="FREE", f=0.9),
            tick(3, block=5, label="FREE", f=0.9, valid=False),
        ]
        focus = statistics(ticks).focus
        assert (focus.switched_on, focus.on_share) == (1, 0.5)
        assert (focus.f_min, focus.f_max, focus.f_mean) == (0.2, 0.9, pytest.approx(0.55))

    def test_range_of_f(self, statistics):
        focus = statistics(free_play([0.25, 0.75, 0.5])).focus
        assert (focus.f_min, focus.f_max, focus.f_mean) == (0.25, 0.75, 0.5)


class TestCuedBlocks:
    """Every REST and FOCUS block counts; a block whose steps had no f yet decides nothing."""

    def test_focus_block_is_on_when_its_mean_f_is_at_least_one_half(self, statistics):
        focus = statistics(ticks_with({0: None, 1: None, 2: 0.2, 3: 0.4, 4: 0.6})).focus
        assert (focus.focus_blocks_on, focus.focus_blocks) == (0, 2)
        assert (focus.rest_blocks_off, focus.rest_blocks) == (1, 3)

    def test_mean_of_exactly_one_half_is_on(self, statistics):
        focus = statistics(ticks_with({0: None, 1: None, 2: 0.5, 3: 0.5, 4: 0.5})).focus
        assert (focus.focus_blocks_on, focus.rest_blocks_off) == (1, 0)

    def test_the_first_blocks_count_like_the_others(self, statistics):
        focus = statistics(ticks_with({0: 0.9, 1: 0.1, 2: 0.2, 3: 0.9, 4: 0.2})).focus
        assert (focus.focus_blocks_on, focus.focus_blocks) == (1, 2)
        assert (focus.rest_blocks_off, focus.rest_blocks) == (2, 3)

    def test_block_that_did_not_run_to_its_end_is_not_counted(self, statistics):
        into_block_4 = (4 * BLOCK - CONTEXT) // H + 1 + 5
        focus = statistics(
            ticks_with({0: None, 1: None, 2: 0.2, 3: 0.9, 4: 0.2}, steps=into_block_4)
        ).focus
        assert (focus.focus_blocks_on, focus.focus_blocks) == (1, 2)
        assert (focus.rest_blocks_off, focus.rest_blocks) == (1, 2)

    def test_block_counts_from_the_step_that_ends_with_it(self, statistics):
        last_of_block_3 = (4 * BLOCK - CONTEXT) // H
        one_short = statistics(
            ticks_with({0: None, 1: None, 2: 0.2, 3: 0.9, 4: 0.2}, steps=last_of_block_3)
        ).focus
        complete = statistics(
            ticks_with({0: None, 1: None, 2: 0.2, 3: 0.9, 4: 0.2}, steps=last_of_block_3 + 1)
        ).focus
        assert (one_short.focus_blocks, complete.focus_blocks) == (1, 2)

    def test_only_valid_full_steps_decide(self, statistics):
        first = (
            3 * BLOCK - CONTEXT
        ) // H + 1  # first step of block 3; its window is still in block 2
        ticks = ticks_with(
            {0: None, 1: None, 2: 0.2, 3: lambda k: 0.0 if k - first < 3 else 0.9, 4: 0.2}
        )
        assert statistics(ticks).focus.focus_blocks_on == 1


class TestFoundSteps:
    """The share of the steps of each state's blocks in which the detector found that state;
    a block cut short counts with the steps it got to."""

    def test_each_state_is_the_share_of_its_steps_that_found_it(self, statistics):
        focus = statistics(ticks_with({0: 0.2, 1: 0.9, 2: 0.2, 3: 0.1, 4: 0.2})).focus
        assert (focus.focus_found, focus.rest_found) == (0.5, 1.0)  # the focus blocks are as long

    def test_exactly_the_threshold_is_focus_found(self, statistics):
        focus = statistics(ticks_with({b: 0.5 for b in range(5)})).focus
        assert (focus.focus_found, focus.rest_found) == (1.0, 0.0)

    def test_a_block_cut_short_counts_with_the_steps_it_got_to(self, statistics):
        into_block_4 = (4 * BLOCK - CONTEXT) // H + 1 + 5
        ticks = ticks_with({0: 0.2, 1: 0.9, 2: 0.2, 3: 0.9, 4: 0.9}, steps=into_block_4)
        focus = statistics(ticks).focus
        assert focus.rest_blocks == 2 and 0 < focus.rest_found < 1  # its five steps found focus

    def test_without_a_block_of_a_state_its_share_is_unknown(self, statistics):
        focus = statistics(free_play([0.2, 0.9])).focus
        assert (focus.focus_found, focus.rest_found) == (None, None)
