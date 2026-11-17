"""console.session_statistics: the facts the console shows after a session."""

import dataclasses

import pytest

from adamus.console.session_statistics import (
    statistics_lines,
    statistics_of,
)
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
from tests.support.statistics_examples import FOCUS, example

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
        return statistics_of(session)

    return build


class TestHeader:
    def test_name_duration_and_reason(self, statistics):
        result = statistics(ticks_with({b: 0.2 for b in range(5)}), reason="end_of_stream")
        assert result.name == "S01"  # the part of the folder name after the time
        assert result.duration_s == 582.0
        assert result.reason == "end_of_stream"
        assert result.directory.name == "20261110-150412_S01"

    def test_name_may_itself_contain_underscores(self, tmp_path, raw_config):
        folder = tmp_path / "20261110-150412_pilot_run_2"
        events = [event("SESSION_START", 100.0), event("SESSION_END", 101.0, reason="user")]
        result = statistics_of(write_session_folder(folder, raw_config, [], events))
        assert result.name == "pilot_run_2"

    @pytest.mark.parametrize(
        "events",
        [[], [event("SESSION_START", 100.0)], [event("SESSION_END", 130.0, reason="user")]],
        ids=["no events", "no end", "no start"],
    )
    def test_record_that_was_cut_off_has_no_duration(self, tmp_path, raw_config, events):
        folder = tmp_path / "20261110-150412_S01"
        result = statistics_of(write_session_folder(folder, raw_config, [], events))
        assert result.duration_s == 0.0

    def test_record_without_its_end_is_an_error(self, tmp_path, raw_config):
        folder = tmp_path / "20261110-150412_S01"
        result = statistics_of(
            write_session_folder(folder, raw_config, [], [event("SESSION_START", 100.0)])
        )
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

    def test_latency_is_the_time_from_sample_to_sent_weights(self, statistics):
        focus = statistics(ticks_with({b: 0.2 for b in range(5)})).focus
        assert focus.latency_median_s == pytest.approx(0.030)
        assert focus.latency_high_s == pytest.approx(0.030)


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


class TestLines:
    @pytest.mark.parametrize(
        ("reason", "text"),
        [("user", "[SPACE]"), ("end_of_stream", "end of file"), ("error", "error")],
    )  # fmt: skip
    def test_reason_and_duration(self, reason, text):
        lines = dict(statistics_lines(example(reason=reason, duration_s=61.4)))
        assert lines["Duration"] == f"01:01   (ended by {text})"

    def test_no_clean_signal_shows_no_focus_and_no_latency(self):
        failed = example(
            verdict="NO CLEAN SIGNAL",
            detail=NO_CLEAN_SIGNAL,
            usable_share=0.43,
            focus=None,
        )
        lines = statistics_lines(failed)
        assert [label for label, _ in lines] == ["Duration", "Result", "Folder"]
        assert lines[1][1] == (
            f"NO CLEAN SIGNAL ({NO_CLEAN_SIGNAL}) · signal usable 43 % of the time"
        )

    def test_error_without_any_step(self):
        error = example(verdict="ERROR", detail="no samples for 5 s", usable_share=None, focus=None)
        assert dict(statistics_lines(error))["Result"] == "ERROR (no samples for 5 s)"

    def test_blocks_line_is_left_out_when_no_evaluation_block_finished(self):
        focus = dataclasses.replace(FOCUS, focus_blocks=0, rest_blocks=0, switched_on=1)
        lines = statistics_lines(example(focus=focus))
        assert [label for label, _ in lines] == [
            "Duration",
            "Result",
            "Focus",
            "Focus range",
            "Latency",
            "Folder",
        ]
        assert lines[2][1] == "switched on 1 time · on 38 % of the time"

    @pytest.mark.parametrize(
        ("changes", "text"),
        [({"focus_blocks": 0, "focus_blocks_on": 0},
          "FOCUS blocks 0 of 0 on · REST blocks 3 of 3 off"),
         ({"rest_blocks": 0, "rest_blocks_off": 0},
          "FOCUS blocks 2 of 3 on · REST blocks 0 of 0 off")],
    )  # fmt: skip
    def test_blocks_line_is_shown_when_blocks_of_one_kind_finished(self, changes, text):
        focus = dataclasses.replace(FOCUS, **changes)
        assert statistics_lines(example(focus=focus))[3] == ("", text)

    def test_shares_and_times_are_rounded_to_whole_percent_and_milliseconds(self):
        focus = dataclasses.replace(
            FOCUS, on_share=0.996, latency_median_s=0.9996, latency_high_s=2.6
        )
        lines = dict(statistics_lines(example(focus=focus, usable_share=0.996)))
        assert lines["Result"] == "OK · signal usable 100 % of the time"
        assert lines["Focus"] == "switched on 7 times · on 100 % of the time"
        assert lines["Latency"] == "median 1000 ms, 95th percentile 2600 ms"

    @pytest.mark.parametrize("missing", ["latency_median_s", "latency_high_s"])
    def test_latency_line_needs_both_of_its_numbers(self, missing):
        focus = dataclasses.replace(FOCUS, **{missing: None})
        assert "Latency" not in dict(statistics_lines(example(focus=focus)))

    def test_folder_outside_the_working_directory_is_shown_in_full(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        elsewhere = tmp_path.parent / "other" / "20261110-150412_S01"
        assert dict(statistics_lines(example(directory=elsewhere)))["Folder"] == str(elsewhere)
