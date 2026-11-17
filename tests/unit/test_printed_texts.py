"""console.printed_texts: the exact lines, columns and colours of what the console prints."""

import io
import re
from pathlib import Path

import pytest
from rich.console import Console

from adamus import constants as c
from adamus.console import printed_texts
from adamus.console.data_sources import Source
from adamus.console.preflight_checks import HOW_TO_FIX, Check
from adamus.errors import AdamusError
from tests.support.statistics_examples import example

GREEN, RED, YELLOW, GREY = "\x1b[32m", "\x1b[31m", "\x1b[33m", "\x1b[90m"
BOLD_RED, REVERSE, PLAIN = "\x1b[1;31m", "\x1b[7m", "\x1b[0m"


def in_colour(colour, text):
    """A pattern for ``text`` shown in ``colour``; the colour may run on to the end of a column."""
    return re.escape(colour + text) + " *" + re.escape(PLAIN) + " *"


def lines(shown, colours=False, width=120):
    """What a terminal shows, line by line; with ``colours`` the colour codes stay in.

    ``shown`` is something to print, or a function that prints to the output it is given.
    """
    out = io.StringIO()
    console = Console(
        file=out,
        width=width,
        highlight=False,
        force_terminal=colours,
        color_system="standard" if colours else None,
    )
    if callable(shown):
        shown(console)
    else:
        console.print(shown)
    return [line.rstrip() for line in out.getvalue().splitlines()]


class TestPrompts:
    def test_welcome_names_the_configuration_file(self):
        shown = lines(lambda out: printed_texts.welcome(out, Path("config.yaml")))
        assert shown == ["AdaMus · config: config.yaml", ""]

    def test_all_ready_is_green_between_two_empty_lines(self):
        assert lines(printed_texts.all_ready) == ["", "All systems ready.", ""]
        assert (
            lines(printed_texts.all_ready, colours=True)[1] == f"{GREEN}All systems ready.{PLAIN}"
        )

    def test_questions(self):
        assert lines(printed_texts.new_session_question) == [
            "",
            "Start a new session?    [Y] yes   [N] quit program",
        ]
        assert lines(printed_texts.session_name_question) == [
            "",
            "Session name (type it, then [ENTER]):",
        ]
        assert lines(printed_texts.invalid_name) == ["Use letters, digits, '_' and '-' only."]
        assert YELLOW in lines(printed_texts.invalid_name, colours=True)[0]

    def test_ready_names_the_session_and_its_source(self):
        shown = lines(
            lambda out: printed_texts.ready(out, "S01", Source("headset", "Muse 2"), None, [])
        )
        assert shown == [
            "",
            "Session S01 · source: Muse 2 (headset)",
            "READY       [SPACE] start   [Q] cancel",
        ]

    def test_ready_says_in_yellow_which_electrodes_are_missing(self):
        dataset = Source("dataset", "tests", Path("data/tests"))

        def ready(out):
            printed_texts.ready(out, "S01", dataset, dataset.dataset / "one", ["TP9", "TP10"])

        shown = lines(ready)
        assert shown[2] == "No TP9, TP10: recorded as zeros, the detector uses the others."
        assert shown[3] == "READY       [SPACE] start   [Q] cancel"
        assert YELLOW in lines(ready, colours=True)[2]

    def test_source_without_any_electrode_of_the_detector_is_refused_in_red(self):
        dataset = Source("dataset", "tests", Path("data/tests"))

        def refused(out):
            printed_texts.lacks_electrodes(out, dataset, dataset.dataset / "one")

        assert lines(refused) == [
            "",
            "tests / one has none of AF7, AF8, TP9, TP10. The detector needs at least one.",
            "",
        ]
        assert RED in lines(refused, colours=True)[1]
        headset = lines(
            lambda out: printed_texts.lacks_electrodes(out, Source("headset", "X"), None)
        )
        assert headset[1] == "X has none of AF7, AF8, TP9, TP10. The detector needs at least one."

    def test_dataset_and_recording_are_named_as_their_folders_are(self):
        dataset = Source("dataset", "[bold]tests[/bold]", Path("data/[bold]tests[/bold]"))
        recording = dataset.dataset / "take 1"
        shown = lines(lambda out: printed_texts.ready(out, "S01", dataset, recording, ["[b]AF7"]))
        assert shown[1] == "Session S01 · source: [bold]tests[/bold] / take 1 (dataset)"

    def test_running_and_cancelled(self):
        assert lines(printed_texts.running) == ["RUNNING     [SPACE] end session"]
        assert lines(printed_texts.cancelled) == ["Session cancelled.", ""]

    def test_session_that_could_not_start(self):
        problem = OSError("bluetooth is off")
        shown = lines(lambda out: printed_texts.could_not_start(out, problem), colours=True)
        assert shown == ["", f"{RED}The session could not start: bluetooth is off{PLAIN}", ""]


class TestSessionFolderNotRead:
    @pytest.mark.parametrize(
        ("what", "heading"),
        [(printed_texts.NO_METRICS, "The metrics of the session could not be computed"),
         (printed_texts.NO_STATISTICS, "The statistics of the session could not be shown")],
    )  # fmt: skip
    def test_says_what_could_not_be_done_and_why_in_red(self, what, heading):
        problem = AdamusError("folder X is not a complete session folder: no meta.json")
        shown = lines(lambda out: printed_texts.not_read(out, what, problem), colours=True)
        assert shown == ["", f"{RED}{heading}: {problem}{PLAIN}", ""]


class TestChecklist:
    CHECKS = [
        Check("Configuration", c.CHECK_PASSED, "config.yaml, 9 protocol blocks"),
        Check("Audio output", c.CHECK_FAILED, "USB, 44100 Hz (need 48000 Hz)"),
        Check("Generator start", c.CHECK_SKIPPED, "skipped"),
        Check("Live headset", c.CHECK_NONE_FOUND, "none found"),
        Check("Data source available"),
    ]

    def test_columns_are_those_of_the_documented_example(self):
        assert lines(printed_texts.checklist(self.CHECKS)) == [
            "Preflight",
            "  ✓ Configuration          config.yaml, 9 protocol blocks",
            "  ✗ Audio output           USB, 44100 Hz (need 48000 Hz)",
            "  ✗ Generator start        skipped",
            "  – Live headset           none found",
            "    Data source available",
        ]

    def test_on_a_narrow_terminal_only_the_last_column_wraps_and_no_word_is_lost(self):
        check = Check("Data source available")
        check.status, check.detail = c.CHECK_PASSED, "one two three four five six seven eight"
        assert lines(printed_texts.checklist([check]), width=40)[1:] == [
            "  ✓ Data source available  one two three",
            "                           four five six",
            "                           seven eight",
        ]

    def test_colours_tell_the_state_of_a_check(self):
        _, passed, failed, skipped, none_found, pending = lines(
            printed_texts.checklist(self.CHECKS), colours=True
        )
        # Passed: green mark and name, the value found in the normal colour.
        assert re.fullmatch(
            "  "
            + in_colour(GREEN, "✓")
            + in_colour(GREEN, "Configuration")
            + re.escape("config.yaml, 9 protocol blocks"),
            passed,
        )
        for row, colour, mark, name, detail in (
            (failed, RED, "✗", "Audio output", "USB, 44100 Hz (need 48000 Hz)"),
            (skipped, RED, "✗", "Generator start", "skipped"),
            (none_found, YELLOW, "–", "Live headset", "none found"),
            (pending, GREY, "", "Data source available", ""),
        ):
            pattern = in_colour(colour, mark) + in_colour(colour, name) + in_colour(colour, detail)
            assert re.fullmatch("  " + pattern, row), row

    def test_running_check_has_a_spinner_in_front_of_its_name(self):
        running = [Check("Generator start", c.CHECK_RUNNING, "loading model ...")]
        (row,) = lines(printed_texts.checklist(running))[1:]
        assert re.fullmatch(r"  [⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏] Generator start        loading model \.\.\.", row)
        (coloured,) = lines(printed_texts.checklist(running), colours=True)[1:]
        assert RED not in coloured and GREY not in coloured  # the normal colour

    def test_checks_after_a_running_one_are_still_listed(self):
        checks = [Check("Storage", c.CHECK_RUNNING), Check("Audio output"), Check("Live headset")]
        assert len(lines(printed_texts.checklist(checks))) == 4


class TestErrors:
    def test_one_paragraph_per_failed_check(self):
        failed = [
            Check("Audio output", c.CHECK_FAILED, "USB, 44100 Hz (need 48000 Hz)"),
            Check("Storage", c.CHECK_FAILED, "[Errno 13] Permission denied: 'sessions'"),
        ]
        assert lines(lambda out: printed_texts.errors(out, failed, HOW_TO_FIX)) == [
            "",
            "Errors",
            "",
            "Audio output",
            "  USB, 44100 Hz (need 48000 Hz)",
            "  Set the output device to 48000 Hz in Audio MIDI Setup.",
            "",
            "Storage",
            "  [Errno 13] Permission denied: 'sessions'",
            "  Start the program in a folder you can write to.",
        ]

    def test_every_line_of_a_problem_with_several_lines_is_indented(self):
        detail = str(AdamusError("invalid configuration:", "  poll_s: must be positive"))
        failed = [Check("Configuration", c.CHECK_FAILED, detail)]
        assert lines(lambda out: printed_texts.errors(out, failed, {}))[3:] == [
            "Configuration",
            "  invalid configuration:",
            "    poll_s: must be positive",
        ]

    def test_heading_and_names_are_red(self):
        failed = [Check("Storage", c.CHECK_FAILED, "full")]
        shown = lines(lambda out: printed_texts.errors(out, failed, HOW_TO_FIX), True)
        assert shown[1] == f"{BOLD_RED}Errors{PLAIN}"
        assert shown[3] == f"{RED}Storage{PLAIN}"
        assert shown[4] == "  full"

    def test_check_without_advice_gets_none(self):
        failed = [Check("Unknown", c.CHECK_FAILED, "why")]
        shown = lines(lambda out: printed_texts.errors(out, failed, {}))
        assert shown[3:] == ["Unknown", "  why"]

    def test_advice_for_the_checks_that_do_not_explain_themselves(self):
        assert HOW_TO_FIX == {
            "Configuration": (
                "Fix the configuration file (config.yaml in the repository lists every key)."
            ),
            "Storage": "Start the program in a folder you can write to.",
            "Focus detector": (
                "Read the log named above; the first training downloads its data from"
                " physionet.org, so it needs the internet."
            ),
            "Audio output": "Set the output device to 48000 Hz in Audio MIDI Setup.",
            "Generator files": (
                "Read the installation log named above; the installation needs Xcode with its"
                " Metal toolchain, cmake, git and curl. Check generator.folder and model."
            ),
            "Data source available": ("Switch the headset on, or put a dataset into data/."),
        }
        assert str(c.AUDIO_RATE) in HOW_TO_FIX["Audio output"]


class TestSourceList:
    SOURCES = [
        Source("headset", "Muse 2"),
        Source("dataset", "wearable_monitoring_n_back"),
        Source("dataset", "synthetic"),
    ]

    def test_columns_are_those_of_the_documented_example(self):
        assert lines(printed_texts.source_list(self.SOURCES, 0)) == [
            "Select data source      [↑] [↓] move   [ENTER] select   [R] rescan headset",
            "> Muse 2                      headset",
            "  wearable_monitoring_n_back  dataset",
            "  synthetic                   dataset",
        ]

    def test_recordings_of_a_dataset_are_listed_under_its_name(self):
        folder = Path("data/wearable_monitoring_n_back")
        recordings = [folder / "Experiment_1_first", folder / "Experiment_1_second"]
        shown = lines(printed_texts.recording_list(recordings, 1))
        assert shown == [
            "Select recording        [↑] [↓] move   [ENTER] select   [Q] back",
            "  Experiment_1_first",
            "> Experiment_1_second",
        ]
        assert lines(printed_texts.recording_list([], 0))[1:] == ["  nothing available"]

    @pytest.mark.parametrize("selected", [0, 1, 2])
    def test_selected_row_has_the_marker_and_is_shown_inverted(self, selected):
        rows = lines(printed_texts.source_list(self.SOURCES, selected), colours=True)[1:]
        for index, row in enumerate(rows):
            assert row.startswith(">") == (index == selected)
            assert (REVERSE in row) == (index == selected)
        label = self.SOURCES[selected].label
        assert f"{REVERSE}{label}" in rows[selected]

    @pytest.mark.parametrize(
        ("selected", "first", "last"),
        [(0, 0, 9), (9, 0, 9), (10, 1, 10), (12, 3, 12), (24, 15, 24)],
    )
    def test_at_most_ten_rows_are_shown_and_the_list_scrolls(self, selected, first, last):
        sources = [Source("dataset", f"rec_{index:02d}.csv") for index in range(25)]
        rows = lines(printed_texts.source_list(sources, selected))[1:]
        assert len(rows) == printed_texts.LIST_ROWS == 10
        assert (rows[0].split()[-2], rows[-1].split()[-2]) == (
            f"rec_{first:02d}.csv",
            f"rec_{last:02d}.csv",
        )
        (marked,) = [row for row in rows if row.startswith(">")]
        assert f"rec_{selected:02d}.csv" in marked

    def test_long_name_keeps_a_space_before_the_kind(self):
        name = "a_recording_with_a_very_long_name_s07.csv"
        (row,) = lines(printed_texts.source_list([Source("dataset", name)], 0))[1:]
        assert row == f"> {name} dataset"

    def test_empty_list_says_so_in_yellow(self):
        assert lines(printed_texts.source_list([], 0))[1:] == ["  nothing available"]
        (row,) = lines(printed_texts.source_list([], 0), colours=True)[1:]
        assert re.fullmatch("  " + in_colour(YELLOW, "nothing available"), row)


class TestStatistics:
    def test_columns_are_those_of_the_documented_example(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        shown = lines(lambda out: printed_texts.statistics(out, example()))
        assert shown == [
            "",
            "Session S01 finished",
            "  Duration        09:42   (ended by [SPACE])",
            "  Result          OK · signal usable 91 % of the time",
            "  Focus           switched on 7 times · on 38 % of the time",
            "                  FOCUS blocks 2 of 3 on · REST blocks 3 of 3 off",
            "  Focus range     f 0.00 to 0.87, mean 0.34",
            "  Latency         median 31 ms, 95th percentile 58 ms",
            "  Folder          sessions/20261110-150412_S01",
            "",
        ]

    def test_a_session_with_the_headset_says_where_its_recording_was_stored(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        stored = example(recording=Path("data/own_recordings/20261110-150412_S01"))
        shown = lines(lambda out: printed_texts.statistics(out, stored))
        assert shown[-3:] == [
            "  Folder          sessions/20261110-150412_S01",
            "  Recording       data/own_recordings/20261110-150412_S01",
            "",
        ]

    def test_ok_is_green_and_only_the_verdict_is_coloured(self):
        shown = lines(lambda out: printed_texts.statistics(out, example()), colours=True)
        assert shown[3] == f"  Result          {GREEN}OK{PLAIN} · signal usable 91 % of the time"
        assert [line for line in shown if "\x1b[" in line] == [shown[3]]

    @pytest.mark.parametrize(
        ("verdict", "detail"),
        [
            ("ERROR", "no samples for 5 s"),
            ("NO CLEAN SIGNAL", "not one clean step for the detector"),
        ],
    )
    def test_other_verdicts_are_red_with_their_reason(self, verdict, detail):
        failed = example(verdict=verdict, detail=detail, focus=None)
        shown = lines(lambda out: printed_texts.statistics(out, failed), colours=True)
        assert shown[3] == (
            f"  Result          {RED}{verdict} ({detail}){PLAIN} · signal usable 91 % of the time"
        )

    def test_verdict_without_any_step_is_coloured_whole(self):
        empty = example(
            verdict="ERROR", detail="the session did not finish",
            usable_share=None, focus=None,
        )  # fmt: skip
        shown = lines(lambda out: printed_texts.statistics(out, empty), colours=True)
        assert shown[3] == f"  Result          {RED}ERROR (the session did not finish){PLAIN}"
