"""console.console_app.ConsoleApp: the whole dialogue, with real sessions over stand-in devices."""

import json
import re

import numpy as np
import pytest
from brainflow.data_filter import DataFilter

from adamus import constants as c
from adamus import own_recordings
from adamus.config import load_config, read_yaml
from adamus.console import console_app, keyboard
from adamus.errors import AdamusError
from adamus.session_files import REASON_END_OF_STREAM, REASON_USER
from tests.support.console_runs import KEY_WAIT_S, LIVE_PROTOCOL
from tests.support.finished_sessions import read_lines, synthetic_signals
from tests.support.synthetic_dataset import write_recording
from tests.support.waiting import wait_until

ENTER, SPACE, UP, DOWN, BACKSPACE = (
    keyboard.ENTER, keyboard.SPACE, keyboard.UP, keyboard.DOWN, keyboard.BACKSPACE,
)  # fmt: skip
NAME_S01 = ("S", "0", "1", ENTER)
SHORT = [["REST", 10], ["FOCUS", 10], ["REST", 10], ["FOCUS", 10]]
NO_CLEAN_SIGNAL = "NO CLEAN SIGNAL (not one clean step for the detector)"


RECORDING = (ENTER, ENTER)  # with the headset off: the first dataset, then its first recording


def lists_after(heading, text, kinds):
    """Every list printed under a heading, as rows of (marker, name, kind).

    Written to a file, a list does not end its last line: the next heading follows it at once.
    """
    lists = []
    for block in text.split(heading)[1:]:
        rows = []
        for line in block.split("Select ")[0].splitlines()[1:]:
            match = re.fullmatch(rf"(.{{2}})(.{{28}})({kinds}) *", line)
            if match is None:
                break
            assert match.group(1) in ("> ", "  ")
            rows.append((match.group(1).strip(), match.group(2).strip(), match.group(3)))
        lists.append(rows)
    return lists


def source_lists(text):
    return lists_after("Select data source", text, "headset|dataset")


def recording_lists(text):
    return [
        [(mark, name) for mark, name, _ in rows]
        for rows in lists_after("Select recording", text, "")
    ]


@pytest.mark.parametrize("key", [None, "while running", "once streaming"])
def test_only_a_session_that_gets_a_key_while_it_runs_takes_real_time(console, key):
    """Every poll then waits a millisecond, so that the key comes while the session runs;
    the other sessions run as fast as they can."""
    if key == "once streaming":
        console.streaming(SPACE)
    console.play_recording(SHORT, while_running=(SPACE,) if key == "while running" else ())
    assert console.machine.paced == (key is not None)


class TestOneSession:
    def test_recording_played_to_its_end_shows_the_statistics(self, console):
        console.machine.headsets_on = set()
        console.add_recording("short", LIVE_PROTOCOL)
        after_session = console.shown("finished")
        code = console.start(*RECORDING, "y", *NAME_S01, SPACE, after_session(ENTER), ENTER, "n")
        assert code == 0
        text = console.text
        assert "Select data source      [↑] [↓] move   [ENTER] select   [R] rescan headset" in text
        assert "Select recording        [↑] [↓] move   [ENTER] select   [Q] back" in text
        assert "Start a new session?    [Y] yes   [N] quit program" in text
        assert "Session name (type it, then [ENTER]): S01" in text
        assert (
            "Session S01 · source: simulated / short (dataset)\n"
            "READY       [SPACE] start   [Q] cancel" in text
        )
        assert "RUNNING     [SPACE] end session" in text
        (folder,) = console.session_folders()
        statistics = text[text.index("Session S01 finished") :]
        assert re.search(r"\n  Duration        02:09   \(ended by end of file\)", statistics)
        assert "\n  Result          OK · signal usable 100 % of the time" in statistics
        assert re.search(
            r"\n  Focus           switched on \d+ times? · on \d+ % of the time", statistics
        )
        assert re.search(
            "\n" + 18 * " " + r"FOCUS blocks \d of 2 on · REST blocks \d of 2 off", statistics
        )
        assert re.search(
            r"\n  Focus range     f [01]\.\d\d to [01]\.\d\d, mean [01]\.\d\d *\n", statistics
        )
        assert "\n  Latency         median" in statistics
        assert f"\n  Folder          sessions/{folder.name}" in statistics

    def test_nothing_is_printed_while_the_session_runs(self, console):
        console.play_recording(SHORT)
        running = console.text.index("RUNNING     [SPACE] end session\n")
        between = console.text[running:].split("Session S01 finished")[0]
        assert between == "RUNNING     [SPACE] end session\n\n"

    def test_session_is_a_complete_session_folder(self, console):
        console.play_recording(SHORT)
        (folder,) = console.session_folders()
        assert folder.name.endswith("_S01")
        events = read_lines(folder / c.FILE_EVENTS)
        assert events[-1]["reason"] == REASON_END_OF_STREAM
        assert len(read_lines(folder / c.FILE_TICKS)) == (40 * 256 - 2 * 512) // 128 + 1
        assert console.machine.sources[0].closed

    def test_a_live_session_without_a_single_sample_shows_why_and_goes_on(self, console):
        """The headset was found by the preflight but sends nothing: the session ends with
        its error, the statistics say so and the operator can start the next one."""
        console.machine.live_samples = 0
        code = console.start(
            ENTER, "y", *NAME_S01, SPACE, console.shown("finished")(ENTER), "n",
        )  # fmt: skip
        assert code == 0
        assert "ERROR (no samples for 5 s)" in " ".join(console.text.split())
        (folder,) = console.session_folders()
        metrics = json.loads((folder / c.FILE_METRICS).read_text())
        assert metrics["record_completeness"] == {
            "ticks": 0, "filtered_records": 0, "expected": 0, "passed": True,
        }  # fmt: skip
        assert metrics["sampling_rate_hz"] is None
        assert not (console.folder / "data" / "own_recordings").exists()  # nothing to keep

    def test_space_ends_a_running_live_session(self, console):
        """The headset delivers one chunk and then waits, so [SPACE] surely comes first."""
        console.machine.live_samples = 64
        console.raw["timeouts"]["stream_s"] = 600  # of fake time: about 15 s of waiting
        code = console.start(
            ENTER, "y", *NAME_S01, SPACE, console.streaming(SPACE),
            console.shown("finished")(ENTER), "n",
        )  # fmt: skip
        assert code == 0
        assert "Session S01 · source: Muse 2 (headset)" in console.text
        assert "(ended by [SPACE])" in console.text
        assert "NO CLEAN SIGNAL (ended before the signal of the first step had arrived)" in (
            console.text
        )
        (folder,) = console.session_folders()
        assert read_lines(folder / c.FILE_EVENTS)[-1]["reason"] == REASON_USER
        assert json.loads((folder / c.FILE_META).read_text())["mode"] == "live"

    def test_recording_without_a_clean_step_shows_why_and_returns_to_the_sources(self, console):
        console.machine.headsets_on = set()
        signals = synthetic_signals(SHORT)
        signals[0] += 1000.0 * np.sign(np.sin(2 * np.pi * 3 * np.arange(signals.shape[1]) / 256))
        blocks = [tuple(block) for block in SHORT]
        write_recording(console.folder / "data" / "simulated" / "loud", signals, blocks)
        after_session = console.shown("finished")
        code = console.start(*RECORDING, "y", *NAME_S01, SPACE, after_session(ENTER), ENTER, "n")
        assert code == 0
        statistics = console.text[console.text.index("Session S01 finished") :]
        assert "(ended by end of file)" in statistics
        assert f"{NO_CLEAN_SIGNAL} · signal usable 0 %" in statistics
        assert "Focus" not in statistics and "Latency" not in statistics
        assert console.text.count("Select data source") == 2

    def test_live_headset_that_falls_silent_is_an_error_in_the_statistics(self, console):
        console.raw["protocol"] = LIVE_PROTOCOL  # the headset delivers 124 s, then nothing
        code = console.start(ENTER, "y", *NAME_S01, SPACE, console.shown("finished")(ENTER), "n")
        assert code == 0
        assert "(ended by error)" in console.text
        assert "Result          ERROR (no samples for 5 s)" in console.text
        assert "Focus range" in console.text  # the steps before the error count

    def test_session_that_cannot_start_returns_to_the_sources(self, console):
        console.machine.source_failure = "bluetooth is off"
        code = console.start(ENTER, "y", *NAME_S01, SPACE, ENTER, "n")
        assert code == 0
        assert "The session could not start: bluetooth is off" in console.text
        assert console.text.count("Select data source") == 2
        assert console.session_folders() == []

    @pytest.mark.parametrize("what", ["metrics", "statistics"])
    def test_a_session_folder_that_cannot_be_read_is_reported_and_the_console_goes_on(
        self, console, monkeypatch, what
    ):
        """Neither the metrics nor the statistics of one session may end the program."""
        from adamus.console import console_app

        def unreadable(directory):
            raise AdamusError(f"{directory} is not a complete session folder: broken")

        monkeypatch.setattr(console_app, "analyze" if what == "metrics" else "statistics_of",
                            unreadable)  # fmt: skip
        code = console.play_recording(SHORT, finished="not a complete session folder")
        assert code == 0
        (folder,) = console.session_folders()
        heading = {
            "metrics": "The metrics of the session could not be computed",
            "statistics": "The statistics of the session could not be shown",
        }[what]
        message = f"{heading}: {folder} is not a complete session folder: broken"
        unwrapped = "".join(console.text.split())  # the long path is folded at the screen edge
        assert "".join(message.split()) in unwrapped
        assert console.text.count("Select data source") == 2  # back at the sources
        assert ("Session S01 finished" in console.text) == (what == "metrics")

    def test_space_pressed_while_the_session_was_starting_does_not_end_it(self, console):
        """A second press of [SPACE] at READY must not cost the session."""
        console.play_recording(SHORT, while_running=(SPACE,))
        (folder,) = console.session_folders()
        assert read_lines(folder / c.FILE_EVENTS)[-1]["reason"] == REASON_END_OF_STREAM
        assert "(ended by end of file)" in console.text

    def test_metrics_of_the_session_are_written_next_to_its_record(self, console):
        console.machine.headsets_on = set()
        console.add_recording("long", LIVE_PROTOCOL)
        after_session = console.shown("finished")
        console.start(*RECORDING, "y", *NAME_S01, SPACE, after_session(ENTER), ENTER, "n")
        (folder,) = console.session_folders()
        metrics = json.loads((folder / c.FILE_METRICS).read_text())
        assert metrics["session"] == folder.name
        assert metrics["state_separation_auc"] > 0.9  # synthetic FOCUS is told from REST
        assert metrics["record_completeness"]["passed"] is True


class TestConfigurationOfASession:
    def test_live_session_runs_on_the_headset_of_the_configuration(self, console):
        console.start(ENTER, "y", *NAME_S01, SPACE, console.streaming(SPACE))
        (folder,) = console.session_folders()
        board = load_config(folder / c.FILE_CONFIG).board
        assert (board.name, board.id) == ("Muse 2", 38)
        meta = json.loads((folder / c.FILE_META).read_text())
        assert meta["mode"] == "live"
        assert meta["electrode_rows"] == {"TP9": 1, "AF7": 2, "AF8": 3, "TP10": 4}

    def test_recording_brings_all_its_electrodes_into_the_session(self, console):
        """The raw file holds every one of them; the record says which row is which."""
        console.machine.headsets_on = set()
        folder = console.add_recording("five", LIVE_PROTOCOL)
        table = np.loadtxt(folder / "eeg.csv", delimiter=",", skiprows=1)  # AF7, AF8, TP9, TP10
        more = np.column_stack([table[:, [2, 0, 1, 3]], table[:, 0] + 1])
        header = "TP9,AF7,AF8,TP10,AUX_RIGHT"
        np.savetxt(folder / "eeg.csv", more, fmt="%.3f", delimiter=",", header=header, comments="")
        console.start(*RECORDING, "y", *NAME_S01, SPACE, console.shown("finished")(ENTER))
        (session,) = console.session_folders()
        meta = json.loads((session / c.FILE_META).read_text())
        assert meta["electrode_rows"] == {"TP9": 0, "AF7": 1, "AF8": 2, "TP10": 3, "AUX_RIGHT": 4}
        assert (meta["timestamp_row"], meta["marker_row"]) == (5, 6)
        raw = DataFilter.read_file(str(session / c.FILE_RAW))
        assert raw.shape == (7, 124 * 256) and np.allclose(raw[:5], more.T, atol=0.0005)
        assert "has no" not in console.text
        assert "\n  Result          OK" in console.text

    def test_recording_without_any_electrode_of_the_detector_is_refused_when_chosen(self, console):
        console.machine.headsets_on = set()
        console.add_recording("a_good", SHORT)
        folder = console.add_recording("other_cap", SHORT)
        text = (folder / "eeg.csv").read_text().replace("AF7,AF8,TP9,TP10", "Fp1,Fp2,T7,T8", 1)
        (folder / "eeg.csv").write_text(text)
        assert console.start(ENTER, DOWN, ENTER, ENTER, ENTER, "n") == 0
        assert (
            "simulated / other_cap has none of AF7, AF8, TP9, TP10. "
            "The detector needs at least one." in console.text
        )
        assert "Session name" not in console.text and console.session_folders() == []
        assert console.text.count("Select data source") == 2  # back to the sources

    def test_recording_without_some_electrodes_runs_on_the_others(self, console):
        console.machine.headsets_on = set()
        folder = console.add_recording("partial", SHORT)
        text = (folder / "eeg.csv").read_text().replace("AF7,AF8,TP9", "Fp1,AF8,T7", 1)
        (folder / "eeg.csv").write_text(text)
        after_session = console.shown("finished")
        code = console.start(*RECORDING, "y", *NAME_S01, SPACE, after_session(ENTER), ENTER, "n")
        assert code == 0
        assert "No AF7, TP9: recorded as zeros, the detector uses the others." in console.text
        (session,) = console.session_folders()
        meta = json.loads((session / c.FILE_META).read_text())
        assert meta["detector_electrodes"] == ["AF8", "TP10"]
        assert "Result          OK" in console.text

    def test_recording_that_cannot_be_read_is_reported_and_the_console_goes_on(self, console):
        console.machine.headsets_on = set()
        folder = console.add_recording("broken", SHORT)
        (folder / "eeg.csv").write_text("AF7,AF7\n" + "1,2\n")
        assert console.start(*RECORDING, "y", *NAME_S01) == 0
        assert "The session could not start: " in console.text
        assert "the first line must name every column, and none twice" in console.text
        assert "READY" not in console.text and console.session_folders() == []

    def test_recording_is_played_under_its_own_protocol(self, console):
        console.machine.headsets_on = set()
        console.add_recording("a_long", SHORT)
        console.add_recording("b_short", [["REST", 10], ["FOCUS", 10]])
        after_a, after_b = console.shown("Session A finished"), console.shown("Session B finished")
        console.start(
            ENTER, ENTER, "y", "A", ENTER, SPACE, after_a(ENTER),
            DOWN, ENTER, "y", "B", ENTER, SPACE, after_b(ENTER),
        )  # fmt: skip
        first, second = console.session_folders()
        assert load_config(first / c.FILE_CONFIG).raw["protocol"] == SHORT
        assert load_config(second / c.FILE_CONFIG).raw["protocol"] == [["REST", 10], ["FOCUS", 10]]
        assert "source: simulated / a_long (dataset)" in console.text
        assert "source: simulated / b_short (dataset)" in console.text
        for folder in (first, second):
            meta = json.loads((folder / c.FILE_META).read_text())
            assert meta["mode"] == "playback"


class TestSourceSelection:
    def test_on_a_terminal_the_list_is_drawn_again_for_every_key(self, console):
        """A file gets the list once, as it was left; a terminal must show every move."""
        console.add_recording("one", dataset="a")
        console.start(DOWN, UP, on="terminal")
        drawn = console.text.split("Select data source")[1:]
        assert [">" in part.split("Muse 2")[0][-4:] for part in drawn][:3] == [True, False, True]

    def test_headset_comes_first_then_the_datasets_in_alphabetical_order(self, console):
        for dataset in ("zeta", "alpha"):
            console.add_recording("one", dataset=dataset)
        (console.folder / "data" / "notes.txt").write_text("")
        (console.folder / "data" / "empty_dataset").mkdir()
        console.start()
        assert source_lists(console.text) == [
            [(">", "Muse 2", "headset"), ("", "alpha", "dataset"), ("", "zeta", "dataset")]
        ]

    def test_arrows_move_the_selection_and_stop_at_the_ends(self, console):
        console.add_recording("one", dataset="a")
        console.add_recording("one", dataset="b")
        console.start(UP, UP, DOWN, DOWN, DOWN, DOWN, UP, ENTER, ENTER, "n")
        assert source_lists(console.text)[0] == [
            ("", "Muse 2", "headset"),
            (">", "a", "dataset"),
            ("", "b", "dataset"),
        ]
        assert console.logged("SOURCE_SELECTED")[0]["label"] == "a"

    def test_each_arrow_moves_the_selection_by_one_row(self, console):
        for dataset in ("a", "b", "c", "d"):
            console.add_recording("one", dataset=dataset)
        console.start(DOWN, DOWN, DOWN, DOWN, UP, ENTER, ENTER, "n")  # down to d, up to c
        assert console.logged("SOURCE_SELECTED")[0]["label"] == "c"

    def test_list_is_redrawn_after_every_key(self, console):
        console.add_recording("one", dataset="a")
        console.start(DOWN, ENTER, ENTER, "n")
        assert source_lists(console.text)[0] == [("", "Muse 2", "headset"), (">", "a", "dataset")]

    def test_r_repeats_the_headset_check(self, console):
        console.machine.headsets_on = set()
        console.add_recording("one")
        switch_on = console.shown("All systems ready.")

        def r_after_switching_the_headset_on():
            switch_on("")()
            console.machine.headsets_on = {"Muse 2"}
            return "r"

        console.start(r_after_switching_the_headset_on, ENTER, "n")
        assert console.machine.probed == [38, 38]  # the preflight check and one rescan
        first, again = (entry for entry in console.logged("PREFLIGHT") if "headset" in entry)
        assert (first["check"], first["result"], first["detail"]) == (
            "Live headset",
            "unavailable",
            "none found",
        )
        assert (again["check"], again["result"], again["detail"], again["headset"]) == (
            "Live headset",
            "ok",
            "Muse 2",
            "Muse 2",
        )
        assert again["duration"] >= 0
        assert console.logged("SOURCE_SELECTED")[0] == {
            **console.logged("SOURCE_SELECTED")[0],
            "kind": "headset",
            "label": "Muse 2",
            "recording": None,
        }


class TestRecordingSelection:
    def test_choosing_a_dataset_asks_which_of_its_recordings(self, console):
        console.machine.headsets_on = set()
        for name in ("Subject02", "Subject00", "Subject01"):
            console.add_recording(name)
        console.start(ENTER, DOWN, ENTER, "n")
        assert recording_lists(console.text) == [
            [("", "Subject00"), (">", "Subject01"), ("", "Subject02")]
        ]
        assert console.logged("SOURCE_SELECTED")[0] == {
            **console.logged("SOURCE_SELECTED")[0],
            "kind": "dataset",
            "label": "simulated",
            "recording": "Subject01",
        }

    def test_choosing_the_headset_asks_nothing_more(self, console):
        console.add_recording("one")
        console.start(ENTER, "n")
        assert "Select recording" not in console.text
        assert console.logged("SOURCE_SELECTED")[0]["kind"] == "headset"

    def test_q_goes_back_to_the_list_of_sources(self, console):
        console.add_recording("one")
        console.start(DOWN, ENTER, "q", ENTER, "n")  # the dataset, back, then the headset
        assert console.text.count("Select data source") == 2
        assert console.text.count("Select recording") == 1
        assert console.logged("SOURCE_SELECTED")[0]["label"] == "Muse 2"
        assert [(e["state"], e["key"]) for e in console.logged("KEY")][:4] == [
            ("SOURCE", "DOWN"), ("SOURCE", "ENTER"), ("RECORDING", "Q"), ("SOURCE", "ENTER"),
        ]  # fmt: skip

    def test_list_shows_ten_rows_and_scrolls_with_the_selection(self, console):
        console.machine.headsets_on = set()
        for index in range(15):
            console.add_recording(f"rec_{index:02d}")
        console.start(ENTER, *[DOWN] * 12, ENTER, "n")
        (rows,) = recording_lists(console.text)
        assert [name for _, name in rows] == [f"rec_{index:02d}" for index in range(3, 13)]
        assert rows[-1] == (">", "rec_12")
        assert console.logged("SOURCE_SELECTED")[0]["recording"] == "rec_12"

    def test_folder_without_both_files_is_not_a_recording(self, console):
        console.machine.headsets_on = set()
        whole = console.add_recording("whole")
        (console.add_recording("no_protocol") / "protocol.yaml").unlink()
        (console.add_recording("no_samples") / "eeg.csv").unlink()
        (whole.parent / "notes.txt").write_text("")
        console.start(ENTER, ENTER, "n")
        assert recording_lists(console.text) == [[(">", "whole")]]

    def test_dataset_whose_recordings_vanished_offers_nothing(self, console):
        console.machine.headsets_on = set()
        recording = console.add_recording("one")
        vanish = console.shown("All systems ready.")

        def enter_after_the_recording_vanished():
            vanish("")()
            (recording / "eeg.csv").unlink()
            return ENTER

        assert console.start(enter_after_the_recording_vanished, ENTER, ENTER) == 0
        assert "nothing available" in console.text
        assert console.logged("SOURCE_SELECTED") == []


class TestDialogue:
    def test_n_quits_the_program_and_stops_the_generator(self, console):
        assert console.start(ENTER, "N") == 0
        assert console.session_folders() == []
        assert "Session name" not in console.text  # it quit at once, not at the [CTRL]+[C]
        assert (console.machine.generator.started, console.machine.generator.stopped) == (1, 1)
        assert console.log()[-1]["type"] == "QUIT"

    def test_keys_that_mean_nothing_in_a_state_are_ignored(self, console):
        console.start("x", SPACE, "q", ENTER, "x", SPACE, ENTER, "n")
        assert [(e["state"], e["key"]) for e in console.logged("KEY")] == [
            ("SOURCE", "ENTER"), ("NEW_SESSION", "N"),
        ]  # fmt: skip

    def test_letters_are_not_case_sensitive(self, console):
        console.start(ENTER, "Y", *NAME_S01, "Q", ENTER, "N")
        assert [e["key"] for e in console.logged("KEY")] == ["ENTER", "Y", "Q", "ENTER", "N"]

    def test_q_cancels_before_the_start_and_creates_no_folder(self, console):
        console.start(ENTER, "y", *NAME_S01, "q", ENTER, "n")
        assert "Session cancelled." in console.text
        assert console.logged("SESSION_CANCELLED")[0]["name"] == "S01"
        assert console.session_folders() == []
        assert console.text.count("Select data source") == 2

    def test_invalid_name_is_asked_again(self, console):
        console.start(ENTER, "y", "a", " ", "b", ENTER, ENTER, "é", ENTER, *NAME_S01, "q")
        assert console.text.count("Session name (type it, then [ENTER]):") == 4
        assert console.text.count("Use letters, digits, '_' and '-' only.") == 3
        assert console.logged("SESSION_NAMED")[0]["name"] == "S01"

    def test_typed_characters_are_shown_as_typed_on_the_line_of_the_question(self, console):
        """Also characters that mean something to the library that prints them."""
        console.start(ENTER, "y", "[", "b", "]", "x", "[", "/", "b", "]", ENTER, *NAME_S01, "q")
        assert "Session name (type it, then [ENTER]): [b]x[/b]\n" in console.text
        assert "Use letters, digits, '_' and '-' only.\n\nSession name (type it" in console.text
        assert console.logged("SESSION_NAMED")[0]["name"] == "S01"

    def test_keys_that_are_not_characters_do_not_become_part_of_the_name(self, console):
        console.start(ENTER, "y", "A", UP, DOWN, "\x1b", "\t", "b", ENTER, "q")
        assert console.logged("SESSION_NAMED")[0]["name"] == "Ab"

    def test_backspace_rubs_the_last_character_out_on_the_screen(self, console):
        console.start(ENTER, "y", "A", "x", BACKSPACE, "b", ENTER, "q")
        assert "Session name (type it, then [ENTER]): Ax\b \bb\n" in console.text

    def test_name_keeps_its_case_and_backspace_corrects_it(self, console):
        console.start(ENTER, "y", "s", "X", BACKSPACE, "_", "q", "-", "1", BACKSPACE, BACKSPACE,
                      BACKSPACE, "Q", ENTER, "q")  # fmt: skip
        assert console.logged("SESSION_NAMED")[0]["name"] == "s_Q"

    def test_a_character_rubbed_out_disappears_from_the_screen_at_once(self, console):
        """Not only with the next character: the terminal shows what was flushed."""
        shown = []

        def b_once_rubbed_out():
            wait_until(lambda: "Ax\b \b" in console.text, "x was never rubbed out", KEY_WAIT_S)
            shown.append(console.output.shown)
            return "b"

        console.start(ENTER, "y", "A", "x", BACKSPACE, b_once_rubbed_out, ENTER, "q")
        assert shown[0].endswith("Ax\b \b")

    def test_backspace_on_an_empty_name_does_nothing(self, console):
        console.start(ENTER, "y", BACKSPACE, BACKSPACE, "A", ENTER, "q")
        assert console.logged("SESSION_NAMED")[0]["name"] == "A"

    def test_two_sessions_share_one_generator(self, console):
        console.machine.headsets_on = set()
        console.add_recording("short", SHORT)
        after_a, after_b = console.shown("Session A finished"), console.shown("Session B finished")
        console.start(
            *RECORDING, "y", "A", ENTER, SPACE, after_a(ENTER),
            ENTER, "y", "B", ENTER, SPACE, after_b(ENTER), ENTER, "n",
        )  # fmt: skip
        assert [folder.name[-2:] for folder in console.session_folders()] == ["_A", "_B"]
        assert (console.machine.generator.started, console.machine.generator.stopped) == (1, 1)
        resets = [m for m in console.machine.generator.messages if m == "reset;"]
        assert len(resets) == 2  # each session resets the model, none restarts Pd

    def test_terminal_is_put_back_as_it_was_when_the_console_ends(self, console, monkeypatch):
        restored = []
        monkeypatch.setattr(
            keyboard.Keyboard, "restore_terminal", lambda self: restored.append("restored")
        )
        console.start(ENTER, "n")
        assert restored == ["restored"]


def test_the_console_of_the_program_colours_only_what_it_was_told_to(console, monkeypatch, capsys):
    """Rich would otherwise colour every number and path in a line on its own."""
    monkeypatch.setenv("FORCE_COLOR", "1")
    console.start(ENTER, "y", *NAME_S01, "q", on="standard output")
    assert "Session S01 · source: Muse 2 (headset)" in capsys.readouterr().out


class TestConsoleLog:
    def test_every_action_of_a_run_is_logged_in_order(self, console):
        console.play_recording(SHORT)
        actions = [entry["type"] for entry in console.log() if entry["type"] != "PREFLIGHT"]
        assert actions == [
            "KEY", "KEY", "SOURCE_SELECTED", "KEY", "SESSION_NAMED", "KEY", "SESSION_STARTED",
            "SESSION_ENDED", "KEY", "KEY", "SOURCE_SELECTED", "KEY", "QUIT",
        ]  # fmt: skip
        assert [(e["state"], e["key"]) for e in console.logged("KEY")] == [
            ("SOURCE", "ENTER"), ("RECORDING", "ENTER"), ("NEW_SESSION", "Y"), ("READY", "SPACE"),
            ("SOURCE", "ENTER"), ("RECORDING", "ENTER"), ("NEW_SESSION", "N"),
        ]  # fmt: skip
        assert console.logged("SESSION_ENDED")[0]["reason"] == REASON_END_OF_STREAM
        assert all(entry["t"] > 0 for entry in console.log())

    def test_session_folder_holds_what_the_console_did_for_that_session(self, console):
        """From the start of the program to the end of the session, the preflight included."""
        console.play_recording(SHORT)
        (folder,) = console.session_folders()
        stored = [entry["type"] for entry in read_lines(folder / "console.jsonl")]
        assert stored[:9] == ["PREFLIGHT"] * 9  # one line per check
        assert stored[9:] == [
            "KEY", "KEY", "SOURCE_SELECTED", "KEY", "SESSION_NAMED", "KEY", "SESSION_STARTED",
            "SESSION_ENDED",
        ]  # fmt: skip
        # What the operator did afterwards led to no session and is stored nowhere.
        assert [entry["type"] for entry in console.app.log.waiting] == [
            "KEY", "KEY", "SOURCE_SELECTED", "KEY", "QUIT",
        ]  # fmt: skip
        assert not (console.folder / "sessions" / "_console").exists()

    def test_every_session_of_a_run_holds_its_own_part(self, console):
        console.machine.headsets_on = set()
        console.add_recording("short", SHORT)
        first_done, second_done = console.shown("Session A finished"), console.shown("B finished")
        console.start(
            *RECORDING, "y", "A", ENTER, "q",  # cancelled at READY
            *RECORDING, "y", "A", ENTER, SPACE,
            first_done(ENTER), ENTER, "y", "B", ENTER, SPACE,
            second_done(ENTER), ENTER, "n",
        )  # fmt: skip
        first, second = console.session_folders()
        first = [entry["type"] for entry in read_lines(first / "console.jsonl")]
        second = [entry["type"] for entry in read_lines(second / "console.jsonl")]
        assert first[9:] == [
            "KEY", "KEY", "SOURCE_SELECTED", "KEY", "SESSION_NAMED", "KEY", "SESSION_CANCELLED",
            "KEY", "KEY", "SOURCE_SELECTED", "KEY", "SESSION_NAMED", "KEY", "SESSION_STARTED",
            "SESSION_ENDED",
        ]  # fmt: skip
        assert second == [
            "KEY", "KEY", "SOURCE_SELECTED", "KEY", "SESSION_NAMED", "KEY", "SESSION_STARTED",
            "SESSION_ENDED",
        ]  # fmt: skip

    def test_space_that_ends_a_session_is_logged_as_a_key_of_the_running_state(self, console):
        console.start(
            ENTER, "y", *NAME_S01, SPACE, console.streaming(SPACE), console.shown("finished")(ENTER)
        )
        (folder,) = console.session_folders()
        keys = [entry for entry in read_lines(folder / "console.jsonl") if entry["type"] == "KEY"]
        assert [(entry["state"], entry["key"]) for entry in keys] == [
            ("SOURCE", "ENTER"), ("NEW_SESSION", "Y"), ("READY", "SPACE"), ("RUNNING", "SPACE"),
        ]  # fmt: skip

    def test_session_that_ends_by_itself_logs_no_key_of_the_running_state(self, console):
        console.play_recording(SHORT)
        assert "RUNNING" not in [entry["state"] for entry in console.logged("KEY")]


class TestInterrupt:
    def test_ctrl_c_during_the_preflight_quits_with_code_zero_and_no_errors(self, console):
        class Interrupted(type(console.machine.generator)):
            def start(self):
                raise KeyboardInterrupt

        console.machine.generator = Interrupted()
        assert console.start() == 0
        assert "Errors" not in console.text and "All systems ready." not in console.text
        assert console.log()[-1]["type"] == "QUIT"

    def test_ctrl_c_after_the_generator_started_stops_it(self, console):
        def interrupted(board):
            raise KeyboardInterrupt

        console.machine.headset_is_available = interrupted
        assert console.start() == 0
        assert (console.machine.generator.started, console.machine.generator.stopped) == (1, 1)
        assert "Errors" not in console.text

    def test_ctrl_c_during_a_session_ends_it_and_quits_the_program(self, console):
        assert console.start(ENTER, "y", *NAME_S01, SPACE) == 0  # then [CTRL]+[C]
        (folder,) = console.session_folders()
        last_event = read_lines(folder / c.FILE_EVENTS)[-1]
        assert (last_event["type"], last_event["reason"]) == ("SESSION_END", REASON_USER)
        assert "finished" not in console.text
        assert console.machine.generator.stopped == 1
        stored = [entry["type"] for entry in read_lines(folder / "console.jsonl")]
        assert stored[-3:] == ["SESSION_STARTED", "SESSION_ENDED", "QUIT"]

    def test_a_headset_session_ended_by_ctrl_c_is_stored_all_the_same(self, console):
        """[CTRL]+[C] quits the program, but what the headset recorded is kept."""
        assert console.start(ENTER, "y", *NAME_S01, SPACE, console.streaming("x")) == 0
        (folder,) = console.session_folders()
        stored = console.folder / "data" / "own_recordings" / folder.name
        raw = DataFilter.read_file(str(folder / c.FILE_RAW))
        table = np.loadtxt(stored / "eeg.csv", delimiter=",", skiprows=1, ndmin=2)
        assert len(table) == raw.shape[1] > 0
        assert "finished" not in console.text  # no statistics: the program quits at once

    @pytest.mark.parametrize(
        "keys", [(), (ENTER,), (ENTER, "y"), (ENTER, "y", "S"), (ENTER, "y", *NAME_S01)]
    )
    def test_ctrl_c_in_any_dialogue_quits_with_code_zero(self, console, keys):
        assert console.start(*keys) == 0
        assert console.machine.generator.stopped == 1
        assert console.log()[-1]["type"] == "QUIT"
        assert console.session_folders() == []


class TestOwnRecordings:
    """A session with the headset is stored as a recording in the data folder, to be played
    again like the recordings of any other dataset."""

    def live_session(self, console):
        console.raw["protocol"] = LIVE_PROTOCOL  # the headset delivers 124 s, then nothing
        after_session = console.shown("finished")
        return console.start(ENTER, "y", *NAME_S01, SPACE, after_session(ENTER), "n")

    def test_the_samples_and_the_protocol_go_into_the_data_folder(self, console):
        assert self.live_session(console) == 0
        (session,) = console.session_folders()
        stored = console.folder / "data" / "own_recordings" / session.name
        assert read_yaml(stored / "protocol.yaml") == {"protocol": LIVE_PROTOCOL}
        table = np.loadtxt(stored / "eeg.csv", delimiter=",", skiprows=1)
        raw = DataFilter.read_file(str(session / c.FILE_RAW))
        assert table.shape == (124 * 256, 4) and np.allclose(table.T, raw[1:5], atol=0.0005)
        assert f"Recording       data/own_recordings/{session.name}" in console.text

    def test_it_is_stored_once_the_headset_was_released(self, console, monkeypatch):
        """BrainFlow completes raw.csv when the headset is released."""
        released = []

        def store(directory, datasets_root):
            released.append(console.machine.sources[-1].closed)
            return own_recordings.store_recording(directory, datasets_root)

        monkeypatch.setattr(console_app, "store_recording", store)
        self.live_session(console)
        assert released == [True]

    def test_the_stored_recording_is_offered_as_a_source_at_once(self, console):
        self.live_session(console)
        first, again = source_lists(console.text)
        assert first == [(">", "Muse 2", "headset")]
        assert again == [(">", "Muse 2", "headset"), ("", "own_recordings", "dataset")]

    def test_a_recording_played_from_a_dataset_is_not_stored_again(self, console):
        console.play_recording(SHORT)
        assert not (console.folder / "data" / "own_recordings").exists()
        assert "Recording " not in console.text

    def test_a_recording_that_cannot_be_stored_is_reported_and_the_console_goes_on(self, console):
        (console.folder / "data").mkdir()
        (console.folder / "data" / "own_recordings").write_text("not a folder")
        assert self.live_session(console) == 0
        flat = " ".join(console.text.split())
        assert "The recording of the session could not be stored in the data folder:" in flat
        assert "Session S01 finished" in console.text and "Recording " not in console.text
        assert console.text.count("Select data source") == 2  # back to the sources

    def test_a_recording_that_cannot_be_read_back_is_reported_too(self, console, monkeypatch):
        def unreadable(directory, datasets_root):
            raise AdamusError(f"{directory} is not a complete session folder: no raw.csv")

        monkeypatch.setattr(console_app, "store_recording", unreadable)
        assert self.live_session(console) == 0
        flat = " ".join(console.text.split())
        assert "could not be stored in the data folder:" in flat and "no raw.csv" in flat
