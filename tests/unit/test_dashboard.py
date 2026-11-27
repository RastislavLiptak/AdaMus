"""server.dashboard: every fact about a session, in the sections of the dashboard."""

import json
import re
import shutil
from dataclasses import replace

import pytest
import yaml

from adamus import constants as c
from adamus.activity_log import ACTION_SESSION_STARTED, ACTION_START_FAILED
from adamus.config import parse_protocol
from adamus.evaluation.session_reader import load_session_folder
from adamus.evaluation.session_statistics import statistics_of
from adamus.server.dashboard import (
    PENDING,
    dashboard_of,
    highlights_of,
    start_problem,
    why_unreadable,
)
from adamus.server.dashboard_sections import (
    blocks_section,
    ended_by,
    failures_of,
    setting_value,
)
from adamus.server.songs import song_view
from adamus.session_format import read_jsonl
from adamus.song_library import song_in
from tests.support.stand_in_songs import ARTIST, SONG

SECTIONS = [
    "Run", "Blocks", "Detection", "Signal", "Music Source", "Music", "Processing", "Configuration",
    "Software", "Activity",
]  # fmt: skip


def facts(dashboard, title):
    """The facts of one section: {label: (value, format)}."""
    (section,) = [s for s in dashboard.sections if s.title == title]
    return {i.label: (i.value.value, i.value.format) for i in section.items if i.kind == "fact"}


def summary(dashboard):
    """The facts of the overview: {label: (value, format)}."""
    return {i.label: (i.value.value, i.value.format) for i in dashboard.summary}


def status(dashboard):
    """How the overview says the session went: {label: status} of the facts that tell it."""
    return {i.label: i.status for i in dashboard.summary if i.status is not None}


def table(dashboard, title):
    (section,) = [s for s in dashboard.sections if s.title == title]
    (found,) = [item for item in section.items if item.kind == "table"]
    return found


@pytest.fixture(scope="module")
def dashboard(played):
    return dashboard_of(played.directory, None, False)


def test_every_section_in_its_order(dashboard, played):
    assert [s.title for s in dashboard.sections] == SECTIONS
    assert dashboard.name == "S01"
    assert dashboard.problem is None and dashboard.pending is None


DETAILS = {
    "Run": ["Focus and Rest From", "Focus Model Trained On", "Ended By"],
    "Detection": [
        "Times Focus Switched On", "Time With Focus On", "Focus Blocks With Focus On",
        "Rest Blocks With Focus Off", "Lowest Chance of Focus", "Highest Chance of Focus",
        "Mean Chance of Focus", "Separation of Focus From Rest (AUC)",
        "Mean Chance of Focus, Focus Less Rest", "Time to Focus On (Median)",
        "Time to Focus Off (Median)",
    ],
    "Signal": [
        "Usable Signal", "Received Sampling Rate", "Longest Gap Between Samples",
        "Steps Recorded", "Record Complete",
    ],
    "Music": [
        "Audio Output", "Control Latency (Median)", "Control Latency (95th Percentile)",
        "Control Latency (Maximum)", "Interval Between Steps (Median)",
        "Interval Between Steps (95th Percentile)", "Music Following Focus (Spearman)",
        "Music Lag", "Time to Audible Focus On", "Time to Audible Focus Off", "Audio Dropouts",
        "Jumps in the Song",
    ],
    "Music Source": ["Title", "Artist", "Length", "Marked Rest", "Marked Focus"],
    "Processing": [
        "Processing Time (Median)", "Processing Time (95th Percentile)",
        "Processing Time (Maximum)",
    ],
    "Configuration": [
        "Headset", "BrainFlow Board", "Headset Serial Number", "Stream Timeout", "Poll Interval",
    ],
}  # fmt: skip


def test_the_details_name_every_fact_and_a_session_with_its_metrics_knows_each(dashboard):
    """A metric read under another name than metrics.json gives it would be unknown here."""
    for title, labels in DETAILS.items():
        shown = facts(dashboard, title)
        assert list(shown) == labels, title
        assert [label for label, (known, _) in shown.items() if known is None] == [], title
    software = facts(dashboard, "Software")
    assert list(software)[:2] == ["AdaMus Commit", "Python"]
    assert table(dashboard, "Activity").columns == ["Time", "Action", "Details"]
    assert table(dashboard, "Activity").label == "Activity"
    assert facts(dashboard, "Signal")["Record Complete"] == (
        "Yes",
        "text",
    )  # the played one is whole


def test_the_audio_output_is_that_of_the_record_and_unknown_of_older_sessions(copied):
    meta = json.loads((copied / c.FILE_META).read_text())
    meta["audio_output"] = "AirPods"
    (copied / c.FILE_META).write_text(json.dumps(meta))
    assert facts(dashboard_of(copied, None, False), "Music")["Audio Output"] == ("AirPods", "text")
    del meta["audio_output"]
    (copied / c.FILE_META).write_text(json.dumps(meta))
    assert facts(dashboard_of(copied, None, False), "Music")["Audio Output"] == (None, "text")


def test_electrodes_the_source_lacked_are_named_as_recorded_as_zeros(copied):
    meta = json.loads((copied / c.FILE_META).read_text())
    meta["detector_electrodes"] = ["AF7", "AF8"]
    (copied / c.FILE_META).write_text(json.dumps(meta))
    signal = facts(dashboard_of(copied, None, False), "Signal")
    assert signal["Electrodes Recorded as Zeros"] == ("TP9, TP10", "text")


def test_a_record_that_is_not_complete_says_so(copied):
    metrics = json.loads((copied / c.FILE_METRICS).read_text())
    metrics["record_completeness"]["passed"] = False
    (copied / c.FILE_METRICS).write_text(json.dumps(metrics))
    signal = facts(dashboard_of(copied, None, False), "Signal")
    assert signal["Record Complete"] == ("No", "text")
    assert signal["Steps Recorded"] == (metrics["record_completeness"]["ticks"], "count")


@pytest.mark.parametrize(
    ("reason", "said"),
    [("user", "Stopped by the operator"), ("end_of_stream", "End of the recording"),
     ("error", "Error"), ("a later reason", "a later reason")],
)  # fmt: skip
def test_what_ended_a_session_is_said_as_its_record_gives_it(reason, said):
    assert ended_by(reason, []) == said


def test_the_app_or_the_program_going_ended_it_the_last_of_them_said():
    going = [{"t": 1.0, "type": "APP_ENDED"}, {"t": 2.0, "type": "PROGRAM_STOPPED"}]
    assert ended_by("user", going) == "The program was stopped"


def test_the_overview_says_when_how_long_on_what_and_how_it_went(dashboard, played):
    run = summary(dashboard)
    meta = json.loads((played.directory / c.FILE_META).read_text())
    assert run["Started"] == (meta["started"], "datetime")
    assert run["Duration"] == (
        statistics_of(load_session_folder(played.directory)).duration_s,
        "duration",
    )
    assert run["Result"] == ("OK", "text")
    assert run["Source"] == ("A recording of a dataset", "text")
    assert run["Recording"] == ("synthetic / played", "text")
    assert list(run) == ["Started", "Duration", "Source", "Recording", "Result"]
    assert status(dashboard) == {"Result": "good"}  # only the result tells how it went
    assert dashboard.caution is None  # the model never saw who the recording is of
    assert dashboard.failures == []  # nothing went wrong
    assert "Folder" not in run  # the folder is the program's own matter
    assert "Why" not in run and "Error" not in run and "Own Recording" not in run


def test_the_details_say_how_it_ran_first(dashboard, played):
    """What told focus from rest, what the detector was trained on, and what ended it."""
    run = facts(dashboard, "Run")
    assert run["Focus and Rest From"] == ("The blocks it was recorded in", "text")
    trained_on = json.loads((played.directory / c.FOCUS_MODEL_FILE).read_text())["trained_on"]
    people = len(trained_on["people"])
    assert run["Focus Model Trained On"] == (f"{trained_on['dataset']}, {people} people", "text")
    assert run["Ended By"] == ("End of the recording", "text")


def test_a_session_without_its_focus_model_does_not_say_what_it_was_trained_on(copied):
    (copied / c.FOCUS_MODEL_FILE).unlink()
    assert list(facts(dashboard_of(copied, None, False), "Run")) == [
        "Focus and Rest From",
        "Ended By",
    ]


def test_the_detection_facts_are_those_of_the_statistics_and_the_metrics(dashboard, played):
    focus = facts(dashboard, "Detection")
    statistics = statistics_of(load_session_folder(played.directory)).focus
    assert statistics is not None
    metrics = json.loads((played.directory / c.FILE_METRICS).read_text())
    assert focus["Times Focus Switched On"] == (statistics.switched_on, "count")
    assert focus["Time With Focus On"] == (statistics.on_share, "percent")
    assert focus["Focus Blocks With Focus On"][0] == (
        f"{statistics.focus_blocks_on} of {statistics.focus_blocks}"
    )
    assert focus["Rest Blocks With Focus Off"][0] == (
        f"{statistics.rest_blocks_off} of {statistics.rest_blocks}"
    )
    assert focus["Separation of Focus From Rest (AUC)"] == (
        metrics["state_separation_auc"], "decimal",
    )  # fmt: skip
    rise = metrics["response_time_s"]["rise_median_s"]
    assert focus["Time to Focus On (Median)"] == (rise, "duration")


def test_the_music_facts_are_those_of_the_metrics(dashboard, played):
    music = facts(dashboard, "Music")
    metrics = json.loads((played.directory / c.FILE_METRICS).read_text())
    assert music["Control Latency (Median)"][0] == metrics["control_latency_s"]["median"]
    assert music["Control Latency (95th Percentile)"][0] == metrics["control_latency_s"]["p95"]
    assert music["Music Following Focus (Spearman)"][0] == metrics["music_response"]["rho"]
    assert music["Audio Dropouts"] == (metrics["audio_dropouts"]["count"], "count")


def test_every_block_is_a_row_with_its_length_its_clean_steps_and_its_response(dashboard, played):
    blocks = table(dashboard, "Blocks")
    metrics = json.loads((played.directory / c.FILE_METRICS).read_text())
    assert blocks.columns == ["Block", "State", "Length", "Clean Steps", "Detector Followed After"]
    rows = [[cell.value for cell in row] for row in blocks.rows]
    assert [row[:3] for row in rows] == [
        [1, "Rest", 20], [2, "Focus", 20], [3, "Rest", 10],
    ]  # fmt: skip  # the state written as the timeline writes it; the recording ended before
    # the free block that follows them
    shares = {block["index"] + 1: block["ratio"] for block in metrics["valid_steps_per_block"]}
    assert [row[3] for row in rows] == [shares[row[0]] for row in rows]
    response = metrics["response_time_s"]
    assert [row[4] for row in rows] == [
        response["fall_s"][0], response["rise_s"][0], response["fall_s"][1],
    ]  # fmt: skip


def test_a_block_that_lasts_until_the_stop_has_that_as_its_length(played):
    session = load_session_folder(played.directory)
    blocks = [["REST", 20], ["FOCUS", 20], ["REST", 5], ["FREE", 0]]  # the free one is begun
    (found,) = blocks_section(replace(session, protocol=parse_protocol(blocks)), {}).items
    assert found.kind == "table"
    assert [row[2].value for row in found.rows] == [20, 20, 5, "Until stopped"]
    assert found.label == "Blocks"


def test_the_configuration_and_the_software_the_session_ran_with(dashboard, played):
    configuration = facts(dashboard, "Configuration")
    assert configuration["Headset"] == ("Muse 2", "text")
    assert configuration["Headset Serial Number"] == ("The first one found", "text")
    assert configuration["BrainFlow Board"] == ("38", "text")  # a name, not grouped by thousands
    assert configuration["Stream Timeout"][1] == "duration"
    settings = yaml.safe_load((played.directory / c.FILE_CONFIG).read_text())
    del settings["protocol"]
    leaves = sum(len(part) if isinstance(part, dict) else 1 for part in settings.values())
    assert len(configuration) == leaves  # every setting; the protocol in the blocks
    software = facts(dashboard, "Software")
    assert "sounddevice" in software
    assert "Focus Model Trained On" not in software  # in the overview
    assert "numpy" in software


def test_the_activity_is_every_line_of_its_log(copied):
    lines = [{"t": 1.5, "type": "PREFLIGHT", "check": "Storage", "result": "ok"},
             {"t": 2.0, "type": "QUIT"}]  # fmt: skip
    (copied / c.FILE_ACTIVITY).write_text("".join(json.dumps(line) + "\n" for line in lines))
    rows = [
        [cell.value for cell in row]
        for row in table(dashboard_of(copied, None, False), "Activity").rows
    ]
    assert rows == [[1.5, "PREFLIGHT", "check: Storage, result: ok"], [2.0, "QUIT", ""]]


def test_a_session_with_the_headset_names_it_and_where_its_recording_went(copied, tmp_path):
    meta = json.loads((copied / c.FILE_META).read_text())
    meta["playback_path"] = None
    (copied / c.FILE_META).write_text(json.dumps(meta))
    own = tmp_path / c.DATASETS_DIR / c.OWN_RECORDINGS / copied.name
    own.mkdir(parents=True)
    dashboard = dashboard_of(copied, own, False)
    run = summary(dashboard)
    assert run["Source"] == ("Headset (Muse 2)", "text") and "Recording" not in run
    assert facts(dashboard, "Run")["Focus and Rest From"] == (
        "The protocol the person was asked to follow", "text",
    )  # fmt: skip
    assert run["Own Recording"] == ("Stored, to be played again", "text")
    assert copied.name not in json.dumps(run)  # the folder is the program's own matter


def test_a_session_of_a_version_with_other_settings_is_shown_with_them(copied):
    """A session keeps the configuration it ran with; a later program may know other keys."""
    settings = yaml.safe_load((copied / c.FILE_CONFIG).read_text())
    del settings["timeouts"]["stream_s"]
    settings["signal"] = {"notch_hz": 50, "ratio": 0.5, "on": True, "bands": [4, 8],
                          "a": {"b": {"c": {"d": {"e": 1}}}}}  # fmt: skip
    (copied / c.FILE_CONFIG).write_text(yaml.safe_dump(settings))
    dashboard = dashboard_of(copied, None, False)
    assert dashboard.problem is None
    configuration = facts(dashboard, "Configuration")
    assert "Stream Timeout" not in configuration
    assert configuration["signal.notch_hz"] == ("50", "text")
    assert configuration["signal.ratio"] == (0.5, "decimal")
    assert configuration["signal.on"] == ("true", "text")
    assert configuration["signal.bands"] == ("[4, 8]", "text")
    assert configuration["signal.a"] == ('{"b": {"c": {"d": {"e": 1}}}}', "text")  # deeper: as text


def test_a_session_stopped_early_has_no_row_of_a_block_it_did_not_get_to(copied):
    ticks = read_jsonl(copied / c.FILE_TICKS)
    first = next(i for i, tick in enumerate(ticks) if tick["block_index"] == 1)
    kept = ticks[: first + 1]  # stopped in the second block
    (copied / c.FILE_TICKS).write_text("".join(json.dumps(tick) + "\n" for tick in kept))
    rows = table(dashboard_of(copied, None, False), "Blocks").rows
    assert [(row[0].value, row[1].value) for row in rows] == [(1, "Rest"), (2, "Focus")]


def test_a_recording_that_could_not_be_stored_is_among_what_went_wrong(copied):
    line = {"t": 3.0, "type": "RECORDING_NOT_STORED", "message": "disk full"}
    (copied / c.FILE_ACTIVITY).write_text(json.dumps(line) + "\n")
    dashboard = dashboard_of(copied, None, False)
    assert dashboard.failures == ["The recording could not be stored: disk full"]
    assert "Recording Not Stored" not in summary(dashboard)  # said once, not again


def test_why_the_result_is_not_ok_comes_before_a_recording_not_stored(played):
    session = load_session_folder(played.directory)
    failed = replace(statistics_of(session), detail="the session did not finish")
    activity = [{"t": 3.0, "type": "RECORDING_NOT_STORED", "message": "disk full"}]
    assert failures_of(session, failed, activity) == [
        "The session did not finish", "The recording could not be stored: disk full",
    ]  # fmt: skip


def test_metrics_that_could_not_be_computed_are_unknown_and_the_reason_is_given(copied):
    (copied / c.FILE_METRICS).unlink()
    line = {"t": 3.0, "type": "ANALYSIS_FAILED", "message": "no filtered.f32"}
    (copied / c.FILE_ACTIVITY).write_text(json.dumps(line) + "\n")
    dashboard = dashboard_of(copied, None, True)
    assert dashboard.problem == "no filtered.f32" and dashboard.pending == PENDING
    assert PENDING == "Computing the metrics of the session"  # as the overview says it
    assert facts(dashboard, "Music")["Music Lag"] == (None, "duration")
    assert [row[4].value for row in table(dashboard, "Blocks").rows] == [None] * 3


def test_an_error_of_the_session_is_named_with_where_it_happened(copied):
    events = read_jsonl(copied / c.FILE_EVENTS)
    end = events.pop()
    events += [{"t": end["t"], "n": end["n"], "type": "ERROR", "where": "loop", "message": "boom"},
               {**end, "reason": "error"}]  # fmt: skip
    (copied / c.FILE_EVENTS).write_text("".join(json.dumps(e) + "\n" for e in events))
    dashboard = dashboard_of(copied, None, False)
    assert dashboard.failures == ["Error in loop: boom"]  # said once, not again among the facts
    assert summary(dashboard)["Result"] == ("ERROR", "text")
    assert status(dashboard) == {"Result": "bad"}
    assert facts(dashboard, "Run")["Ended By"] == ("Error", "text")
    assert "Why" not in summary(dashboard) and "Error" not in summary(dashboard)


@pytest.mark.parametrize(
    ("detail", "said"),
    [("not one clean step for the detector", "Not one clean step for the detector"),
     ("the session did not finish", "The session did not finish")],
)  # fmt: skip
def test_a_result_that_is_not_ok_without_an_error_says_why_as_a_sentence(played, detail, said):
    session = load_session_folder(played.directory)
    assert failures_of(session, replace(statistics_of(session), detail=detail), []) == [said]


def test_a_folder_that_cannot_be_read_shows_its_files_and_why_without_its_path(copied):
    (copied / c.FILE_CONFIG).write_text("calibration: 1\n")
    dashboard = dashboard_of(copied, None, False)
    assert dashboard.problem is not None
    assert dashboard.problem.startswith("This session cannot be read: invalid configuration:")
    assert "protocol: expected a non-empty list" in dashboard.problem
    assert str(copied) not in dashboard.problem
    assert [s.title for s in dashboard.sections] == ["Files"] and dashboard.summary == []
    assert list(facts(dashboard, "Files")) == ["Files"]
    names = sorted(path.name for path in copied.iterdir())
    assert facts(dashboard, "Files")["Files"] == (", ".join(names), "text")


def test_a_missing_value_is_named_and_another_problem_as_python_names_it(copied):
    unreadable = "This session cannot be read"
    assert why_unreadable(copied, KeyError("fs")) == f"{unreadable}: its files have no 'fs'"
    assert why_unreadable(copied, ValueError("x")) == f"{unreadable}: ValueError('x')"


def test_the_first_look_is_the_main_results(dashboard, played):
    session = load_session_folder(played.directory)
    statistics = statistics_of(session)
    metrics = json.loads((played.directory / c.FILE_METRICS).read_text())
    shown = {h.title: (h.value.value, h.value.format) for h in dashboard.highlights}
    focus = statistics.focus
    assert focus is not None
    assert shown == {
        "Focus Detected": (focus.focus_found, "percent"),
        "Rest Detected": (focus.rest_found, "percent"),
        "Music Lag": (metrics["music_response"]["lag_s"], "duration"),
        "Music Follows Detection": (metrics["music_response"]["rho"], "decimal"),
        "Usable Signal": (statistics.usable_share, "percent"),
    }  # fmt: skip
    captions = {h.title: h.caption for h in dashboard.highlights}
    assert captions["Focus Detected"].endswith(f" · {focus.focus_blocks_on} of 1 block")
    assert captions["Rest Detected"].endswith(f" · {focus.rest_blocks_off} of 2 blocks")
    assert all(captions.values())  # each says what it means
    # the interface writes numbers as the computer's language does; a caption cannot
    assert not [h.caption for h in dashboard.highlights if re.search(r"\d[.,]\d", h.caption)]
    kinds = {h.title: h.colour for h in dashboard.highlights}
    assert kinds["Focus Detected"] == "orange" and kinds["Usable Signal"] == "teal"
    assert kinds["Rest Detected"] == "indigo"  # as the timeline draws them
    assert {kinds[t] for t in ("Music Follows Detection", "Music Lag")} == {"pink"}  # one kind


def test_a_session_that_cannot_be_read_has_no_first_look(copied):
    (copied / c.FILE_META).write_text("{")
    unreadable = dashboard_of(copied, None, analysis_pending=False)
    assert (unreadable.highlights, unreadable.summary, unreadable.caution) == ([], [], None)
    assert unreadable.failures == []  # why it cannot be read is its problem


def test_a_recording_of_a_person_the_focus_model_learned_from_is_cautioned_of(copied, tmp_path):
    people = json.loads((copied / c.FOCUS_MODEL_FILE).read_text())["trained_on"]["people"]
    recording = tmp_path / c.DATASETS_DIR / "a_dataset" / "a_recording"
    recording.mkdir(parents=True)

    def caution_of_playing(person):
        protocol = {"protocol": [["REST", 1]]} | ({"person": person} if person else {})
        (recording / c.FILE_DATASET_PROTOCOL).write_text(yaml.safe_dump(protocol))
        meta = json.loads((copied / c.FILE_META).read_text())
        meta["playback_path"] = str(recording / c.FILE_DATASET_EEG)
        (copied / c.FILE_META).write_text(json.dumps(meta))
        return dashboard_of(copied, None, False).caution

    assert caution_of_playing(people[0]) == (
        f"The focus model was trained on this person: {people[0]} is one of the {len(people)} "
        "people it learned from, so the detection does better here than on someone it never saw."
    )
    assert caution_of_playing("someone it never saw") is None
    assert caution_of_playing(None) is None  # the recording does not say whose it is
    (recording / c.FILE_DATASET_PROTOCOL).unlink()
    assert dashboard_of(copied, None, False).caution is None  # nor is it there any more


def test_the_overview_has_the_song_the_session_kept_as_it_was_then(dashboard, played):
    """Its own copy, whatever the library does with the song later: its sound, its parts as
    they were marked, its title, artist and picture."""
    assert dashboard.song == song_view(song_in(played.directory / c.SONG_DIR))
    assert dashboard.song is not None and dashboard.song.title == SONG
    assert [(p.mark, p.title, p.colour) for p in dashboard.song.parts] == [
        (c.REST, "Rest", "indigo"), (c.FOCUS, "Focus", "orange"),
    ]  # fmt: skip
    assert "Music" not in summary(dashboard)  # the song has its own box, under the results


def test_the_details_say_what_the_session_kept_of_its_song(dashboard, played):
    song = song_in(played.directory / c.SONG_DIR)
    assert facts(dashboard, "Music Source") == {
        "Title": (SONG, "text"),
        "Artist": (ARTIST, "text"),
        "Length": (song.duration_s, "duration"),
        "Marked Rest": (song.duration_s / 2, "duration"),
        "Marked Focus": (song.duration_s / 2, "duration"),
    }
    assert facts(dashboard, "Music")["Jumps in the Song"] == (0, "count")  # switched in place


def test_only_a_move_to_another_place_of_the_song_is_a_jump(copied):
    went = [
        ("start", c.REST, None, 0.0),  # where it began
        ("continue", c.REST, 8.0, 20.0),  # the part ran out: elsewhere
        ("switch", c.FOCUS, 24.0, 24.0),  # the song itself went on into focus
        ("switch", c.REST, 30.0, 4.0),  # elsewhere
    ]
    lines = [{"t": 1.0 + i, "n": i, "kind": kind, "mark": mark, "from_s": a, "to_s": b}
             for i, (kind, mark, a, b) in enumerate(went)]  # fmt: skip
    text = "".join(json.dumps(line) + c.LINE_END for line in lines)
    (copied / c.FILE_MUSIC).write_text(text, encoding="utf-8")
    jumps = facts(dashboard_of(copied, None, False), "Music")["Jumps in the Song"]
    assert jumps == (2, "count")


def test_a_session_that_kept_no_song_has_none_on_its_overview_nor_in_its_details(copied):
    shutil.rmtree(copied / c.SONG_DIR)
    shown = dashboard_of(copied, None, False)
    assert shown.song is None and "Music Source" not in [s.title for s in shown.sections]


@pytest.mark.parametrize(
    ("path", "raw", "shown"),
    [("board.serial_number", "", ("The first one found", "text")),
     ("board.serial_number", "MuseS-1A2B", ("MuseS-1A2B", "text")),
     ("board.name", "", ("", "text")), ("timeouts.stream_s", 5, (5, "duration"))],
)  # fmt: skip
def test_only_a_serial_number_not_given_is_the_first_headset_found(path, raw, shown):
    found = setting_value(path, raw)
    assert (found.value, found.format) == shown


def test_the_main_results_say_what_they_mean_also_without_a_clean_step(played):
    statistics = statistics_of(load_session_folder(played.directory))
    captions = [h.caption for h in highlights_of(replace(statistics, focus=None), {})]
    assert captions == [
        "of the steps of focus blocks", "of the steps of rest blocks",
        "until the music followed the detection", "Spearman ρ, 1 when it follows every change",
        "of the steps clean enough to use",
    ]  # fmt: skip


def test_a_session_without_a_step_got_to_no_block(copied):
    (copied / c.FILE_TICKS).write_text("")
    assert table(dashboard_of(copied, None, False), "Blocks").rows == []


def test_a_start_that_failed_after_the_folder_was_made_is_said_with_its_reason():
    tried = {"t": 1.0, "type": ACTION_SESSION_STARTED}
    failed = {"t": 2.0, "type": ACTION_START_FAILED, "message": "TV is not connected"}
    assert start_problem([tried, failed]) == "This session did not start: TV is not connected"
    assert start_problem([failed, tried]) is None  # that was an attempt before this session
