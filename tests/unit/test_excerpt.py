"""server.excerpt: a part of a session, cut out of its data as one JSON document."""

import json
import math
from datetime import datetime

import numpy as np
import pytest
import yaml

from adamus import constants as c
from adamus.evaluation.session_reader import load_session_folder
from adamus.server.api_models import AudioTrack
from adamus.server.excerpt import clock, dumps, excerpt_of, found_in, loudness, part_data, rounded
from adamus.server.timeline import session_tracks, step_times
from adamus.session_format import read_jsonl

START_S, END_S = 15.0, 25.0  # across the end of the first block and the start of the second
FS = c.DETECTOR_FS


@pytest.fixture(scope="module")
def tracks(played):
    return session_tracks(played.directory)


@pytest.fixture(scope="module")
def written(played, tracks):
    return excerpt_of(played.directory, tracks, START_S, END_S)


@pytest.fixture(scope="module")
def data(written):
    return json.loads(written)


def cut(folder, tracks, start_s=START_S, end_s=END_S):
    return json.loads(excerpt_of(folder, tracks, start_s, end_s))


def rows_of(found):
    """The rows of a table, each as {column: value}."""
    return [dict(zip(found["columns"], row, strict=True)) for row in found["rows"]]


def audio_of(tracks):
    (audio,) = [t for g in tracks.timeline.groups for t in g.tracks if t.kind == "audio"]
    return audio


def steps_over(played, start_s=START_S, end_s=END_S):
    session = load_session_folder(played.directory)
    return [
        (end, tick) for (start, end), tick in zip(step_times(session), session.ticks, strict=True)
        if start < end_s and end > start_s
    ]  # fmt: skip


def known(value, decimals):
    """A value the played session has, rounded as the excerpt keeps it."""
    assert value is not None
    return round(value, decimals)


def with_meta(folder, **changed):
    meta = json.loads((folder / c.FILE_META).read_text())
    (folder / c.FILE_META).write_text(json.dumps(meta | changed))
    return meta | changed


@pytest.mark.parametrize(
    ("seconds", "shown"),
    [(0, "0:00.0"), (-0.04, "0:00.0"), (-3.2, "−0:03.2"), (0.5, "0:00.5"), (-0.5, "−0:00.5"),
     (62.45, "1:02.5"), (59.96, "1:00.0"), (3723.5, "1:02:03.5")],
)  # fmt: skip
def test_a_time_is_shown_as_the_app_shows_it(seconds, shown):
    assert clock(seconds) == shown


def test_the_parts_of_the_data_in_their_order(data, written):
    assert list(data) == [
        "session", "part", "summary", "protocol", "detector", "music", "eeg", "log", "events",
        "generator", "configuration", "files",
    ]  # fmt: skip
    assert written.endswith("}\n")


def test_a_row_of_a_table_stays_on_one_line_and_the_rest_is_indented():
    shown = dumps({"a": {"columns": ["x", "y"], "rows": [[1, None], [2.5, "Š"]]}, "b": {}})
    assert shown == (
        '{\n  "a": {\n    "columns": ["x", "y"],\n    "rows": [\n      [1, null],\n'
        '      [2.5, "Š"]\n    ]\n  },\n  "b": {}\n}'
    )


def test_the_session_its_folder_and_what_it_played(data, played):
    meta = json.loads((played.directory / c.FILE_META).read_text())
    trained = json.loads((played.directory / c.FOCUS_MODEL_FILE).read_text())["trained_on"]
    assert data["session"] == {
        "name": "S01",
        "folder": str(played.directory.resolve()),
        "started": datetime.fromtimestamp(meta["started"]).isoformat(sep=" "),
        "adamus_commit": meta["adamus_commit"],
        "source": "recording",
        "recording": {
            "dataset": "synthetic", "name": "played", "eeg_csv": meta["playback_path"],
            "line_of_sample_0": 2, "person": None,
        },
        "focus_model_trained_on": {"dataset": trained["dataset"], "people": len(trained["people"])},
        "focus_model_trained_on_this_person": False,
        "sampling_rate_hz": FS,
        "electrodes": list(meta["electrode_rows"]),
        "detector_electrodes": ["AF7", "AF8", "TP9", "TP10"],
    }  # fmt: skip


def test_a_person_the_focus_model_learned_from_is_said(copied, tracks):
    people = json.loads((copied / c.FOCUS_MODEL_FILE).read_text())["trained_on"]["people"]
    recording = copied.parent.parent / "a_recording"
    recording.mkdir()
    (recording / c.FILE_DATASET_PROTOCOL).write_text(yaml.safe_dump({"person": people[0]}))
    with_meta(copied, playback_path=str(recording / c.FILE_DATASET_EEG))
    session = cut(copied, tracks)["session"]
    assert session["recording"]["person"] == people[0]
    assert session["focus_model_trained_on_this_person"] is True


def test_a_session_of_the_headset_names_the_headset_and_one_without_its_model_no_model(
    copied, tracks
):
    with_meta(copied, playback_path=None)
    (copied / c.FOCUS_MODEL_FILE).unlink()
    session = cut(copied, tracks)["session"]
    assert (session["source"], session["headset"]) == ("headset", "Muse 2")
    assert "recording" not in session and session["focus_model_trained_on"] is None


def test_the_part_in_the_time_of_the_timeline_as_shown_and_in_samples(data):
    assert data["part"] == {
        "start_s": 15.0, "end_s": 25.0, "duration_s": 10.0, "start_shown": "0:15.0",
        "end_shown": "0:25.0", "first_sample": 15 * FS, "last_sample": 25 * FS - 1,
    }  # fmt: skip


def test_a_part_from_before_the_first_sample_has_its_samples_from_the_first(played):
    session = load_session_folder(played.directory)
    assert part_data(session, -1.0, 2.0)["first_sample"] == 0
    assert part_data(session, -1.0, 2.0)["last_sample"] == 2 * FS - 1
    assert part_data(session, -1.0, 1 / FS)["last_sample"] == 0
    before = part_data(session, -2.0, -1.0)
    assert (before["first_sample"], before["last_sample"]) == (None, None)
    assert part_data(session, -2.0, -1.0)["start_s"] == -2.0


def test_the_blocks_over_the_part_whole_and_cut_to_it(data):
    assert data["protocol"] == [
        {"index": 0, "label": "REST", "start_s": 0.0, "end_s": 20.0},
        {"index": 1, "label": "FOCUS", "start_s": 20.0, "end_s": 40.0},
    ]
    assert data["summary"]["blocks"] == [
        {"index": 0, "label": "REST", "from_s": 15.0, "to_s": 20.0},
        {"index": 1, "label": "FOCUS", "from_s": 20.0, "to_s": 25.0},
    ]


def test_a_block_until_the_stop_has_no_end(played, tracks):
    last = cut(played.directory, tracks, 51.0, tracks.timeline.end_s)
    assert last["protocol"] == [{"index": 3, "label": "FREE", "start_s": 50.0, "end_s": None}]
    assert last["summary"]["blocks"][0]["to_s"] == round(tracks.timeline.end_s, 3)


def test_what_the_detector_found_in_each_block_of_the_part(data, played):
    found = []
    for index, label in ((0, "REST"), (1, "FOCUS")):
        own = [
            f
            for _, t in steps_over(played)
            if t["block_index"] == index and (f := t["f"]) is not None
        ]
        focus = sum(f >= c.FOCUS_THRESHOLD for f in own)
        found.append({"block": index, "label": label, "steps": len(own), "valid": len(own),
                      "focus": focus, "rest": len(own) - focus, "rejected": 0})  # fmt: skip
    assert data["summary"]["detection_by_block"] == found


def test_a_rejected_step_and_an_error_are_counted_in_the_summary(copied):
    ticks = read_jsonl(copied / c.FILE_TICKS)
    inside = [i for i, t in enumerate(ticks) if START_S < t["n_end"] / FS <= 20.0]
    ticks[inside[0]]["valid"] = False
    (copied / c.FILE_TICKS).write_text("".join(json.dumps(tick) + "\n" for tick in ticks))
    events = read_jsonl(copied / c.FILE_EVENTS)
    events.append({"t": events[-1]["t"], "n": 16 * FS, "type": "ERROR", "message": "boom"})
    (copied / c.FILE_EVENTS).write_text("".join(json.dumps(e) + "\n" for e in events))
    summary = cut(copied, session_tracks(copied))["summary"]
    rest = summary["detection_by_block"][0]
    assert (rest["steps"], rest["valid"], rest["rejected"]) == (len(inside), len(inside) - 1, 1)
    assert summary["events"]["ERROR"] == 1


def test_the_lines_of_the_log_are_counted_by_how_grave_they_are(data, tracks):
    (log,) = [shown for shown in tracks.timeline.consoles if shown.title == "Log"]
    inside = [line for line in log.lines if START_S <= line.time_s <= END_S]
    assert data["summary"]["log_lines"] == {
        "all": len(inside),
        "warnings": sum(line.severity == "warning" for line in inside),
        "errors": sum(line.severity == "error" for line in inside),
    }


def test_how_far_the_music_was_steered_over_the_part(data, tracks):
    levels = [s for s in audio_of(tracks).level if s.start_s < END_S and s.end_s > START_S]
    values = [s.value for s in levels]
    assert data["summary"]["music_toward_focus"] == {
        "first": round(values[0], 4), "last": round(values[-1], 4),
        "lowest": round(min(values), 4), "highest": round(max(values), 4),
    }  # fmt: skip
    assert data["music"]["toward_focus"]["columns"] == ["from_s", "to_s", "value"]
    assert data["music"]["toward_focus"]["rows"] == [
        [round(max(s.start_s, START_S), 3), round(min(s.end_s, END_S), 3), round(s.value, 4)]
        for s in levels
    ]


def test_the_limits_of_the_detector_and_every_step_of_the_part_whole(data, played):
    detector = data["detector"]
    assert {key: value for key, value in detector.items() if key != "steps"} == {
        "hop_s": 0.5, "window_s": 2.0, "focus_threshold": 0.5,
        "artifact_electrodes": ["AF7", "AF8"], "artifact_uv": 400.0, "noise_uv": 20.0,
    }  # fmt: skip
    steps = rows_of(detector["steps"])
    expected = steps_over(played)
    assert [step["k"] for step in steps] == [tick["k"] for _, tick in expected]
    end, tick = expected[0]
    sent = tick["t_sent"]
    assert sent is not None  # every step of the session sent its weights
    assert steps[0] == {
        "k": tick["k"], "end_s": round(end, 3), "t_eeg": tick["t_eeg"],
        "block_index": tick["block_index"], "block_label": tick["block_label"],
        "full_window": tick["full_window"], "valid": True, "f": known(tick["f"], 4),
        "f_held": False, "score": known(tick["score"], 4), "w_focus": known(tick["w_focus"], 4),
        "sent_after_s": round(sent - tick["t_eeg"], 4),
        **{f"max_abs_uv_{name}": known(value, 2)
           for name, value in zip(c.DETECTOR_ELECTRODES, tick["max_abs_uv"], strict=True)},
    }  # fmt: skip


def test_a_part_before_any_step_and_block_has_none(played, tracks):
    """As of a session from when the music began before the first sample."""
    early = cut(played.directory, tracks, -1.0, -0.5)
    assert early["detector"]["steps"]["rows"] == [] and early["protocol"] == []
    assert early["summary"]["detection_by_block"] == []


def test_the_music_and_its_loudness_over_the_part(data, tracks):
    audio = audio_of(tracks)
    peaks = tracks.samples["music"]
    low, high = peaks[: audio.buckets], peaks[audio.buckets :]
    first = math.floor((START_S - audio.start_s) * audio.rate_hz)
    last = math.ceil((END_S - audio.start_s) * audio.rate_hz)
    level = np.maximum(np.abs(low[first:last]), np.abs(high[first:last]))
    music = {key: value for key, value in data["music"].items() if key != "toward_focus"}
    assert music == {
        "recorded": True, "start_s": round(audio.start_s, 3) + 0.0,
        "end_s": round(audio.start_s + audio.duration_s, 3),
        "peak_dbfs": round(20 * math.log10(level.max()), 2),
        "median_peak_dbfs": round(20 * math.log10(np.median(level)), 2),
    }  # fmt: skip


def test_the_loudness_is_of_the_buckets_of_the_part_wherever_the_music_begins():
    audio = AudioTrack(id="music", title="Music", caption=None, start_s=-2.0, duration_s=4.0,
                       rate_hz=1, buckets=4)  # fmt: skip
    peaks = np.array([-0.1, -0.2, -0.5, -1.0, 0.1, 0.2, 0.25, 0.5])  # lowest, then highest
    assert loudness(audio, peaks, 0.0, 2.0) == {"peak_dbfs": 0.0, "median_peak_dbfs": -2.5}
    assert loudness(audio, peaks, -3.0, -1.5) == {"peak_dbfs": -20.0, "median_peak_dbfs": -20.0}
    assert loudness(audio, peaks, -4.0, -2.0) == {"peak_dbfs": None, "median_peak_dbfs": None}
    silent = np.zeros(8)
    assert loudness(audio, silent, 0.0, 2.0) == {"peak_dbfs": None, "median_peak_dbfs": None}


def test_a_session_without_music_says_so(copied):
    (copied / c.FILE_AUDIO).unlink()
    music = cut(copied, session_tracks(copied))["music"]
    assert music["recorded"] is False and "peak_dbfs" not in music
    assert music["toward_focus"]["rows"]  # how far it was steered, all the same


def test_every_electrode_raw_and_filtered_over_the_part(data, tracks):
    eeg = data["eeg"]
    assert eeg["raw_less"] == "median of the session"
    assert eeg["columns"] == [
        "electrode", "raw_min_uv", "raw_max_uv", "raw_sd_uv", "filtered_min_uv",
        "filtered_max_uv", "filtered_sd_uv", "filtered_share",
    ]  # fmt: skip
    signals = [t for g in tracks.timeline.groups for t in g.tracks if t.kind == "signal"]
    assert [row[0] for row in eeg["rows"]] == [t.title for t in signals]
    track = signals[1]
    values = tracks.samples[track.id].reshape(track.layers, track.samples)
    raw, filtered = values[0, 15 * FS : 25 * FS], values[1, 15 * FS : 25 * FS]
    assert eeg["rows"][1][1:] == [
        round(float(raw.min()), 2), round(float(raw.max()), 2), round(float(raw.std()), 2),
        round(float(np.nanmin(filtered)), 2), round(float(np.nanmax(filtered)), 2),
        round(float(np.nanstd(filtered)), 2), round(float(np.isfinite(filtered).mean()), 3),
    ]  # fmt: skip


def test_the_eeg_of_a_part_from_before_the_first_sample_is_from_the_first(played, tracks):
    track = [t for g in tracks.timeline.groups for t in g.tracks if t.kind == "signal"][0]
    raw = tracks.samples[track.id].reshape(track.layers, track.samples)[0, : 2 * FS]
    row = cut(played.directory, tracks, -1.0, 2.0)["eeg"]["rows"][0]
    assert row[1:4] == [round(float(stat), 2) for stat in (raw.min(), raw.max(), raw.std())]


def test_an_electrode_the_detector_does_not_read_has_no_filtered_signal(copied):
    meta = json.loads((copied / c.FILE_META).read_text())
    with_meta(copied, electrode_rows={**meta["electrode_rows"], "AUX": meta["timestamp_row"]})
    last = cut(copied, session_tracks(copied))["eeg"]["rows"][-1]
    assert last[0] == "AUX" and last[4:] == [None, None, None, None]


def test_the_lines_of_the_log_in_the_part_with_either_end(data, tracks, played):
    (log,) = [shown for shown in tracks.timeline.consoles if shown.title == "Log"]
    inside = [line for line in log.lines if START_S <= line.time_s <= END_S]
    assert data["log"]["columns"] == ["time_s", "source", "severity", "text"]
    assert data["log"]["rows"] == [
        [round(line.time_s, 3), line.source, line.severity, line.text] for line in inside
    ]
    first, last = log.lines[10], log.lines[20]
    rows = cut(played.directory, tracks, first.time_s, last.time_s)["log"]["rows"]
    assert (rows[0][3], rows[-1][3]) == (first.text, last.text)


def test_the_events_of_the_part_as_they_were_stored(data, played):
    events = read_jsonl(played.directory / c.FILE_EVENTS)
    inside = [e for e in events if START_S <= e["n"] / FS <= END_S]
    assert inside, "events in the part"
    assert data["events"] == [{"time_s": round(e["n"] / FS, 3), **e} for e in inside]
    assert data["summary"]["events"] == {
        kind: sum(e["type"] == kind for e in inside)
        for kind in dict.fromkeys(e["type"] for e in inside)
    }


def test_what_the_generator_was_sent_in_the_part_as_it_was_stored(data, played, tracks):
    sent = read_jsonl(played.directory / c.FILE_GENERATOR)
    inside = [line for line in sent if START_S <= line["n"] / FS <= END_S]
    assert any("noteon" in line["msg"] for line in inside), "chords in the part"
    assert any("prompt 0" in line["msg"] for line in inside), "steps in the part"
    assert data["generator"] == [{"time_s": round(line["n"] / FS, 3), **line} for line in inside]
    first, last = sent[5], sent[-5]
    ends = cut(played.directory, tracks, first["n"] / FS, last["n"] / FS)["generator"]
    assert ends[0]["n"] == first["n"] and ends[-1]["n"] == last["n"]  # both ends belong to it


def test_the_configuration_it_ran_with_whole_in_its_order(copied, tracks):
    settings = yaml.safe_load((copied / c.FILE_CONFIG).read_text())
    settings["styles"][0]["name"] = "Tichý klavír"
    (copied / c.FILE_CONFIG).write_text(yaml.safe_dump(settings, sort_keys=False))
    written = excerpt_of(copied, tracks, START_S, END_S)
    assert json.loads(written)["configuration"] == settings
    assert list(json.loads(written)["configuration"]) == list(settings)
    assert '"name": "Tichý klavír"' in written  # its letters as they are


def test_the_files_of_the_session_there(copied, tracks):
    (copied / c.FILE_PD_LOG).unlink()
    names = [c.FILE_TICKS, c.FILE_FILTERED, c.FILE_RAW, c.FILE_EVENTS, c.FILE_GENERATOR,
             c.FILE_LOG, c.FILE_AUDIO, c.FILE_META, c.FILE_METRICS, c.FILE_CONFIG]  # fmt: skip
    assert cut(copied, tracks)["files"] == {name: str((copied / name).resolve()) for name in names}


@pytest.mark.parametrize(
    ("value", "kept"), [(None, None), (float("nan"), None), (1.23456, 1.235), (2.0, 2.0)]
)
def test_a_number_is_kept_to_its_decimals_and_one_not_known_is_none(value, kept):
    assert rounded(value, 3) == kept


def test_a_number_that_rounds_to_zero_is_zero_not_minus_zero():
    assert math.copysign(1, rounded(-0.0001, 3) or 0.0) == 1
    assert '"start_s": 0.0' in dumps({"start_s": rounded(-0.0001, 3)})


def test_an_event_at_either_end_of_the_part_is_in_it(played, tracks):
    events = read_jsonl(played.directory / c.FILE_EVENTS)
    first, last = events[2], events[-2]
    inside = cut(played.directory, tracks, first["n"] / FS, last["n"] / FS)["events"]
    assert (inside[0]["n"], inside[-1]["n"]) == (first["n"], last["n"])


def test_warnings_and_errors_of_the_log_are_counted(played, tracks):
    timeline = tracks.timeline.model_copy(deep=True)
    (log,) = [shown for shown in timeline.consoles if shown.title == "Log"]
    for severity in ("warning", "error", "error"):
        log.lines.append(log.lines[0].model_copy(update={"time_s": 20.0, "severity": severity}))
    log.lines.sort(key=lambda line: line.time_s)
    counted = json.loads(
        excerpt_of(played.directory, type(tracks)(timeline, tracks.samples), START_S, END_S)
    )
    assert counted["summary"]["log_lines"]["warnings"] == 1
    assert counted["summary"]["log_lines"]["errors"] == 2


def test_a_step_at_the_threshold_found_focus():
    steps = [{"block_index": 0, "valid": True, "f": c.FOCUS_THRESHOLD},
             {"block_index": 0, "valid": True, "f": 0.1}]  # fmt: skip
    found = found_in(0, "REST", steps)  # type: ignore[arg-type]  # only what it reads
    assert (found["focus"], found["rest"]) == (1, 1)


def test_the_eeg_of_a_part_before_the_first_sample_has_nothing_to_tell(played, tracks):
    rows = cut(played.directory, tracks, -2.0, -1.0)["eeg"]["rows"]
    assert all(row[1:] == [None] * 7 for row in rows)
