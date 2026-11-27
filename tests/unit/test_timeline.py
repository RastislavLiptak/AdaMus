"""server.timeline: the tracks of the timeline of a session, as they lie in time."""

import json

import numpy as np
import pytest
import soundfile

from adamus import constants as c
from adamus.datasets import read_recording
from adamus.evaluation.session_reader import load_session_folder
from adamus.server.api_models import CurveTrack
from adamus.server.timeline import (
    envelope,
    nice_scale,
    on_the_samples,
    protocol_bands,
    session_log,
    session_tracks,
    severity_of,
    song_time,
)
from adamus.session_format import EVENT_MUSIC_START, log_time, read_jsonl
from tests.support.finished_sessions import CONTEXT, FS, H, N

ELECTRODES = list(c.DETECTOR_ELECTRODES)
MARKED = "INFO adamus: a line the test wrote"


@pytest.fixture(scope="module")
def tracks(played):
    return session_tracks(played.directory)


def every_track(tracks):
    return [track for group in tracks.timeline.groups for track in group.tracks]


def by_id(tracks):
    return {track.id: track for track in every_track(tracks)}


def console(tracks, title):
    (lines,) = [shown.lines for shown in tracks.timeline.consoles if shown.title == title]
    return lines


def test_the_tracks_in_their_order_in_their_groups(tracks, played):
    """The music on top, how it was steered faded behind it; under it how focus was found,
    under the blocks it was asked for in; then every electrode the source recorded, in the
    order of its rows."""
    electrodes = list(load_session_folder(played.directory).meta["electrode_rows"])
    groups = [(group.title, [t.id for t in group.tracks]) for group in tracks.timeline.groups]
    assert groups == [
        ("Music", ["music"]), ("Focus", ["detection"]),
        ("EEG", [f"signal-{name}" for name in electrodes]),
    ]  # fmt: skip
    tints = [group.colour for group in tracks.timeline.groups]
    assert tints == ["pink", "purple", "teal"]  # the music, the detection, the signal
    titles = [(t.title, t.caption) for t in every_track(tracks)][:2]
    assert titles == [
        ("Music", "faded: the state it followed"),
        ("Detection", "chance of focus under the protocol"),
    ]  # fmt: skip
    assert [(e.colour, e.title) for e in tracks.timeline.legend] == [
        ("orange", "Focus"), ("indigo", "Rest"),
    ]  # fmt: skip  # no step was rejected: no colour for it


def test_an_electrode_is_its_raw_signal_less_its_median_then_its_filtered_signal(tracks, played):
    names, recorded = read_recording(played.root / "data" / "synthetic" / "played" / "eeg.csv")
    samples = tracks.samples["signal-AF8"]
    track = by_id(tracks)["signal-AF8"]
    assert (track.rate_hz, track.samples) == (FS, recorded.shape[1])
    raw, filtered = samples[: track.samples], samples[track.samples :]
    expected = recorded[names.index("AF8")]
    np.testing.assert_allclose(raw, expected - np.median(expected), atol=1e-3)
    assert samples.dtype == np.float32 and len(filtered) == track.samples


def test_the_filtered_signal_is_the_window_of_each_step_where_it_ends(tracks, played):
    """The first step brings its whole window, every later one the hop it ends with."""
    records = np.fromfile(played.directory / c.FILE_FILTERED, dtype="<f4").reshape(-1, 4, N)
    track = by_id(tracks)["signal-TP9"]
    filtered = tracks.samples["signal-TP9"][track.samples :]
    row = ELECTRODES.index("TP9")
    assert np.isnan(filtered[: CONTEXT - N]).all()
    np.testing.assert_array_equal(filtered[CONTEXT - N : CONTEXT], records[0, row])
    k = 7
    end = CONTEXT + k * H
    np.testing.assert_array_equal(filtered[end - H : end], records[k, row, -H:])
    assert not np.isnan(filtered[CONTEXT - N : CONTEXT + (len(records) - 1) * H]).any()


def test_the_scale_of_an_electrode_holds_its_filtered_signal(tracks):
    for name in ELECTRODES:
        track = by_id(tracks)[f"signal-{name}"]
        filtered = tracks.samples[track.id][track.samples :]
        assert np.nanpercentile(np.abs(filtered), 99.5) <= track.scale
        assert (track.layers, track.caption) == (2, f"±{track.scale:g} µV")


@pytest.mark.parametrize(
    ("values", "scale"),
    [([0.3, -0.7], 1.0), ([3.0, 1.5], 5.0), ([12.0], 20.0), ([49.0, -20.0], 50.0),
     ([51.0], 100.0), ([100.0], 100.0), ([0.0, 0.0], 1.0), ([np.nan], 1.0), ([], 1.0),
     ([0.3], 0.5), ([0.04], 0.05)],  # under 1, a power of ten under 1
)  # fmt: skip
def test_a_scale_is_one_two_or_five_times_a_power_of_ten(values, scale):
    assert nice_scale(np.array(values)) == pytest.approx(scale)


def test_the_music_begins_when_its_first_frame_went_to_the_output(tracks, played):
    events = read_jsonl(played.directory / c.FILE_EVENTS)
    rec_start = next(e["t"] for e in events if e["type"] == EVENT_MUSIC_START)
    raw = np.loadtxt(played.directory / c.FILE_RAW, delimiter="\t", ndmin=2).T
    meta = json.loads((played.directory / c.FILE_META).read_text())
    music = by_id(tracks)["music"]
    assert music.start_s == pytest.approx(rec_start - raw[meta["timestamp_row"]][0], abs=1e-6)
    assert music.start_s >= 0  # the music begins once the stream has
    assert music.rate_hz == FS and len(tracks.samples["music"]) == 2 * music.buckets
    assert tracks.timeline.start_s == 0.0  # the first sample
    sound = soundfile.info(played.directory / c.FILE_AUDIO)
    assert music.duration_s == pytest.approx(sound.frames / sound.samplerate)


def test_the_timeline_ends_with_what_ended_last_the_samples_the_music_or_the_steps(tracks, played):
    music = by_id(tracks)["music"]
    samples = len(np.loadtxt(played.directory / c.FILE_RAW, delimiter="\t", ndmin=2))
    ends = [samples / FS, music.start_s + music.duration_s, played.ticks[-1]["n_end"] / FS]
    assert tracks.timeline.end_s == pytest.approx(max(ends))


def test_music_without_the_moment_it_began_is_not_shown(copied):
    events = [e for e in read_jsonl(copied / c.FILE_EVENTS) if e["type"] != EVENT_MUSIC_START]
    (copied / c.FILE_EVENTS).write_text("".join(json.dumps(e) + "\n" for e in events))
    shown = session_tracks(copied)
    assert "music" not in by_id(shown) and "control" in by_id(shown)


def test_the_envelope_is_the_lowest_of_every_bucket_then_the_highest():
    mono = np.array([0.0, 0.5, -0.2, 0.1, 0.9, -0.9, 0.3], dtype=np.float32)
    peaks = envelope(mono, rate=4, buckets_per_s=2)  # buckets of two samples
    np.testing.assert_allclose(peaks, [0.0, -0.2, -0.9, 0.3, 0.5, 0.1, 0.9, 0.3])
    assert len(envelope(np.empty(0, dtype=np.float32), 4, 2)) == 0


def test_a_sound_shorter_than_a_bucket_is_one_bucket_and_none_reaches_past_its_end():
    np.testing.assert_allclose(envelope(np.array([0.4], dtype=np.float32), 4, 2), [0.4, 0.4])
    # two samples at three a second, two buckets a second: the second bucket would begin at
    # sample 1.5, rounded to 2, past the end; it begins at the last sample instead
    peaks = envelope(np.array([0.1, -0.4], dtype=np.float32), rate=3, buckets_per_s=2)
    np.testing.assert_allclose(peaks, [0.1, -0.4, 0.1, -0.4])


def test_the_protocol_lies_along_the_top_of_the_detection_and_ends_with_the_timeline(tracks):
    bands = [(b.start_s, b.end_s, b.title, b.colour) for b in by_id(tracks)["detection"].bands]
    assert bands == [
        (0.0, 20.0, "Rest", "indigo"), (20.0, 40.0, "Focus", "orange"),
        (40.0, 50.0, "Rest", "indigo"),
        (50.0, tracks.timeline.end_s, "Free", None),
    ]  # fmt: skip


def test_every_step_is_the_hop_it_ends_with_and_its_chance_of_focus(tracks, played):
    """Tinted as focus over the threshold and as rest under it, as the metrics count it."""
    detection = by_id(tracks)["detection"]
    assert (detection.reference, detection.stepped) == (c.FOCUS_THRESHOLD, False)
    assert (detection.above, detection.below) == ("orange", "indigo")
    assert len(detection.steps) == len(played.ticks)
    for step, tick in zip(detection.steps, played.ticks, strict=True):
        assert step.end_s == tick["n_end"] / FS
        assert step.end_s - step.start_s == pytest.approx(H / FS)
        assert step.value == tick["f"]
        assert step.colour == (None if tick["valid"] else "gray")  # its signal too disturbed


def with_ticks(folder, change):
    """The session in ``folder``, its ticks altered by ``change``."""
    ticks = read_jsonl(folder / c.FILE_TICKS)
    change(ticks)
    (folder / c.FILE_TICKS).write_text("".join(json.dumps(tick) + "\n" for tick in ticks))
    return folder


def test_a_step_whose_signal_was_not_clean_is_drawn_in_the_tone_of_rejected_ones(copied):
    def rejected(ticks):
        ticks[3]["valid"] = False

    shown = session_tracks(with_ticks(copied, rejected))
    steps = by_id(shown)["detection"].steps
    assert [index for index, step in enumerate(steps) if step.colour == "gray"] == [3]
    assert ("gray", "Rejected") in [(e.colour, e.title) for e in shown.timeline.legend]


def test_the_music_control_holds_the_state_told_from_one_step_to_the_next(tracks, played):
    control = by_id(tracks)["music"].level
    ends = [tick["n_end"] / FS for tick in played.ticks]
    timeline = tracks.timeline
    assert [step.start_s for step in control] == [timeline.start_s, *ends]
    assert [step.end_s for step in control] == [*ends, timeline.end_s]
    focus = False  # the music starts at rest
    assert (control[0].value, control[0].colour) == (0.0, "indigo")
    for step, tick in zip(control[1:], played.ticks, strict=True):
        focus = tick["told"] == c.FOCUS if tick["told"] is not None else focus
        assert (step.value, step.colour) == ((1.0, "orange") if focus else (0.0, "indigo"))
    assert {step.value for step in control} == {0.0, 1.0}


def test_a_step_that_told_nothing_leaves_the_music_as_it_was(copied, played):
    def untold(ticks):
        ticks[0]["t_sent"], ticks[0]["told"] = None, None
        ticks[3]["t_sent"], ticks[3]["told"] = None, None  # never reached the music

    steps = by_id(session_tracks(with_ticks(copied, untold)))["music"].level
    assert steps[1].value == 0.0  # step 0: still at rest, as the session started it
    told = 1.0 if played.ticks[2]["told"] == c.FOCUS else 0.0
    assert steps[4].value == told  # step 3: as step 2 left it


def test_a_skipped_step_leaves_the_filtered_signal_where_its_steps_ended(copied, played):
    """A step that failed writes neither its tick nor its window; the next ones keep theirs."""
    size = len(ELECTRODES) * N  # the values of the window of one step
    folder = copied
    ticks = read_jsonl(folder / c.FILE_TICKS)
    records = np.fromfile(folder / c.FILE_FILTERED, dtype=c.FILTERED_DTYPE).reshape(-1, size)
    kept = [k for k in range(len(ticks)) if k != 3]
    (folder / c.FILE_TICKS).write_text("".join(json.dumps(ticks[k]) + "\n" for k in kept))
    records[kept].tofile(folder / c.FILE_FILTERED)
    samples = len(session_tracks(folder).samples["signal-AF7"]) // 2
    whole = session_tracks(played.directory).samples["signal-AF7"][samples:]
    gapped = session_tracks(folder).samples["signal-AF7"][samples:]
    end_2, end_3 = ticks[2]["n_end"], ticks[3]["n_end"]
    np.testing.assert_array_equal(gapped[:end_2], whole[:end_2])
    np.testing.assert_array_equal(gapped[end_3:], whole[end_3:])  # the later steps stay put
    assert not np.isnan(gapped[end_2:end_3]).any()  # step 4's window covers 3's hop


def test_a_window_cut_short_when_the_session_was_killed_is_left_out(copied, played):
    folder = copied
    with (folder / c.FILE_FILTERED).open("ab") as filtered:
        filtered.write(np.zeros(5, dtype=c.FILTERED_DTYPE).tobytes())
    np.testing.assert_array_equal(
        session_tracks(folder).samples["signal-AF7"],
        session_tracks(played.directory).samples["signal-AF7"],
    )


def test_the_last_window_cut_short_leaves_the_steps_before_it_as_they_were(copied, played):
    """Killed while it wrote the window of its last step: that step has no window to draw."""
    path = copied / c.FILE_FILTERED
    whole = path.read_bytes()
    window = len(ELECTRODES) * N * np.dtype(c.FILTERED_DTYPE).itemsize
    path.write_bytes(whole[: len(whole) - window // 2])
    cut = session_tracks(copied).samples["signal-AF7"]
    kept = session_tracks(played.directory).samples["signal-AF7"]
    end_before = played.ticks[-2]["n_end"]
    half = len(kept) // 2  # the filtered layer follows the raw one
    np.testing.assert_array_equal(cut[half : half + end_before], kept[half : half + end_before])


def test_a_block_that_would_begin_as_the_timeline_ends_is_not_drawn(played):
    session = load_session_folder(played.directory)
    second = session.spans[1].start / session.fs
    assert [band.title for band in protocol_bands(session, second)] == ["Rest"]


def test_a_session_without_music_shows_how_it_was_steered_alone(copied, tracks):
    folder = copied
    (folder / c.FILE_AUDIO).unlink()
    alone = session_tracks(folder)
    assert "music" not in by_id(alone) and "music" not in alone.samples
    assert alone.timeline.start_s == 0.0
    (control,) = alone.timeline.groups[0].tracks
    assert isinstance(control, CurveTrack)
    assert (control.id, control.title, control.stepped) == ("control", "Music Control", True)
    steered = by_id(tracks)["music"].level
    assert [step.value for step in control.steps] == [step.value for step in steered]


def test_a_session_that_received_nothing_has_empty_tracks(copied):
    folder = copied
    for name in (c.FILE_RAW, c.FILE_FILTERED, c.FILE_AUDIO):
        (folder / name).unlink()
    (folder / c.FILE_TICKS).write_text("")
    tracks = session_tracks(folder)
    signal = by_id(tracks)["signal-AF7"]
    assert signal.samples == 0 and len(tracks.samples["signal-AF7"]) == 0
    assert (tracks.timeline.start_s, tracks.timeline.end_s) == (0.0, 0.0)
    assert by_id(tracks)["detection"].steps == []
    (quiet,) = by_id(tracks)["control"].steps  # the music as the session started it
    assert (quiet.start_s, quiet.end_s, quiet.value) == (0.0, 0.0, 0.0)


def add_to_the_log(folder, moment, said=MARKED):
    with (folder / c.FILE_LOG).open("a", encoding="utf-8") as log:
        print(log_time(moment), said, file=log)


def test_the_lines_of_the_log_lie_where_the_samples_came_when_they_were_written(copied):
    folder = copied
    session = load_session_folder(folder)
    first = float(session.raw_samples()[session.meta["timestamp_row"]][0])
    add_to_the_log(folder, first + 2.5)
    log = console(session_tracks(folder), "Log")
    (line,) = [line for line in log if line.text == MARKED]
    assert line.time_s == pytest.approx(2.5, abs=0.001)  # the time of a log is to the millisecond
    assert line.source == "AdaMus"
    assert any(line.text.startswith("DEBUG adamus.realtime.session: step 0:") for line in log)


def test_without_samples_the_log_lies_where_the_music_began(copied):
    folder = copied
    (folder / c.FILE_RAW).unlink()
    began = load_session_folder(folder).music_began
    assert began is not None
    add_to_the_log(folder, began + 1)
    log = console(session_tracks(folder), "Log")
    (line,) = [line for line in log if line.text == MARKED]
    assert line.time_s == pytest.approx(1.0, abs=0.001)


# The log, at the moments its lines were written.

ZERO = 1_800_000_000.0  # the moment time 0 of the timeline stands for


def FROM_ZERO(moment):  # noqa: N802 - stands for a constant: the clock from ZERO on
    return moment - ZERO


def write_log(path, *lines):
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def test_the_lines_of_the_log_lie_at_the_moments_they_were_written(tmp_path):
    write_log(tmp_path / c.FILE_LOG, f"{log_time(ZERO - 2)} INFO adamus: opened",
              f"{log_time(ZERO + 1.5)} DEBUG adamus.realtime.session: step 0")  # fmt: skip
    lines = session_log(tmp_path, FROM_ZERO)
    assert [(line.time_s, line.source, line.text) for line in lines] == [
        (-2.0, "AdaMus", "INFO adamus: opened"),
        (1.5, "AdaMus", "DEBUG adamus.realtime.session: step 0"),
    ]


def test_lines_written_in_the_same_millisecond_keep_the_order_of_their_log(tmp_path):
    write_log(tmp_path / c.FILE_LOG, *(f"{log_time(ZERO)} INFO adamus: {n}" for n in "abc"))
    assert [line.text[-1] for line in session_log(tmp_path, FROM_ZERO)] == ["a", "b", "c"]


def test_a_traceback_stays_with_the_line_it_belongs_to(tmp_path):
    write_log(
        tmp_path / c.FILE_LOG,
        f"{log_time(ZERO)} ERROR adamus.realtime.session: session failed in a step",
        "Traceback (most recent call last):",
        "ValueError: no",
        f"{log_time(ZERO + 1)} INFO adamus: ended",
    )
    failed, ended = session_log(tmp_path, FROM_ZERO)
    assert failed.text.splitlines()[1:] == ["Traceback (most recent call last):", "ValueError: no"]
    assert (ended.time_s, ended.text) == (1.0, "INFO adamus: ended")


def test_a_warning_in_a_log_is_a_line_of_a_warning(tmp_path):
    write_log(tmp_path / c.FILE_LOG, f"{log_time(ZERO)} WARNING adamus: the disk is nearly full")
    (line,) = session_log(tmp_path, FROM_ZERO)
    assert line.severity == "warning"


def test_a_line_that_is_not_utf_8_is_read_all_the_same(tmp_path):
    (tmp_path / c.FILE_LOG).write_bytes(log_time(ZERO).encode() + b" INFO adamus: caf\xe9\n")
    (line,) = session_log(tmp_path, FROM_ZERO)
    assert line.text == "INFO adamus: caf\ufffd"  # the byte it could not read, marked


def test_a_log_that_was_not_written_has_no_lines(tmp_path):
    assert session_log(tmp_path, FROM_ZERO) == []


def test_a_session_with_no_time_0_shows_no_lines(tmp_path):
    write_log(tmp_path / c.FILE_LOG, f"{log_time(ZERO)} INFO adamus: opened")
    assert session_log(tmp_path, None) == []


def test_a_moment_between_the_first_two_or_the_last_two_samples_lies_between_them():
    stamps = np.array([0.0, 1.0, 2.0, 4.0]) + ZERO  # a sample lost before the last
    place = on_the_samples(stamps, 1)
    assert place(ZERO + 0.5) == pytest.approx(0.5)
    assert place(ZERO + 3.0) == pytest.approx(2.5)  # half way between the last two samples


def test_a_moment_lies_at_the_samples_that_came_then_also_when_samples_were_lost():
    """Four samples a second, the third second lost: what happened at the time of day of the
    sample after the gap lies at that sample, not a second later."""
    stamps = np.concatenate([np.arange(8) / 4, 3 + np.arange(4) / 4]) + ZERO
    place = on_the_samples(stamps, 4)
    assert place(stamps[8]) == pytest.approx(8 / 4)  # the gap is not on the timeline
    assert place(stamps[2] + 0.125) == pytest.approx(2.5 / 4)  # between two samples
    assert place(ZERO - 1) == pytest.approx(-1)  # before the first, at the pace of the clock
    assert place(stamps[-1] + 2) == pytest.approx(11 / 4 + 2)  # after the last, likewise


def test_the_music_and_the_logs_lie_at_the_samples_also_when_the_headset_lost_some(copied):
    folder = copied
    raw = (folder / c.FILE_RAW).read_text().splitlines()
    lost = FS  # a second of samples, from the tenth second on
    kept = raw[: 10 * FS] + raw[10 * FS + lost :]
    (folder / c.FILE_RAW).write_text("".join(row + "\n" for row in kept))
    session = load_session_folder(folder)
    after_the_gap = float(session.raw_samples()[session.meta["timestamp_row"]][12 * FS])
    add_to_the_log(folder, after_the_gap)
    log = console(session_tracks(folder), "Log")
    (line,) = [line for line in log if line.text == MARKED]
    assert line.time_s == pytest.approx(12.0, abs=0.001)  # at sample 12 * FS, not a second later


def test_every_event_lies_at_the_samples_received_when_it_happened_as_a_console_shows_it(
    tracks, played
):
    lines = console(tracks, "Events")
    assert len(lines) == len(played.events)
    for line, event in zip(lines, played.events, strict=True):
        assert line.time_s == event["n"] / FS
        assert line.text.startswith(event["type"])
    texts = [line.text for line in lines]
    assert "MUSIC_START" in texts and "MUSIC_END dropouts=0" in texts
    block = next(line.text for line in lines if line.text.startswith("BLOCK_START"))
    assert block == "BLOCK_START index=0 label=REST marker=1"  # what it holds, field by field
    assert {line.severity for line in lines} == {"info"}


def test_the_timeline_of_a_session_whose_music_began_before_its_stream_begins_with_the_music(
    copied,
):
    """As sessions from before the music waited for the stream."""
    events = read_jsonl(copied / c.FILE_EVENTS)
    for event in events:
        if event["type"] == EVENT_MUSIC_START:
            event["t"] -= 3.0
    (copied / c.FILE_EVENTS).write_text("".join(json.dumps(e) + "\n" for e in events))
    tracks = session_tracks(copied)
    music = by_id(tracks)["music"]
    assert music.start_s == pytest.approx(-3.0, abs=0.01)
    assert tracks.timeline.start_s == music.start_s


def test_where_the_music_went_in_its_song_lies_when_it_went_there(tracks, played):
    lines = console(tracks, "Music")
    went = read_jsonl(played.directory / c.FILE_MUSIC)
    assert len(lines) == len(went) and went[0]["kind"] == "start"
    assert lines[0].text == "start rest: 0:00.0"
    switch = next(index for index, line in enumerate(went) if line["kind"] == "switch")
    assert lines[switch].text.startswith(f"switch {went[switch]['mark'].lower()}: ")
    assert " → " in lines[switch].text
    times = [line.time_s for line in lines]
    assert times == sorted(times) and times[0] == by_id(tracks)["music"].start_s
    assert {line.severity for line in lines} == {"info"}


def test_a_place_in_the_song_is_written_in_minutes_and_seconds():
    assert [song_time(seconds) for seconds in (0, 5.25, 65.3, 600)] == [
        "0:00.0", "0:05.2", "1:05.3", "10:00.0",
    ]  # fmt: skip


def test_a_session_whose_music_never_began_shows_nowhere_it_went(copied):
    (copied / c.FILE_MUSIC).unlink()
    assert console(session_tracks(copied), "Music") == []


def test_an_error_of_the_session_is_an_error_line_that_says_what_went_wrong(copied):
    events = read_jsonl(copied / c.FILE_EVENTS)
    end = events.pop()
    error = {"t": end["t"], "n": end["n"], "type": "ERROR", "where": "loop", "message": "boom"}
    (copied / c.FILE_EVENTS).write_text(
        "".join(json.dumps(e) + "\n" for e in [*events, error, end])
    )
    line = console(session_tracks(copied), "Events")[-2]
    assert (line.severity, line.text) == ("error", "ERROR where=loop boom")


def test_an_electrode_the_session_did_not_filter_is_shown_raw_alone(copied):
    """Whatever the program computes with, the timeline shows every electrode recorded."""
    folder = copied
    rows = [line.split("\t") for line in (folder / c.FILE_RAW).read_text().splitlines()]
    (folder / c.FILE_RAW).write_text("".join("\t".join([*row, "7.0"]) + "\n" for row in rows))
    meta = json.loads((folder / c.FILE_META).read_text())
    meta["electrode_rows"]["Fp1"] = len(rows[0])  # a column of its own, after the others
    (folder / c.FILE_META).write_text(json.dumps(meta))
    shown = session_tracks(folder)
    track = by_id(shown)["signal-Fp1"]
    assert (track.title, track.layers) == ("Fp1", 1)
    assert track in shown.timeline.groups[-1].tracks  # with the other electrodes
    assert len(shown.samples["signal-Fp1"]) == track.samples  # its raw signal, less its median
    assert not shown.samples["signal-Fp1"].any()


def test_a_line_of_a_log_says_how_grave_it_is():
    assert severity_of("DEBUG adamus.realtime.session: step 0") == "debug"
    assert severity_of("WARNING adamus.realtime.session: the disk is full") == "warning"
    assert severity_of("ERROR adamus.realtime.session: session failed") == "error"
    assert severity_of("CRITICAL adamus.main: the program could not go on") == "error"
    assert severity_of("a line without a level") == "info"
