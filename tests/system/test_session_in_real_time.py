"""A whole session in real time over a recording, and its metrics.

BrainFlow plays the recording at its sampling rate, the real player plays a song of the
library, and an output that takes the music in real time stands in for the speakers;
everything else is the real program. The song is quiet in its part marked rest and loud in
the part marked focus. The project folder has a space in its name.
"""

import json
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile

from adamus import constants as c
from adamus.config import load_config, read_yaml, write_yaml
from adamus.evaluation.metrics import analyze
from adamus.realtime.session import SessionResult, real_devices, run_session
from adamus.realtime.song_player import SongPlayer
from adamus.session_format import EVENT_MUSIC_END, REASON_END_OF_STREAM, START, SWITCH, read_jsonl
from adamus.song_library import Library, song_in
from tests.support.finished_sessions import synthetic_signals
from tests.support.project_paths import CONFIG_FILE, require_reference_model
from tests.support.stand_in_machine import place_focus_model
from tests.support.stand_in_outputs import OUTPUT, RATE, stand_in_output
from tests.support.synthetic_dataset import write_recording
from tests.support.synthetic_songs import add_marked, write_song

# A session in real time takes its time: run by the full run only.
pytestmark = pytest.mark.slow

# The music follows the blocks from the first step on.
PROTOCOL = [["REST", 20], ["FOCUS", 20], ["REST", 20], ["FOCUS", 20]]
SECONDS = 80
STEPS = (SECONDS * 256 - 2 * 512) // 128 + 1


def run_batch_session(config_path: Path, name: str, recording: Path, song: str) -> SessionResult:
    """Play ``recording`` through one session with the music of ``song``: the program without
    its app."""
    config = load_config(config_path)
    music = SongPlayer(Library(c.LIBRARY_DIR), song, OUTPUT, lambda: [OUTPUT])
    devices = real_devices(config, recording, music, threading.Event())  # never set
    return run_session(config, name, devices, c.SESSIONS_DIR, lambda folder: None)


@pytest.fixture(scope="module")
def runs(tmp_path_factory, request):
    """Run a session over a recording and compute its metrics."""
    work = tmp_path_factory.mktemp("system") / "my project"
    work.mkdir()
    raw = read_yaml(CONFIG_FILE)
    raw["protocol"] = PROTOCOL
    place_focus_model(work)
    raw["timeouts"]["stream_s"] = 2
    write_yaml(raw, work / "config.yaml")
    patcher = pytest.MonkeyPatch()
    request.addfinalizer(patcher.undo)
    patcher.chdir(work)
    stand_in_output(patcher)
    write_recording(work / "data" / "simulated" / "short", synthetic_signals(PROTOCOL), PROTOCOL)
    library = Library(c.LIBRARY_DIR)
    song = add_marked(library, write_song(work / "Beat.wav"))

    def run(name, recording, config):
        result = run_batch_session(config, name, recording, song)
        analyze(result.directory)
        return {
            "reason": result.reason,
            "folder": result.directory,
            "metrics": json.loads((result.directory / c.FILE_METRICS).read_text()),
        }

    first = run("first", work / "data" / "simulated" / "short", work / "config.yaml")
    return {"work": work, "first": first}


def test_session_ends_with_the_recording(runs):
    assert runs["first"]["reason"] == REASON_END_OF_STREAM
    events = read_jsonl(runs["first"]["folder"] / c.FILE_EVENTS)
    assert events[-1]["reason"] == REASON_END_OF_STREAM
    assert events[-1]["n"] == SECONDS * 256


def test_session_folder_holds_every_file(runs):
    """A session without the user interface: no activity.jsonl."""
    assert {path.name for path in runs["first"]["folder"].iterdir()} == {
        "config.yaml", "focus_model.json", "meta.json", "raw.csv", "filtered.f32", "ticks.jsonl",
        "events.jsonl", "music.jsonl", "audio.wav", "song", "adamus.log", "metrics.json",
    }  # fmt: skip


def test_meta_describes_the_run(runs):
    meta = json.loads((runs["first"]["folder"] / c.FILE_META).read_text())
    assert (meta["playback_path"] is not None, meta["board_id"], meta["fs"]) == (True, 38, 256)
    played = runs["work"] / "data" / "simulated" / "short" / "eeg.csv"
    assert meta["playback_path"] == str(played.resolve())
    assert meta["audio_output"] == OUTPUT


def test_every_sample_was_received_and_processed(runs):
    assert runs["first"]["metrics"]["record_completeness"] == {
        "ticks": STEPS, "filtered_records": STEPS, "expected": STEPS, "passed": True,
    }  # fmt: skip


def test_samples_arrive_at_the_sampling_rate_without_gaps(runs):
    metrics = runs["first"]["metrics"]
    assert metrics["sampling_rate_hz"] == pytest.approx(256, rel=0.03)
    assert metrics["longest_sample_gap_s"] < 0.5


def test_control_separates_rest_from_focus(runs):
    require_reference_model()
    metrics = runs["first"]["metrics"]
    # the same session played at once gives 0.762 and 0.335 (blocks of 20 s are short)
    assert metrics["state_separation_auc"] == pytest.approx(0.762, abs=0.002)
    assert metrics["mean_f_difference"] == pytest.approx(0.335, abs=0.002)


def test_the_state_reaches_the_music_within_a_poll_of_the_last_sample(runs):
    metrics = runs["first"]["metrics"]
    assert 0 <= metrics["control_latency_s"]["median"] < 0.15
    assert metrics["step_interval_s"]["median"] == pytest.approx(0.5, abs=0.1)
    assert metrics["processing_time_s"]["max"] < 0.5


def test_block_markers_are_in_the_raw_file(runs):
    """The record says which column holds what: here four electrodes, the time, the markers."""
    meta = json.loads((runs["first"]["folder"] / c.FILE_META).read_text())
    assert meta["electrode_rows"] == {"AF7": 0, "AF8": 1, "TP9": 2, "TP10": 3}
    assert (meta["timestamp_row"], meta["marker_row"]) == (4, 5)
    raw = np.loadtxt(runs["first"]["folder"] / c.FILE_RAW, delimiter="\t")
    assert raw.shape == (SECONDS * 256, 6)
    positions = np.flatnonzero(raw[:, 5])
    assert raw[positions, 5].tolist() == [1.0, 2.0, 1.0, 2.0]
    starts = np.cumsum([0] + [seconds * 256 for _, seconds in PROTOCOL[:-1]])
    for position, start in zip(positions, starts, strict=True):
        assert start <= position <= start + 64


def test_the_music_is_recorded_for_the_length_of_the_session(runs):
    info = soundfile.info(runs["first"]["folder"] / c.FILE_AUDIO)
    assert (info.samplerate, info.channels, info.subtype) == (RATE, 2, "PCM_16")
    assert info.duration == pytest.approx(SECONDS + 2, abs=1.5)  # plus the end-of-stream wait
    events = read_jsonl(runs["first"]["folder"] / c.FILE_EVENTS)
    ended = [event for event in events if event["type"] == EVENT_MUSIC_END]
    assert ended == [ended[0]] and ended[0]["dropouts"] == 0
    assert runs["first"]["metrics"]["audio_dropouts"] == {"count": 0}


def test_the_music_follows_the_control(runs):
    """The song is loud in its part marked focus: its onsets follow the chance of focus."""
    metrics = runs["first"]["metrics"]
    assert metrics["music_response"]["rho"] > 0.5
    assert metrics["music_response"]["lag_s"] <= 1.0
    expected = metrics["response_time_s"]["rise_median_s"] + metrics["music_response"]["lag_s"]
    assert metrics["total_response_s"]["rise_s"] == expected


def test_the_music_notes_where_it_went_in_its_song(runs):
    went = read_jsonl(runs["first"]["folder"] / c.FILE_MUSIC)
    assert (went[0]["kind"], went[0]["mark"], went[0]["from_s"]) == (START, c.REST, None)
    switches = [line["mark"] for line in went if line["kind"] == SWITCH]
    assert switches[:3] == [c.FOCUS, c.REST, c.FOCUS]  # the blocks, each found by the detector
    times = [line["t"] for line in went]
    assert times == sorted(times)
    song = song_in(runs["first"]["folder"] / c.SONG_DIR)
    assert [m.mark for m in song.marks] == [c.REST, c.FOCUS]
