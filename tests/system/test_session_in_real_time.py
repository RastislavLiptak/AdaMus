"""A whole session in real time over a recording, and its metrics.

BrainFlow plays the recording at its sampling rate and a stand-in for the pd
binary receives the messages and records audio; everything else is the real
program. The project folder has a space in its name.
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
from adamus.realtime.music_generator import MusicGenerator
from adamus.realtime.session import SessionResult, real_devices, run_session
from adamus.session_format import REASON_END_OF_STREAM, read_jsonl
from tests.support.finished_sessions import synthetic_signals
from tests.support.project_paths import CONFIG_FILE, require_reference_model
from tests.support.stand_in_generator_files import install_stand_in_generator
from tests.support.stand_in_machine import place_focus_model
from tests.support.stand_in_pure_data import OUTPUTS
from tests.support.synthetic_dataset import write_recording

# A session in real time takes its time: run by the full run only.
pytestmark = pytest.mark.slow

# The music follows the blocks from the first step on.
PROTOCOL = [["REST", 20], ["FOCUS", 20], ["REST", 20], ["FOCUS", 20]]
SECONDS = 80
STEPS = (SECONDS * 256 - 2 * 512) // 128 + 1


def run_batch_session(config_path: Path, name: str, recording: Path) -> SessionResult:
    """Start the generator, play ``recording`` through one session, stop the generator: the
    program without its app."""
    config = load_config(config_path)
    generator = MusicGenerator(
        config.generator,
        config.styles,
        config.timeouts,
        lambda kind, /, **fields: None,
        lambda: OUTPUTS,
    )
    try:
        generator.start(OUTPUTS[0])
        devices = real_devices(config, recording, generator, threading.Event())  # never set
        return run_session(config, name, devices, c.SESSIONS_DIR, lambda folder: None)
    finally:
        generator.stop()


@pytest.fixture(scope="module")
def runs(tmp_path_factory, request):
    """Run a session over a recording and compute its metrics."""
    work = tmp_path_factory.mktemp("system") / "my project"
    work.mkdir()
    raw = read_yaml(CONFIG_FILE)
    raw["protocol"] = PROTOCOL
    raw["generator"].update(install_stand_in_generator(work / "generator"))
    place_focus_model(work)
    raw["timeouts"]["stream_s"] = 2
    write_yaml(raw, work / "config.yaml")
    patcher = pytest.MonkeyPatch()
    request.addfinalizer(patcher.undo)
    patcher.chdir(work)
    write_recording(work / "data" / "simulated" / "short", synthetic_signals(PROTOCOL), PROTOCOL)

    def run(name, recording, config):
        result = run_batch_session(config, name, recording)
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
        "events.jsonl", "audio.wav", "pd.log", "adamus.log", "metrics.json",
    }  # fmt: skip


def test_meta_describes_the_run(runs):
    meta = json.loads((runs["first"]["folder"] / c.FILE_META).read_text())
    assert (meta["playback_path"] is not None, meta["board_id"], meta["fs"]) == (True, 38, 256)
    played = runs["work"] / "data" / "simulated" / "short" / "eeg.csv"
    assert meta["playback_path"] == str(played.resolve())
    assert meta["pd"] == "Pd-0.57-0 (stand-in)"


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


def test_weights_reach_the_generator_within_a_poll_of_the_last_sample(runs):
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


def test_audio_is_recorded_for_the_length_of_the_session(runs):
    info = soundfile.info(runs["first"]["folder"] / c.FILE_AUDIO)
    assert (info.samplerate, info.channels, info.subtype) == (48000, 2, "PCM_16")
    assert info.duration == pytest.approx(SECONDS + 2, abs=1.5)  # plus the end-of-recording timeout
    assert abs(runs["first"]["metrics"]["recording_length_error_s"]) < 0.5


def test_music_follows_the_control(runs):
    """The stand-in generator clicks faster the higher the focus weight."""
    metrics = runs["first"]["metrics"]
    assert metrics["music_response"]["rho"] > 0.5
    assert metrics["music_response"]["lag_s"] <= 1.0
    expected = metrics["response_time_s"]["rise_median_s"] + metrics["music_response"]["lag_s"]
    assert metrics["total_response_s"]["rise_s"] == expected


def test_session_folder_holds_what_pd_printed(runs):
    folder = runs["first"]["folder"]
    printed = (folder / c.FILE_PD_LOG).read_text()
    assert printed.index("mrt2~: model loaded.") < printed.index("mrt2~: state reset.")
    assert sorted(path.name for path in (runs["work"] / "sessions").iterdir()) == [folder.name]
