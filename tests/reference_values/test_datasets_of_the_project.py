"""The datasets in the data folder of the project: every recording can be played, and the
computation gives the known results on them.

The recordings were converted once from the original datasets: at the rate of
the headset, cut into blocks of rest and task. A recording is one table of
all the electrodes of its original, and the program takes from it the ones
of the detector.
"""

import numpy as np
import pytest

from adamus import constants as c
from adamus.datasets import electrodes_of_recording, person_of, protocol_of, recordings_of
from adamus.detection.focus_training import LOG_FILE, TRAINING_DATA, blocks_of
from adamus.evaluation import metrics
from adamus.session_format import REASON_END_OF_STREAM
from adamus.session_sources import (
    missing_electrodes,
    session_config,
)
from tests.support.finished_sessions import CONTEXT, FS, FinishedSession, H, synthetic_signals
from tests.support.project_paths import DATASETS, ROOT, shipped_datasets

SYNTHETIC = DATASETS / "synthetic"
N_BACK = DATASETS / "wearable_monitoring_n_back"
HEADBAND = DATASETS / "office_mental_workload"
RECORDINGS = [recording for dataset in shipped_datasets() for recording in recordings_of(dataset)]


def table_of(recording):
    """The samples of a recording as [electrodes x samples]."""
    return np.loadtxt(recording / c.FILE_DATASET_EEG, delimiter=",", skiprows=1).T


def test_the_three_datasets_and_their_recordings():
    names = {dataset.name: len(recordings_of(dataset)) for dataset in shipped_datasets()}
    assert names == {
        "office_mental_workload": 12,
        "synthetic": 1,
        "wearable_monitoring_n_back": 23,
    }


def test_the_n_back_recordings_are_the_people_the_models_were_trained_on():
    from adamus.detection.focus_detector import load_models

    people = load_models().raw["trained_on"]["people"]
    assert sorted(str(person_of(r)) for r in recordings_of(N_BACK)) == sorted(people)
    assert {person_of(r) for r in recordings_of(HEADBAND)} == {None}  # the model never saw them


ORIGINAL = ROOT / c.FOCUS_DETECTOR_DIR / TRAINING_DATA


@pytest.mark.skipif(not ORIGINAL.is_dir(), reason="the original dataset is not downloaded")
def test_each_n_back_recording_is_the_person_it_names_to_the_millisecond_of_their_blocks():
    """The task blocks of the protocol of each recording are those of its person in the original
    dataset, and of no one else: the name of a person is known, not guessed from a folder."""
    original = {
        f"{person.parent.name}/{person.name}": [end - start for start, end in blocks_of(person)]
        for person in ORIGINAL.glob("*/Experiment_*/*")
        if (person / LOG_FILE).is_file()
    }
    for recording in recordings_of(N_BACK):
        tasks = [seconds for label, seconds in protocol_of(recording) if label == c.FOCUS]
        alike = [
            name for name, blocks in original.items()
            if len(blocks) >= len(tasks) and np.allclose(blocks[: len(tasks)], tasks, atol=2e-3)
        ]  # fmt: skip
        assert alike == [person_of(recording)], recording.name


@pytest.mark.parametrize(
    "recording", RECORDINGS, ids=lambda path: f"{path.parent.name}/{path.name}"
)
class TestEveryRecording:
    def test_is_one_table_of_samples_and_its_protocol(self, recording):
        assert sorted(path.name for path in recording.iterdir()) == ["eeg.csv", "protocol.yaml"]

    def test_protocol_can_be_run_and_the_samples_fill_it_exactly(self, recording, config):
        """One line per sample at the rate of the headset, with a value for every electrode."""
        protocol = session_config(config, recording).protocol
        assert all(block.duration_s > 0 for block in protocol)
        names = electrodes_of_recording(recording / c.FILE_DATASET_EEG)
        with (recording / c.FILE_DATASET_EEG).open(encoding="utf-8") as lines:
            next(lines)
            values_per_line = {line.count(",") + 1 for line in lines}
            lines.seek(0)
            count = sum(1 for _ in lines) - 1
        assert values_per_line == {len(names)}
        assert count == round(FS * sum(block.duration_s for block in protocol))


class TestElectrodes:
    """Every electrode of an original is in its recordings, under its own name."""

    def test_synthetic_recording_has_the_electrodes_of_the_detector(self):
        names = electrodes_of_recording(SYNTHETIC / "rest_focus_blocks" / c.FILE_DATASET_EEG)
        assert names == list(c.DETECTOR_ELECTRODES) == ["AF7", "AF8", "TP9", "TP10"]

    @pytest.mark.parametrize("recording", recordings_of(HEADBAND), ids=lambda path: path.name)
    def test_headband_recording_has_its_four_electrodes_and_the_auxiliary_input(
        self, recording, config
    ):
        names = electrodes_of_recording(recording / c.FILE_DATASET_EEG)
        assert names == ["TP9", "AF7", "AF8", "TP10", "AUX_RIGHT"]
        assert missing_electrodes(config, recording) == []

    @pytest.mark.parametrize("recording", recordings_of(N_BACK), ids=lambda path: path.name)
    def test_n_back_recording_has_its_four_electrodes_and_the_auxiliary_input(
        self, recording, config
    ):
        names = electrodes_of_recording(recording / c.FILE_DATASET_EEG)
        assert names == ["TP9", "AF7", "AF8", "TP10", "Right AUX"]
        assert missing_electrodes(config, recording) == []

    @pytest.mark.parametrize(
        "recording",
        [HEADBAND / "Participant01_Lab1", N_BACK / "Experiment_1_A5", N_BACK / "Experiment_2_B9"],
        ids=lambda path: path.name,
    )
    def test_every_electrode_carries_a_signal_in_microvolts(self, recording):
        table = table_of(recording)
        assert np.isfinite(table).all()
        assert (table.std(axis=1) > 1).all() and (table.std(axis=1) < 5000).all()


def test_n_back_protocol_is_the_blocks_of_the_task_and_the_breaks(config):
    """Ten seconds before the first block, then 16 blocks of about 42 s with breaks of about
    15 s (with the instruction for the next block) and half a minute of rest in the middle."""
    protocol = session_config(config, N_BACK / "Experiment_1_A5").protocol
    assert (protocol[0].label, protocol[0].duration_s) == (c.REST, 10.0)
    focus = [b.duration_s for b in protocol if b.label == c.FOCUS]
    rest = [b.duration_s for b in protocol[1:-1] if b.label == c.REST]
    assert len(focus) == 16 and all(40 < seconds < 45 for seconds in focus)
    assert sorted(rest)[0] > 13 and max(rest) > 25
    assert [b.label for b in protocol[1:]] == [c.FOCUS, c.REST] * 16


def test_synthetic_recording_is_the_synthetic_signal_of_the_configured_protocol(config):
    recording = SYNTHETIC / "rest_focus_blocks"
    assert session_config(config, recording).protocol == config.protocol[:8]  # without free play
    expected = synthetic_signals(config.raw["protocol"])
    assert np.abs(table_of(recording) - expected).max() <= 0.0005  # written with three decimals


def finished(tmp_path, config, recording):
    session = FinishedSession(tmp_path, session_config(config, recording), recording=recording)
    mean_f = {
        label: float(
            np.mean([t["f"] for t in session.ticks if t["f"] is not None and t["full_window"]
                     and t["block_label"] == label])
        )
        for label in (c.REST, c.FOCUS)
    }  # fmt: skip
    return session, mean_f, metrics.compute_metrics(session.directory)["state_separation_auc"]


def test_session_over_a_muse_s_recording_of_a_person_the_model_never_saw(tmp_path, config):
    """Relaxing video and n-back, on another headband: the blocks are told apart, but the video
    is far from the breaks of the training data and keeps f around a half."""
    recording = HEADBAND / "Participant01_Lab1"
    assert session_config(config, recording).protocol == config.protocol[:8]  # without free play
    session, mean_f, auc = finished(tmp_path, config, recording)
    assert session.result.reason == REASON_END_OF_STREAM
    assert len(session.ticks) == (480 * FS - CONTEXT) // H + 1
    assert session.ticks[0]["f"] is not None  # no calibration
    assert mean_f == pytest.approx({c.REST: 0.536, c.FOCUS: 0.893}, abs=0.005)
    assert auc == pytest.approx(0.831, abs=0.003)
    assert session.source.electrode_rows == {
        "TP9": 0, "AF7": 1, "AF8": 2, "TP10": 3, "AUX_RIGHT": 4,
    }  # fmt: skip


def test_session_over_an_n_back_recording_follows_the_blocks(tmp_path, config):
    """A person of the training data, so the numbers are not those of a new person (the
    cross-validation in tests/training gives those)."""
    session, mean_f, auc = finished(tmp_path, config, N_BACK / "Experiment_1_A5")
    assert session.result.reason == REASON_END_OF_STREAM
    assert mean_f == pytest.approx({c.REST: 0.068, c.FOCUS: 0.876}, abs=0.005)
    assert auc == pytest.approx(0.972, abs=0.003)
    assert session.source.electrode_rows == {
        "TP9": 0, "AF7": 1, "AF8": 2, "TP10": 3, "Right AUX": 4,
    }  # fmt: skip
