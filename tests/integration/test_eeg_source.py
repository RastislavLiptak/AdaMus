"""realtime.eeg_source: the headset as BrainFlow describes it, and the player of recordings."""

# A failure of the headset is made in the BrainFlow board the source wraps, which nothing but
# the source itself holds.
# pyright: reportPrivateUsage=false

import time

import numpy as np
import pytest
from brainflow.data_filter import DataFilter

from adamus.config import Board
from adamus.errors import AdamusError
from adamus.realtime import eeg_source
from adamus.realtime.eeg_source import (
    HeadsetSource,
    RecordingSource,
    electrodes_of_headset,
    headset_is_available,
)
from tests.support.finished_sessions import FS, synthetic_signals
from tests.support.stand_in_devices import StandInClock
from tests.support.synthetic_dataset import write_recording

MUSE = Board("Muse 2", 38, "")


class TestHeadset:
    def test_rate_and_rows_come_from_the_board_description(self):
        source = HeadsetSource(MUSE)
        assert (source.board_id, source.fs) == (38, 256)
        assert source.electrode_rows == {"TP9": 1, "AF7": 2, "AF8": 3, "TP10": 4}
        assert (source.timestamp_row, source.marker_row) == (6, 7)
        assert source.playback_file is None

    def test_another_board_has_its_own_rate_and_rows(self):
        cyton = HeadsetSource(Board("Cyton", 6, ""))
        assert (cyton.fs, cyton.timestamp_row, cyton.marker_row) == (1000, 30, 31)

    def test_board_whose_electrodes_brainflow_does_not_name_has_them_named_by_their_place(self):
        """None to choose for the detector, but every row is recorded and shown."""
        cyton = Board("Cyton", 6, "")
        assert electrodes_of_headset(cyton) == []
        rows = HeadsetSource(cyton).electrode_rows
        assert list(rows) == [f"EEG {place}" for place in range(1, len(rows) + 1)]
        assert len(set(rows.values())) == len(rows) > 0

    def test_electrodes_of_a_headset_are_those_brainflow_names(self):
        assert electrodes_of_headset(MUSE) == ["TP9", "AF7", "AF8", "TP10"]

    def test_unknown_board(self):
        with pytest.raises(AdamusError) as error:
            HeadsetSource(Board("Unknown", 123456, ""))
        assert str(error.value).startswith("unknown board id 123456: UNSUPPORTED_BOARD_ERROR")
        with pytest.raises(AdamusError, match="^unknown board id 123456"):
            electrodes_of_headset(Board("Unknown", 123456, ""))

    def test_headset_is_opened_by_its_serial_number(self):
        parameters = eeg_source._open_parameters(Board("Muse 2", 38, "Muse-ABCD"))
        assert (parameters.serial_number, parameters.file) == ("Muse-ABCD", "")
        assert HeadsetSource(MUSE)._shim.board_id == 38

    def test_stopping_a_headset_that_never_started_is_harmless(self):
        source = HeadsetSource(MUSE)
        source.stop()
        source.close()

    def test_probe_of_a_headset_opens_and_releases_it(self, monkeypatch, capsys):
        calls = []

        class Shim:
            def __init__(self, board_id, parameters):
                calls.append(("open", board_id, parameters.serial_number))

            def prepare_session(self):
                calls.append("prepare")

            def release_session(self):
                calls.append("release")

        monkeypatch.setattr(eeg_source, "BoardShim", Shim)
        monkeypatch.setattr(Shim, "disable_board_logger", staticmethod(lambda: None), False)
        assert headset_is_available(Board("Muse 2", 38, "Muse-ABCD")) is True
        assert calls == [("open", 38, "Muse-ABCD"), "prepare", "release"]
        # said first, for the app to show should macOS end the program over Bluetooth
        assert capsys.readouterr().err == "Looking for Muse 2 over Bluetooth.\n"

    def test_headset_that_cannot_be_reached_is_not_available(self, monkeypatch):
        class Unreachable:
            def __init__(self, board_id, parameters):
                pass

            def prepare_session(self):
                raise eeg_source.BrainFlowError("BOARD_NOT_READY_ERROR", 7)

        monkeypatch.setattr(eeg_source, "BoardShim", Unreachable)
        monkeypatch.setattr(Unreachable, "disable_board_logger", staticmethod(lambda: None), False)
        assert headset_is_available(MUSE) is False


SYNTHETIC_BOARD = Board("BrainFlow's synthetic board", -1, "")


def test_brainflow_prints_nothing_of_its_own(capfd):
    """Its own log ("[board_logger] [info] ...") would fill the last lines the app shows when
    the program ends; only what the program says before Bluetooth is there."""
    assert headset_is_available(SYNTHETIC_BOARD) is True
    assert capfd.readouterr().err == f"Looking for {SYNTHETIC_BOARD.name} over Bluetooth.\n"


def test_opening_the_headset_is_said_first_as_it_goes_over_bluetooth(tmp_path, capfd):
    source = HeadsetSource(SYNTHETIC_BOARD)
    source.start(tmp_path / "raw.csv")
    try:
        assert capfd.readouterr().err == f"Opening {SYNTHETIC_BOARD.name} over Bluetooth.\n"
    finally:
        source.stop()
        source.close()


@pytest.fixture(scope="module")
def streamed(tmp_path_factory):
    """A second of the stream of a headset BrainFlow makes up, with a marker in the middle."""
    raw = tmp_path_factory.mktemp("headset") / "raw.csv"
    source = HeadsetSource(SYNTHETIC_BOARD)
    source.start(raw)
    time.sleep(0.5)
    first = source.read()
    source.insert_marker(2)
    time.sleep(0.5)
    second = source.read()
    source.stop()
    after_stop = source.read()
    source.close()
    return {"source": source, "raw": raw, "chunks": [first, second, after_stop]}


class TestHeadsetStream:
    """The code a live headset runs through, on the virtual board of BrainFlow."""

    def test_samples_arrive_at_the_rate_of_the_board(self, streamed):
        source, (first, second, _after_stop) = streamed["source"], streamed["chunks"]
        assert first.shape[0] == second.shape[0] == source.marker_row + 1
        assert first.shape[1] == pytest.approx(0.5 * source.fs, abs=0.2 * source.fs)
        assert second.shape[1] == pytest.approx(0.5 * source.fs, abs=0.2 * source.fs)
        times = np.hstack([first, second])[source.timestamp_row]
        assert np.all(np.diff(times) >= 0) and times[-1] - times[0] == pytest.approx(1.0, abs=0.3)

    def test_named_electrodes_carry_a_signal(self, streamed):
        source, (first, _second, _after_stop) = streamed["source"], streamed["chunks"]
        assert len(source.electrode_rows) == len(set(source.electrode_rows.values())) > 0
        assert all(first[row].std() > 0 for row in source.electrode_rows.values())

    def test_inserted_marker_lands_on_a_sample_that_follows(self, streamed):
        source, (first, second, _after_stop) = streamed["source"], streamed["chunks"]
        assert not first[source.marker_row].any()
        marked = second[source.marker_row]
        assert marked[np.flatnonzero(marked)].tolist() == [2.0]

    def test_brainflow_writes_what_was_delivered_to_the_raw_file(self, streamed):
        delivered = np.hstack(streamed["chunks"])
        written = DataFilter.read_file(str(streamed["raw"]))
        assert written is not None and written.shape == delivered.shape
        assert np.allclose(written, delivered, atol=1e-4)

    def test_headset_that_does_not_answer_cannot_be_started(self, monkeypatch, tmp_path):
        source = HeadsetSource(MUSE)

        def no_answer():
            raise eeg_source.BrainFlowError("BOARD_NOT_READY_ERROR", 7)

        monkeypatch.setattr(source._shim, "prepare_session", no_answer)
        with pytest.raises(AdamusError) as error:
            source.start(tmp_path / "raw.csv")
        assert str(error.value).startswith("cannot start the EEG stream: BOARD_NOT_READY_ERROR")
        source.close()  # nothing left open

    def test_stream_that_cannot_be_stopped_is_only_noted(self, monkeypatch, caplog):
        source = HeadsetSource(MUSE)
        with caplog.at_level("WARNING", logger="adamus"):
            source.stop()  # it never started
        assert "stopping the stream failed: " in caplog.text

    def test_a_session_that_cannot_be_released_is_only_noted(self, monkeypatch, caplog):
        """close() runs while a session ends: a lost Bluetooth link must not end the program."""
        source = HeadsetSource(MUSE)

        def link_lost():
            raise eeg_source.BrainFlowError("BOARD_NOT_CREATED_ERROR", 15)

        monkeypatch.setattr(source._shim, "is_prepared", lambda: True)
        monkeypatch.setattr(source._shim, "release_session", link_lost)
        with caplog.at_level("WARNING", logger="adamus"):
            source.close()
        assert "releasing the headset failed: BOARD_NOT_CREATED_ERROR" in caplog.text


# --- The player of recordings ----------------------------------------------------

SECONDS = 3
NAMES = ["TP9", "AF7", "AF8", "TP10", "AUX"]


def write_table(path, names, samples):
    """A recording as the datasets hold it: a line of names, then one line per sample."""
    np.savetxt(path, samples.T, fmt="%.3f", delimiter=",", header=",".join(names), comments="")
    return path


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """Three seconds of five channels, each with its own values."""
    samples = np.arange(5 * SECONDS * FS).reshape(5, SECONDS * FS) / 8
    folder = tmp_path_factory.mktemp("recording")
    write_table(folder / "eeg.csv", NAMES, samples)
    return folder, samples


@pytest.fixture
def playing(recording, tmp_path):
    """The recording being played on a clock the test moves."""
    path, samples = recording
    clock = StandInClock()
    source = RecordingSource(MUSE, path, clock)
    source.start(tmp_path / "raw.csv")
    yield source, clock, samples, tmp_path / "raw.csv"
    source.close()


class TestRecording:
    def test_description(self, playing, recording):
        source = playing[0]
        assert (source.board_id, source.fs) == (38, 256)  # played at the rate of the headset
        assert source.playback_file == recording[0] / "eeg.csv"

    def test_every_column_is_an_electrode_followed_by_time_and_marker(self, playing):
        source = playing[0]
        assert source.electrode_rows == {"TP9": 0, "AF7": 1, "AF8": 2, "TP10": 3, "AUX": 4}
        assert (source.timestamp_row, source.marker_row) == (5, 6)

    def test_nothing_is_due_at_the_start(self, playing):
        source, _clock, _samples, _raw = playing
        assert source.read().shape == (7, 0)

    def test_samples_become_due_at_the_sampling_rate(self, playing):
        source, clock, samples, _raw = playing
        clock.sleep(0.5)
        first = source.read()
        assert first.shape == (7, 128) and first.dtype == np.float64
        assert np.array_equal(first[:5], samples[:, :128])
        clock.sleep(0.25)
        second = source.read()
        assert np.array_equal(second[:5], samples[:, 128:192])
        assert source.read().shape == (7, 0)  # nothing twice

    def test_sample_k_carries_the_time_it_was_due(self, playing):
        source, clock, _samples, _raw = playing
        started = clock.time
        clock.sleep(1.0)
        times = source.read()[source.timestamp_row]
        assert times == pytest.approx(started + np.arange(256) / 256, abs=1e-5)

    def test_recording_ends_with_its_last_sample(self, playing):
        source, clock, _samples, _raw = playing
        clock.sleep(SECONDS + 10)
        assert source.read().shape == (7, SECONDS * FS)
        clock.sleep(10)
        assert source.read().shape == (7, 0)
        source.stop()
        assert source.read().shape == (7, 0)

    def test_what_is_due_can_still_be_read_after_stop(self, playing):
        source, clock, _samples, _raw = playing
        clock.sleep(0.25)
        source.stop()
        assert source.read().shape == (7, 64)

    def test_inserted_marker_lands_on_the_next_sample(self, playing):
        source, clock, _samples, _raw = playing
        clock.sleep(0.25)
        source.read()
        source.insert_marker(2)
        assert source.read().shape == (7, 0)  # the marker waits for a sample
        clock.sleep(0.25)
        markers = source.read()[source.marker_row]
        assert (markers[0], markers[1:].any()) == (2.0, False)
        clock.sleep(0.25)
        assert not source.read()[source.marker_row].any()  # once

    def test_no_sample_carries_a_marker_unless_one_was_inserted(self, playing):
        source, clock, _samples, _raw = playing
        clock.sleep(0.25)
        assert not source.read()[source.marker_row].any()

    def test_marker_lands_on_a_sample_that_arrives_alone(self, playing):
        source, clock, _samples, _raw = playing
        source.insert_marker(2)
        clock.sleep(1.5 / FS)  # one sample is due, the second not yet
        alone = source.read()
        assert alone.shape == (7, 1) and alone[source.marker_row, 0] == 2.0

    def test_everything_delivered_is_in_the_raw_file_as_brainflow_would_write_it(self, playing):
        """All electrodes, the times and the markers: one line per sample, tab-separated."""
        source, clock, _samples, raw = playing
        delivered = []
        for marker in (1, 0, 2):
            source.insert_marker(marker)
            clock.sleep(0.5)
            delivered.append(source.read())
        source.close()
        delivered = np.hstack(delivered)
        written = DataFilter.read_file(str(raw))
        assert written is not None and written.shape == delivered.shape == (7, 384)
        assert np.allclose(written[:5], delivered[:5], atol=0.0005)
        assert np.allclose(written[5], delivered[5], atol=1e-6)
        assert np.array_equal(written[6], delivered[6])
        assert written[6, [0, 128, 256]].tolist() == [1.0, 0.0, 2.0]

    def test_closing_twice_is_harmless(self, playing):
        playing[0].close()
        playing[0].close()

    def test_recording_of_the_test_support_is_played_as_it_was_written(self, tmp_path):
        signals = synthetic_signals([["REST", SECONDS]])
        folder = write_recording(tmp_path / "in", signals, [("REST", SECONDS)])
        clock = StandInClock()
        source = RecordingSource(MUSE, folder, clock)
        assert source.electrode_rows == {"AF7": 0, "AF8": 1, "TP9": 2, "TP10": 3}
        source.start(tmp_path / "raw.csv")
        clock.sleep(SECONDS)
        assert np.allclose(source.read()[:4], signals, atol=0.0005)
        source.close()


class TestRecordingsThatCannotBePlayed:
    """How a recording is read is tested with datasets.read_recording; the player says why."""

    def test_missing_file(self, tmp_path):
        with pytest.raises(AdamusError) as error:
            RecordingSource(MUSE, tmp_path / "missing", StandInClock())
        assert str(error.value).startswith(f"cannot read {tmp_path / 'missing' / 'eeg.csv'}: ")

    def test_lines_that_do_not_fit_the_names(self, tmp_path):
        path = tmp_path / "eeg.csv"
        path.write_text("AF7,AF8,TP9\n1,2\n3,4\n")
        with pytest.raises(AdamusError) as error:
            RecordingSource(MUSE, tmp_path, StandInClock())
        assert str(error.value) == f"{path}: every line must hold 3 values"
