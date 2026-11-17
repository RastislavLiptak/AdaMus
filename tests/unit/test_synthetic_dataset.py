"""The synthetic signal of the tests, and the recording folders made of it."""

import numpy as np
import pytest

from adamus import constants as c
from adamus.config import Block, read_yaml
from adamus.errors import AdamusError
from adamus.protocol import block_spans
from tests.support import synthetic_dataset
from tests.support.finished_sessions import FS, synthetic_signals
from tests.support.synthetic_dataset import board_matrix, write_recording

BLOCK = 60 * FS
REST_BLOCKS, FOCUS_BLOCKS = (0, 2, 4, 6), (1, 3, 5, 7)


def band_correlation(segments, band):
    """Correlation of the electrodes in exactly the frequencies of a band, over the segments."""
    low, high = synthetic_dataset.BANDS_HZ[band]
    cross = 0
    for segment in segments:
        hz = np.fft.rfftfreq(segment.shape[-1], 1 / FS)
        part = np.fft.rfft(segment, axis=-1)[:, (hz >= low) & (hz < high)]
        cross = cross + np.real(part @ part.conj().T)
    scale = np.sqrt(np.diag(cross))
    return cross / np.outer(scale, scale)


def power(segment, band):
    """Power [uV^2] in a band: the spectrum summed over its frequencies."""
    low, high = synthetic_dataset.BANDS_HZ[band]
    spectrum = np.abs(np.fft.rfft(segment, axis=-1)) ** 2 / segment.shape[-1] ** 2 * 2
    hz = np.fft.rfftfreq(segment.shape[-1], 1 / FS)
    return spectrum[..., (hz >= low) & (hz < high)].sum(axis=-1)


def block(signals, index):
    return signals[:, index * BLOCK : (index + 1) * BLOCK]


def mean_power(signals, indices, band):
    return np.mean([power(block(signals, k), band) for k in indices], axis=0)


class TestSynthetic:
    def test_default_recording_is_eight_blocks_of_60_s_on_four_electrodes(
        self, synthetic_default_signals
    ):
        assert synthetic_default_signals.shape == (4, 8 * BLOCK)

    def test_is_deterministic(self, synthetic_default_signals, default_protocol):
        assert np.array_equal(synthetic_signals(default_protocol), synthetic_default_signals)

    @pytest.mark.parametrize("band", range(8))
    def test_rest_has_the_median_power_of_the_training_people(
        self, synthetic_default_signals, band
    ):
        found = mean_power(synthetic_default_signals, REST_BLOCKS, band)
        assert found == pytest.approx(synthetic_dataset.REST_POWER_UV2[band], rel=0.15)

    @pytest.mark.parametrize("band", range(8))
    def test_focus_has_their_power_during_the_task(self, synthetic_default_signals, band):
        found = mean_power(synthetic_default_signals, FOCUS_BLOCKS, band)
        expected = synthetic_dataset.REST_POWER_UV2[band] * synthetic_dataset.FOCUS_SHARE[band]
        assert found == pytest.approx(expected, rel=0.15)

    def test_the_task_lowers_the_slow_rhythms_on_the_forehead(self):
        """What the detector reads: below 13 Hz the frontal power falls during the task."""
        shares = synthetic_dataset.FOCUS_SHARE[1:5, :2]  # 2 to 13 Hz, AF7 and AF8
        assert (shares < 0.85).all()

    @pytest.mark.parametrize("band", [0, 2, 4, 7])
    def test_electrodes_are_correlated_as_theirs_were(self, synthetic_default_signals, band):
        rest = [block(synthetic_default_signals, k) for k in REST_BLOCKS]
        found = band_correlation(rest, band)[np.triu_indices(4, 1)]
        expected = synthetic_dataset.CORRELATION_PAIRS[band] * (1 - synthetic_dataset.SHRINK)
        assert found == pytest.approx(expected, abs=0.1)  # 2.5 standard errors

    def test_every_correlation_is_positive_definite(self):
        for band in range(len(synthetic_dataset.BANDS_HZ)):
            assert np.linalg.eigvalsh(synthetic_dataset.correlation(band)).min() > 0

    def test_bands_are_the_eye_band_and_those_of_the_detector(self):
        assert synthetic_dataset.BANDS_HZ == (c.EYE_BAND_HZ, *c.BANDS_HZ)

    def test_nothing_below_the_eye_band(self, synthetic_default_signals):
        spectrum = np.abs(np.fft.rfft(block(synthetic_default_signals, 0), axis=-1))
        hz = np.fft.rfftfreq(BLOCK, 1 / FS)
        assert spectrum[:, hz < c.EYE_BAND_HZ[0]].max() == pytest.approx(0, abs=1e-6)

    def test_frontal_electrodes_stay_clean(self, synthetic_default_signals):
        assert np.abs(synthetic_default_signals[:2]).max() < c.ARTIFACT_UV / 2

    def test_block_boundaries_are_those_of_a_session(self):
        blocks = [("REST", 0.3), ("FOCUS", 0.3)] * 2
        spans = block_spans([Block(label, seconds) for label, seconds in blocks], FS)
        assert synthetic_signals(blocks).shape[1] == spans[-1].end == 307

    def test_block_of_duration_zero_adds_no_samples(self):
        with_free_play = synthetic_signals([("REST", 2), ("FOCUS", 2), ("FREE", 0)])
        assert np.array_equal(with_free_play, synthetic_signals([("REST", 2), ("FOCUS", 2)]))

    def test_protocol_without_a_timed_block_gives_no_samples(self):
        assert synthetic_dataset.signals([("FREE", 0)], FS).shape == (4, 0)

    def test_timed_free_play_has_no_synthetic_signal(self):
        with pytest.raises(
            AdamusError, match="there is no synthetic signal for a timed FREE block"
        ):
            synthetic_dataset.signals([("FREE", 5)], FS)


class TestBoardMatrix:
    def test_muse_2_layout(self, synthetic_default_signals, config):
        signals = synthetic_default_signals[:, :600]
        matrix = board_matrix(signals, config)
        assert matrix.shape == (8, 600)
        assert np.array_equal(matrix[0], np.arange(600) % 256)  # package number
        # the board's rows are TP9, AF7, AF8, TP10; the signals come in the detector's order
        assert np.array_equal(matrix[[2, 3, 1, 4]], signals)
        assert np.allclose(matrix[6], 1.7e9 + np.arange(600) / 256)  # time
        for row in (5, 7):  # other, marker
            assert not matrix[row].any()

    def test_number_of_signals_must_match_the_electrodes_of_the_detector(self, config, tmp_path):
        with pytest.raises(AdamusError, match="the signals of the 4 electrodes of the detector"):
            board_matrix(np.zeros((3, 10)), config)
        with pytest.raises(AdamusError, match="the signals of the 4 electrodes of the detector"):
            write_recording(tmp_path / "nowhere", np.zeros((3, 10)), [("REST", 1)])
        assert not (tmp_path / "nowhere").exists()  # refused before anything was written


class TestRecordingFolder:
    def test_holds_the_samples_and_their_protocol(
        self, tmp_path, synthetic_default_signals, config
    ):
        signals = synthetic_default_signals[:, :512]
        folder = write_recording(tmp_path / "dataset" / "one", signals, [("REST", 1), ("FOCUS", 1)])
        assert folder == tmp_path / "dataset" / "one"
        assert sorted(path.name for path in folder.iterdir()) == ["eeg.csv", "protocol.yaml"]
        assert read_yaml(folder / "protocol.yaml") == {"protocol": [["REST", 1], ["FOCUS", 1]]}

    def test_samples_are_a_table_under_the_names_of_the_electrodes(
        self, tmp_path, synthetic_default_signals, config
    ):
        """Like every recording of a dataset: microvolts with three decimals, one line a sample."""
        signals = synthetic_default_signals[:, :512]
        folder = write_recording(tmp_path / "one", signals, [("REST", 2)])
        lines = (folder / "eeg.csv").read_text().splitlines()
        assert lines[0] == "AF7,AF8,TP9,TP10" and len(lines) == 1 + 512
        assert lines[1] == ",".join(f"{value:.3f}" for value in signals[:, 0])
        table = np.loadtxt(folder / "eeg.csv", delimiter=",", skiprows=1)
        assert np.abs(table.T - signals).max() <= 0.0005
