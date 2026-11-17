"""signal_processing: the filters, the validity and the covariances of one step."""

import numpy as np
import pytest
from scipy.signal import sosfreqz

from adamus import constants as c
from adamus.detection.signal_processing import (
    Filters,
    Steps,
    StepSignal,
    artifact_rows,
    design_filters,
    present_rows,
    samples_of,
)
from adamus.detection.signal_processing import process as process_some
from tests.support.finished_sessions import CONTEXT, FS, N

FILTERS = design_filters(FS)
AF7, AF8, TP9, TP10 = range(4)
EVERY = c.DETECTOR_ELECTRODES


def process(context, window, filters, electrodes=EVERY):
    return process_some(context, window, filters, electrodes)


def tones(amplitudes_uv, hz=10.0, noise_uv=2.0, seed=0, length=CONTEXT):
    """A sine of ``hz`` on every electrode with its own amplitude, plus noise."""
    t = np.arange(length) / FS
    noise = np.random.default_rng(seed)
    return np.vstack(
        [a * np.sin(2 * np.pi * hz * t + 0.5 * k) + noise.normal(0, noise_uv, length)
         for k, a in enumerate(amplitudes_uv)]
    )  # fmt: skip


QUIET = tones([10.0, 10.0, 10.0, 10.0])


def gain(sections, hz):
    _, response = sosfreqz(sections, worN=[hz], fs=FS)
    return float(abs(response[0]))


class TestDurations:
    @pytest.mark.parametrize(
        ("seconds", "fs", "samples"),
        [(2.0, 256, 512), (0.5, 256, 128), (0.025, 256, 6), (0.05, 256, 13), (2.0, 1000, 2000)],
    )
    def test_seconds_in_samples(self, seconds, fs, samples):
        assert samples_of(seconds, fs) == samples


class TestSteps:
    """Where the steps lie in a stream: a step every hop, from the first whole context."""

    def test_window_hop_and_context_in_samples_of_the_rate(self):
        assert Steps.at(256) == Steps(window=512, hop=128, context=1024)
        assert Steps.at(1000) == Steps(window=2000, hop=500, context=4000)

    def test_step_k_ends_k_hops_after_the_first_whole_context(self):
        steps = Steps.at(256)
        assert [steps.end(k) for k in range(3)] == [1024, 1152, 1280]

    @pytest.mark.parametrize(
        ("samples", "count"), [(0, 0), (1023, 0), (1024, 1), (1151, 1), (1152, 2), (2048, 9)]
    )
    def test_steps_whose_whole_context_has_arrived(self, samples, count):
        steps = Steps.at(256)
        assert steps.count(samples) == count
        assert all(steps.end(k) <= samples for k in range(count))
        assert steps.end(count) > samples


class TestFilters:
    def test_one_filter_per_band_then_the_eye_band_and_the_delays_in_samples(self):
        assert isinstance(FILTERS, Filters)
        assert len(FILTERS.bands) == len(c.BANDS_HZ) + 1 == 8
        assert FILTERS.delays == [0, 6, 13]

    def test_eye_band_passes_blinks_and_stops_the_rhythms(self):
        eye = FILTERS.bands[-1]
        assert gain(eye, 1.0) > 0.9
        assert gain(eye, 0.1) < 0.1 and gain(eye, 6.0) < 0.1

    @pytest.mark.parametrize("hz", [50.0, 60.0])
    def test_mains_hum_is_removed_in_every_filter(self, hz):
        assert gain(FILTERS.broadband, hz) < 0.01
        assert all(gain(sections, hz) < 0.01 for sections in FILTERS.bands)

    @pytest.mark.parametrize("hz", [5.0, 10.0, 20.0, 30.0])
    def test_broadband_passes_the_eeg(self, hz):
        assert gain(FILTERS.broadband, hz) > 0.9

    @pytest.mark.parametrize("hz", [0.3, 70.0, 120.0])
    def test_broadband_stops_drift_and_high_frequencies(self, hz):
        assert gain(FILTERS.broadband, hz) < 0.1

    @pytest.mark.parametrize(("band", "inside", "outside"), [
        (7, 1.0, 4.0),
        (0, 3.0, 8.0), (1, 6.0, 12.0), (2, 9.0, 13.0), (3, 11.5, 8.0),
        (4, 16.0, 10.0), (5, 25.0, 15.0), (6, 37.0, 25.0),
    ])  # fmt: skip
    def test_each_band_passes_its_centre_and_stops_its_neighbours(self, band, inside, outside):
        assert gain(FILTERS.bands[band], inside) > 0.7
        assert gain(FILTERS.bands[band], outside) < 0.3

    def test_filters_follow_the_sampling_rate(self):
        other = design_filters(512)
        assert other.delays == [0, 13, 26]
        assert abs(gain(other.broadband, 10.0) - gain(FILTERS.broadband, 10.0)) < 0.01


class TestResult:
    def test_types_and_shapes(self):
        result = process(QUIET, N, FILTERS)
        assert isinstance(result, StepSignal)
        assert type(result.valid) is bool and result.valid
        assert result.max_abs_uv.shape == (4,)
        assert result.filtered.shape == (4, N)
        shapes = [cov.shape for cov in result.covariances]
        assert shapes == [(4, 4)] * 8 + [(12, 12)]
        assert result.noise_uv.shape == result.carries_signal.shape == (4,)

    def test_covariances_are_symmetric_and_positive_definite(self):
        for cov in process(QUIET, N, FILTERS).covariances:
            assert np.allclose(cov, cov.T)
            assert np.linalg.eigvalsh(cov).min() > 0

    def test_input_is_left_untouched(self):
        before = QUIET.copy()
        process(QUIET, N, FILTERS)
        assert np.array_equal(QUIET, before)

    def test_function_keeps_no_state(self):
        first = process(QUIET, N, FILTERS)
        process(tones([50.0, 1.0, 1.0, 1.0], seed=3), N, FILTERS)
        second = process(QUIET, N, FILTERS)
        assert all(
            np.array_equal(a, b) for a, b in zip(first.covariances, second.covariances, strict=True)
        )

    def test_a_constant_offset_changes_nothing(self):
        """The electrodes of a Muse sit around 800 uV; only the changes are signal."""
        shifted = process(QUIET + 800.0, N, FILTERS)
        plain = process(QUIET, N, FILTERS)
        assert all(
            np.allclose(a, b) for a, b in zip(shifted.covariances, plain.covariances, strict=True)
        )

    def test_context_too_short_for_the_delays_is_refused(self):
        with pytest.raises(AssertionError):
            process(QUIET[:, : N + 12], N, FILTERS)

    def test_other_number_of_electrodes_is_refused(self):
        with pytest.raises(AssertionError):
            process(QUIET[:2], N, FILTERS)


class TestCovariances:
    def test_power_of_a_tone_lies_in_its_band(self):
        """A 10 Hz sine of 20 uV has a power of 200 uV^2: in the 10-13 Hz band, not elsewhere."""
        result = process(tones([20.0, 20.0, 20.0, 20.0], hz=11.0, noise_uv=0.0), N, FILTERS)
        powers = [cov[AF7, AF7] for cov in result.covariances[:-1]]
        assert powers[3] == pytest.approx(200.0, rel=0.1)
        assert max(powers[:2] + powers[5:]) < 2.0  # the eye band too

    def test_band_covariance_is_that_of_the_filtered_window(self):
        from scipy.signal import sosfilt

        x = QUIET - QUIET.mean(axis=1, keepdims=True)
        part = sosfilt(FILTERS.bands[2], x, axis=1)[:, -N:]
        assert np.allclose(process(QUIET, N, FILTERS).covariances[2], part @ part.T / N)

    def test_covariance_of_the_delayed_copies(self):
        """Blocks of 4 x 4: the broadband covariance, then the covariance with its delayed self.

        For a sine of f Hz, x(t) and x(t - d) correlate as cos(2 pi f d)."""
        hz = 20.0
        result = process(tones([20.0, 20.0, 20.0, 20.0], hz=hz, noise_uv=0.0), N, FILTERS)
        delayed = result.covariances[-1]
        broadband = result.filtered @ result.filtered.T / N
        assert np.allclose(delayed[:4, :4], broadband)
        for block, delay in enumerate(FILTERS.delays):
            ratio = delayed[0, 4 * block] / delayed[0, 0]
            assert ratio == pytest.approx(np.cos(2 * np.pi * hz * delay / FS), abs=0.02)

    def test_reference_values_pin_the_whole_chain(self):
        """Any change of a filter, of the order of the steps or of a constant moves these."""
        result = process(tones([10.0, 20.0, 30.0, 40.0], noise_uv=5.0, seed=7), N, FILTERS)
        traces = [float(np.trace(cov)) for cov in result.covariances]
        assert traces == pytest.approx(REFERENCE_TRACES, rel=1e-6)
        assert result.max_abs_uv == pytest.approx(REFERENCE_PEAKS, rel=1e-6)
        assert result.noise_uv == pytest.approx(REFERENCE_NOISE, rel=1e-6)


REFERENCE_TRACES = [
    1.6607016624564657, 25.106391533121986, 761.4280431307263, 758.0697060380364,
    7.104881427461558, 6.795075017842111, 11.166358062870339, 2.308149721060361,
    4655.381600495083,
]  # fmt: skip
REFERENCE_PEAKS = [20.237660053030247, 26.09353142869547, 35.827864161214876, 47.88778679495547]
REFERENCE_NOISE = [1.6191621360164559, 1.4537835496301985, 1.745487500546002, 1.8396898696310904]


class TestValidity:
    def test_clean_signal_is_valid(self):
        assert process(QUIET, N, FILTERS).valid

    @pytest.mark.parametrize("electrode", [AF7, AF8])
    def test_large_signal_on_a_frontal_electrode_makes_the_step_invalid(self, electrode):
        amplitudes = [10.0] * 4
        amplitudes[electrode] = 500.0
        assert not process(tones(amplitudes), N, FILTERS).valid

    def test_blinks_and_eye_movements_are_part_of_the_signal(self):
        """150 uV on the forehead: a blink, which says something about the task."""
        assert process(tones([150.0, 150.0, 10.0, 10.0], hz=1.0), N, FILTERS).valid

    @pytest.mark.parametrize("electrode", [TP9, TP10])
    def test_large_signal_behind_the_ears_does_not(self, electrode):
        """The ears' electrodes of a Muse are often loud; the model was trained with them so."""
        amplitudes = [10.0] * 4
        amplitudes[electrode] = 500.0
        assert process(tones(amplitudes), N, FILTERS).valid

    def test_limit_is_the_artifact_limit_on_the_filtered_window(self):
        unit = tones([1.0, 1.0, 1.0, 1.0], noise_uv=0.0)
        peak = process(unit, N, FILTERS).max_abs_uv[:2].max()
        assert process(unit * (c.ARTIFACT_UV * 0.999 / peak), N, FILTERS).valid
        assert not process(unit * (c.ARTIFACT_UV * 1.001 / peak), N, FILTERS).valid

    def test_signal_exactly_at_the_limit_is_still_valid(self, monkeypatch):
        from adamus import constants

        signal = tones([30.0, 30.0, 10.0, 10.0])
        peak = float(process(signal, N, FILTERS).max_abs_uv[:2].max())
        monkeypatch.setattr(constants, "ARTIFACT_UV", peak)
        assert process(signal, N, FILTERS).valid

    @pytest.mark.parametrize("electrode", [AF7, AF8])
    def test_an_electrode_without_contact_fills_the_noise_band(self, electrode):
        """An open input gives broadband noise of tens of uV; muscle does the same."""
        signal = QUIET.copy()
        signal[electrode] += np.random.default_rng(5).normal(0, 60.0, CONTEXT)
        result = process(signal, N, FILTERS)
        assert result.noise_uv[electrode] > c.NOISE_UV and not result.valid

    def test_noise_is_the_rms_in_the_noise_band(self):
        result = process(tones([0.0, 0.0, 10.0, 10.0], hz=37.0, noise_uv=0.0), N, FILTERS)
        band = c.BANDS_HZ.index(c.NOISE_BAND_HZ)
        assert result.noise_uv == pytest.approx(np.sqrt(np.diag(result.covariances[band])))
        assert result.noise_uv[TP9] == pytest.approx(10.0 / np.sqrt(2), rel=0.1)

    def test_noise_exactly_at_the_limit_is_still_valid(self, monkeypatch):
        from adamus import constants

        signal = tones([30.0, 30.0, 10.0, 10.0], hz=37.0)
        noise = float(process(signal, N, FILTERS).noise_uv[:2].max())
        monkeypatch.setattr(constants, "NOISE_UV", noise)
        assert process(signal, N, FILTERS).valid
        monkeypatch.setattr(constants, "NOISE_UV", noise * 0.999)
        assert not process(signal, N, FILTERS).valid

    def test_noise_behind_the_ears_does_not_count(self):
        signal = QUIET.copy()
        signal[TP9] += np.random.default_rng(5).normal(0, 60.0, CONTEXT)
        assert process(signal, N, FILTERS).valid

    def test_a_very_weak_signal_is_still_a_signal(self):
        """Only a flat electrode carries nothing; a weak one (under 1 uV) does."""
        weak = QUIET.copy()
        weak[TP9] = 0.2 * np.sin(np.arange(CONTEXT))
        assert np.ptp(weak[TP9]) < 1 and process(weak, N, FILTERS).valid

    def test_artifact_before_the_window_does_not_count(self):
        signal = QUIET.copy()
        signal[AF7, : N // 2] += 2000.0 * np.sin(np.arange(N // 2) / 30)
        assert process(signal, N, FILTERS).valid

    @pytest.mark.parametrize("electrode", [AF7, TP10])
    def test_an_electrode_without_signal_makes_the_step_invalid(self, electrode):
        """A flat electrode (all zeros, or a constant) carries nothing to compare."""
        for flat in (0.0, 812.5):
            signal = QUIET.copy()
            signal[electrode] = flat
            result = process(signal, N, FILTERS)
            assert not result.valid and not result.carries_signal[electrode]

    @pytest.mark.parametrize("electrode", [AF7, AF8, TP9, TP10])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_a_sample_that_is_not_a_number_makes_the_step_invalid(self, electrode, value):
        """A recording or a headset may hand over NaN or an infinity: that is no signal."""
        signal = QUIET.copy()
        signal[electrode, N] = value
        result = process(signal, N, FILTERS)
        assert not result.valid and not result.carries_signal[electrode]
        others = [row for row in range(4) if row != electrode]
        assert result.carries_signal[others].all()

    @pytest.mark.parametrize("electrode", [TP9, TP10])
    def test_a_value_too_large_to_compute_with_carries_no_signal(self, electrode):
        """Only a file can hold it (no headset measures 1e170 µV); its power is no number."""
        signal = QUIET.copy()
        signal[electrode, N] = 1e170
        result = process(signal, N, FILTERS)
        assert not result.valid and not result.carries_signal[electrode]
        others = [row for row in range(4) if row != electrode]
        assert result.carries_signal[others].all()


class TestMissingElectrodes:
    """A source without an electrode gives zeros there; it is left out, not judged."""

    @pytest.mark.parametrize("electrodes", [("AF7", "TP9", "TP10"), ("AF8",), ("TP9", "TP10")])
    def test_covariances_are_those_of_the_present_electrodes_only(self, electrodes):
        """That they are those of all electrodes, cut down, is checked against the training's
        own cutting (test_focus_training, TestSubsetCovariances)."""
        rows = present_rows(electrodes)
        signal = QUIET.copy()
        signal[[k for k in range(4) if k not in rows]] = 0.0
        some = process(signal, N, FILTERS, electrodes)
        assert some.valid
        delays = len(c.DELAYS_S)
        shapes = [(len(rows), len(rows))] * (len(some.covariances) - 1)
        assert [m.shape for m in some.covariances] == [*shapes, (delays * len(rows),) * 2]

    def test_the_noise_of_a_missing_electrode_is_zero(self):
        signal = QUIET.copy()
        signal[AF8] = 0.0
        result = process(signal, N, FILTERS, ("AF7", "TP9", "TP10"))
        assert result.noise_uv[AF8] == 0.0 and result.noise_uv[AF7] > 0

    def test_artifacts_are_looked_for_on_the_frontal_electrodes_there_are(self):
        assert artifact_rows(EVERY) == [AF7, AF8]
        assert artifact_rows(("AF8", "TP10")) == [AF8]

    def test_without_a_frontal_electrode_every_present_one_is_checked(self):
        assert artifact_rows(("TP9", "TP10")) == [TP9, TP10]
        loud = tones([10.0, 10.0, 500.0, 10.0])
        loud[[AF7, AF8]] = 0.0
        assert not process(loud, N, FILTERS, ("TP9", "TP10")).valid
        assert process(loud, N, FILTERS, ("TP10",)).valid  # the loud TP9 is not one of them

    def test_present_rows_follow_the_detector_order(self):
        assert present_rows(("AF8", "TP10")) == [AF8, TP10]
