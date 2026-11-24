"""Signal processing: the latest EEG in, the validity and the covariances of one step out.

The electrodes are those of the detector, in its order; one a source lacks is
all zeros and left out of the covariances. A step is computed from the last
window; its filters run over the windows before it as well, so that their start
has died away by the window. Every filter is causal: a step knows nothing of the
samples after it.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

from adamus import constants as c


@dataclass(frozen=True)
class Filters:
    """The filters of one sampling rate, made once per session."""

    broadband: np.ndarray  # second-order sections: mains notches, then the band-pass
    bands: list[np.ndarray]  # the same for every band of the detector, then the eye band
    delays: list[int]  # of the delayed copies, in samples


@dataclass(frozen=True)
class StepSignal:
    valid: bool
    max_abs_uv: np.ndarray  # per electrode, broadband, in the window; NaN where no number came
    noise_uv: np.ndarray  # per electrode, RMS in the highest band (muscle, no contact)
    carries_signal: np.ndarray  # per electrode: numbers only, and not flat over the context
    covariances: list[np.ndarray]  # of the present electrodes: every band, then the delayed copies
    filtered: np.ndarray  # the broadband window [electrodes x samples]


def samples_of(seconds: float, fs: int) -> int:
    """A duration in samples; the durations of the detector are whole samples at its rate."""
    return round(seconds * fs)


@dataclass(frozen=True)
class Steps:
    """Where the steps of the detector lie in a stream of samples: one every hop, the
    first where the context of its filters has arrived whole."""

    window: int  # a step is computed from this many samples
    hop: int  # between two steps
    context: int  # the samples its filters run over, the window last

    @classmethod
    def at(cls, fs: int) -> "Steps":
        window = samples_of(c.WINDOW_S, fs)
        return cls(window, samples_of(c.HOP_S, fs), c.CONTEXT_WINDOWS * window)

    def end(self, k: int) -> int:
        """The sample after the last of step k."""
        return self.context + k * self.hop

    def count(self, samples: int) -> int:
        """The steps whose whole context lies in the first ``samples`` samples."""
        return max(0, (samples - self.context) // self.hop + 1)


def design_filters(fs: int) -> Filters:
    notches = np.vstack(
        [np.asarray(tf2sos(*iirnotch(hz, c.NOTCH_QUALITY, fs))) for hz in c.MAINS_HZ]
    )

    def band_pass(band: tuple[float, float], order: int) -> np.ndarray:
        sections = np.asarray(butter(order, band, btype="bandpass", fs=fs, output="sos"))
        return np.vstack([notches, sections])

    bands = [band_pass(band, c.FILTER_ORDER) for band in c.BANDS_HZ]
    bands.append(band_pass(c.EYE_BAND_HZ, c.EYE_FILTER_ORDER))
    delays = [samples_of(seconds, fs) for seconds in c.DELAYS_S]
    return Filters(band_pass(c.BROADBAND_HZ, c.FILTER_ORDER), bands, delays)


def present_rows(electrodes: tuple[str, ...]) -> list[int]:
    """Rows of the given electrodes among the detector's."""
    return [c.DETECTOR_ELECTRODES.index(name) for name in electrodes]


def artifact_rows(electrodes: tuple[str, ...]) -> list[int]:
    """Rows checked for artifacts: the artifact electrodes a source has, or all it has."""
    named = tuple(name for name in electrodes if name in c.ARTIFACT_ELECTRODES)
    return present_rows(named or electrodes)


def clean(signal: "StepSignal", electrodes: tuple[str, ...]) -> bool:
    """Every present electrode carries a signal; the artifact rows stay within both limits."""
    one = np.newaxis  # one step
    return bool(
        clean_steps(
            signal.max_abs_uv[one], signal.noise_uv[one], signal.carries_signal[one], electrodes
        )[0]
    )


def clean_steps(
    max_abs_uv: np.ndarray, noise_uv: np.ndarray, carries_signal: np.ndarray,
    electrodes: tuple[str, ...],
) -> np.ndarray:  # fmt: skip
    """``clean`` of many steps at once; every argument but ``electrodes`` is [steps x
    electrodes of the detector]."""
    rows = artifact_rows(electrodes)
    carries = carries_signal[:, present_rows(electrodes)].all(axis=1)
    quiet = (noise_uv[:, rows] <= c.NOISE_UV).all(axis=1)
    small = (max_abs_uv[:, rows] <= c.ARTIFACT_UV).all(axis=1)
    return np.asarray(carries & quiet & small)


def process(
    context: np.ndarray, window: int, filters: Filters, electrodes: tuple[str, ...]
) -> StepSignal:
    """One step from ``context`` [electrodes x samples]; its last ``window`` samples are the window.

    ``electrodes`` are the ones the source has; the step is valid when they are clean.
    An electrode that delivered a NaN or an infinity carries no signal: it is filtered as
    zeros, so that the others are computed as usual, and its window is NaN. Neither does
    one whose power is too large to be a number.
    The function keeps no state and leaves ``context`` untouched.
    """
    assert context.ndim == 2 and context.shape[0] == len(c.DETECTOR_ELECTRODES)
    assert window + max(filters.delays) <= context.shape[1], "too little signal before the window"
    present = present_rows(electrodes)
    numbers = np.asarray(np.isfinite(context).all(axis=1))  # per electrode: only numbers
    x = np.where(numbers[:, None], context, 0.0)
    x = x - x.mean(axis=1, keepdims=True)
    broadband: np.ndarray = np.asarray(sosfilt(filters.broadband, x, axis=1))
    filtered = np.where(numbers[:, None], broadband[:, -window:], np.nan)
    max_abs_uv = np.abs(filtered).max(axis=1)
    carries_signal = numbers & (np.ptp(x, axis=1) > 0)

    covariances = []
    with np.errstate(over="ignore", invalid="ignore"):  # too large a sample: judged below
        for sections in filters.bands:
            part = np.asarray(sosfilt(sections, x[present], axis=1))[:, -window:]
            covariances.append(part @ part.T / window)
        noise_uv = np.zeros(len(c.DETECTOR_ELECTRODES))
        noise_uv[present] = np.sqrt(np.diag(covariances[c.BANDS_HZ.index(c.NOISE_BAND_HZ)]))
        end = broadband.shape[1]
        delayed = np.vstack(
            [broadband[present, end - window - delay : end - delay] for delay in filters.delays]
        )
        covariances.append(delayed @ delayed.T / window)
    # An electrode whose power is too large for a number carries no signal either.
    powers = np.vstack([np.diag(m).reshape(-1, len(present)) for m in covariances])
    carries_signal[present] &= np.isfinite(powers).all(axis=0)
    step = StepSignal(False, max_abs_uv, noise_uv, carries_signal, covariances, filtered)
    return replace(step, valid=clean(step, electrodes))
