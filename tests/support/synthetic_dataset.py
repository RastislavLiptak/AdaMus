"""A synthetic EEG signal with a known ground truth, and recordings made of it.

The signal is the typical person of the data the focus detector was trained on
(a Muse during an n-back task and its breaks): in every frequency band each
electrode has the median power of those people at rest, and in FOCUS blocks the
median power they had during the task. In every band the electrodes are
correlated as theirs were. The recording in the data folder of the project (dataset "synthetic") is
this signal for the default protocol.
"""

from pathlib import Path

import numpy as np
from brainflow.board_shim import BoardShim

from adamus import constants as c
from adamus import datasets
from adamus.config import Block, Config
from adamus.errors import AdamusError

Protocol = list[tuple[str, float]]  # (label, seconds) of every block
TIME_ORIGIN = 1.7e9  # time of the first sample
PACKAGE_MODULO = 256  # the package number of a board counts modulo this
SEED = 0
#: The bands the powers are given for: the eye band, then the detector's bands.
BANDS_HZ = (c.EYE_BAND_HZ, *c.BANDS_HZ)
#: Median power at rest [uV^2] per band (rows) of AF7, AF8, TP9, TP10 (columns).
REST_POWER_UV2 = np.array([
    [17.0, 19.9, 334.0, 343.0],
    [21.9, 17.3, 414.2, 404.6],
    [13.2, 11.6, 238.9, 249.8],
    [2.1, 2.0, 23.3, 24.8],
    [2.3, 2.1, 17.2, 15.7],
    [8.0, 8.4, 13.6, 13.5],
    [19.3, 19.9, 15.8, 16.9],
    [19.3, 18.8, 15.6, 15.5],
])  # fmt: skip
#: Median power during the task, as a share of that at rest.
FOCUS_SHARE = np.array([
    [0.88, 0.49, 1.30, 1.17],
    [0.58, 0.54, 1.19, 1.23],
    [0.62, 0.47, 0.83, 0.75],
    [0.70, 0.72, 0.64, 0.61],
    [0.80, 0.73, 0.61, 0.70],
    [0.80, 0.80, 0.91, 0.72],
    [1.18, 1.10, 0.85, 0.78],
    [1.09, 0.94, 0.85, 0.70],
])  # fmt: skip
#: Median correlation of the electrodes at rest per band, as the pairs AF7-AF8, AF7-TP9,
#: AF7-TP10, AF8-TP9, AF8-TP10, TP9-TP10.
CORRELATION_PAIRS = np.array([
    [-0.32, -0.32, -0.46, -0.14, 0.10, 0.96],
    [-0.21, -0.47, -0.58, -0.14, 0.04, 0.98],
    [-0.24, -0.37, -0.46, -0.11, 0.04, 0.97],
    [-0.18, 0.11, -0.12, -0.05, 0.17, 0.87],
    [-0.14, 0.20, -0.01, -0.01, 0.22, 0.78],
    [-0.02, 0.17, 0.05, 0.00, 0.17, 0.66],
    [0.01, 0.13, 0.09, 0.10, 0.13, 0.62],
    [0.04, 0.19, 0.11, 0.11, 0.15, 0.67],
])  # fmt: skip
SHRINK = 0.02  # towards no correlation: a median of correlations need not be a correlation
POWER_OF = {c.REST: REST_POWER_UV2, c.FOCUS: REST_POWER_UV2 * FOCUS_SHARE}


def correlation(band: int) -> np.ndarray:
    """The correlation matrix of a band, made positive definite."""
    electrodes = REST_POWER_UV2.shape[1]
    matrix = np.eye(electrodes)
    rows, columns = np.triu_indices(electrodes, 1)
    matrix[rows, columns] = matrix[columns, rows] = CORRELATION_PAIRS[band]
    return (1 - SHRINK) * matrix + SHRINK * np.eye(electrodes)


def band_mask(band: int, hz: np.ndarray) -> np.ndarray:
    """The frequencies of a band; above the highest band its density goes on."""
    low, high = BANDS_HZ[band]
    return (hz >= low) & ((hz < high) | (band == len(BANDS_HZ) - 1))


def block_signal(label: str, samples: int, fs: int, noise: np.random.Generator) -> np.ndarray:
    """One block [electrodes x samples]: in every band its own noise, correlated as the
    electrodes are in that band and shaped to the band's power.

    White noise of unit variance shaped by a constant a over a band of width w
    gets the power a^2 * w / (fs / 2) in it. Nothing lies below the lowest band.
    """
    hz = np.fft.rfftfreq(samples, 1 / fs)
    total = np.zeros((REST_POWER_UV2.shape[1], hz.size), dtype=complex)
    for band, ((low, high), power) in enumerate(zip(BANDS_HZ, POWER_OF[label], strict=True)):
        mixing = np.linalg.cholesky(correlation(band))
        white = np.fft.rfft(mixing @ noise.normal(0.0, 1.0, (len(mixing), samples)), axis=1)
        amplitude = np.sqrt(power * (fs / 2) / (high - low))[:, None]
        total += white * amplitude * band_mask(band, hz)
    result: np.ndarray = np.fft.irfft(total, samples, axis=1)
    return result


def signals(protocol: list[tuple[str, float]], fs: int) -> np.ndarray:
    """Signals [electrodes x n] of the detector's electrodes for the timed blocks of a protocol.

    Each block draws its own noise. A block of duration 0 (free play) adds nothing.
    """
    noise = np.random.default_rng(SEED)
    blocks, start, elapsed_s = [], 0, 0.0
    for label, seconds in protocol:
        if seconds > 0 and label not in POWER_OF:
            raise AdamusError(f"there is no synthetic signal for a timed {label} block")
        elapsed_s += seconds
        samples = round(fs * elapsed_s) - start  # the block boundaries of a session
        start += samples
        if samples > 0:
            blocks.append(block_signal(label, samples, fs, noise))
    return np.hstack(blocks) if blocks else np.zeros((REST_POWER_UV2.shape[1], 0))


def board_matrix(signals: np.ndarray, config: Config) -> np.ndarray:
    """What the configured headset would deliver: the signals in the rows of its electrodes.

    The signals [electrodes x n] are those of the detector's electrodes. All other
    rows stay zero, the package number counts modulo 256 and the time advances
    by one sample period.
    """
    names = detector_electrodes_of(signals)
    board = BoardShim.get_board_descr(config.board.id)
    rows = dict(zip(board["eeg_names"].split(","), board["eeg_channels"], strict=True))
    index = np.arange(signals.shape[1])
    matrix = np.zeros((board["num_rows"], signals.shape[1]))
    matrix[board["package_num_channel"]] = index % PACKAGE_MODULO
    matrix[board["timestamp_channel"]] = TIME_ORIGIN + index / board["sampling_rate"]
    matrix[[rows[name] for name in names]] = signals
    return matrix


def write_recording(folder: Path, signals: np.ndarray, protocol: Protocol) -> Path:
    """A recording folder of the signals of the detector's electrodes and their protocol,
    written as the program writes every recording of a dataset."""
    names = detector_electrodes_of(signals)
    blocks = [Block(label, seconds) for label, seconds in protocol]
    return datasets.write_recording(Path(folder), names, signals, blocks)


def detector_electrodes_of(signals: np.ndarray) -> list[str]:
    """The names of the rows of ``signals``: the electrodes of the detector, one per row."""
    names = list(c.DETECTOR_ELECTRODES)
    if signals.ndim != 2 or signals.shape[0] != len(names):
        raise AdamusError(f"the signals of the {len(names)} electrodes of the detector are needed")
    return names
