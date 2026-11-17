"""The focus detector: from the covariances of one step to a score, with no calibration.

Every covariance of a step becomes the upper triangle of its matrix logarithm.
From these the detector subtracts the person's own running mean: at first the
mean of the people the model was trained on, then more and more the person's
own signal. What remains goes into a logistic regression trained on other
people (adamus.detection.focus_training); its threshold gave the best balanced
accuracy on people the model had not seen. There is a model for every set of
electrodes a source may have, so a missing electrode does not stop the detector.
"""

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from adamus import constants as c
from adamus.errors import AdamusError

RIDGE = 1e-5  # added to a covariance, as a share of its mean variance, so it is never singular
EIGEN_FLOOR = 1e-10  # eigenvalues below this count as this, against rounding
FILE_KEYS = ("features", "trained_on", "models")
MODEL_KEYS = ("electrodes", "prior", "mean", "scale", "coef", "intercept", "threshold")


def features() -> dict[str, Any]:
    """What the detector computes; a model file holds the values it was trained with."""
    return {
        "electrodes": list(c.DETECTOR_ELECTRODES),
        "artifact_electrodes": list(c.ARTIFACT_ELECTRODES),
        "artifact_uv": c.ARTIFACT_UV,
        "noise_band_hz": list(c.NOISE_BAND_HZ),
        "noise_uv": c.NOISE_UV,
        "fs": c.DETECTOR_FS,
        "window_s": c.WINDOW_S,
        "hop_s": c.HOP_S,
        "context_windows": c.CONTEXT_WINDOWS,
        "broadband_hz": list(c.BROADBAND_HZ),
        "bands_hz": [list(band) for band in c.BANDS_HZ],
        "filter_order": c.FILTER_ORDER,
        "eye_band_hz": list(c.EYE_BAND_HZ),
        "eye_filter_order": c.EYE_FILTER_ORDER,
        "mains_hz": list(c.MAINS_HZ),
        "notch_quality": c.NOTCH_QUALITY,
        "delays_s": list(c.DELAYS_S),
        "prior_s": c.PRIOR_S,
        "memory_s": c.MEMORY_S,
    }


def electrode_sets() -> list[tuple[str, ...]]:
    """Every non-empty set of the detector's electrodes, in the detector's order."""
    names = c.DETECTOR_ELECTRODES
    return [subset for size in range(1, len(names) + 1) for subset in combinations(names, size)]


def vector_length(electrodes: int) -> int:
    """One triangle per band and of the eye band, and the triangle of the delayed copies."""
    delayed = electrodes * len(c.DELAYS_S)
    single = electrodes * (electrodes + 1) // 2
    return (len(c.BANDS_HZ) + 1) * single + delayed * (delayed + 1) // 2


def log_vector(covariances: list[np.ndarray]) -> np.ndarray:
    """The upper triangles of the matrix logarithms, the off-diagonal entries times the
    square root of 2 (so that the length is the log-Euclidean distance). Works on a stack
    of steps too: every covariance may be [steps x n x n]."""
    parts = []
    for covariance in covariances:
        size = covariance.shape[-1]
        trace = np.trace(covariance, axis1=-2, axis2=-1)[..., None, None]
        ridged = covariance + RIDGE * trace / size * np.eye(size)
        values, vectors = np.linalg.eigh(ridged)
        logs = np.log(np.maximum(values, EIGEN_FLOOR))
        log = (vectors * logs[..., None, :]) @ np.swapaxes(vectors, -1, -2)
        rows, columns = np.triu_indices(size)
        parts.append(log[..., rows, columns] * np.where(rows == columns, 1.0, math.sqrt(2)))
    return np.concatenate(parts, axis=-1)


class RunningMean:
    """The person's mean so far. It starts at the prior, which counts as ``prior_steps``
    steps of the person, and then takes in every clean step with the weight
    max(1 / (prior_steps + steps taken), 1 / memory_steps)."""

    def __init__(self, prior: np.ndarray, prior_steps: int, memory_steps: int) -> None:
        assert 0 < prior_steps <= memory_steps
        self.mean = np.array(prior, dtype=float)
        self._prior_steps = prior_steps
        self._memory_steps = memory_steps
        self._taken = 0

    def add(self, vector: np.ndarray) -> None:
        self._taken += 1
        share = max(1.0 / (self._prior_steps + self._taken), 1.0 / self._memory_steps)
        self.mean += share * (vector - self.mean)


@dataclass(frozen=True)
class FocusModel:
    """The model of one set of electrodes."""

    electrodes: tuple[str, ...]
    prior: np.ndarray  # mean log vector of the people the model was trained on
    mean: np.ndarray  # of each feature in the training data
    scale: np.ndarray  # its standard deviation
    coef: np.ndarray  # weights of the standardised features
    intercept: float
    threshold: float  # scores above it are focus

    def score(self, vector: np.ndarray) -> float:
        """Positive for focus, negative for rest."""
        z = (vector - self.mean) / self.scale
        return float(z @ self.coef + self.intercept - self.threshold)


@dataclass(frozen=True)
class FocusModels:
    by_electrodes: dict[tuple[str, ...], FocusModel]
    raw: dict[str, Any]  # what the file holds, as it was read


def load_models(path: Path | None = None) -> FocusModels:
    """Read the models and check that they fit the detector of this program."""
    path = c.FOCUS_MODEL_PATH if path is None else Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as problem:
        raise AdamusError(f"cannot read the focus model {path}: {problem}") from problem
    if type(raw) is not dict or sorted(raw) != sorted(FILE_KEYS):
        raise AdamusError(f"{path}: a focus model file needs exactly {', '.join(FILE_KEYS)}")
    if raw["features"] != features():
        raise AdamusError(f"{path}: the model was trained on other features than these")
    trained_on = raw["trained_on"]
    people = trained_on.get("people") if type(trained_on) is dict else None
    if type(people) is not list or not people or not all(type(p) is str for p in people):
        raise AdamusError(f"{path}: trained_on must name the people the model was trained on")
    entries = raw["models"]
    if type(entries) is not list or not all(type(e) is dict for e in entries):
        raise AdamusError(f"{path}: models must be a list of models")
    models = {}
    for entry in entries:
        model = _model_of(entry, path)
        models[model.electrodes] = model
    if sorted(models) != sorted(electrode_sets()) or len(entries) != len(models):
        raise AdamusError(f"{path}: there must be one model for every set of electrodes")
    return FocusModels(models, raw)


def _model_of(entry: dict[str, Any], path: Path) -> FocusModel:
    if sorted(entry) != sorted(MODEL_KEYS):
        raise AdamusError(f"{path}: a model needs exactly {', '.join(MODEL_KEYS)}")
    names = entry["electrodes"]
    if type(names) is not list or tuple(names) not in electrode_sets():
        raise AdamusError(f"{path}: unknown set of electrodes {names!r}")
    try:
        prior, mean, scale, coef = (
            np.array(entry[key], dtype=float) for key in ("prior", "mean", "scale", "coef")
        )
        intercept, threshold = float(entry["intercept"]), float(entry["threshold"])
    except (TypeError, ValueError) as problem:
        raise AdamusError(f"{path}: numbers expected: {problem}") from problem
    arrays = (prior, mean, scale, coef)
    shapes_fit = all(array.shape == (vector_length(len(names)),) for array in arrays)
    finite = all(np.isfinite(array).all() for array in arrays)
    finite = finite and math.isfinite(intercept) and math.isfinite(threshold)
    if not (shapes_fit and finite and bool(np.all(scale > 0))):
        raise AdamusError(f"{path}: the numbers of the model of {', '.join(names)} do not fit")
    return FocusModel(tuple(names), prior, mean, scale, coef, intercept, threshold)


class Detector:
    """One person's detector: the model of their electrodes and their running mean."""

    def __init__(self, model: FocusModel, prior_steps: int, memory_steps: int) -> None:
        self._model = model
        self._running = RunningMean(model.prior, prior_steps, memory_steps)

    def step(self, covariances: list[np.ndarray]) -> float:
        """The score of a clean step, which then joins the person's mean."""
        vector = log_vector(covariances)
        score = self._model.score(vector - self._running.mean)
        self._running.add(vector)
        return score
