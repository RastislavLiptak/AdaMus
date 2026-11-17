"""Training of the focus models: the program runs it when it starts without a usable model.

The data are the public dataset "Regulation of Brain Cognitive States through
Auditory, Gustatory, and Olfactory Stimulation with Wearable Monitoring" 1.0.0 on
PhysioNet (Open Data Commons Attribution License v1.0): a Muse headband (TP9, AF7,
AF8, TP10 at 256 Hz) while people do n-back blocks of about 42 s, each followed by a
short break. They are downloaded once, next to the model.

Every step is computed by the program's own signal processing, so the models see
exactly what they will see in a session. Focus is a whole window inside an n-back
block, rest a whole window inside a break after one (the instruction shown before
the next block is left out). A person trains the model of a set of electrodes when
those electrodes give clean signal at least half of the time. The threshold of a
model is the one with the best balanced accuracy on people the model had not seen
(four groups, each scored by a model of the others).
"""

import csv
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from scipy.optimize import minimize

from adamus import constants as c
from adamus.detection.focus_detector import RunningMean, electrode_sets, features, log_vector
from adamus.detection.signal_processing import (
    Steps,
    clean_steps,
    design_filters,
    present_rows,
    process,
)

DATASET = "PhysioNet brain-wearable-monitoring 1.0.0 (Muse, n-back blocks and breaks)"
DATASET_URL = "https://physionet.org/content/brain-wearable-monitoring/get-zip/1.0.0/"
TRAINING_DATA = "training_data"  # in the folder of the model
DOWNLOAD_CHUNK = 1 << 20  # bytes
DOWNLOAD_TIMEOUT_S = 60.0  # the longest wait for the server, so a stalled download ends
EEG_FILE, LOG_FILE = "EEG_recording.csv", "n_back_responses.csv"
COLUMNS = ("TP9", "AF7", "AF8", "TP10")  # after the time column of the EEG file
PRACTICE = "PracticeSession"  # the blocks before the sessions, left out
STIMULUS_S = 2.0  # a letter and its fixation: the block ends this long after the last onset
BREAK_START_S = 1.0  # rest starts this long after a block
INSTRUCTION_S = 6.0  # rest ends this long before the next block (5 s instruction and margin)
LONGEST_BREAK_S = 40.0  # longer gaps are pauses between sessions, of unknown content
HOLE_S = 0.1  # a gap between two samples longer than this makes the steps over it invalid
LEAST_VALID_SHARE = 0.5  # of clean steps, for a person to train the model of a set
LEAST_STEPS = 30  # of each state, for a person to train a model
C_REGULARISATION = 0.1  # inverse strength of the squared penalty, as in scikit-learn
GROUPS = 4  # for the threshold, on people the model had not seen
ITERATIONS = 3000


def train(folder: Path, log: TextIO) -> Path:
    """Download the data if they are not there yet, train every model, write the file."""
    data = download(folder / TRAINING_DATA, log)
    people = []
    for person in sorted(p for p in data.glob("Experiment_*/*") if p.is_dir()):
        people.append(person_steps(person))
        print(f"{people[-1]['name']} read", file=log, flush=True)
    written = trained_models(people, log)
    target = folder / c.FOCUS_MODEL_FILE
    target.write_text(json.dumps(written, indent=1), encoding="utf-8")
    print(f"written: {target}", file=log, flush=True)
    return target


# --- The recordings ---------------------------------------------------------------


def download(target: Path, log: TextIO) -> Path:
    """The dataset unpacked into ``target``; downloaded once. An interrupted download or
    unpacking leaves no ``target`` behind, so the next start begins again."""
    if not target.is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        archive = target.with_name(target.name + ".zip")
        unpacking = target.with_name(target.name + ".part")
        print(f"downloading {DATASET_URL} ...", file=log, flush=True)
        with (
            urllib.request.urlopen(DATASET_URL, timeout=DOWNLOAD_TIMEOUT_S) as response,
            archive.open("wb") as file,
        ):
            shutil.copyfileobj(response, file, DOWNLOAD_CHUNK)
        shutil.rmtree(unpacking, ignore_errors=True)
        with zipfile.ZipFile(archive) as unpacked:
            unpacked.extractall(unpacking)
        unpacking.rename(target)
        archive.unlink()
    return next(p for p in target.iterdir() if p.is_dir())


def signal_of(person: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """The EEG of a person on an even grid at the detector's rate, in its electrode order,
    whether each sample lies in a hole of the recording, and the time of the first sample."""
    table = np.loadtxt(
        person / EEG_FILE, delimiter=",", skiprows=1, usecols=range(len(COLUMNS) + 1)
    )
    order = np.argsort(table[:, 0])
    stamps, raw = table[order, 0], table[order, 1:].T
    grid = stamps[0] + np.arange(int((stamps[-1] - stamps[0]) * c.DETECTOR_FS)) / c.DETECTOR_FS
    rows = [COLUMNS.index(name) for name in c.DETECTOR_ELECTRODES]
    signal = np.stack([np.interp(grid, stamps, raw[row]) for row in rows])
    after = np.clip(np.searchsorted(stamps, grid, side="right"), 1, len(stamps) - 1)
    hole = stamps[after] - stamps[after - 1] > HOLE_S  # the gap each grid point falls into
    return signal, hole, float(grid[0])


def blocks_of(person: Path) -> list[tuple[float, float]]:
    """Start and end of every n-back block (practice left out), in seconds since 1970."""
    with (person / LOG_FILE).open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    start = float(rows[0]["Start_time_unix"])
    onset_columns = [k for k in rows[0] if k.startswith("Stimulus") and k.endswith(".OnsetTime")]
    onsets: dict[str, list[float]] = {}
    for row in rows:
        onset = next((float(row[k]) for k in onset_columns if row[k]), None)
        if onset is not None and row["Running[Block]"] != PRACTICE:
            onsets.setdefault(row["Block"], []).append(start + onset / 1000)
    return [(times[0], times[-1] + STIMULUS_S) for times in onsets.values()]


def states_of(blocks: list[tuple[float, float]]) -> list[tuple[float, float, int]]:
    """(start, end, 1 focus / 0 rest) of every block and of the break after it."""
    spans = []
    for k, (begin, end) in enumerate(blocks):
        spans.append((begin, end, 1))
        if k + 1 < len(blocks) and blocks[k + 1][0] - end < LONGEST_BREAK_S:
            spans.append((end + BREAK_START_S, blocks[k + 1][0] - INSTRUCTION_S, 0))
    return spans


def subset_covariances(
    covariances: list[np.ndarray], electrodes: tuple[str, ...]
) -> list[np.ndarray]:
    """The covariances of a set of electrodes, cut out of those of all of them."""
    rows = present_rows(electrodes)
    count = len(c.DETECTOR_ELECTRODES)
    delayed = [d * count + row for d in range(len(c.DELAYS_S)) for row in rows]
    *bands, last = covariances
    return [m[..., rows, :][..., rows] for m in bands] + [last[..., delayed, :][..., delayed]]


def person_steps(person: Path) -> dict[str, Any]:
    """What every step of a person gives, before any set of electrodes is chosen: the
    covariances of all electrodes and what decides whether the step is clean; the states.

    The log vectors of a set are computed from these when the set is trained
    (``in_set``): those of all sets at once would not fit into memory.
    """
    signal, hole, first = signal_of(person)
    filters = design_filters(c.DETECTOR_FS)
    steps = Steps.at(c.DETECTOR_FS)
    ends = [steps.end(k) for k in range(steps.count(signal.shape[1]))]
    every = c.DETECTOR_ELECTRODES
    covariances: list[list[np.ndarray]] = []
    judged: dict[str, list[np.ndarray]] = {"max_abs_uv": [], "noise_uv": [], "carries_signal": []}
    for e in ends:  # one step at a time: its filtered signal is not kept
        step = process(signal[:, e - steps.context : e], steps.window, filters, every)
        covariances.append(step.covariances)
        for key, values in judged.items():
            values.append(getattr(step, key))
    times = first + (np.array(ends) - 1) / c.DETECTOR_FS
    states = np.full(len(ends), -1)
    for begin, end, state in states_of(blocks_of(person)):
        states[(times - c.WINDOW_S >= begin) & (times <= end)] = state
    return {
        "name": f"{person.parent.name}/{person.name}",
        "states": states,
        "covariances": [np.array(stack) for stack in zip(*covariances, strict=True)],
        **{key: np.array(values) for key, values in judged.items()},
        "gaps": np.array([hole[e - steps.window : e].any() for e in ends]),
    }


def in_set(person: dict[str, Any], electrodes: tuple[str, ...]) -> dict[str, Any]:
    """A person as the model of a set of electrodes sees them: the log vector of every
    step and whether the step is clean on these electrodes."""
    clean = clean_steps(
        person["max_abs_uv"], person["noise_uv"], person["carries_signal"], electrodes
    )
    return {
        "name": person["name"],
        "states": person["states"],
        "electrodes": electrodes,
        "logs": log_vector(subset_covariances(person["covariances"], electrodes)),
        "valid": clean & ~person["gaps"],
    }


# --- The models ---------------------------------------------------------------------


def centred(logs: np.ndarray, valid: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Every step minus the person's running mean before it, as the detector computes it."""
    running = RunningMean(prior, c.PRIOR_STEPS, c.MEMORY_STEPS)
    out = np.empty_like(logs)
    for k, vector in enumerate(logs):
        out[k] = vector - running.mean
        if valid[k]:
            running.add(vector)
    return out


def labelled(person: dict[str, Any], prior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Features and states of the clean labelled steps of a person in a set."""
    x = centred(person["logs"], person["valid"], prior)
    use = person["valid"] & (person["states"] >= 0)
    return x[use], person["states"][use]


def usable(person: dict[str, Any]) -> bool:
    """Clean at least half of the time, and enough clean steps of each state."""
    counts = [np.sum(person["valid"] & (person["states"] == state)) for state in (0, 1)]
    return bool(person["valid"].mean() >= LEAST_VALID_SHARE and min(counts) >= LEAST_STEPS)


def fit_logistic(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Standardised features, then a logistic regression with balanced class weights and a
    penalty on the squared weights: the same as scikit-learn's
    LogisticRegression(C, class_weight="balanced")."""
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    counts = np.bincount(y, minlength=2)
    weights = np.where(y == 1, len(y) / (2 * counts[1]), len(y) / (2 * counts[0]))
    start = np.zeros(z.shape[1] + 1)
    fitted = minimize(
        penalised_loss, start, args=(z, y, weights), jac=True, method="L-BFGS-B",
        options={"maxiter": ITERATIONS},
    )  # fmt: skip
    return {"mean": mean, "scale": scale, "coef": fitted.x[:-1], "intercept": fitted.x[-1]}


def penalised_loss(
    theta: np.ndarray, z: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """The weighted log loss times C plus half the squared weights (not the intercept), and
    its gradient; ``theta`` is the weights followed by the intercept."""
    margin = z @ theta[:-1] + theta[-1]
    signed = np.where(y == 1, margin, -margin)
    value = C_REGULARISATION * np.sum(weights * np.logaddexp(0, -signed))
    residual = C_REGULARISATION * weights * (1 / (1 + np.exp(-margin)) - y)
    gradient = np.concatenate([z.T @ residual + theta[:-1], [residual.sum()]])
    return float(value + theta[:-1] @ theta[:-1] / 2), gradient


def scores(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    result: np.ndarray = ((x - model["mean"]) / model["scale"]) @ model["coef"] + model["intercept"]
    return result


def prior_of(people: list[dict[str, Any]]) -> np.ndarray:
    result: np.ndarray = np.vstack([p["logs"][p["valid"]] for p in people]).mean(axis=0)
    return result


def fit_people(people: list[dict[str, Any]]) -> dict[str, Any]:
    """The model of people in a set: prior and logistic regression, without threshold."""
    prior = prior_of(people)
    pairs = [labelled(p, prior) for p in people]
    model = fit_logistic(np.vstack([x for x, _ in pairs]), np.concatenate([y for _, y in pairs]))
    return {**model, "prior": prior}


def balanced_threshold(values: np.ndarray, states: np.ndarray) -> float:
    """The threshold with the best mean of focus caught and rest right (focus is above it)."""
    order = np.argsort(values)
    ranked, focus = values[order], states[order] == 1
    rest_right = np.cumsum(~focus) / max((~focus).sum(), 1)  # at or below each value
    focus_caught = 1 - np.cumsum(focus) / max(focus.sum(), 1)  # above each value
    best = int(np.argmax(rest_right + focus_caught))
    upper = ranked[best + 1] if best + 1 < len(ranked) else ranked[best] + 1.0
    return float((ranked[best] + upper) / 2)


def held_out(
    people: list[dict[str, Any]], groups: int = GROUPS
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """For every group of people: the others, whom a model is fitted on, and the group, whom
    that model scores. Every person is in one group."""
    return [
        ([p for k, p in enumerate(people) if k % groups != group],
         [p for k, p in enumerate(people) if k % groups == group])
        for group in range(groups)
    ]  # fmt: skip


def threshold_of(people: list[dict[str, Any]], groups: int = GROUPS) -> float:
    """Balanced threshold on the steps of every person, scored by a model of the others."""
    held_values, held_states = [], []
    for fitted_on, scored in held_out(people, groups):
        model = fit_people(fitted_on)
        for person in scored:
            x, y = labelled(person, model["prior"])
            held_values.append(scores(model, x))
            held_states.append(y)
    return balanced_threshold(np.concatenate(held_values), np.concatenate(held_states))


def model_entry(people: list[dict[str, Any]], electrodes: tuple[str, ...]) -> dict[str, Any]:
    model = fit_people(people)
    return {
        "electrodes": list(electrodes),
        "prior": model["prior"].tolist(),
        "mean": model["mean"].tolist(),
        "scale": model["scale"].tolist(),
        "coef": model["coef"].tolist(),
        "intercept": float(model["intercept"]),
        "threshold": threshold_of(people),
    }


def trained_models(people: list[dict[str, Any]], log: TextIO) -> dict[str, Any]:
    """The content of a model file: one model per set of electrodes, of its usable people."""
    entries, trained_on = [], {}
    for electrodes in electrode_sets():  # the vectors of one set at a time are in memory
        chosen = [view for view in (in_set(p, electrodes) for p in people) if usable(view)]
        entries.append(model_entry(chosen, electrodes))
        name = "+".join(electrodes)
        trained_on[name] = len(chosen)
        print(f"{name} model of {len(chosen)} people", file=log, flush=True)
        if electrodes == c.DETECTOR_ELECTRODES:
            everyone = sorted(p["name"] for p in chosen)
    return {
        "features": features(),
        "trained_on": {
            "dataset": DATASET,
            "people": everyone,
            "left_out": sorted(p["name"] for p in people if p["name"] not in everyone),
            "people per set": trained_on,
        },
        "models": entries,
    }
