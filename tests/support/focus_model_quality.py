"""How well the focus models do on people they had not seen, as the detector would meet them."""

import numpy as np
from scipy.stats import mannwhitneyu

from adamus import constants as c
from adamus.detection import focus_training as train
from adamus.detection.focus_control import focus_of, next_level

CHECK_GROUPS = 7  # every group is scored by models of the others


def auc(focus: np.ndarray, rest: np.ndarray) -> float:
    """Probability that a focus step scores above a rest step (ties count half)."""
    return float(mannwhitneyu(focus, rest).statistic / (len(focus) * len(rest)))


def controlled(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """f of every step, as the controller makes it from the scores of the clean steps."""
    level, f, started = 0.0, np.full(len(values), np.nan), False
    for k in range(len(values)):
        if valid[k]:
            level, started = next_level(level, float(values[k])), True
        if started:
            f[k] = focus_of(level)
    return f


def judged(person: dict, model: dict, threshold: float) -> dict[str, float]:
    """How a person in a set fares with a model and its threshold."""
    centred = train.centred(person["logs"], person["valid"], model["prior"])
    s = train.scores(model, centred) - threshold
    f = controlled(s, person["valid"])
    use = person["valid"] & (person["states"] >= 0)
    y = person["states"][use]
    on = f[use] > c.FOCUS_THRESHOLD
    s, f = s[use], f[use]
    return {
        "auc": auc(s[y == 1], s[y == 0]),
        "auc of f": auc(f[y == 1], f[y == 0]),
        "focus caught": float(on[y == 1].mean()),
        "rest right": float((~on[y == 0]).mean()),
    }


def check(people: list[dict]) -> dict[str, float]:
    """Every group of people in a set scored by a model of the others, with a threshold of
    the others; the means over the people."""
    names = sorted(p["name"] for p in people)
    rows = []
    for group in range(CHECK_GROUPS):
        held = set(names[group::CHECK_GROUPS])
        others = [p for p in people if p["name"] not in held]
        model, threshold = train.fit_people(others), train.threshold_of(others)
        rows += [judged(p, model, threshold) for p in people if p["name"] in held]
    mean = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
    return {**mean, "balanced": (mean["focus caught"] + mean["rest right"]) / 2}
