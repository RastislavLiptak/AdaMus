"""Export the head the app draws the electrodes on, once, into the program.

The head is the average head of FreeSurfer's fsaverage that MNE-Python ships (its scalp,
fsaverage-head.fif), and the electrodes are every place of the 10-05 system fitted to that
head (fsaverage_1005.tsv), both from MNE-Python (BSD-3-Clause). AdaMus does not need MNE to
run: this script writes them into src/adamus/server/head_model.json, which the program reads.

Run it with MNE installed, in an environment of its own:
    python -m venv /tmp/mne && /tmp/mne/bin/pip install mne
    /tmp/mne/bin/python scripts/export_head_model.py

Every length is in millimetres, in MNE's frame of the head: x to the right ear, y to the nose,
z up. The triangles wind so that the skin faces outward. A place of the system lies a few
millimetres off this coarse skin; it is moved onto the skin, along the line from the middle of
the head, and given the way the skin faces there, so that a dot drawn there lies on it.
"""

import csv
import json
from importlib.resources import files
from pathlib import Path

import mne  # type: ignore[import-not-found]  # only here, see above
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "adamus" / "server" / "head_model.json"
MNE = files("mne")
HEAD = MNE / "data" / "fsaverage" / "fsaverage-head.fif"
POSITIONS = MNE / "channels" / "data" / "montages" / "fsaverage_1005.tsv"
LANDMARKS = {"LPA", "RPA", "NAS", "Nz"}  # points of the anatomy, not places of electrodes
MILLIMETRES = 1000.0
DECIMALS = 1  # a tenth of a millimetre
NORMAL_DECIMALS = 3
MIDDLE = np.array(
    [0.0, -12.0, -10.0]
)  # mm: between the ears, a little up; places are moved from here
FAR = 100.0  # mm outside the skin, where a line toward the middle starts
CORNERS = 3  # of a triangle
PARALLEL = 1e-9  # a line this close to lying in a triangle's plane misses it


def onto_skin(
    place: np.ndarray, points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Where the line from outside toward the middle of the head through ``place`` first meets
    the skin, and the outward normal of the triangle it meets there (Moller and Trumbore)."""
    inward = MIDDLE - place
    inward /= np.linalg.norm(inward)
    start = place - inward * FAR
    a, b, c = (points[triangles[:, corner]] for corner in range(CORNERS))
    ab, ac = b - a, c - a
    p = np.cross(inward, ac)
    det = np.einsum("ij,ij->i", ab, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_vec = start - a
        u = np.einsum("ij,ij->i", t_vec, p) / det
        q = np.cross(t_vec, ab)
        v = np.einsum("j,ij->i", inward, q) / det
        distance = np.einsum("ij,ij->i", ac, q) / det
    hits = (np.abs(det) > PARALLEL) & (u >= 0) & (v >= 0) & (u + v <= 1) & (distance > 0)
    first = np.flatnonzero(hits)[np.argmin(distance[hits])]
    normal = np.cross(ab[first], ac[first])
    return start + inward * distance[first], normal / np.linalg.norm(normal)


def main() -> None:
    (skin,) = mne.read_bem_surfaces(str(HEAD))
    with POSITIONS.open(encoding="utf-8") as table:
        rows = list(csv.DictReader(table, delimiter="\t"))
    points, triangles = skin["rr"] * MILLIMETRES, skin["tris"]
    places = {}
    for row in rows:
        if row["label"] in LANDMARKS:
            continue
        place = np.array([float(row[axis]) for axis in "xyz"]) * MILLIMETRES
        at, normal = onto_skin(place, points, triangles)
        places[row["label"]] = [
            *np.round(at, DECIMALS).tolist(),
            *np.round(normal, NORMAL_DECIMALS).tolist(),
        ]
    model = {
        "source": f"MNE-Python {mne.__version__}: {HEAD.name}, {POSITIONS.name}",
        "vertices": [round(float(v), DECIMALS) for v in points.ravel()],
        "triangles": [int(i) for i in triangles.ravel()],
        "places": places,  # each: where on the skin, then the way the skin faces there
    }
    with OUT.open("w", encoding="utf-8") as out:
        print(json.dumps(model, separators=(",", ":")), file=out)
    print(f"{OUT.relative_to(ROOT)}: {len(places)} places, {len(triangles)} triangles")


if __name__ == "__main__":
    main()
