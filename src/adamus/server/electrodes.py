"""The electrodes of a session on a head: where each one is, and what it records.

The head and the places of the 10-05 system on it come from MNE-Python
(scripts/export_head_model.py). Every electrode the source recorded is given with its place, or
none when its name is not one of the system (an auxiliary input); the reference the headset
measures them against comes last, where the headset's is known.

What an electrode records is a cheat sheet, the same in every session: the part of the brain
under it and what that part does, the rhythms and responses measured there, what it is used
for, and which disturbances show in it. The four of the Muse are written out; any other place
gets the text of its region, read from the letters of its name, the system's own rule.
"""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from brainflow.board_shim import BoardIds

from adamus.evaluation.session_reader import SessionFolder
from adamus.server.api_models import Electrode, ElectrodeKind, HeadModel

HEAD_MODEL = Path(__file__).with_name("head_model.json")
AXES = 3  # x, y and z: a place is where it is, then the way the skin faces there
MUSE_REFERENCE = "Fpz"
# The reference electrode of each headset whose reference is known, by its BrainFlow board id
REFERENCES = {board.value: MUSE_REFERENCE for board in BoardIds if board.name.startswith("MUSE")}

WRITTEN = {
    "AF7": (
        "AF7 records the left prefrontal cortex, the part of the brain that holds attention, "
        "keeps information in working memory and holds back impulses. When you concentrate, "
        "alpha waves (8 to 12 Hz) weaken here and theta waves (4 to 8 Hz) grow with mental "
        "effort, so AF7 is one of the main electrodes for telling focus from rest and for "
        "measuring workload, and theta that keeps rising over a long task points to fatigue. "
        "Comparing alpha between AF7 and AF8 gives frontal alpha asymmetry, a widely used though "
        "debated marker of motivation and mood, in which relatively more activity on the left "
        "goes with approaching things and with positive feelings. Being so close to the eyes, "
        "AF7 records every blink as a large slow wave of up to a few hundred microvolts and "
        "every sideways glance as a step in the signal, and frowning or raising the eyebrows "
        "adds fast muscle noise above about 20 Hz. Its signal therefore needs cleaning or "
        "rejecting more often than that of any other electrode, and its dry sensor loses "
        "contact easily when the band sits loosely."
    ),
    "AF8": (
        "AF8 records the right prefrontal cortex, which is linked with staying vigilant, "
        "noticing errors and holding back a response that is already under way. Like AF7 it "
        "shows alpha weakening and theta growing with concentration and mental effort, and "
        "together the two give the most reliable frontal measure of focus, workload and fatigue "
        "that a headband can get. In frontal alpha asymmetry the right side stands for "
        "withdrawal, so relatively more activity here goes with avoidance, stress and anxiety. "
        "It records blinks, eye movements and frowning as strongly as AF7, and comparing the "
        "two tells them apart. A blink looks almost the same on both sides, a sideways glance "
        "shows on them with opposite signs, and brain activity usually differs between them in "
        "its own way."
    ),
    "TP9": (
        "TP9 records where the left temporal and parietal lobes meet, close to the auditory "
        "cortex and the language areas of the left hemisphere. It shows the alpha rhythm "
        "clearly, rising when you relax or close your eyes and falling when you engage with "
        "something, and it picks up how the brain responds to sound, speech and music. Its "
        "signal is often combined with the frontal electrodes to judge relaxation, drowsiness "
        "and engagement. Muscle is its main source of noise, as chewing, clenching the jaw, "
        "swallowing, talking and turning the head all add bursts of fast activity. Poor contact, "
        "common when the band shifts or hair gets under the sensor, shows as slow drift or "
        "sudden jumps."
    ),
    "TP10": (
        "TP10 records where the right temporal and parietal lobes meet. The right auditory "
        "cortex is especially tuned to pitch and melody, and the right temporoparietal junction "
        "turns attention to unexpected events and helps in understanding what others think and "
        "feel. Like TP9 it shows the alpha rhythm clearly and is combined with the frontal "
        "electrodes to judge relaxation, drowsiness and engagement, and comparing it with TP9 "
        "shows how the two hemispheres differ. Chewing, clenching the jaw, swallowing, talking "
        "and turning the head add bursts of muscle noise, and poor contact, common when the band "
        "shifts or hair gets under the sensor, shows as slow drift or sudden jumps."
    ),
    MUSE_REFERENCE: (
        "Fpz is the reference of the headset, which every other electrode is measured against. "
        "Each channel is the voltage between its electrode and Fpz, so whatever happens at Fpz "
        "shows in all of them at once. Blinks are strong here and reach every channel through "
        "it, inverted in the electrodes behind the ears. AF7 and AF8 lie close to it and share "
        "much of its activity, so their signals are smaller than they would be against a "
        "reference on the ear. Fpz has no signal of its own in the recording."
    ),
}
# The regions of the 10-05 system by the letters a name begins with; the longer letters first,
# as Fp is not F and TP is not T.
REGIONS = (
    ("Fp", "records the frontal pole, the foremost part of the prefrontal cortex, involved in "
           "planning, weighing decisions and keeping several goals in mind at once. Blinks and "
           "eye movements dominate its signal, which is why the Fp electrodes are mostly used to "
           "detect those artifacts and remove them from the other channels rather than to study "
           "the brain itself."),
    ("AF", "records the prefrontal cortex, which holds attention and working memory. Theta waves "
           "(4 to 8 Hz) grow here with mental effort and alpha waves (8 to 12 Hz) weaken with "
           "concentration, which makes it useful for measuring focus and workload. Blinks and "
           "eye movements show in it strongly as large slow waves."),
    ("FT", "records where the frontal and temporal lobes meet, close to the areas that produce "
           "speech on the left side. The jaw muscle lies under it, so chewing, talking and "
           "clenching add strong fast noise."),
    ("FC", "records the premotor and supplementary motor areas, which prepare movements. The "
           "readiness potential builds up here about a second before a deliberate movement, and "
           "the error related negativity that follows a mistake is strongest close to FCz."),
    ("F", "records the frontal lobe, where planning, decisions and mental effort take place. "
          "Frontal midline theta, strongest at Fz, is the best known sign of concentration and "
          "working memory load, and comparing alpha between the left and right frontal "
          "electrodes relates to motivation and mood."),
    ("TP", "records where the temporal and parietal lobes meet, a hub for hearing, language and "
           "turning attention to sounds. It shows the alpha rhythm well, and the jaw and neck "
           "muscles add bursts of fast noise."),
    ("T", "records the temporal lobe with the auditory cortex, involved in hearing, understanding "
          "speech and memory. The temporalis muscle lies under it, so any movement of the jaw "
          "makes its signal noisy."),
    ("CP", "records the sensory areas behind the motor cortex, which process touch and the "
           "position of the body. The P300 response to rare or important events is often "
           "measured here and along the parietal midline."),
    ("C", "records the sensorimotor cortex, which controls movement and feels touch. Its mu "
          "rhythm, at the frequency of alpha, weakens when you move or only imagine moving, "
          "which is the basis of most brain computer interfaces driven by imagined movement."),
    ("PO", "records the passage from the parietal to the visual cortex, which works out where "
           "things are and guides the eyes. The alpha rhythm is very strong here with the eyes "
           "closed."),
    ("P", "records the parietal lobe, which directs spatial attention and combines what the "
          "senses report. Alpha here weakens when attention is engaged, and the P300 response to "
          "rare or important events is strongest over Pz."),
    ("O", "records the visual cortex. Alpha is strongest here with the eyes closed and nearly "
          "vanishes when they open, and a flickering light produces a steady state visual "
          "response at its own frequency, which brain computer interfaces use to tell what "
          "someone looks at."),
    ("I", "records the lowest part of the visual cortex, next to the cerebellum. The neck "
          "muscles under it make its signal noisy, so it serves mainly to complete the coverage "
          "of the whole head."),
)  # fmt: skip
UNPLACED = (
    "{name} is not one of the standard electrode positions, so nothing general can be said "
    "about it."
)


@dataclass(frozen=True)
class Head:
    vertices: list[float]  # x, y and z of every point, in millimetres
    triangles: list[int]  # three points each
    # every place of the 10-05 system, by its name in small letters: where on the skin, then the
    # way the skin faces there
    places: dict[str, list[float]]


@cache
def head() -> Head:
    """The head, read once; a headset writes some names in other cases (FPZ, Fpz), so the places
    are found by their names in small letters."""
    model = json.loads(HEAD_MODEL.read_text(encoding="utf-8"))
    places = {name.lower(): place for name, place in model["places"].items()}
    return Head(model["vertices"], model["triangles"], places)


def head_model() -> HeadModel:
    return HeadModel(vertices=head().vertices, triangles=head().triangles)


def about(name: str) -> str:
    """What the electrode of this name records, and what is worth knowing about its signal."""
    if name in WRITTEN:
        return WRITTEN[name]
    if name.lower() not in head().places:
        return UNPLACED.format(name=name)
    text = next(text for prefix, text in REGIONS if name.lower().startswith(prefix.lower()))
    return f"{name} {text}"


def electrode(name: str, kind: ElectrodeKind = "recorded") -> Electrode:
    place = head().places.get(name.lower())
    return Electrode(
        name=name, kind=kind, position=place[:AXES] if place else None,
        normal=place[AXES:] if place else None, about=about(name),
    )  # fmt: skip


def electrodes_of(session: SessionFolder) -> list[Electrode]:
    """The electrodes the source recorded, in the order of its rows, then the reference of the
    headset, where it is known and was not recorded itself."""
    recorded = [electrode(name) for name in session.meta["electrode_rows"]]
    reference = REFERENCES.get(session.meta["board_id"])
    if reference is not None and reference.lower() not in {e.name.lower() for e in recorded}:
        recorded.append(electrode(reference, "reference"))
    return recorded
