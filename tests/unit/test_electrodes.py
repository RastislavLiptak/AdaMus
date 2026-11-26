"""server.electrodes: the electrodes of a session on the head, and what each records."""

import json

import numpy as np
import pytest
from brainflow.board_shim import BoardIds

from adamus import constants as c
from adamus.evaluation.session_reader import load_session_folder
from adamus.server.electrodes import (
    AUXILIARY_NAMES,
    MUSE_REFERENCE,
    REGIONS,
    WRITTEN,
    about,
    electrodes_of,
    head,
    head_model,
)

MUSE = ["AF7", "AF8", "TP9", "TP10"]
SKIN_MM = 10  # how far from a point of the head's skin a place may lie
TEXT_MARKS = ":;-"  # the texts read as plain sentences, without these


def with_meta(folder, **changed):
    meta = json.loads((folder / c.FILE_META).read_text())
    (folder / c.FILE_META).write_text(json.dumps(meta | changed))
    return load_session_folder(folder)


def test_the_head_is_whole_triangles_of_its_own_points():
    model = head_model()
    assert len(model.vertices) % 3 == 0 and len(model.triangles) % 3 == 0
    assert min(model.triangles) == 0 and max(model.triangles) == len(model.vertices) // 3 - 1


def test_the_skin_faces_outward():
    points = np.array(head().vertices).reshape(-1, 3)
    corners = points[np.array(head().triangles).reshape(-1, 3)]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    outward = np.einsum("ij,ij->i", normals, corners.mean(axis=1) - points.mean(axis=0))
    assert (outward > 0).mean() > 0.99


def test_every_place_lies_on_the_skin_facing_out_in_millimetres():
    points = np.array(head().vertices).reshape(-1, 3)
    places = np.array(list(head().places.values()))
    at, normals = places[:, :3], places[:, 3:]
    nearest = np.linalg.norm(at[:, None, :] - points[None, :, :], axis=2).min(axis=1)
    assert len(places) == 336 and nearest.max() < SKIN_MM
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=0.01)
    assert (np.einsum("ij,ij->i", normals, at - points.mean(axis=0)) > 0).all()  # outward
    nose, crown = head().places["fpz"], head().places["cz"]
    assert nose[1] > 80 and crown[2] > 90  # y to the nose, z up, in mm


def test_the_electrodes_of_a_muse_session_in_the_order_of_its_rows_then_its_reference(played):
    session = load_session_folder(played.directory)
    found = electrodes_of(session)
    rows = list(session.meta["electrode_rows"])
    assert [e.name for e in found] == [*rows, MUSE_REFERENCE]
    assert [e.kind for e in found] == ["recorded"] * len(rows) + ["reference"]
    assert {e.name: [*e.position, *e.normal] for e in found if e.position and e.normal} == {
        name: head().places[name.lower()] for name in [*MUSE, "Fpz"]
    }


@pytest.mark.parametrize("name", AUXILIARY_NAMES)
def test_the_auxiliary_input_has_no_place_on_the_head_but_says_what_it_is(copied, name):
    session = with_meta(copied, electrode_rows={"AF7": 0, name: 4})
    aux = electrodes_of(session)[1]
    assert (aux.name, aux.position, aux.normal) == (name, None, None)
    assert aux.about.startswith(f"{name} is the auxiliary input of the Muse headband")


def test_an_input_that_is_no_place_on_the_head_and_is_not_known_says_so(copied):
    session = with_meta(copied, electrode_rows={"AF7": 0, "EXT": 4})
    assert electrodes_of(session)[1].about == (
        "EXT is not one of the standard electrode positions, so nothing general can be said "
        "about it."
    )


def test_a_name_in_other_letters_is_found(copied):
    session = with_meta(copied, electrode_rows={"FPZ": 0, "cz": 1})
    found = electrodes_of(session)
    assert [e.position for e in found] == [head().places["fpz"][:3], head().places["cz"][:3]]
    assert [e.name for e in found] == ["FPZ", "cz"]  # Fpz recorded: no reference added


@pytest.mark.parametrize("board", [BoardIds.SYNTHETIC_BOARD.value, BoardIds.CYTON_BOARD.value])
def test_a_headset_whose_reference_is_not_known_has_none(copied, board):
    assert all(e.kind == "recorded" for e in electrodes_of(with_meta(copied, board_id=board)))


def test_every_muse_takes_fpz_as_its_reference(copied):
    for board in (BoardIds.MUSE_S_BOARD, BoardIds.MUSE_2016_BOARD, BoardIds.MUSE_S_ATHENA_BOARD):
        assert electrodes_of(with_meta(copied, board_id=board.value))[-1].name == "Fpz"


def test_the_electrodes_of_muse_and_its_reference_are_written_out():
    assert set(WRITTEN) == {*MUSE, MUSE_REFERENCE, *AUXILIARY_NAMES}
    for name, text in WRITTEN.items():
        assert about(name) == text and text.startswith(f"{name} ")


@pytest.mark.parametrize(
    ("name", "region"),
    [("Fp1", "Fp"), ("AF3", "AF"), ("AFF1h", "AF"), ("FT7", "FT"), ("FCz", "FC"), ("F3", "F"),
     ("TP7", "TP"), ("T8", "T"), ("CP1", "CP"), ("C5", "C"), ("PO7", "PO"), ("Pz", "P"),
     ("O1", "O"), ("Iz", "I"), ("fpz", "Fp")],
)  # fmt: skip
def test_any_other_place_has_the_text_of_its_region_by_the_letters_of_its_name(name, region):
    assert about(name) == f"{name} {dict(REGIONS)[region]}"


def test_every_place_of_the_system_has_a_text():
    assert all(about(name).startswith(f"{name} ") for name in head().places)


def test_the_texts_are_plain_sentences():
    texts = [*WRITTEN.values(), *(text for _, text in REGIONS)]
    assert [text for text in texts if any(mark in text for mark in TEXT_MARKS)] == []


def test_every_text_is_whole_sentences():
    texts = [*WRITTEN.values(), *(text for _, text in REGIONS)]
    assert [text for text in texts if not text.endswith(".") or ".." in text] == []
