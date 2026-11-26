"""The music of a session: its chords struck on their bars, and its full form (drums, power
chords) from focus on, changing at most once a bar; everything as generator.jsonl recorded it."""

import re

import pytest

from adamus import constants as c
from adamus.config import with_style
from adamus.session_format import read_jsonl
from tests.support.finished_sessions import FinishedSession, config_with, synthetic_matrix
from tests.support.short_sessions import CONTROL, STYLE

FOCUS_WEIGHT = re.compile(r"prompt 1 [a-z0-9 -]+ (\d\.\d{3});")


def sent(session):
    return read_jsonl(session.directory / c.FILE_GENERATOR)


def strikes(lines):
    """(time, notes) of every chord struck."""
    found = []
    for line in lines:
        notes = [int(n) for n in re.findall(r"noteon (\d+);", line["msg"])]
        if notes:
            found.append((line["t"], notes))
    return found


def steps(lines):
    """(time, weight toward focus, the rest of the message) of every step."""
    found = []
    for line in lines:
        weight = FOCUS_WEIGHT.search(line["msg"])
        if weight and "prompt 0" in line["msg"]:
            found.append((line["t"], float(weight.group(1)), line["msg"]))
    return found


@pytest.fixture(scope="module", params=["rock", "country"])
def styled(request, tmp_path_factory):
    """A session over 120 s of blocks of 30 s, in a style with power chords or drums in focus."""
    config = with_style(config_with(CONTROL), request.param)
    session = FinishedSession(
        tmp_path_factory.mktemp(request.param), config, synthetic_matrix(CONTROL)
    )
    return session, config.style


def test_the_chords_are_struck_on_their_bars_after_their_gap(short):
    """The loop wakes for the chords: a strike waits for no poll of the samples."""
    struck = strikes(sent(short))
    first = struck[0][0]
    assert len(struck) >= 5
    for moment, _ in struck[1:]:
        bars = (moment - first - STYLE.chord_gap_s) / STYLE.bar_s
        assert bars == pytest.approx(round(bars), abs=1e-4), moment - first


def form_switches(session, style):
    """When the full form came and went, and what it was at each step: by the power chords of a
    style that has them, by the drums of one that has them."""
    lines = sent(session)
    if style.focus_power_chords:
        held = [(t, len(notes) == 3) for t, notes in strikes(lines)]  # a power chord: 3 notes
    else:
        held = [(t, "drumless 0;" in rest) for t, _, rest in steps(lines)]
    switches = [t for (t, full), (_, before) in zip(held[1:], held, strict=False) if full != before]
    return held, switches


def test_the_full_form_comes_at_once_with_the_first_step_of_focus(styled):
    session, style = styled
    first_focus = next(t for t, weight, _ in steps(sent(session)) if weight >= c.FOCUS_THRESHOLD)
    held, switches = form_switches(session, style)
    assert not any(full for t, full in held if t < first_focus)  # never before focus
    assert switches[0] == pytest.approx(first_focus, abs=1e-3)  # with that step, not a bar later


def test_the_form_changes_at_most_once_a_bar_and_goes_with_the_focus_again(styled):
    session, style = styled
    _, switches = form_switches(session, style)
    assert len(switches) >= 3  # on in focus, off at rest, on again
    gaps = [later - earlier for earlier, later in zip(switches, switches[1:], strict=False)]
    assert min(gaps) >= style.bar_s - 1e-6
