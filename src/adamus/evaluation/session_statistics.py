"""The short statistics of a session: how it ended, how clean its signal was, how the focus went.

Computed from ticks.jsonl and events.jsonl only; the full metrics, in
metrics.json, are computed by evaluation.metrics.
"""

from dataclasses import dataclass

import numpy as np

from adamus import constants as c
from adamus.evaluation.session_reader import SessionFolder
from adamus.session_format import (
    EVENT_ERROR,
    EVENT_SESSION_END,
    EVENT_SESSION_START,
    REASON_ERROR,
)

OK = "OK"
ERROR = "ERROR"
NO_CLEAN_SIGNAL = "NO CLEAN SIGNAL"
NO_CLEAN_SIGNAL_DETAIL = "not one clean step for the detector"
NO_STEP_DETAIL = "ended before the signal of the first step had arrived"


@dataclass(frozen=True)
class FocusStatistics:
    switched_on: int  # times f went from below FOCUS_THRESHOLD to at least it
    on_share: float  # share of controlled steps with f at least FOCUS_THRESHOLD
    focus_blocks_on: int
    focus_blocks: int
    rest_blocks_off: int
    rest_blocks: int
    focus_found: float | None  # share of the steps of focus blocks with f at least the threshold
    rest_found: float | None  # share of the steps of rest blocks with f under it; None: no step
    f_min: float
    f_max: float
    f_mean: float


@dataclass(frozen=True)
class Statistics:
    duration_s: float
    reason: str
    verdict: str  # OK, NO CLEAN SIGNAL or ERROR
    detail: str | None  # why the verdict is not OK
    usable_share: float | None  # valid steps / all steps
    focus: FocusStatistics | None  # None without a clean step


def statistics_of(session: SessionFolder) -> Statistics:
    start, end = session.event(EVENT_SESSION_START), session.event(EVENT_SESSION_END)
    error = session.event(EVENT_ERROR)
    reason = end.get("reason", REASON_ERROR) if end else REASON_ERROR
    if reason == REASON_ERROR or error:
        unfinished = "the session did not finish"
        verdict, detail = ERROR, error.get("message", unfinished) if error else unfinished
    elif session.controlled_ticks():
        verdict, detail = OK, None
    else:
        verdict = NO_CLEAN_SIGNAL
        detail = NO_CLEAN_SIGNAL_DETAIL if session.ticks else NO_STEP_DETAIL
    ticks = session.ticks
    return Statistics(
        duration_s=end["t"] - start["t"] if start and end else 0.0,
        reason=reason,
        verdict=verdict,
        detail=detail,
        usable_share=sum(tick["valid"] for tick in ticks) / len(ticks) if ticks else None,
        focus=_focus_statistics(session),
    )


def _focus_statistics(session: SessionFolder) -> FocusStatistics | None:
    f = np.array([tick["f"] for tick in session.controlled_ticks()])
    if len(f) == 0:
        return None
    on = f >= c.FOCUS_THRESHOLD
    as_planned = {c.FOCUS: 0, c.REST: 0}
    finished = {c.FOCUS: 0, c.REST: 0}
    stepped: dict[str, list[float]] = {c.FOCUS: [], c.REST: []}
    for span in session.evaluation_spans:
        stepped[span.label] += session.block_f(span)  # also of a block cut short
        assert span.end is not None
        if session.ticks[-1]["n_end"] < span.end:
            continue  # the block did not run to its end
        finished[span.label] += 1
        block_f = session.block_f(span)
        if block_f:
            block_on = float(np.mean(block_f)) >= c.FOCUS_THRESHOLD
            as_planned[span.label] += block_on == (span.label == c.FOCUS)
    return FocusStatistics(
        switched_on=int((~on[:-1] & on[1:]).sum()),
        on_share=float(on.mean()),
        focus_blocks_on=as_planned[c.FOCUS],
        focus_blocks=finished[c.FOCUS],
        rest_blocks_off=as_planned[c.REST],
        rest_blocks=finished[c.REST],
        focus_found=found_share(stepped[c.FOCUS], focus=True),
        rest_found=found_share(stepped[c.REST], focus=False),
        f_min=float(f.min()),
        f_max=float(f.max()),
        f_mean=float(f.mean()),
    )


def found_share(f: list[float], focus: bool) -> float | None:
    """The share of the steps of blocks of one state in which the detector found it: focus from
    the threshold up, as the metrics count it; None without a step."""
    if not f:
        return None
    on = np.array(f) >= c.FOCUS_THRESHOLD
    return float((on if focus else ~on).mean())
