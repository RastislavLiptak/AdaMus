"""The short statistics the console shows after a session.

Computed from ticks.jsonl and events.jsonl only; the full metrics, in
metrics.json, are computed by evaluation.metrics.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adamus import constants as c
from adamus.evaluation.metrics import PERCENTILE_KEY, spread
from adamus.evaluation.session_folder import SessionFolder, load_session_folder
from adamus.session_files import (
    EVENT_ERROR,
    EVENT_SESSION_END,
    EVENT_SESSION_START,
    REASON_END_OF_STREAM,
    REASON_ERROR,
    REASON_USER,
    name_of,
)

OK = "OK"
ERROR = "ERROR"
NO_CLEAN_SIGNAL = "NO CLEAN SIGNAL"
NO_CLEAN_SIGNAL_DETAIL = "not one clean step for the detector"
NO_STEP_DETAIL = "ended before the signal of the first step had arrived"
ENDED_BY = {
    REASON_USER: "[SPACE]",
    REASON_END_OF_STREAM: "end of file",
    REASON_ERROR: "error",
}


@dataclass(frozen=True)
class FocusStatistics:
    switched_on: int  # times f went from below FOCUS_THRESHOLD to at least it
    on_share: float  # share of controlled steps with f at least FOCUS_THRESHOLD
    focus_blocks_on: int
    focus_blocks: int
    rest_blocks_off: int
    rest_blocks: int
    f_min: float
    f_max: float
    f_mean: float
    latency_median_s: float | None
    latency_high_s: float | None


@dataclass(frozen=True)
class Statistics:
    name: str
    directory: Path
    duration_s: float
    reason: str
    verdict: str  # OK, NO CLEAN SIGNAL or ERROR
    detail: str | None  # why the verdict is not OK
    usable_share: float | None  # valid steps / all steps
    focus: FocusStatistics | None  # None without a clean step
    recording: Path | None = None  # where the samples were stored as a recording of a dataset


def statistics_of(directory: Path) -> Statistics:
    session = load_session_folder(directory)
    start, end = session.event(EVENT_SESSION_START), session.event(EVENT_SESSION_END)
    error = session.event(EVENT_ERROR)
    reason = end["reason"] if end else REASON_ERROR
    if reason == REASON_ERROR or error:
        verdict, detail = ERROR, error["message"] if error else "the session did not finish"
    elif session.controlled_ticks():
        verdict, detail = OK, None
    else:
        verdict = NO_CLEAN_SIGNAL
        detail = NO_CLEAN_SIGNAL_DETAIL if session.ticks else NO_STEP_DETAIL
    ticks = session.ticks
    return Statistics(
        name=name_of(session.directory),
        directory=session.directory,
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
    for span in session.evaluation_spans:
        assert span.end is not None
        if session.ticks[-1]["n_end"] < span.end:
            continue  # the block did not run to its end
        finished[span.label] += 1
        block_f = session.block_f(span)
        if block_f:
            block_on = float(np.mean(block_f)) >= c.FOCUS_THRESHOLD
            as_planned[span.label] += block_on == (span.label == c.FOCUS)
    latency = spread(session.control_latencies())
    return FocusStatistics(
        switched_on=int((~on[:-1] & on[1:]).sum()),
        on_share=float(on.mean()),
        focus_blocks_on=as_planned[c.FOCUS],
        focus_blocks=finished[c.FOCUS],
        rest_blocks_off=as_planned[c.REST],
        rest_blocks=finished[c.REST],
        f_min=float(f.min()),
        f_max=float(f.max()),
        f_mean=float(f.mean()),
        latency_median_s=latency["median"] if latency else None,
        latency_high_s=latency[PERCENTILE_KEY] if latency else None,
    )


def verdict_text(statistics: Statistics) -> str:
    """OK, or what went wrong and why: the coloured beginning of the Result row."""
    if statistics.detail is None:
        return statistics.verdict
    return f"{statistics.verdict} ({statistics.detail})"


def statistics_lines(statistics: Statistics) -> list[tuple[str, str]]:
    """The statistics as (label, text) rows; an empty label continues the row above."""
    minutes, seconds = divmod(round(statistics.duration_s), 60)
    result = verdict_text(statistics)
    if statistics.usable_share is not None:
        result += f" · signal usable {round(100 * statistics.usable_share)} % of the time"
    lines = [
        ("Duration", f"{minutes:02d}:{seconds:02d}   (ended by {ENDED_BY[statistics.reason]})"),
        ("Result", result),
    ]
    focus = statistics.focus
    if focus is not None:
        times = "time" if focus.switched_on == 1 else "times"
        lines.append(
            ("Focus", f"switched on {focus.switched_on} {times} · "
                      f"on {round(100 * focus.on_share)} % of the time")
        )  # fmt: skip
        if focus.focus_blocks or focus.rest_blocks:
            lines.append(
                ("", f"FOCUS blocks {focus.focus_blocks_on} of {focus.focus_blocks} on · "
                     f"REST blocks {focus.rest_blocks_off} of {focus.rest_blocks} off")
            )  # fmt: skip
        f_range = f"f {focus.f_min:.2f} to {focus.f_max:.2f}, mean {focus.f_mean:.2f}"
        lines.append(("Focus range", f_range))
        if focus.latency_median_s is not None and focus.latency_high_s is not None:
            lines.append(
                ("Latency", f"median {round(1000 * focus.latency_median_s)} ms, "
                            f"{c.PERCENTILE}th percentile {round(1000 * focus.latency_high_s)} ms")
            )  # fmt: skip
    lines.append(("Folder", shown_path(statistics.directory)))
    if statistics.recording is not None:
        lines.append(("Recording", shown_path(statistics.recording)))
    return lines


def shown_path(path: Path) -> str:
    """A path as the operator reads it: from the current folder, when it lies inside it."""
    whole, here = path.resolve(), Path.cwd().resolve()
    return str(whole.relative_to(here) if whole.is_relative_to(here) else whole)
