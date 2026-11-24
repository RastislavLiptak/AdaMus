"""The dashboard of one session. Its overview: what it is, what bears on how its results are
read, and its main results; its details: every other fact in sections (dashboard_sections).

A session that cannot be read whole (one recorded by an older version of the program, or one
that failed early) still gets what can be read, and the reason for the rest.
"""

from pathlib import Path
from typing import Any

from adamus import constants as c
from adamus.activity_log import ACTION_ANALYSIS_FAILED, ACTION_SESSION_STARTED, ACTION_START_FAILED
from adamus.errors import AdamusError
from adamus.evaluation.session_reader import NOT_COMPLETE, load_session_folder
from adamus.evaluation.session_statistics import Statistics, statistics_of
from adamus.server.api_models import (
    Colour,
    Dashboard,
    Format,
    Highlight,
)
from adamus.server.dashboard_sections import (
    activity_section,
    blocks_section,
    caution_of,
    configuration_section,
    detection_section,
    files_section,
    music_section,
    part,
    processing_section,
    read_metrics,
    run_section,
    signal_section,
    software_section,
    summary_facts,
    value,
)
from adamus.server.timeline import EEG_GROUP, FOCUSED, GROUP_COLOURS, MUSIC_GROUP, RESTING
from adamus.session_format import name_of, read_jsonl

UNREADABLE = "This session cannot be read"
DID_NOT_START = "This session did not start"
# The colours of the kinds of result: those of their groups of tracks on the timeline, and of
# focus and rest the colours they are drawn in there.
SIGNAL, MUSIC = GROUP_COLOURS[EEG_GROUP], GROUP_COLOURS[MUSIC_GROUP]
PENDING = "Computing the metrics of the session"  # while the session is running or analysed
#: What reading the files of a session raises when they are incomplete or of another version.
UNREADABLE_PROBLEMS = (AdamusError, OSError, KeyError, ValueError)

# --- What cannot be read ----------------------------------------------------------


def why_unreadable(folder: Path, problem: Exception) -> str:
    """What keeps the session in ``folder`` from being read, without the path of the folder,
    which is the program's own matter and not shown."""
    if isinstance(problem, KeyError):
        return f"{UNREADABLE}: its files have no {problem.args[0]!r}"
    reason = str(problem) if isinstance(problem, AdamusError) else repr(problem)
    return f"{UNREADABLE}: {reason.removeprefix(f'{folder} {NOT_COMPLETE}: ')}"


# --- The first look -------------------------------------------------------------


def blocks_found(found: int, blocks: int) -> str:
    """Of how many blocks of a state the detector found it in, after what the share is of."""
    return f"{found} of {blocks} {'block' if blocks == 1 else 'blocks'}"


def highlights_of(statistics: Statistics, metrics: dict[str, Any]) -> list[Highlight]:
    """The results that tell first how the session went, each with what it means, in the
    colour of its kind: focus, rest, music, signal."""
    focus = statistics.focus
    music = part(metrics, "music_response")
    in_focus = f" · {blocks_found(focus.focus_blocks_on, focus.focus_blocks)}" if focus else ""
    in_rest = f" · {blocks_found(focus.rest_blocks_off, focus.rest_blocks)}" if focus else ""
    shown: list[tuple[str, Any, Format, str, Colour]] = [
        ("Focus Detected", focus and focus.focus_found, "percent",
         f"of the steps of focus blocks{in_focus}", FOCUSED),
        ("Rest Detected", focus and focus.rest_found, "percent",
         f"of the steps of rest blocks{in_rest}", RESTING),
        ("Music Lag", music.get("lag_s"), "duration", "until the music followed the detection",
         MUSIC),
        ("Music Follows Detection", music.get("rho"), "decimal",
         "Spearman ρ, 1 when it follows every change", MUSIC),
        ("Usable Signal", statistics.usable_share, "percent",
         "of the steps clean enough to use", SIGNAL),
    ]  # fmt: skip
    return [
        Highlight(title=title, value=value(raw, form), caption=caption, colour=colour)
        for title, raw, form, caption, colour in shown
    ]


# --- The whole, readable ----------------------------------------------------------


def readable_dashboard(
    folder: Path, own_recording: Path | None, activity: list[dict[str, Any]], pending: str | None
) -> Dashboard:
    """The dashboard of a session that can be read."""
    session = load_session_folder(folder)
    statistics = statistics_of(session)
    metrics = read_metrics(folder)
    return Dashboard(
        name=name_of(folder),
        problem=start_problem(activity) or metrics_problem(activity),
        pending=pending,
        summary=summary_facts(session, statistics, activity, own_recording),
        caution=caution_of(session),
        highlights=highlights_of(statistics, metrics),
        sections=[
            run_section(session, statistics, activity),
            blocks_section(session, metrics),
            detection_section(statistics, metrics),
            signal_section(session, statistics, metrics),
            music_section(metrics),
            processing_section(metrics),
            configuration_section(session),
            software_section(session),
            activity_section(activity),
        ],
    )


def metrics_problem(activity: list[dict[str, Any]]) -> str | None:
    """Why the session has no metrics, when their computation failed."""
    failed = [action for action in activity if action["type"] == ACTION_ANALYSIS_FAILED]
    return failed[-1]["message"] if failed else None


def start_problem(activity: list[dict[str, Any]]) -> str | None:
    """Why the session did not run, when it failed after its folder was made. A start that
    failed before (no folder made) is logged with the next session, before it started."""
    why = None
    for action in activity:
        if action["type"] == ACTION_SESSION_STARTED:
            why = None  # what failed before was another attempt
        elif action["type"] == ACTION_START_FAILED:
            why = action["message"]
    return f"{DID_NOT_START}: {why}" if why else None


def dashboard_of(folder: Path, own_recording: Path | None, analysis_pending: bool) -> Dashboard:
    """The dashboard of the session in ``folder``."""
    name = name_of(folder)
    pending = PENDING if analysis_pending else None
    activity: list[dict[str, Any]] = []
    try:
        activity_file = folder / c.FILE_ACTIVITY
        activity = read_jsonl(activity_file) if activity_file.is_file() else []
        return readable_dashboard(folder, own_recording, activity, pending)
    except UNREADABLE_PROBLEMS as problem:
        said = start_problem(activity) or why_unreadable(folder, problem)
        return Dashboard(
            name=name, problem=said, pending=pending, summary=[],
            caution=None, highlights=[], sections=[files_section(folder)],
        )  # fmt: skip
