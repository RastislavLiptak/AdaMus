"""The facts of the dashboard of one session: what it is, for its overview, and every other
fact the program knows about it, in the sections of its details.

The facts come from the files of the session folder alone: its statistics, its
metrics.json, its meta.json, its configuration, the focus model it ran with, its
events and its activity.jsonl.
"""

import json
import re
from pathlib import Path
from typing import Any

from adamus import constants as c
from adamus.activity_log import (
    ACTION_APP_ENDED,
    ACTION_PROGRAM_STOPPED,
    ACTION_RECORDING_NOT_STORED,
    ACTION_SESSION_STARTED,
)
from adamus.datasets import person_of
from adamus.evaluation.metrics import PERCENTILE_KEY
from adamus.evaluation.session_reader import SessionFolder
from adamus.evaluation.session_statistics import OK, FocusStatistics, Statistics
from adamus.realtime.music_generator import REPORT_OUTPUT_LOST
from adamus.server.api_models import (
    Fact,
    Format,
    Section,
    Table,
    Value,
)
from adamus.session_format import (
    EVENT_ERROR,
    REASON_END_OF_STREAM,
    REASON_ERROR,
    REASON_USER,
    read_json,
)

ENDED_BY = {
    REASON_USER: "Stopped by the operator",
    REASON_END_OF_STREAM: "End of the recording",
    REASON_ERROR: "Error",
}
# A session ended by the app or the program going: its record says only that it was stopped.
ENDED_BY_GOING = {
    ACTION_APP_ENDED: "The app was closed",
    ACTION_PROGRAM_STOPPED: "The program was stopped",
}
UNTIL_STOPPED = "Until stopped"  # a block of no fixed length
RECORDING_SOURCE = "A recording of a dataset"
RECORDING_LABELS = "The blocks it was recorded in"  # the protocol of the recording
HEADSET_LABELS = "The protocol the person was asked to follow"
TRAINED_ON_PERSON = (
    "The focus model was trained on this person: {person} is one of the {people} people it "
    "learned from, so the detection does better here than on someone it never saw."
)
NOT_STORED = "The recording could not be stored"  # what the headset delivered stays in the session
FIRST_FOUND = "The first one found"  # a headset without a serial number in the configuration
HIGH = f"{c.PERCENTILE}th Percentile"
#: The settings of config.yaml by the names the dashboard gives them; any other setting
#: (one of an older or newer version of the program) is shown by its path.
SETTING_LABELS = {
    "board.name": "Headset",
    "board.id": "BrainFlow Board",
    "board.serial_number": "Headset Serial Number",
    "generator.folder": "Generator Folder",
    "generator.model": "Music Model",
    "generator.prompt_idle": "Music at Rest",
    "generator.prompt_focus": "Music in Focus",
    "timeouts.stream_s": "Stream Timeout",
    "timeouts.pd_connect_s": "Pure Data Connection Timeout",
    "timeouts.model_load_s": "Model Load Timeout",
    "poll_s": "Poll Interval",
}
SERIAL_NUMBER = "board.serial_number"
SECONDS = "_s"  # the ending of a setting given in seconds
SHOWN_ELSEWHERE = {"protocol"}  # in the section of the blocks
RELEASE = r"\d+(\.\d+)+"  # the numbers of a release, as 0.57.0
Item = Fact | Table


def value(raw: Any, form: Format) -> Value:
    return Value(value=raw, format=form)


def fact(label: str, raw: Any, form: Format = "text") -> Fact:
    return Fact(label=label, value=value(raw, form))


def part(metrics: dict[str, Any], key: str) -> dict[str, Any]:
    """A part of metrics.json that holds several numbers; empty when it holds none."""
    found = metrics.get(key)
    return found if isinstance(found, dict) else {}


def spread_facts(label: str, numbers: dict[str, Any], keys: tuple[str, ...]) -> list[Fact]:
    """The median, the high percentile and the maximum of something, as far as known."""
    names = {"median": "Median", PERCENTILE_KEY: HIGH, "max": "Maximum"}
    return [fact(f"{label} ({names[key]})", numbers.get(key), "duration") for key in keys]


def read_metrics(folder: Path) -> dict[str, Any]:
    path = folder / c.FILE_METRICS
    return read_json(path) if path.is_file() else {}


# --- The sections ----------------------------------------------------------------


def source_facts(session: SessionFolder) -> list[Fact]:
    """The headset, or the recording of a dataset played in its place."""
    played = session.meta["playback_path"]
    if played is None:
        return [fact("Source", f"Headset ({session.settings['board']['name']})")]
    recording = Path(played).parent
    return [
        fact("Source", RECORDING_SOURCE),
        fact("Recording", f"{recording.parent.name} / {recording.name}"),
    ]


def run_section(
    session: SessionFolder, statistics: Statistics, activity: list[dict[str, Any]]
) -> Section:
    """How the session ran: what told focus from rest, what the detector was trained on, and
    what ended it."""
    labels = HEADSET_LABELS if session.meta["playback_path"] is None else RECORDING_LABELS
    items: list[Item] = [fact("Focus and Rest From", labels)]
    trained = trained_on(session)
    if trained is not None:
        people = len(trained["people"])
        items.append(fact("Focus Model Trained On", f"{trained['dataset']}, {people} people"))
    items.append(fact("Ended By", ended_by(statistics.reason, activity)))
    return Section(title="Run", items=items)


def trained_on(session: SessionFolder) -> dict[str, Any] | None:
    """What the focus model the session ran with was trained on; None without its copy."""
    model = session.directory / c.FOCUS_MODEL_FILE
    return read_json(model)["trained_on"] if model.is_file() else None


def caution_of(session: SessionFolder) -> str | None:
    """That the focus model learned from the very person a played recording is of, so that its
    results there are better than on someone it never saw; None otherwise, also where the
    recording no longer says whose it is."""
    played, trained = session.meta["playback_path"], trained_on(session)
    recording = Path(played).parent if played is not None else None
    if recording is None or trained is None or not (recording / c.FILE_DATASET_PROTOCOL).is_file():
        return None
    person, people = person_of(recording), trained["people"]
    if person not in people:
        return None
    return TRAINED_ON_PERSON.format(person=person, people=len(people))


def failures_of(
    session: SessionFolder, statistics: Statistics, activity: list[dict[str, Any]]
) -> list[str]:
    """What went wrong: first why the result is not OK (the error and where it happened, or why
    there was nothing to detect from), then each time what the headset delivered could not be
    stored as a recording; none when it all went well."""
    failures = []
    error = session.event(EVENT_ERROR)
    if error is not None:
        failures.append(f"Error in {error.get('where')}: {error.get('message')}")
    elif statistics.detail is not None:
        failures.append(statistics.detail[:1].upper() + statistics.detail[1:])
    failures += [
        f"{NOT_STORED}: {entry['message']}"
        for entry in activity
        if entry["type"] == ACTION_RECORDING_NOT_STORED
    ]
    return failures


def ended_by(reason: str, activity: list[dict[str, Any]]) -> str:
    """What ended the session: the app or the program going, or the reason its record gives
    (as it gives it, when a later program no longer knows it)."""
    going = [ENDED_BY_GOING[entry["type"]] for entry in activity if entry["type"] in ENDED_BY_GOING]
    return going[-1] if going else ENDED_BY.get(reason, reason)


def summary_facts(
    session: SessionFolder, statistics: Statistics, own_recording: Path | None
) -> list[Fact]:
    """When, how long and on what the session ran, its result and where it is kept."""
    items = [
        fact("Started", session.meta["started"], "datetime"),
        fact("Duration", statistics.duration_s, "duration"),
        *source_facts(session),
        Fact(
            label="Result",
            value=value(statistics.verdict, "text"),
            status="good" if statistics.verdict == OK else "bad",
        ),  # fmt: skip
    ]
    if own_recording is not None:
        items.append(fact("Own Recording", "Stored, to be played again"))
    return items


def focus_facts(focus: FocusStatistics) -> list[Item]:
    return [
        fact("Times Focus Switched On", focus.switched_on, "count"),
        fact("Time With Focus On", focus.on_share, "percent"),
        fact("Focus Blocks With Focus On", f"{focus.focus_blocks_on} of {focus.focus_blocks}"),
        fact("Rest Blocks With Focus Off", f"{focus.rest_blocks_off} of {focus.rest_blocks}"),
        fact("Lowest Chance of Focus", focus.f_min, "decimal"),
        fact("Highest Chance of Focus", focus.f_max, "decimal"),
        fact("Mean Chance of Focus", focus.f_mean, "decimal"),
    ]


def detection_section(statistics: Statistics, metrics: dict[str, Any]) -> Section:
    """How the detector followed the person."""
    items = focus_facts(statistics.focus) if statistics.focus is not None else []
    response = part(metrics, "response_time_s")
    items += [
        fact("Separation of Focus From Rest (AUC)", metrics.get("state_separation_auc"), "decimal"),
        fact("Mean Chance of Focus, Focus Less Rest", metrics.get("mean_f_difference"), "decimal"),
        fact("Time to Focus On (Median)", response.get("rise_median_s"), "duration"),
        fact("Time to Focus Off (Median)", response.get("fall_median_s"), "duration"),
    ]
    return Section(title="Detection", items=items)


def signal_section(
    session: SessionFolder, statistics: Statistics, metrics: dict[str, Any]
) -> Section:
    """What the source delivered and how much of it the detector could use."""
    present = session.meta["detector_electrodes"]
    missing = [name for name in c.DETECTOR_ELECTRODES if name not in present]
    complete = part(metrics, "record_completeness")
    items: list[Item] = [
        fact("Usable Signal", statistics.usable_share, "percent"),
        fact("Received Sampling Rate", metrics.get("sampling_rate_hz"), "frequency"),
        fact("Longest Gap Between Samples", metrics.get("longest_sample_gap_s"), "duration"),
    ]
    if missing:
        items.append(fact("Electrodes Recorded as Zeros", ", ".join(missing)))
    if complete:
        items.append(fact("Steps Recorded", complete["ticks"], "count"))
        items.append(fact("Record Complete", "Yes" if complete["passed"] else "No"))
    return Section(title="Signal", items=items)


def output_lost_at(activity: list[dict[str, Any]]) -> float | None:
    """When the audio output was disconnected while the session ran, if it was."""
    started = [k for k, entry in enumerate(activity) if entry["type"] == ACTION_SESSION_STARTED]
    after = activity[started[0] :] if started else []
    lost = [entry["t"] for entry in after if entry["type"] == REPORT_OUTPUT_LOST]
    return lost[0] if lost else None


def music_section(
    session: SessionFolder, metrics: dict[str, Any], activity: list[dict[str, Any]]
) -> Section:
    """Where the music played, how quickly it was told, and how it followed."""
    latency, interval = part(metrics, "control_latency_s"), part(metrics, "step_interval_s")
    response, total = part(metrics, "music_response"), part(metrics, "total_response_s")
    dropouts = part(metrics, "audio_dropouts")
    lost = output_lost_at(activity)
    items: list[Item] = [
        fact("Audio Output", session.meta.get("audio_output")),
        *([fact("Audio Output Disconnected", lost, "datetime")] if lost is not None else []),
        *spread_facts("Control Latency", latency, ("median", PERCENTILE_KEY, "max")),
        *spread_facts("Interval Between Steps", interval, ("median", PERCENTILE_KEY)),
        fact("Music Following Focus (Spearman)", response.get("rho"), "decimal"),
        fact("Music Lag", response.get("lag_s"), "duration"),
        fact("Time to Audible Focus On", total.get("rise_s"), "duration"),
        fact("Time to Audible Focus Off", total.get("fall_s"), "duration"),
        fact("Audio Dropouts", dropouts.get("count"), "count"),
        fact("Audio Dropouts in Total", dropouts.get("total_s"), "duration"),
        fact("First Sound", dropouts.get("first_sound_s"), "duration"),
        fact("Recording Length Error", metrics.get("recording_length_error_s"), "duration"),
    ]
    return Section(title="Music", items=items)


def processing_section(metrics: dict[str, Any]) -> Section:
    keys = ("median", PERCENTILE_KEY, "max")
    items: list[Item] = [*spread_facts("Processing Time", part(metrics, "processing_time_s"), keys)]
    return Section(title="Processing", items=items)


def blocks_section(session: SessionFolder, metrics: dict[str, Any]) -> Section:
    """Every block the session got to: how long, how clean, and how soon the detector followed."""
    valid = {block["index"]: block for block in metrics.get("valid_steps_per_block") or []}
    response = part(metrics, "response_time_s")
    times = {
        c.FOCUS: iter(response.get("rise_s") or []),
        c.REST: iter(response.get("fall_s") or []),
    }
    rows = []
    for span in session.begun_spans:
        block = session.protocol[span.index]
        length = (
            value(block.duration_s, "duration")
            if block.duration_s
            else value(UNTIL_STOPPED, "text")
        )
        share = valid.get(span.index, {}).get("ratio")
        reached = next(times[span.label], None) if span.label in times else None
        rows.append([
            value(span.index + 1, "count"), value(span.label.capitalize(), "text"), length,
            value(share, "percent"), value(reached, "duration"),
        ])  # fmt: skip
    columns = ["Block", "State", "Length", "Clean Steps", "Detector Followed After"]
    return Section(title="Blocks", items=[Table(label="Blocks", columns=columns, rows=rows)])


def setting_value(path: str, raw: Any) -> Value:
    """A setting as the dashboard shows it: seconds as a time, a fraction as a number, a
    whole number as written (most name something, as a port or a board does, and are not
    grouped by thousands), anything else as its YAML text."""
    if path == SERIAL_NUMBER and not raw:
        return value(FIRST_FOUND, "text")
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        if path.endswith(SECONDS):
            return value(raw, "duration")
        return value(str(raw), "text") if isinstance(raw, int) else value(raw, "decimal")
    return value(raw if isinstance(raw, str) else json.dumps(raw), "text")


def opened(path: str, raw: Any) -> list[tuple[str, Any]]:
    """The settings inside a section, by their paths; a setting that is no section, alone."""
    if isinstance(raw, dict):
        return [(f"{path}.{key}", inner) for key, inner in raw.items()]
    return [(path, raw)]


def settings_of(settings: dict[str, Any]) -> list[tuple[str, Any]]:
    """Every setting, by its path (e.g. timeouts.stream_s), in the order of the file."""
    found = [setting for path, raw in settings.items() for setting in opened(path, raw)]
    return [(path, raw) for path, raw in found if path not in SHOWN_ELSEWHERE]


def configuration_section(session: SessionFolder) -> Section:
    """Every setting the session ran with, also those a later version no longer knows."""
    items: list[Item] = [
        Fact(label=SETTING_LABELS.get(path, path), value=setting_value(path, raw))
        for path, raw in settings_of(session.settings)
    ]
    return Section(title="Configuration", items=items)


def release(version_line: str | None) -> str | None:
    """The release in what a program says of its version: 0.57.0 of 'Pd-0.57.0 ("")'."""
    found = re.search(RELEASE, version_line or "")
    return found.group() if found else version_line


def software_section(session: SessionFolder) -> Section:
    """The program and the libraries the session ran with."""
    meta = session.meta
    items: list[Item] = [
        fact("AdaMus Commit", meta["adamus_commit"]),
        fact("Python", meta["python"]),
        fact("Pure Data", release(meta["pd"])),
        *(fact(name, version) for name, version in meta["versions"].items()),
    ]
    return Section(title="Software", items=items)


def activity_section(activity: list[dict[str, Any]]) -> Section:
    """What the program did around the session, from the checks at its start on."""
    rows = []
    for entry in activity:
        details = ", ".join(
            f"{key}: {field}" for key, field in entry.items() if key not in ("t", "type")
        )
        rows.append(
            [value(entry["t"], "datetime"), value(entry["type"], "text"), value(details, "text")]
        )
    table = Table(label="Activity", columns=["Time", "Action", "Details"], rows=rows)
    return Section(title="Activity", items=[table])


# --- The folder, when nothing more can be read -------------------------------------


def files_section(folder: Path) -> Section:
    """What lies in the folder: all that can be shown of a session that cannot be read."""
    names = sorted(path.name for path in folder.iterdir())
    return Section(title="Files", items=[fact("Files", ", ".join(names))])
