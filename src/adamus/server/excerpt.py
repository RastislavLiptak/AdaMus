"""A part of a session, cut out of its data: what the operator marked on the timeline, and
everything the session holds between those two moments, as one JSON document.

It reads without the window: the session and where its folder is, the part in the time of the
timeline (as the app shows it too) and in samples, a summary of the part, then its data: the
blocks of the protocol over it, every step of the detector in it, the music and how far it was
steered, every electrode, the lines of the log, the events, what the music generator was sent,
the configuration the session ran with and its files. Times are seconds of the timeline: 0 is
the first sample of EEG. A table is its columns and its rows; a list of numbers stays on one
line, so that it reads as a table.
"""

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from adamus import constants as c
from adamus.datasets import person_of
from adamus.evaluation.session_reader import SessionFolder, load_session_folder
from adamus.server.api_models import AudioTrack, Console, ConsoleLine, SignalTrack, Track
from adamus.server.dashboard_sections import caution_of, trained_on
from adamus.server.timeline import (
    LOG,
    MUSIC,
    SessionTracks,
    control_steps,
    step_times,
)
from adamus.session_format import Tick, name_of

TENTHS = 10  # the app shows times to a tenth of a second
SECONDS_PER_MINUTE, SECONDS_PER_HOUR = 60, 3600
HALF = 0.5
SECONDS, MICROVOLTS, CHANCE = 3, 2, 4  # decimals kept of each
DECIBELS = 20.0  # of a ratio of amplitudes
INDENT = "  "
ROW_MARK = "<row {}>"  # where a row of a table waits to be written in
ROW_MARKED = r'"<row (\d+)>"'
SPREAD = (np.min, np.max, np.std)  # of the samples of an electrode in the part
SAMPLE_0_LINE = 2  # of eeg.csv: its first line names the electrodes
FILES = (
    c.FILE_TICKS, c.FILE_FILTERED, c.FILE_RAW, c.FILE_EVENTS, c.FILE_GENERATOR, c.FILE_LOG,
    c.FILE_PD_LOG, c.FILE_AUDIO, c.FILE_META, c.FILE_METRICS, c.FILE_CONFIG,
)  # fmt: skip
STEP_COLUMNS = [
    "k", "end_s", "t_eeg", "block_index", "block_label", "full_window", "valid", "f", "f_held",
    "score", "w_focus", "sent_after_s",
]  # fmt: skip
EEG_COLUMNS = [
    "electrode", "raw_min_uv", "raw_max_uv", "raw_sd_uv", "filtered_min_uv", "filtered_max_uv",
    "filtered_sd_uv", "filtered_share",
]  # fmt: skip
Table = dict[str, list[Any]]


def clock(seconds: float) -> str:
    """A time of the timeline as the app writes it, to a tenth: 1:02.5, 1:02:03.5 from an hour
    on, or −0:03.0 before the first sample."""
    tenths = math.floor(abs(seconds) * TENTHS + HALF)  # halves away from zero, as the app rounds
    whole, tenth = divmod(tenths, TENTHS)
    hours, rest = divmod(whole, SECONDS_PER_HOUR)
    minutes, second = divmod(rest, SECONDS_PER_MINUTE)
    sign = "−" if seconds < 0 and tenths else ""
    shown = f"{hours}:{minutes:02d}:{second:02d}" if hours else f"{minutes}:{second:02d}"
    return f"{sign}{shown}.{tenth}"


def rounded(value: float | None, decimals: int) -> float | None:
    """A number to ``decimals``; None for one not known (JSON has no NaN)."""
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), decimals) + 0.0  # not −0.0


def overlaps(start: float, end: float, start_s: float, end_s: float) -> bool:
    return start < end_s and end > start_s


def table(columns: list[str], rows: list[list[Any]]) -> Table:
    return {"columns": columns, "rows": rows}


def dumps(value: Any) -> str:
    """JSON, indented, with a list of numbers, texts and the like on one line: a row of a table.
    Each such list is first put aside, a mark in its place, and written in after."""
    value = json.loads(json.dumps(value))  # a copy, to put the marks into
    rows: list[str] = []
    holders = [value]
    for holder in holders:  # the list grows by the dicts and lists found in it, so all are seen
        for key, item in list(holder.items() if isinstance(holder, dict) else enumerate(holder)):
            if isinstance(item, list) and not any(isinstance(inner, dict | list) for inner in item):
                holder[key] = ROW_MARK.format(len(rows))
                rows.append(json.dumps(item, ensure_ascii=False))
            elif isinstance(item, dict | list):
                holders.append(item)
    written = json.dumps(value, indent=len(INDENT), ensure_ascii=False)
    return re.sub(ROW_MARKED, lambda found: rows[int(found.group(1))], written)


# --- The parts of the excerpt -------------------------------------------------------


def session_data(session: SessionFolder) -> dict[str, Any]:
    meta = session.meta
    played = meta["playback_path"]
    data: dict[str, Any] = {
        "name": name_of(session.directory),
        "folder": str(session.directory.resolve()),
        "started": datetime.fromtimestamp(meta["started"]).isoformat(sep=" "),
        "adamus_commit": meta["adamus_commit"],
        "source": "headset" if played is None else "recording",
    }
    if played is None:
        data["headset"] = session.settings["board"]["name"]
    else:
        recording = Path(played).parent
        data["recording"] = {
            "dataset": recording.parent.name, "name": recording.name, "eeg_csv": played,
            "line_of_sample_0": SAMPLE_0_LINE, "person": person_of(recording),
        }  # fmt: skip
    trained = trained_on(session)
    data["focus_model_trained_on"] = (
        None
        if trained is None
        else {"dataset": trained["dataset"], "people": len(trained["people"])}
    )
    data["focus_model_trained_on_this_person"] = caution_of(session) is not None
    data["sampling_rate_hz"] = session.fs
    data["electrodes"] = list(meta["electrode_rows"])
    data["detector_electrodes"] = meta["detector_electrodes"]
    return data


def part_data(session: SessionFolder, start_s: float, end_s: float) -> dict[str, Any]:
    fs = session.fs
    first, last = max(0, math.ceil(start_s * fs)), math.floor(end_s * fs) - 1
    has_samples = last >= first
    return {
        "start_s": rounded(start_s, SECONDS), "end_s": rounded(end_s, SECONDS),
        "duration_s": rounded(end_s - start_s, SECONDS),
        "start_shown": clock(start_s), "end_shown": clock(end_s),
        "first_sample": first if has_samples else None,
        "last_sample": last if has_samples else None,
    }  # fmt: skip


def steps_in(
    session: SessionFolder, start_s: float, end_s: float
) -> list[tuple[float, float, Tick]]:
    """The steps whose hop lies in the part: where it begins and ends, and what it holds."""
    return [
        (start, end, tick)
        for (start, end), tick in zip(step_times(session), session.ticks, strict=True)
        if overlaps(start, end, start_s, end_s)
    ]


def blocks_in(
    session: SessionFolder, start_s: float, end_s: float, timeline_end: float
) -> list[tuple[int, str, float, float | None]]:
    """The blocks of the protocol over the part: index, label, from and to (None: until the
    stop, which is the end of the timeline)."""
    fs = session.fs
    return [
        (block.index, block.label, block.start / fs, None if block.end is None else block.end / fs)
        for block in session.spans
        if overlaps(block.start / fs, timeline_end if block.end is None else block.end / fs,
                    start_s, end_s)
    ]  # fmt: skip


def log_lines_in(tracks: SessionTracks, start_s: float, end_s: float) -> list[ConsoleLine]:
    (log,) = [console for console in tracks.timeline.consoles if console.title == LOG]
    return lines_in(log, start_s, end_s)


def lines_in(console: Console, start_s: float, end_s: float) -> list[ConsoleLine]:
    return [line for line in console.lines if start_s <= line.time_s <= end_s]


def events_in(session: SessionFolder, start_s: float, end_s: float) -> list[dict[str, Any]]:
    """The events of the part as events.jsonl holds them, each with its time on the timeline."""
    return [
        {"time_s": rounded(event["n"] / session.fs, SECONDS), **event}
        for event in session.events
        if start_s <= event["n"] / session.fs <= end_s
    ]


def generator_in(session: SessionFolder, start_s: float, end_s: float) -> list[dict[str, Any]]:
    """What the music generator was sent in the part, as generator.jsonl holds it, each with its
    time on the timeline."""
    return [
        {"time_s": rounded(sent["n"] / session.fs, SECONDS), **sent}
        for sent in session.generator_messages
        if start_s <= sent["n"] / session.fs <= end_s
    ]


def summary_data(
    session: SessionFolder, tracks: SessionTracks, start_s: float, end_s: float
) -> dict[str, Any]:
    """The part in short: the blocks asked for in it, what the detector found in each, how the
    music was steered, and how many lines and events of which kind it holds."""
    timeline = tracks.timeline
    blocks = [
        {"index": index, "label": label, "from_s": rounded(max(start, start_s), SECONDS),
         "to_s": rounded(end_s if end is None else min(end, end_s), SECONDS)}
        for index, label, start, end in blocks_in(session, start_s, end_s, timeline.end_s)
    ]  # fmt: skip
    steps = [tick for _, _, tick in steps_in(session, start_s, end_s)]
    stepped = dict.fromkeys((tick["block_index"], tick["block_label"]) for tick in steps)
    levels = [
        step.value
        for step in control_steps(session, timeline.start_s, timeline.end_s)
        if overlaps(step.start_s, step.end_s, start_s, end_s) and step.value is not None
    ]
    said = Counter(line.severity for line in log_lines_in(tracks, start_s, end_s))
    events = Counter(event["type"] for event in events_in(session, start_s, end_s))
    return {
        "blocks": blocks,
        "detection_by_block": [found_in(index, label, steps) for index, label in stepped],
        "music_toward_focus": None if not levels else {
            "first": rounded(levels[0], CHANCE), "last": rounded(levels[-1], CHANCE),
            "lowest": rounded(min(levels), CHANCE), "highest": rounded(max(levels), CHANCE),
        },
        "log_lines": {
            "all": sum(said.values()), "warnings": said["warning"], "errors": said["error"],
        },
        "events": dict(events),
    }  # fmt: skip


def found_in(index: int, label: str, steps: list[Tick]) -> dict[str, Any]:
    """What the detector found in one block of the part."""
    own = [tick for tick in steps if tick["block_index"] == index]
    valid = [f for tick in own if tick["valid"] and (f := tick["f"]) is not None]
    focus = sum(f >= c.FOCUS_THRESHOLD for f in valid)
    return {
        "block": index, "label": label, "steps": len(own), "valid": len(valid), "focus": focus,
        "rest": len(valid) - focus, "rejected": len(own) - len(valid),
    }  # fmt: skip


def protocol_data(
    session: SessionFolder, start_s: float, end_s: float, timeline_end: float
) -> list[dict[str, Any]]:
    return [
        {"index": index, "label": label, "start_s": rounded(start, SECONDS),
         "end_s": None if end is None else rounded(end, SECONDS)}
        for index, label, start, end in blocks_in(session, start_s, end_s, timeline_end)
    ]  # fmt: skip


def detector_data(session: SessionFolder, start_s: float, end_s: float) -> dict[str, Any]:
    largest = [f"max_abs_uv_{name}" for name in c.DETECTOR_ELECTRODES]
    rows = [step_row(end, tick) for _, end, tick in steps_in(session, start_s, end_s)]
    return {
        "hop_s": session.steps.hop / session.fs, "window_s": c.WINDOW_S,
        "focus_threshold": c.FOCUS_THRESHOLD, "artifact_electrodes": list(c.ARTIFACT_ELECTRODES),
        "artifact_uv": c.ARTIFACT_UV, "noise_uv": c.NOISE_UV,
        "steps": table([*STEP_COLUMNS, *largest], rows),
    }  # fmt: skip


def step_row(end: float, tick: Tick) -> list[Any]:
    sent = tick["t_sent"]
    return [
        tick["k"], rounded(end, SECONDS), tick["t_eeg"], tick["block_index"], tick["block_label"],
        tick["full_window"], tick["valid"], rounded(tick["f"], CHANCE), tick["f_held"],
        rounded(tick["score"], CHANCE), rounded(tick["w_focus"], CHANCE),
        None if sent is None else rounded(sent - tick["t_eeg"], CHANCE),
        *(rounded(value, MICROVOLTS) for value in tick["max_abs_uv"]),
    ]  # fmt: skip


def music_data(
    session: SessionFolder, tracks: SessionTracks, start_s: float, end_s: float
) -> dict[str, Any]:
    timeline = tracks.timeline
    audio = next((track for track in all_tracks(tracks) if isinstance(track, AudioTrack)), None)
    steered = [
        [rounded(max(step.start_s, start_s), SECONDS), rounded(min(step.end_s, end_s), SECONDS),
         rounded(step.value, CHANCE)]
        for step in control_steps(session, timeline.start_s, timeline.end_s)
        if overlaps(step.start_s, step.end_s, start_s, end_s)
    ]  # fmt: skip
    data: dict[str, Any] = {"recorded": audio is not None}
    if audio is not None:
        data |= {
            "start_s": rounded(audio.start_s, SECONDS),
            "end_s": rounded(audio.start_s + audio.duration_s, SECONDS),
            **loudness(audio, tracks.samples[MUSIC], start_s, end_s),
        }
    data["toward_focus"] = table(["from_s", "to_s", "value"], steered)
    return data


def loudness(
    audio: AudioTrack, peaks: np.ndarray, start_s: float, end_s: float
) -> dict[str, float | None]:
    """The loudest peak of the music in the part, and the middle of its peaks, in dBFS; None
    where the music does not reach the part."""
    low, high = peaks[: audio.buckets], peaks[audio.buckets :]
    first = max(0, math.floor((start_s - audio.start_s) * audio.rate_hz))
    last = min(audio.buckets, math.ceil((end_s - audio.start_s) * audio.rate_hz))
    if first >= last:
        return {"peak_dbfs": None, "median_peak_dbfs": None}
    level = np.maximum(np.abs(low[first:last]), np.abs(high[first:last]))
    return {
        "peak_dbfs": dbfs(float(level.max())),
        "median_peak_dbfs": dbfs(float(np.median(level))),
    }


def dbfs(amplitude: float) -> float | None:
    return rounded(DECIBELS * math.log10(amplitude), MICROVOLTS) if amplitude > 0 else None


def all_tracks(tracks: SessionTracks) -> list[Track]:
    return [track for group in tracks.timeline.groups for track in group.tracks]


def eeg_data(tracks: SessionTracks, start_s: float, end_s: float) -> dict[str, Any]:
    """Every electrode over the part: raw as the source gave it less its median over the whole
    session, and filtered, what the detector worked on, for its electrodes only."""
    rows = []
    for track in all_tracks(tracks):
        if not isinstance(track, SignalTrack):
            continue
        values = tracks.samples[track.id].reshape(track.layers, track.samples)
        first = min(track.samples, max(0, math.ceil(start_s * track.rate_hz)))
        last = min(track.samples, max(0, math.floor(end_s * track.rate_hz)))
        filtered = values[1, first:last] if track.layers > 1 else np.empty(0)
        share = rounded(float(np.isfinite(filtered).mean()), SECONDS) if filtered.size else None
        rows.append([track.title, *spread(values[0, first:last]), *spread(filtered), share])
    return {"raw_less": "median of the session", **table(EEG_COLUMNS, rows)}


def spread(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return [None] * len(SPREAD)
    return [rounded(float(stat(finite)), MICROVOLTS) for stat in SPREAD]


def log_data(tracks: SessionTracks, start_s: float, end_s: float) -> Table:
    rows: list[list[Any]] = [
        [rounded(line.time_s, SECONDS), line.source, line.severity, line.text]
        for line in log_lines_in(tracks, start_s, end_s)
    ]
    return table(["time_s", "source", "severity", "text"], rows)


def excerpt_of(folder: Path, tracks: SessionTracks, start_s: float, end_s: float) -> str:
    """The part of the session in ``folder`` from ``start_s`` to ``end_s`` of its timeline, cut
    out of its data; ``tracks`` are what its timeline shows."""
    session = load_session_folder(folder)
    present = [name for name in FILES if (folder / name).is_file()]
    data = {
        "session": session_data(session),
        "part": part_data(session, start_s, end_s),
        "summary": summary_data(session, tracks, start_s, end_s),
        "protocol": protocol_data(session, start_s, end_s, tracks.timeline.end_s),
        "detector": detector_data(session, start_s, end_s),
        "music": music_data(session, tracks, start_s, end_s),
        "eeg": eeg_data(tracks, start_s, end_s),
        "log": log_data(tracks, start_s, end_s),
        "events": events_in(session, start_s, end_s),
        "generator": generator_in(session, start_s, end_s),
        "configuration": session.settings,
        "files": {name: str((folder / name).resolve()) for name in present},
    }
    return dumps(data) + c.LINE_END
