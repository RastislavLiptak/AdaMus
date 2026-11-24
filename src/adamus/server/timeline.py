"""The tracks of the timeline of one session, read from its folder.

Time 0 is the first sample of EEG, and the time of the timeline is that of the samples: what
has a time of day (the music, the lines of the logs) lies at the samples that came then.

Every electrode the source recorded is a track: its raw signal, less its median so that it
lies around zero as the filtered one does, and for an electrode of the detector the filtered
signal the detector worked on. filtered.f32 holds the window of every step; the steps
overlap, so each adds what it ends with that the step before did not cover: the hop, or its
whole window when it is the first or follows a gap. The music is an envelope with one bucket
per sample of EEG; it begins when the generator started recording, before the first sample.
Then come the blocks the person was asked to keep, the chance of focus after every step (and
which steps had a signal too disturbed to use), and how far toward focus the music was sent.
The lines of adamus.log and pd.log come with the moment each was written, so that playing
the session shows them as they came; a line without a time goes on with the one before it,
as a traceback does. So do the events of the session, each at the sample it came with. The
app is told what to draw only: titles, groups, captions, the colours of the palette and
what they stand for.
"""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile

from adamus import constants as c
from adamus.evaluation.metrics import read_audio
from adamus.evaluation.session_reader import SessionFolder, load_session_folder
from adamus.realtime.generator_messages import REC_START, START_W_FOCUS
from adamus.server.api_models import (
    AudioTrack,
    Band,
    Colour,
    Console,
    ConsoleLine,
    CurveStep,
    CurveTrack,
    LegendEntry,
    Severity,
    SignalTrack,
    Timeline,
    Track,
    TrackGroup,
)
from adamus.session_format import EVENT_ERROR, read_log_line

SCALE_PERCENTILE = 99.5  # of the filtered signal that fits into its track; spikes may leave it
NICE_STEPS = (1.0, 2.0, 5.0)  # a scale is one of these times a power of ten
DECIMAL = 10.0
MUSIC, DETECTION, CONTROL = "music", "detection", "control"  # track ids
SIGNAL_PREFIX = "signal-"  # then the name of the electrode
TITLES = {MUSIC: "Music", DETECTION: "Detection", CONTROL: "Music Control"}
# the groups of the tracks, from the top: what was heard, the focus that steered it, the EEG
# the focus was found in
MUSIC_GROUP, FOCUS_GROUP, EEG_GROUP = "Music", "Focus", "EEG"
# Their colours, which the results of each kind take on the dashboard too.
GROUP_COLOURS: dict[str, Colour] = {EEG_GROUP: "teal", FOCUS_GROUP: "purple", MUSIC_GROUP: "pink"}
CAPTIONS = {DETECTION: "chance of focus under the protocol", CONTROL: "0–100 % toward focus"}
SIGNAL_UNIT = "µV"
# the colours the states are drawn in
FOCUSED: Colour = "orange"
RESTING: Colour = "indigo"
REJECTED: Colour = "gray"
STATE_COLOURS: dict[str, Colour] = {c.FOCUS: FOCUSED, c.REST: RESTING}  # of the blocks
LEGEND: dict[Colour, str] = {FOCUSED: "Focus", RESTING: "Rest", REJECTED: "Rejected"}
LOG, EVENTS = "Log", "Events"  # the titles of the consoles
SAMPLE_TYPE = np.float32  # of the samples sent to the app
Placing = Callable[[float], float]  # a moment of the time of day to its time on the timeline
LOG_SOURCES = ((c.PROGRAM_NAME, c.FILE_LOG), ("Pd", c.FILE_PD_LOG))  # who wrote each log
SEVERITIES: dict[str, Severity] = {
    "DEBUG": "debug", "INFO": "info", "WARNING": "warning", "ERROR": "error", "CRITICAL": "error",
}  # fmt: skip  # the levels adamus.log writes; Pd marks an error with ERROR_MARK
ERROR_MARK = "error:"


@dataclass(frozen=True)
class SessionTracks:
    """Everything the timeline of one session shows; the samples, ready to be sent."""

    timeline: Timeline
    samples: dict[str, np.ndarray]  # by track id


def nice_scale(values: np.ndarray) -> float:
    """The smallest of 1, 2 and 5 times a power of ten that holds most of ``values``."""
    finite = np.abs(values[np.isfinite(values)])
    if not finite.any():
        return NICE_STEPS[0]
    high = float(np.percentile(finite, SCALE_PERCENTILE))
    power = DECIMAL ** math.floor(math.log10(high)) if high > 0 else 1.0
    return next(step * power for step in (*NICE_STEPS, DECIMAL) if step * power >= high)


def filtered_signal(session: SessionFolder, samples: int) -> np.ndarray:
    """The filtered signal of every electrode of the detector [electrodes x samples]; NaN
    where no step covered a sample."""
    steps = session.steps
    signal = np.full((len(c.DETECTOR_ELECTRODES), samples), np.nan)
    path = session.directory / c.FILE_FILTERED
    if not path.is_file():  # the session ended before it was opened
        return signal
    size = len(c.DETECTOR_ELECTRODES) * steps.window
    values = np.fromfile(path, dtype=c.FILTERED_DTYPE)
    whole = len(values) // size  # a record cut short when the session was killed is left out
    records = values[: whole * size].reshape(whole, len(c.DETECTOR_ELECTRODES), steps.window)
    previous_end: int | None = None
    # Record k is the window of the step of line k of ticks.jsonl: both are written together.
    for tick, record in zip(session.ticks, records, strict=False):
        end = int(tick["n_end"])
        if end > samples:
            break
        new = steps.window if previous_end is None else min(steps.window, end - previous_end)
        signal[:, end - new : end] = record[:, -new:]
        previous_end = end
    return signal


def signal_tracks(
    session: SessionFolder, raw: np.ndarray
) -> tuple[list[Track], dict[str, np.ndarray]]:
    """One track per electrode the source recorded, whatever the program computes with: its
    raw signal, and over it the filtered one where the session filtered it."""
    samples = raw.shape[1] if raw.size else 0
    filtered = filtered_signal(session, samples)
    tracks: list[Track] = []
    data: dict[str, np.ndarray] = {}
    for name, row_index in session.meta["electrode_rows"].items():
        row = raw[row_index] if samples else np.empty(0)
        centered = row - np.nanmedian(row) if samples else row
        layers = [centered]
        if name in c.DETECTOR_ELECTRODES:
            own = filtered[c.DETECTOR_ELECTRODES.index(name)]
            layers += [own] if np.isfinite(own).any() else []
        scale = nice_scale(layers[-1])
        tracks.append(
            SignalTrack(
                id=SIGNAL_PREFIX + name, title=name, caption=f"±{scale:g} {SIGNAL_UNIT}",
                scale=scale, rate_hz=session.fs, samples=samples, layers=len(layers),
            )
        )  # fmt: skip
        data[SIGNAL_PREFIX + name] = np.concatenate(layers).astype(SAMPLE_TYPE)
    return tracks, data


def envelope(mono: np.ndarray, rate: int, buckets_per_s: int) -> np.ndarray:
    """The lowest sample of every bucket, then the highest of every bucket."""
    count = math.ceil(len(mono) * buckets_per_s / rate)
    if count == 0:
        return np.empty(0, dtype=SAMPLE_TYPE)
    edges = np.minimum(np.round(np.arange(count) * rate / buckets_per_s).astype(int), len(mono) - 1)
    low, high = np.minimum.reduceat(mono, edges), np.maximum.reduceat(mono, edges)
    peaks: np.ndarray = np.concatenate([low, high]).astype(SAMPLE_TYPE)
    return peaks


def on_the_samples(stamps: np.ndarray, fs: float) -> Placing:
    """Where a moment of the time of day lies on the timeline, whose time is that of the samples
    (sample k at k / fs): between two samples, at the samples that came then; before the first
    and after the last, at the pace of the clock. So samples a headset lost do not shift the
    music and the logs against the signals."""
    on_the_axis = np.arange(len(stamps)) / fs

    def place(moment: float) -> float:
        if moment <= stamps[0]:
            return moment - float(stamps[0])
        if moment >= stamps[-1]:
            return float(on_the_axis[-1]) + moment - float(stamps[-1])
        return float(np.interp(moment, stamps, on_the_axis))

    return place


def placing(stamps: np.ndarray | None, fs: float, rec_start: float | None) -> Placing | None:
    """How moments are put on the timeline: by the samples that came then; without samples
    from the start of the music; None when there is neither."""
    if stamps is not None:
        return on_the_samples(stamps, fs)
    return None if rec_start is None else (lambda moment: moment - rec_start)


def music_track(
    session: SessionFolder, place: Placing | None, rec_start: float | None
) -> tuple[AudioTrack, np.ndarray] | None:
    """The music and its envelope, placed where it began (``rec_start``); None without audio or
    without a start of it."""
    try:
        mono, _, rate = read_audio(session.directory / c.FILE_AUDIO)
    except soundfile.SoundFileError:  # also when the file is not there
        return None
    if rec_start is None or len(mono) == 0:
        return None
    peaks = envelope(mono, rate, session.fs)
    start = place(rec_start) if place is not None else 0.0
    track = AudioTrack(
        id=MUSIC, title=TITLES[MUSIC], caption=None, start_s=start, duration_s=len(mono) / rate,
        rate_hz=session.fs, buckets=len(peaks) // 2,
    )  # fmt: skip
    return track, peaks


def protocol_bands(session: SessionFolder, end_s: float) -> list[Band]:
    """The blocks of the protocol; one that lasts until the stop ends with the timeline."""
    return [
        Band(
            start_s=span.start / session.fs,
            end_s=end_s if span.end is None else span.end / session.fs,
            title=span.label.capitalize(), colour=STATE_COLOURS.get(span.label),
        )
        for span in session.spans
        if span.start / session.fs < end_s
    ]  # fmt: skip


def step_times(session: SessionFolder) -> list[tuple[float, float]]:
    """The hop every step ends with: (start, end) in seconds."""
    hop_s = session.steps.hop / session.fs
    ends = [tick["n_end"] / session.fs for tick in session.ticks]
    return [(end - hop_s, end) for end in ends]


def detection_track(session: SessionFolder, end_s: float) -> CurveTrack:
    """The chance of focus after every step under the blocks asked for, the area from it to the
    threshold tinted by what the step found: focus over the threshold, rest under it. A step
    whose signal was too disturbed to use is tinted as rejected (f was kept from the step
    before)."""
    steps = [
        CurveStep(
            start_s=start, end_s=end, value=tick["f"], colour=None if tick["valid"] else REJECTED
        )
        for (start, end), tick in zip(step_times(session), session.ticks, strict=True)
    ]
    return CurveTrack(
        id=DETECTION, title=TITLES[DETECTION], caption=CAPTIONS[DETECTION], steps=steps,
        reference=c.FOCUS_THRESHOLD, above=FOCUSED, below=RESTING,
        bands=protocol_bands(session, end_s), stepped=False,
    )  # fmt: skip


def control_steps(session: SessionFolder, start_s: float, end_s: float) -> list[CurveStep]:
    """How far toward focus the music was sent. The session starts it in its quiet form, before
    it records; the weights of a step go out at its end and hold until the next ones, so a
    step that sent none leaves the music where it was."""
    ends = [tick["n_end"] / session.fs for tick in session.ticks]
    level = START_W_FOCUS
    steps = [CurveStep(start_s=start_s, end_s=ends[0] if ends else end_s, value=level, colour=None)]
    for index, tick in enumerate(session.ticks):
        if tick["t_sent"] is not None:
            level = tick["w_focus"]
        until = ends[index + 1] if index + 1 < len(ends) else end_s
        steps.append(CurveStep(start_s=ends[index], end_s=until, value=level, colour=None))
    return steps


def sound_tracks(
    session: SessionFolder, audio: AudioTrack | None, start_s: float, end_s: float
) -> list[Track]:
    """The music, with how far it was steered faded behind it in one lane; without a sound,
    how far it was steered alone."""
    steps = control_steps(session, start_s, end_s)
    if audio is None:
        return [CurveTrack(
            id=CONTROL, title=TITLES[CONTROL], caption=CAPTIONS[CONTROL], steps=steps,
            reference=None, above=None, below=None, bands=[], stepped=True,
        )]  # fmt: skip
    return [audio.model_copy(update={"caption": f"faded: {CAPTIONS[CONTROL]}", "level": steps})]


def severity_of(said: str) -> Severity:
    """How grave a line of a log is: the level adamus.log begins it with; an error Pd printed;
    otherwise information."""
    level = said.split(" ", 1)[0]
    if level in SEVERITIES:
        return SEVERITIES[level]
    return "error" if said.startswith(ERROR_MARK) else "info"


def log_lines(path: Path, source: str, place: Placing) -> list[ConsoleLine]:
    """The lines of one log, each where ``place`` puts its moment; none when it was not written."""
    lines: list[ConsoleLine] = []
    if not path.is_file():
        return lines
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        read = read_log_line(line)
        if read is not None:
            moment, said = read
            lines.append(
                ConsoleLine(
                    time_s=place(moment), source=source, severity=severity_of(said), text=said
                )
            )
        elif lines:
            lines[-1].text += c.LINE_END + line
    return lines


def session_log(folder: Path, place: Placing | None) -> list[ConsoleLine]:
    """The lines of every log of the session in ``folder``, in the order they were written;
    ``place`` puts a moment on the timeline, None when it has no time to put it at."""
    if place is None:
        return []
    lines = [
        line for source, name in LOG_SOURCES for line in log_lines(folder / name, source, place)
    ]
    return sorted(lines, key=lambda line: line.time_s)  # stable: a log keeps its own order


SAID = ("msg", "message")  # what an event says, shown as it is after what else it holds
NOT_SHOWN = ("t", "n", "type", *SAID)  # its time and type have places of their own


def event_lines(session: SessionFolder) -> list[ConsoleLine]:
    """The events of the session as a console shows them: the type, ``field=value`` of what it
    holds, then what it says; each at the samples received when it happened."""
    lines = []
    for event in session.events:
        held = [f"{key}={value}" for key, value in event.items() if key not in NOT_SHOWN]
        said = [str(value) for key, value in event.items() if key in SAID]
        text = " ".join([event["type"], *held, *said])
        severity: Severity = "error" if event["type"] == EVENT_ERROR else "info"
        lines.append(
            ConsoleLine(time_s=event["n"] / session.fs, source=None, severity=severity, text=text)
        )
    return lines


def colours_of(track: Track) -> list[Colour | None]:
    """The colours a track tints what it draws with."""
    if not isinstance(track, CurveTrack):
        return []
    parts = [*track.bands, *track.steps]
    return [track.above, track.below, *(part.colour for part in parts)]


def legend_of(colours: Iterable[Colour | None]) -> list[LegendEntry]:
    """What each of ``colours`` stands for, in the order of the legend."""
    used = set(colours)
    return [
        LegendEntry(colour=colour, title=title)
        for colour, title in LEGEND.items()
        if colour in used
    ]


def session_tracks(folder: Path) -> SessionTracks:
    """Every track of the session in ``folder``."""
    session = load_session_folder(folder)
    raw = session.raw_samples()
    signals, data = signal_tracks(session, raw)
    rec_start = session.fudi_time(REC_START)
    place = placing(raw[session.meta["timestamp_row"]] if raw.size else None, session.fs, rec_start)
    music = music_track(session, place, rec_start)
    audio = [] if music is None else [music[0]]
    ends = [raw.shape[1] / session.fs if raw.size else 0.0]
    ends += [track.start_s + track.duration_s for track in audio]
    ends += [tick["n_end"] / session.fs for tick in session.ticks[-1:]]
    start_s, end_s = min([0.0, *(track.start_s for track in audio)]), max(ends)
    focus: list[Track] = [detection_track(session, end_s)]
    sound = sound_tracks(session, music[0] if music else None, start_s, end_s)
    grouped = ((MUSIC_GROUP, sound), (FOCUS_GROUP, focus), (EEG_GROUP, signals))
    groups = [TrackGroup(title=title, colour=GROUP_COLOURS[title], tracks=tracks)
              for title, tracks in grouped]  # fmt: skip
    consoles = [Console(title=LOG, lines=session_log(folder, place)),
                Console(title=EVENTS, lines=event_lines(session))]  # fmt: skip
    drawn = [track for group in groups for track in group.tracks]
    colours: list[Colour | None] = [colour for track in drawn for colour in colours_of(track)]
    timeline = Timeline(
        start_s=start_s, end_s=end_s, groups=groups, legend=legend_of(colours), consoles=consoles,
    )  # fmt: skip
    samples = data if music is None else {**data, MUSIC: music[1]}
    return SessionTracks(timeline, samples)
