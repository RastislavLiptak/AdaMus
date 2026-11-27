"""What the app and the program say to each other: every answer and request of the API.

The app (src/ui) has the same models in Swift; its tests decode what the real program answers
(tests/app), so that the two cannot drift apart.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

# --- The list of sessions -------------------------------------------------------


class SessionSummary(BaseModel):
    id: str  # the folder of the session: for the program only
    name: str  # the name the operator gave it


class SessionRename(BaseModel):
    name: str  # the session is to be called by from now on


class SessionPage(BaseModel):
    total: int  # sessions in all
    sessions: list[SessionSummary]  # newest first


# --- The dashboard: every fact about one session ----------------------------------

#: How the app writes a value: a text as it is, a whole number, a number, a share
#: (0 to 1) in percent, seconds as a duration, seconds since 1970 as a date and time, hertz.
Format = Literal["text", "count", "decimal", "percent", "duration", "datetime", "frequency"]


class Value(BaseModel):
    value: str | float | None  # None: not known
    format: Format


Status = Literal["good", "bad"]


class Fact(BaseModel):
    kind: Literal["fact"] = "fact"
    label: str
    value: Value
    status: Status | None = None  # how it went, where the fact tells it, as the result does


class Table(BaseModel):
    kind: Literal["table"] = "table"
    label: str
    columns: list[str]
    rows: list[list[Value]]


class Section(BaseModel):
    title: str
    items: list[Annotated[Fact | Table, Field(discriminator="kind")]]


# --- The timeline: tracks over the time of a session ------------------------------
# Times are seconds from the first sample of EEG; music may begin before it. The app
# draws what it is given and knows nothing of what the program computes: the tracks come in
# titled groups, each says what it is called and what it is read by, and its parts name the
# colours of the palette they are drawn in; the legend says what the colours stand for.

#: A colour of the palette of the app, by the name of its hue: what tints a span of time
#: or a column, a group of tracks and its marks, a kind of result. What it stands for, the
#: program says (a legend, a title); the app knows only how to draw it.
Colour = Literal["orange", "indigo", "gray", "teal", "pink", "purple"]


class LegendEntry(BaseModel):
    colour: Colour
    title: str


class TrackBase(BaseModel):
    id: str
    title: str
    caption: str | None  # what the track is read by, e.g. its scale


class SignalTrack(TrackBase):
    """A signal; its samples come separately: each layer, the first drawn faded behind the
    next (as the raw signal behind the filtered one)."""

    kind: Literal["signal"] = "signal"
    scale: float  # half the height of the track, in the unit of the samples
    rate_hz: float
    samples: int  # of each layer
    layers: int


class Band(BaseModel):
    start_s: float
    end_s: float
    title: str | None  # written in it, where there is room
    colour: Colour | None  # None: not tinted


class CurveStep(BaseModel):
    start_s: float
    end_s: float
    value: float | None  # from 0 to 1; None where there is none
    colour: Colour | None  # of the step's span, behind the line; None: not tinted


class AudioTrack(TrackBase):
    """Sound, its samples from -1 to 1; its envelope comes separately: the lowest sample of
    every bucket, then the highest of every bucket. The sound itself is played from the
    track's own address. ``level`` is drawn behind it, faded, as an area up to the value of
    each step, held until the next: how far the sound was steered."""

    kind: Literal["audio"] = "audio"
    start_s: float
    duration_s: float
    rate_hz: float  # buckets per second
    buckets: int
    level: list[CurveStep] = []  # from 0 to 1; empty: none


class CurveTrack(TrackBase):
    """Values from 0 to 1, one for each step of time: as a line through the steps, or held from
    one step to the next (``stepped``). ``reference`` is drawn as a dashed line, and the area
    between it and the line is tinted ``above`` where the line is over it and ``below`` where
    it is under it. ``bands`` lie in a strip along the top, each tinted and named."""

    kind: Literal["curve"] = "curve"
    steps: list[CurveStep]
    reference: float | None
    above: Colour | None  # None: not tinted
    below: Colour | None
    bands: list[Band]
    stepped: bool


Track = Annotated[
    SignalTrack | AudioTrack | CurveTrack,
    Field(discriminator="kind"),
]

Severity = Literal["debug", "info", "warning", "error"]


class ConsoleLine(BaseModel):
    """A line of a console under the tracks, at the moment it came on the timeline."""

    time_s: float
    source: str | None  # who wrote it, where a console shows several
    severity: Severity
    text: str  # with the lines that went on with it, as a traceback does


class TrackGroup(BaseModel):
    title: str
    colour: Colour  # of the group, and of the marks of its tracks
    tracks: list[Track]


class Console(BaseModel):
    title: str
    lines: list[ConsoleLine]  # in the order of their times


class Timeline(BaseModel):
    start_s: float
    end_s: float
    groups: list[TrackGroup]
    legend: list[LegendEntry]  # of the colours the tracks use
    consoles: list[Console]  # side by side under the tracks


ElectrodeKind = Literal[
    "recorded", "reference"
]  # the reference: what the others are measured against


class Electrode(BaseModel):
    """An electrode of a session: where it lies on the head, and what it records."""

    name: str
    kind: ElectrodeKind
    position: list[float] | None  # mm, on the skin of the head (HeadModel); None: not on it
    normal: list[float] | None  # the way the skin faces there, a unit vector
    about: str


class HeadModel(BaseModel):
    """The head the electrodes are drawn on, in millimetres: x to the right ear, y to the nose,
    z up."""

    vertices: list[float]  # x, y and z of every point
    triangles: list[int]  # three points each, the skin facing outward


class Excerpt(BaseModel):
    """A part of the timeline, cut out of the data of the session (excerpt)."""

    text: str  # JSON, indented to be read


# --- The overview: its main results ------------------------------------------------


class Highlight(BaseModel):
    """One of the few results the overview shows first and large, with what it means."""

    title: str
    value: Value
    caption: str
    colour: Colour  # of the kind of result, as the timeline draws it


class SongPart(BaseModel):
    """A stretch of a song, or of the music of a session, that belongs to one state: in its
    colour, as the timeline draws the state, and called as the state is."""

    start_s: float
    end_s: float
    mark: str  # the id of the mark: rest or focus
    title: str
    colour: Colour


class Song(BaseModel):
    """A song of the library, or a draft of one, as it is shown and marked: how loud it is along
    it, its marks."""

    id: str
    title: str
    artist: str | None
    length_s: float
    loudness: list[float]  # of equal slices of the song, the loudest 1
    parts: list[SongPart]  # what is marked, in the order of the song
    cover: str | None  # the version of its picture, which changes with it; none without one


class Dashboard(BaseModel):
    """The dashboard: its overview first, what the session is (``summary``), what went wrong
    (``failures``), anything that bears on how its results are read (``caution``) and its main
    results; then its details, every other fact in sections."""

    name: str
    problem: str | None  # why some of the session could not be read
    pending: str | None  # what is still being worked on, while it is
    summary: list[Fact]  # what the session is and how it ended
    failures: list[str]  # what went wrong, each on its own
    caution: str | None
    highlights: list[Highlight]
    song: Song | None  # the source of its music, as it kept it; none if its music never began
    sections: list[Section]


# --- A new session -----------------------------------------------------------------


class SourceChoice(BaseModel):
    id: str
    title: str  # as the form shows it
    live: bool  # the headset: no session with it starts while it is being looked for
    recordings: list[str]  # one is chosen from them; none to choose from for a live source


class NewSessionForm(BaseModel):
    """What the form for a new session offers, and the rule its name must keep."""

    sources: list[SourceChoice]
    sources_hint: str  # what to tell the operator when there is none
    name_pattern: str  # a regular expression the whole name must match
    name_hint: str  # what to tell the operator when it does not


class SourceCheck(BaseModel):
    warning: str | None  # what the operator should know, though a session can run
    problem: str | None  # why a session cannot run from it


class AudioOutputChoice(BaseModel):
    id: str
    title: str  # as the form shows it


class AudioOutputs(BaseModel):
    """The outputs the music can play on, and the one the system plays on: chosen at first."""

    outputs: list[AudioOutputChoice]
    system: str | None  # the id of the one the system plays on; None when it plays on none
    hint: str  # what to tell the operator when there is none


class SessionRequest(BaseModel):
    source_id: str
    recording: str | None = None  # of a dataset; none of the headset
    song_id: str  # the song its music is made of
    name: str
    output: str  # the id of the audio output the music is to play on


Phase = Literal["idle", "starting", "running", "finishing"]


class RunState(BaseModel):
    run: int  # how many sessions have been asked for so far
    phase: Phase
    session_id: str | None  # the folder of the session of this run, once it has one
    problem: str | None  # why the session of this run could not start, or ended in an error


# --- The library of songs ------------------------------------------------------------


class MarkChoice(BaseModel):
    """What a part of a song can be marked: its id, as it is called and its colour."""

    id: str
    title: str
    colour: Colour


class SongPage(BaseModel):
    total: int  # songs in all
    songs: list[Song]  # by title


class SongMarks(BaseModel):
    marks: list[MarkChoice]  # what a part can be marked, in this order
    unmarked: str  # what a part no mark covers is called


class DraftKept(BaseModel):
    """What a draft that goes into the library is called, and its picture."""

    title: str
    artist: str | None = None  # none, or empty, when it is not known
    picture: str | None = None  # the path of a picture on this computer; none keeps its own


class DraftOfFile(BaseModel):
    path: str  # of the song file on this computer, which the draft copies


class SongMark(BaseModel):
    """Mark the stretch from ``start_s`` to ``end_s`` as ``mark``; None unmarks it."""

    start_s: float
    end_s: float
    mark: str | None = None
