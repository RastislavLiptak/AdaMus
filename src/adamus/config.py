"""Loading and validation of config.yaml.

A configuration that loads is usable: besides types and ranges, the values are
checked against each other and against the EEG board they name.
"""

import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml
from brainflow.board_shim import BoardShim, BrainFlowError

from adamus import constants as c
from adamus.errors import AdamusError

STYLE_TEXT = r"[a-z][a-z0-9-]*( [a-z][a-z0-9-]*)*"  # Pd would turn a numeric word into a float
STYLE_ID = r"[a-z]+"  # in the addresses of the API and the names of the files of a style
# A root, then its kind: major, minor, seventh, minor seventh, major seventh, or an open fifth.
CHORD = r"[A-G](m|7|m7|maj7|5)?"
BEATS_PER_BAR = 4  # a chord lasts a bar
FASTEST_BPM = 300.0  # faster, the chords would flood the generator with notes
LOUDEST_DB = 2.0  # of the generator's output: measured, the loudest moments of focus then clip
INVALID = "invalid configuration:"  # heads the list of what is wrong with one
LONGEST_S = 24 * 60 * 60  # no block and no wait lasts a day; longer ones overflow in samples


@dataclass(frozen=True)
class Board:
    name: str  # how the headset is called in the program
    id: int  # BrainFlow board id
    serial_number: str


@dataclass(frozen=True)
class Block:
    label: str  # REST, FOCUS or FREE
    duration_s: float  # 0 = lasts until the session is stopped


@dataclass(frozen=True)
class Generator:
    folder: str  # where scripts/install_generator.sh installed the generator
    model: str  # name of the model, e.g. mrt2_small
    style_guidance: float  # how closely the music follows its style text
    note_guidance: float  # how closely it follows the chords

    @property
    def directory(self) -> Path:
        """generator.folder, as the path every file of the generator lies below."""
        return Path(self.folder)

    @property
    def pd_binary(self) -> Path:
        return self.directory / c.GENERATOR_PD_BINARY

    @property
    def externals(self) -> Path:
        return self.directory / c.GENERATOR_EXTERNALS

    @property
    def external_file(self) -> Path:
        return self.directory / c.GENERATOR_EXTERNAL_FILE

    @property
    def assets(self) -> Path:
        return self.directory / c.GENERATOR_ASSETS

    @property
    def asset_files(self) -> list[Path]:
        return [self.assets / file for file in c.GENERATOR_ASSET_FILES]

    @property
    def model_file(self) -> Path:
        return self.directory / c.GENERATOR_MODEL_FILE.format(model=self.model)


@dataclass(frozen=True)
class Style:
    """One piece of music in two forms: the quiet one at rest, the full one in focus."""

    id: str  # how the program and the app know it
    name: str  # as the app shows it
    about: str  # how it sounds, in a sentence
    rest: str  # the style text of the quiet form
    focus: str  # of the full form
    tempo_bpm: float
    chords: list[str]  # one a bar, again and again
    chord_gap_s: float  # between letting a chord go and striking the next
    focus_power_chords: bool  # in focus only root, fifth and octave
    focus_drums: bool  # drums may come in focus; at rest they never do
    rest_volume_db: float
    focus_volume_db: float
    temperature: float
    top_k: int

    @property
    def bar_s(self) -> float:
        return BEATS_PER_BAR * 60.0 / self.tempo_bpm


@dataclass(frozen=True)
class Timeouts:
    stream_s: float
    pd_connect_s: float
    model_load_s: float


@dataclass(frozen=True)
class Config:
    board: Board
    protocol: list[Block]
    generator: Generator
    styles: list[Style]
    timeouts: Timeouts
    poll_s: float
    raw: dict[str, Any]  # the mapping this object was built from

    @property
    def style(self) -> Style:
        """The style a session of this configuration plays: its first. The form gives a session
        the one chosen alone (with_style), so that the copy it keeps says what played."""
        return self.styles[0]


def load_config(path: Path) -> Config:
    return parse_config(read_yaml(path))


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as problem:
        raise AdamusError(f"cannot read {path}: {problem}") from problem
    if type(data) is not dict:
        raise AdamusError(f"{path} must contain a mapping")
    return data


def write_yaml(data: dict[str, Any], path: Path) -> None:
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=None)
    Path(path).write_text(text, encoding="utf-8")


def is_style_text(text: str) -> bool:
    return re.fullmatch(STYLE_TEXT, text) is not None


def parse_config(raw: dict[str, Any]) -> Config:
    """Turn a mapping into a Config; every problem found is reported at once."""
    problems: list[str] = []
    allowed = {f.name for f in fields(Config)} - {"raw"}
    problems += [f"{key}: unknown key" for key in sorted(set(raw) - allowed, key=str)]

    board = _section(Board, raw.get("board"), "board", problems)
    generator = _section(Generator, raw.get("generator"), "generator", problems)
    timeouts = _section(Timeouts, raw.get("timeouts"), "timeouts", problems)
    protocol = _protocol(raw.get("protocol"), problems)
    styles = _styles(raw.get("styles"), problems)
    if not _has_type(raw.get("poll_s"), float):
        problems.append(f"poll_s: expected a number, got {raw.get('poll_s')!r}")
    if not problems:
        problems += _range_problems(generator, timeouts, raw["poll_s"])
    if not problems:
        problems += _style_problems(styles)
    if not problems:
        problems += _board_problems(board)
    if not problems:
        problems += _consistency_problems(board, timeouts, raw["poll_s"])
    if problems:
        raise AdamusError(INVALID, *(f"  {problem}" for problem in problems))
    return Config(board, protocol, generator, styles, timeouts, float(raw["poll_s"]), raw)


# --- Structure and types ------------------------------------------------------


def _has_type(value: Any, kind: Any) -> bool:
    """Is a YAML value an int, float, bool, str or list of texts as expected? A whole number will
    do for a float, and a number means a finite one: YAML also knows .nan and .inf."""
    if kind is float:
        return type(value) is int or (type(value) is float and math.isfinite(value))
    if kind == list[str]:
        return type(value) is list and all(type(item) is str for item in value)
    return type(value) is kind


def _type_problem(name: str, value: Any, kind: Any) -> str:
    if kind is float and type(value) is float:  # a number, but not a finite one
        return f"{name}: must be a finite number, got {value!r}"
    return f"{name}: wrong type, got {value!r}"


def _section(cls: Any, data: Any, name: str, problems: list[str]) -> Any:
    """Build one dataclass from a mapping, or report what is wrong with it."""
    if type(data) is not dict:
        problems.append(f"{name}: missing or not a mapping")
        return None
    kinds = {f.name: f.type for f in fields(cls)}
    found = [f"{name}.{key}: missing" for key in sorted(kinds.keys() - data.keys())]
    found += [f"{name}.{key}: unknown key" for key in sorted(data.keys() - kinds.keys(), key=str)]
    for key in sorted(data.keys() & kinds.keys()):
        if not _has_type(data[key], kinds[key]):
            found.append(_type_problem(f"{name}.{key}", data[key], kinds[key]))
    problems += found
    return None if found else cls(**data)


def parse_protocol(data: Any) -> list[Block]:
    """The protocol alone, checked as in a whole configuration."""
    problems: list[str] = []
    protocol = _protocol(data, problems)
    if problems:
        raise AdamusError(INVALID, *(f"  {problem}" for problem in problems))
    return protocol


def _protocol(data: Any, problems: list[str]) -> list[Block]:
    is_list_of_lists = type(data) is list and all(type(item) is list for item in data)
    if not is_list_of_lists or not data or any(len(item) != 2 for item in data):
        problems.append("protocol: expected a non-empty list of [label, duration_s]")
        return []
    for index, (label, duration) in enumerate(data):
        last = index == len(data) - 1
        known = type(label) is str and label in c.BLOCK_MARKERS
        if not known or not _has_type(duration, float) or duration < 0:
            problems.append(f"protocol[{index}]: expected [REST|FOCUS|FREE, seconds >= 0]")
        elif duration == 0 and not (label == c.FREE and last):
            problems.append(f"protocol[{index}]: only a final FREE block may have duration 0")
        elif duration >= LONGEST_S:
            problems.append(f"protocol[{index}]: a block must last less than a day")
    return [Block(label, duration) for label, duration in data]


def _styles(data: Any, problems: list[str]) -> list[Style]:
    if type(data) is not list or not data:
        problems.append("styles: expected a non-empty list of styles")
        return []
    built = [_section(Style, item, f"styles[{index}]", problems) for index, item in enumerate(data)]
    return [style for style in built if style is not None]


# --- Values -------------------------------------------------------------------


def _failed(checks: list[tuple[bool, str]]) -> list[str]:
    return [message for passed, message in checks if not passed]


def _range_problems(generator: Generator, timeouts: Timeouts, poll_s: float) -> list[str]:
    """Each value on its own: is it in the range that makes sense?"""
    every_timeout = (
        timeouts.stream_s,
        timeouts.pd_connect_s,
        timeouts.model_load_s,
    )
    checks = [
        (generator.folder != "", "generator.folder: must name a folder"),
        (generator.model != "" and "/" not in generator.model,
         "generator.model: must be the name of a model, not a path"),
        (generator.style_guidance > 0 and generator.note_guidance > 0,
         "generator: guidance must be positive"),
        (min(every_timeout) > 0, "timeouts: every timeout must be positive"),
        (max(every_timeout) <= LONGEST_S, "timeouts: none may be longer than a day"),
        (poll_s > 0, "poll_s: must be positive"),
    ]  # fmt: skip
    return _failed(checks)


def _style_problems(styles: list[Style]) -> list[str]:
    """Each style on its own, and the ids and names of all of them, which must differ."""
    checks: list[tuple[bool, str]] = []
    for index, style in enumerate(styles):
        at = f"styles[{index}]"
        checks += [
            (re.fullmatch(STYLE_ID, style.id) is not None, f"{at}.id: lower-case letters only"),
            (style.name.strip() != "", f"{at}.name: must not be empty"),
            (is_style_text(style.rest), f"{at}.rest: not a valid style text"),
            (is_style_text(style.focus), f"{at}.focus: not a valid style text"),
            (style.rest != style.focus, f"{at}.rest, focus: must be two different styles"),
            (0 < style.tempo_bpm <= FASTEST_BPM,
             f"{at}.tempo_bpm: above 0, at most {FASTEST_BPM:g}"),
            (style.chords != [] and all(re.fullmatch(CHORD, chord) for chord in style.chords),
             f"{at}.chords: at least one, each a root A to G, then nothing, m, 7, m7, maj7 or 5"),
            (style.tempo_bpm <= 0 or 0 <= style.chord_gap_s < style.bar_s,  # no tempo, no bar
             f"{at}.chord_gap_s: from 0, shorter than a bar"),
            (max(style.rest_volume_db, style.focus_volume_db) <= LOUDEST_DB,
             f"{at}: volumes at most {LOUDEST_DB:g} dB"),
            (style.temperature > 0, f"{at}.temperature: must be positive"),
            (style.top_k >= 1, f"{at}.top_k: at least 1"),
        ]  # fmt: skip
    ids, names = [style.id for style in styles], [style.name for style in styles]
    checks += [
        (len(set(ids)) == len(ids), "styles: every id must differ"),
        (len(set(names)) == len(names), "styles: every name must differ"),
    ]
    return _failed(checks)


def with_style(config: Config, style_id: str) -> Config:
    """The configuration of a session that plays the style ``style_id``: its only style, so that
    the copy the session keeps says exactly what played."""
    chosen = [style for style in config.raw["styles"] if style["id"] == style_id]
    if not chosen:
        raise AdamusError(f"{style_id}: no such style")
    return parse_config({**config.raw, "styles": chosen})


def _board_problems(board: Board) -> list[str]:
    """Does BrainFlow know the board, and has it a name to be shown by?"""
    try:
        BoardShim.get_board_descr(board.id)
    except BrainFlowError:
        return [f"board.id: BrainFlow knows no board {board.id}"]
    return _failed([(board.name != "", "board.name: must not be empty")])


def _consistency_problems(board: Board, timeouts: Timeouts, poll_s: float) -> list[str]:
    """Values against each other and against the board the detector was trained for."""
    fs = BoardShim.get_sampling_rate(board.id)
    checks = [
        (fs == c.DETECTOR_FS,
         f"board.id: the detector needs {c.DETECTOR_FS} Hz, the board delivers {fs} Hz"),
        (timeouts.stream_s > poll_s, "timeouts.stream_s: must be longer than poll_s"),
        (poll_s <= c.HOP_S, f"poll_s: must not be longer than a step ({c.HOP_S:g} s)"),
    ]  # fmt: skip
    return _failed(checks)
