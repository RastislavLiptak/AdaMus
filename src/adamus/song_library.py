"""The library of songs a session can play, each with the parts its owner marked as rest or focus.

A song is a folder: the song file as it was added (audio, with its suffix), song.json (what the
song is called and its marks), analysis.json (its beats, and what each beat sounds like, which
the player joins its parts by) and the picture the file carries, if any. The library is a folder
of songs; a session keeps a copy of the song it played, as the library had it then.

A song is changed in a draft: a file is analysed into one, a song of the library is copied into
one; its parts are marked there, and it goes into the library, called as its owner says, with
the picture they chose, in place of the song it was copied from, once a beat of it is marked
each state. A mark is a stretch of the song, from where the owner put it: one
marked over another replaces it where they overlap, as paint does. The drafts are put aside as
the library is opened: one the app left behind is never finished. The server changes the
library from several threads, so a change is made under a lock, and a file is written whole or
not at all.
"""

import json
import math
import shutil
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile
from tinytag import TinyTag, TinyTagException

from adamus import constants as c
from adamus.errors import AdamusError

AUDIO_FILE = "audio"  # then the suffix of the song file as it was added
FILE_ANALYSIS = "analysis.json"
COVER_SUFFIXES = {"image/jpeg": ".jpg", "image/png": ".png"}  # of a picture the tags carry
PICTURE_SUFFIXES = {".jpg": ".jpg", ".jpeg": ".jpg", ".png": ".png"}  # of a picture chosen
CHANNELS = 2  # a song is decoded to stereo, and played so
MARKS = (c.REST, c.FOCUS)  # what a part of a song is marked: the states of the protocol
ANALYSIS_RATE = 22050  # Hz: the beats and the sound of each beat are found at this rate
HOP = 512  # samples between two analysis frames
BEATS_PER_BAR = 4
LOW_HZ = 200.0  # the bass and the kick drum, which fall on the first beat of a bar
LOW_BANDS = 32
TIMBRE_COEFFICIENTS = 13
# How loud the song is along it, in this many slices, for drawing it: a tenth of a second or so
# of a song of a few minutes, as the app draws it zoomed in to mark it exactly.
LOUDNESS_BARS = 2000
LOUDNESS_DECIMALS = 3  # as many as a drawing shows
DECODED = "decoded.wav"
WRITING = ".writing"  # after the name of a file while it is written
TINY = 1e-9  # keeps a silent beat from being divided by zero
DRAFTS = ".drafts"  # in the library: hidden, and never taken for a song
DRAFT_OF = "draft_of"  # in the song.json of a draft: the song it was copied from, if any
TITLE_HINT = "A song has a title."


@dataclass(frozen=True)
class Mark:
    start_s: float
    end_s: float
    mark: str  # rest or focus


@dataclass(frozen=True)
class Song:
    id: str  # the name of its folder
    title: str
    artist: str | None
    duration_s: float
    marks: list[Mark]
    loudness: list[float]  # of equal slices of the song, the loudest 1
    cover: Path | None


@dataclass(frozen=True)
class Analysis:
    beats: list[float]  # s
    first_downbeat: int  # the beat of these that begins a bar
    sound: list[list[float]]  # of every beat: its harmony and its timbre, each of length one


def decode(path: Path, rate: int) -> np.ndarray:
    """The audio of a song file [frames x 2] at ``rate``, whatever its kind: macOS decodes it."""
    with tempfile.TemporaryDirectory() as folder:
        wav = Path(folder) / DECODED
        command = [
            "afconvert",
            "-f",
            "WAVE",
            "-d",
            f"LEF32@{rate}",
            "-c",
            str(CHANNELS),
            str(path),
            str(wav),
        ]
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise AdamusError(f"{path.name} is not a song macOS can read: {done.stderr.strip()}")
        audio, _ = soundfile.read(wav, dtype="float32", always_2d=True)
    return audio


def analyse(mono: np.ndarray) -> Analysis:
    """The beats of a song at ANALYSIS_RATE, the beat that begins a bar (the one with the most
    bass), and the harmony and timbre of each beat."""
    onset = librosa.onset.onset_strength(y=mono, sr=ANALYSIS_RATE, hop_length=HOP)
    # Not trimmed: the tracker would drop the beats of quiet stretches at the ends, and quiet
    # stretches are what a song's rest is made of.
    _, frames = librosa.beat.beat_track(
        onset_envelope=onset, sr=ANALYSIS_RATE, hop_length=HOP, trim=False
    )
    if len(frames) < 2 * BEATS_PER_BAR:
        raise AdamusError("the song has no beat that could be followed")
    low = librosa.onset.onset_strength(
        y=mono, sr=ANALYSIS_RATE, hop_length=HOP, fmax=LOW_HZ, n_mels=LOW_BANDS
    )
    by_phase = [low[frames[phase::BEATS_PER_BAR]].mean() for phase in range(BEATS_PER_BAR)]
    harmony = librosa.feature.chroma_cqt(y=mono, sr=ANALYSIS_RATE, hop_length=HOP)
    timbre = librosa.feature.mfcc(
        y=mono, sr=ANALYSIS_RATE, n_mfcc=TIMBRE_COEFFICIENTS, hop_length=HOP
    )
    per_beat = [
        librosa.util.sync(feature, frames.tolist())[:, 1:].T for feature in (harmony, timbre)
    ]
    sound = np.hstack([part / (np.linalg.norm(part, axis=1, keepdims=True) + TINY)
                       for part in per_beat])  # fmt: skip
    beats = librosa.frames_to_time(frames, sr=ANALYSIS_RATE, hop_length=HOP)
    return Analysis([float(t) for t in beats], int(np.argmax(by_phase)), sound.tolist())


def loudness_of(mono: np.ndarray) -> list[float]:
    """How loud ``mono`` is in each of LOUDNESS_BARS equal slices of it, the loudest 1."""
    levels = np.array([np.sqrt(np.mean(part**2)) if len(part) else 0.0
                       for part in np.array_split(mono, LOUDNESS_BARS)])  # fmt: skip
    return [round(float(level), LOUDNESS_DECIMALS) for level in levels / (levels.max() or 1.0)]


def song_in(folder: Path) -> Song:
    """The song kept in ``folder``: of the library, a draft of one, or the copy a session keeps."""
    try:
        kept = json.loads((folder / c.FILE_SONG).read_text(encoding="utf-8"))
        return Song(
            id=folder.name, title=kept["title"], artist=kept["artist"],
            duration_s=kept["duration_s"], marks=[Mark(**mark) for mark in kept["marks"]],
            loudness=kept["loudness"], cover=cover_in(folder),
        )  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError) as problem:
        raise AdamusError(f"{folder.name}: {c.FILE_SONG} cannot be read: {problem}") from problem


def analysis_in(folder: Path) -> Analysis:
    return Analysis(**json.loads((folder / FILE_ANALYSIS).read_text(encoding="utf-8")))


def audio_in(folder: Path) -> Path:
    """The song file kept in ``folder``, as it was added."""
    return next(folder.glob(AUDIO_FILE + ".*"))


def cover_in(folder: Path) -> Path | None:
    """The picture of the song kept in ``folder``, if it carries one."""
    return next(iter(sorted(folder.glob(c.FILE_COVER + ".*"))), None)


def beat_marks(starts: np.ndarray, ends: np.ndarray, marks: list[Mark]) -> list[str | None]:
    """The mark of every beat: of the part its middle lies in, if any (times in s)."""
    middles = (starts + ends) / 2
    found: list[str | None] = [None] * len(starts)
    for mark in marks:
        for index in np.flatnonzero((middles >= mark.start_s) & (middles < mark.end_s)):
            found[int(index)] = mark.mark
    return found


def unplayed(song: Song, analysis: Analysis) -> list[str]:
    """The states no beat of the song is marked: a session cannot play it until there is none,
    as it may switch to either at any time."""
    starts = np.array(analysis.beats)
    marked = beat_marks(starts, np.append(starts[1:], song.duration_s), song.marks)
    return [state for state in MARKS if state not in marked]


def painted(marks: list[Mark], new: Mark | None, start_s: float, end_s: float) -> list[Mark]:
    """``marks`` with the stretch from ``start_s`` to ``end_s`` marked ``new`` (or unmarked),
    and marks of the same kind that now touch joined into one."""
    kept: list[Mark] = []
    for old in marks:
        if old.end_s <= start_s or old.start_s >= end_s:
            kept.append(old)
            continue
        if old.start_s < start_s:
            kept.append(Mark(old.start_s, start_s, old.mark))
        if old.end_s > end_s:
            kept.append(Mark(end_s, old.end_s, old.mark))
    kept += [new] if new is not None else []
    joined: list[Mark] = []
    for mark in sorted(kept, key=lambda m: m.start_s):
        if joined and joined[-1].mark == mark.mark and joined[-1].end_s >= mark.start_s:
            joined[-1] = Mark(joined[-1].start_s, max(joined[-1].end_s, mark.end_s), mark.mark)
        else:
            joined.append(mark)
    return joined


def write_whole(path: Path, text: str) -> None:
    """Write ``path`` so that it is never seen half written: beside it first, then put in place."""
    beside = path.with_name(path.name + WRITING)
    beside.write_text(text, encoding="utf-8")
    beside.replace(path)


def tags_of(path: Path) -> TinyTag | None:
    """The title, artist and pictures the file carries; None when its tags cannot be read."""
    try:
        return TinyTag.get(path, image=True)
    except (TinyTagException, OSError, ValueError):
        return None


class Library:
    """The songs in one folder, and the drafts in its DRAFTS folder."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.drafts = self.root / DRAFTS
        self._changing = threading.Lock()
        shutil.rmtree(self.drafts, ignore_errors=True)

    def songs(self) -> list[Song]:
        """Every song, by title; songs of one title by the names of their folders. Read while
        nothing changes: a song deleted meanwhile would be listed but not found."""
        if not self.root.is_dir():
            return []
        with self._changing:
            folders = [path for path in self.root.iterdir() if (path / c.FILE_SONG).is_file()]
            found = [song_in(folder) for folder in folders]
        return sorted(found, key=lambda s: (s.title.casefold(), s.id))

    def folder(self, song_id: str) -> Path:
        return kept_in(self.root, song_id, "song")

    def draft_folder(self, draft_id: str) -> Path:
        return kept_in(self.drafts, draft_id, "draft")

    def song(self, song_id: str) -> Song:
        return song_in(self.folder(song_id))

    def draft(self, path: Path) -> Song:
        """A draft of the song in the file at ``path``, analysed. It is called as its tags say,
        or by its file name."""
        path = Path(path)
        if not path.suffix:  # the library keeps it by its suffix, which tells players its kind
            raise AdamusError(f"{path.name}: a song file has a suffix that tells its kind, as .mp3")
        mono = decode(path, ANALYSIS_RATE).mean(axis=1)  # also refuses what is not a song
        analysis = analyse(mono)
        tags = tags_of(path)
        picture = None if tags is None else tags.images.any
        self.drafts.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(dir=self.drafts))
        try:
            shutil.copyfile(path, folder / (AUDIO_FILE + path.suffix.lower()))
            if picture is not None and picture.mime_type in COVER_SUFFIXES:
                cover = folder / (c.FILE_COVER + COVER_SUFFIXES[picture.mime_type])
                cover.write_bytes(picture.data)
            write_whole(folder / FILE_ANALYSIS, json.dumps(asdict(analysis)))
            kept = {"title": (tags and tags.title) or path.stem,
                    "artist": (tags and tags.artist) or None,
                    "duration_s": len(mono) / ANALYSIS_RATE, "marks": [],
                    "loudness": loudness_of(mono), DRAFT_OF: None}  # fmt: skip
            write_whole(folder / c.FILE_SONG, json.dumps(kept, ensure_ascii=False))
        except BaseException:
            shutil.rmtree(folder)
            raise
        return song_in(folder)

    def draft_of(self, song_id: str) -> Song:
        """A draft of the song ``song_id`` of the library, whose marks are changed in it."""
        self.drafts.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(dir=self.drafts))
        try:
            with self._changing:
                shutil.copytree(self.folder(song_id), folder, dirs_exist_ok=True)
            self._change(folder, {DRAFT_OF: song_id})
        except BaseException:
            shutil.rmtree(folder)
            raise
        return song_in(folder)

    def mark(self, draft_id: str, start_s: float, end_s: float, mark: str | None) -> Song:
        """Mark the stretch of a draft from ``start_s`` to ``end_s`` as ``mark`` (rest or
        focus), or unmark it with None; as far as the song reaches."""
        if mark is not None and mark not in MARKS:
            raise AdamusError(f"{mark}: a part is marked rest or focus")
        if not (math.isfinite(start_s) and math.isfinite(end_s)):
            raise AdamusError("a part begins and ends at a time in the song")
        with self._changing:
            folder = self.draft_folder(draft_id)
            song = song_in(folder)
            start, end = sorted(min(max(t, 0.0), song.duration_s) for t in (start_s, end_s))
            if end > start:
                new = None if mark is None else Mark(start, end, mark)
                marks = [asdict(m) for m in painted(song.marks, new, start, end)]
                self._change(folder, {"marks": marks})
        return song_in(folder)

    def keep(
        self, draft_id: str, title: str, artist: str | None, picture: Path | None = None
    ) -> Song:
        """Put the draft into the library, called ``title``, by ``artist`` (none when it is left
        empty), with the picture in the file ``picture`` if one is given, once a beat of it is
        marked each state: a new song, or in place of the song it was copied from."""
        title = title.strip()
        if not title:
            raise AdamusError(TITLE_HINT)
        if picture is not None and Path(picture).suffix.lower() not in PICTURE_SUFFIXES:
            raise AdamusError(f"{Path(picture).name}: a picture is a JPEG or PNG file")
        with self._changing:
            folder = self.draft_folder(draft_id)
            missing = unplayed(song_in(folder), analysis_in(folder))
            if missing:
                states = " and ".join(state.lower() for state in missing)
                raise AdamusError(f"No beat of the song is marked {states}: mark a part so.")
            if picture is not None:
                for old in folder.glob(c.FILE_COVER + ".*"):
                    old.unlink()
                suffix = PICTURE_SUFFIXES[Path(picture).suffix.lower()]
                shutil.copyfile(picture, folder / (c.FILE_COVER + suffix))
            of = json.loads((folder / c.FILE_SONG).read_text(encoding="utf-8"))[DRAFT_OF]
            artist = (artist or "").strip() or None
            self._change(folder, {"title": title, "artist": artist}, drop=DRAFT_OF)
            if of is None:
                song = self._new_folder(title)
                folder.replace(song)  # onto the empty folder that holds its name
            else:  # the song as it was is put aside, the draft put in its place, then it goes
                song = self.folder(of)
                aside = Path(tempfile.mkdtemp(dir=self.drafts))
                song.replace(aside)
                folder.replace(song)
                shutil.rmtree(aside)
            return song_in(song)

    def discard(self, draft_id: str) -> None:
        with self._changing:
            shutil.rmtree(self.draft_folder(draft_id))

    def copy(self, song_id: str, to: Path) -> Path:
        """A copy of the song ``song_id`` in the new folder ``to``, as the library has it now."""
        with self._changing:
            return Path(shutil.copytree(self.folder(song_id), to))

    def delete(self, song_id: str) -> None:
        with self._changing:
            shutil.rmtree(self.folder(song_id))

    @staticmethod
    def _change(folder: Path, changes: dict[str, Any], drop: str | None = None) -> None:
        """Write the song.json of ``folder`` with ``changes``, and without the key ``drop``."""
        kept = json.loads((folder / c.FILE_SONG).read_text(encoding="utf-8"))
        kept.update(changes)
        kept.pop(drop, None)
        write_whole(folder / c.FILE_SONG, json.dumps(kept, ensure_ascii=False))

    def _new_folder(self, title: str) -> Path:
        """A new folder for a song, named as its title, or with a number after it when a song
        is called so already: as the file system compares names, which on a Mac ignores case
        and how a letter with an accent is written."""
        name = title.replace("/", "-").replace(":", "-").strip(" .") or AUDIO_FILE
        taken = len(list(self.root.iterdir()))
        for number in range(1, taken + 2):  # one of these is free: there are only so many
            folder = self.root / (name if number == 1 else f"{name} {number}")
            try:
                folder.mkdir()
                return folder
            except FileExistsError:
                continue
        raise AdamusError(f"{name}: no free name for the song in {self.root}")


def kept_in(root: Path, name: str, kind: str) -> Path:
    """The folder ``name`` of a song or a draft in ``root``: never one outside it."""
    folder = root / name
    named = name.strip(" .") != "" and "/" not in name and not name.startswith(".")
    if not named or not (folder / c.FILE_SONG).is_file():
        raise AdamusError(f"{name}: no such {kind}")
    return folder
