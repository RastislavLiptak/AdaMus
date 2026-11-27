"""Song files with a known beat, for the tests of the library and of the player.

A song is a beat at TEMPO_BPM: a short high click on every beat, and on the first beat of each
bar a low thump too, as a kick drum gives it. Its first half is quiet and its second loud. A FLAC
song can carry a title, an artist and a picture, as a real song file does.
"""

import struct
from pathlib import Path

import numpy as np
import soundfile

from adamus import constants as c
from adamus.song_library import Library

RATE = 44100  # Hz: not the player's rate, so that a song is converted as a real one is
TEMPO_BPM = 120.0
BEAT_S = 60.0 / TEMPO_BPM
BEATS_PER_BAR = 4
FIRST_BEAT_S = 0.25  # where the first beat is; the first bar begins on it
CLICK_HZ, THUMP_HZ = 2000.0, 60.0
CLICK_S, THUMP_S = 0.03, 0.12
QUIET, LOUD = 0.1, 0.5  # a click and a thump together stay below full scale
SECONDS = 24.0
PICTURE_BLOCK = 6  # the kind of a FLAC metadata block that holds a picture
FRONT_COVER = 3
LAST_BLOCK = 0x80  # set on the header of the last metadata block of a FLAC file
FLAC_HEADER = 4  # "fLaC"
BLOCK_HEADER = 4  # its kind, then its length in three bytes
LENGTH_BYTES = 3
PICTURE = b"a stand-in for a picture"  # the library keeps it as it is, never shows it
FADE = 4.0  # a sound dies away this many times faster than it lasts


def beat_times(seconds: float = SECONDS) -> np.ndarray:
    return np.arange(FIRST_BEAT_S, seconds - BEAT_S, BEAT_S)


def song_audio(seconds: float = SECONDS) -> np.ndarray:
    """[frames x 2] of the beat described above."""
    audio = np.zeros(round(seconds * RATE))
    for index, start in enumerate(beat_times(seconds)):
        level = QUIET if start < seconds / 2 else LOUD
        sounds = [(CLICK_HZ, CLICK_S)] + (
            [(THUMP_HZ, THUMP_S)] if index % BEATS_PER_BAR == 0 else []
        )
        for hz, length in sounds:
            t = np.arange(round(length * RATE)) / RATE
            at = round(start * RATE)
            audio[at : at + len(t)] += (
                level * np.sin(2 * np.pi * hz * t) * np.exp(-t / length * FADE)
            )
    return np.column_stack([audio, audio]).astype(np.float32)


def write_song(path: Path, seconds: float = SECONDS, title: str = "", artist: str = "") -> Path:
    """The song as ``path``'s kind of file (WAV or FLAC); a FLAC one with a title and artist."""
    with soundfile.SoundFile(path, "w", RATE, 2) as song:
        if title:
            song.title = title
        if artist:
            song.artist = artist
        song.write(song_audio(seconds))
    return path


def with_picture(flac: Path, picture: bytes = PICTURE, mime: str = "image/png") -> Path:
    """Put ``picture`` into the FLAC file as its front cover, right after its first block."""
    data = flac.read_bytes()
    at = FLAC_HEADER
    first_length = int.from_bytes(data[at + 1 : at + BLOCK_HEADER], "big")
    after_first = at + BLOCK_HEADER + first_length
    body = struct.pack(">II", FRONT_COVER, len(mime)) + mime.encode() + struct.pack(">I", 0)
    body += struct.pack(">IIIII", 0, 0, 0, 0, len(picture)) + picture
    header = bytes([PICTURE_BLOCK]) + len(body).to_bytes(LENGTH_BYTES, "big")
    first = bytes([data[at] & ~LAST_BLOCK]) + data[at + 1 : after_first]  # no longer the last
    last = data[at] & LAST_BLOCK  # the picture is last when the first block was
    header = bytes([header[0] | last]) + header[1:]
    flac.write_bytes(data[:at] + first + header + body + data[after_first:])
    return flac


def add_marked(library: Library, path: Path) -> str:
    """Put the song in the file at ``path`` into ``library``, its first half marked rest and its
    second focus, as the owner marks it in a draft; the id of the song."""
    draft = library.draft(path)
    library.mark(draft.id, 0.0, draft.duration_s / 2, c.REST)
    library.mark(draft.id, draft.duration_s / 2, draft.duration_s, c.FOCUS)
    return library.keep(draft.id, draft.title, draft.artist).id
