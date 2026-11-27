"""Songs placed into a library as the library keeps them, without analysing a song: for the
tests of what uses the library (the form, the API, the sessions), not of the library itself."""

import json
import shutil
from pathlib import Path

import numpy as np
import soundfile

from adamus import constants as c
from adamus.song_library import AUDIO_FILE, BEATS_PER_BAR, FILE_ANALYSIS
from tests.support.project_paths import ROOT

SONG = "Stand-in Song"
ARTIST = "Nobody"  # of the song the stand-in music plays
BEAT_S = 0.5
BEATS = 40
RATE = 8000  # Hz of the short silent file that stands in for the song
SOUND_SIZE = 25
# a picture as the tags of a song may carry one: the app's icon, as large as a cover shows it
PICTURE = ROOT / "src/ui/AdaMus/Assets.xcassets/AppIcon.appiconset/icon_128x128.png"


def place_song(
    library: Path, song_id: str = SONG, pictured: bool = False, title: str | None = None,
    artist: str | None = None,
) -> Path:  # fmt: skip
    """A song of BEATS beats in the folder ``song_id`` of ``library``, called ``title`` or as its
    folder, by ``artist``: its first half marked rest, its second focus; with a picture when
    ``pictured``."""
    folder = library / song_id
    folder.mkdir(parents=True)
    duration = BEATS * BEAT_S
    soundfile.write(folder / f"{AUDIO_FILE}.wav", np.zeros((round(duration * RATE), 2)), RATE)
    sound = np.ones((BEATS, SOUND_SIZE)) / np.sqrt(SOUND_SIZE)
    beats = (np.arange(BEATS) * BEAT_S).tolist()
    analysis = {"beats": beats, "first_downbeat": 0, "sound": sound.tolist()}
    (folder / FILE_ANALYSIS).write_text(json.dumps(analysis), encoding="utf-8")
    half = duration / 2
    kept = {
        "title": title or song_id, "artist": artist, "duration_s": duration,
        "marks": [{"start_s": 0.0, "end_s": half, "mark": c.REST},
                  {"start_s": half, "end_s": duration, "mark": c.FOCUS}],
        # louder beat by beat through each bar, so that the app draws a track
        "loudness": [(1 + beat % BEATS_PER_BAR) / BEATS_PER_BAR for beat in range(BEATS)],
    }  # fmt: skip
    (folder / c.FILE_SONG).write_text(json.dumps(kept), encoding="utf-8")
    if pictured:
        shutil.copyfile(PICTURE, folder / (c.FILE_COVER + ".png"))
    return folder
