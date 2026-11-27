"""The songs of the library as the app shows them: how loud each is along it, and which of its
stretches is marked which state, in the colour the timeline draws that state in."""

from adamus import song_library
from adamus.server import api_models
from adamus.server.timeline import STATE_COLOURS
from adamus.song_library import MARKS

UNMARKED = "Unmarked"  # what the parts of a song no mark covers are called


def part_of(start_s: float, end_s: float, mark: str) -> api_models.SongPart:
    return api_models.SongPart(
        start_s=start_s, end_s=end_s, mark=mark, title=mark.capitalize(),
        colour=STATE_COLOURS[mark],
    )  # fmt: skip


def song_view(song: song_library.Song) -> api_models.Song:
    return api_models.Song(
        id=song.id, title=song.title, artist=song.artist, length_s=song.duration_s,
        loudness=song.loudness, parts=[part_of(m.start_s, m.end_s, m.mark) for m in song.marks],
        cover=None if song.cover is None else str(song.cover.stat().st_mtime_ns),
    )  # fmt: skip


def page_of(songs: list[song_library.Song], offset: int, limit: int) -> api_models.SongPage:
    shown = songs[offset : offset + limit]
    return api_models.SongPage(total=len(songs), songs=[song_view(song) for song in shown])


def marks_view() -> api_models.SongMarks:
    marks = [api_models.MarkChoice(id=m, title=m.capitalize(), colour=STATE_COLOURS[m])
             for m in MARKS]  # fmt: skip
    return api_models.SongMarks(marks=marks, unmarked=UNMARKED)
