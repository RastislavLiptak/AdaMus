"""song_sequencer: which beat of a song plays when, and how a jump sounds.

The song of these tests has 40 beats of half a second; every beat is one level, (beat + 1) / 100,
so the output tells which beat played. Beats 0-7 and 16-23 are marked rest, 8-15 and 24-39 focus.
"""

import numpy as np
import pytest

from adamus import constants as c
from adamus.errors import AdamusError
from adamus.realtime.song_sequencer import JUMP_FADE_S, Sequencer
from adamus.session_format import CONTINUE, START, SWITCH
from adamus.song_library import Analysis, Mark

RATE = 1000
BEAT_S = 0.5
BEATS = 40
FRAMES_PER_BEAT = round(BEAT_S * RATE)
FADE = round(JUMP_FADE_S * RATE)
SOUND_SIZE = 25
LEVEL = 100.0  # beat k plays (k + 1) / LEVEL
REST = [Mark(0.0, 4.0, c.REST), Mark(8.0, 12.0, c.REST)]
FOCUS = [Mark(4.0, 8.0, c.FOCUS), Mark(12.0, 20.0, c.FOCUS)]


def unit_vectors(seed: int) -> np.ndarray:
    vectors = np.random.default_rng(seed).normal(size=(BEATS, SOUND_SIZE))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def song_audio() -> np.ndarray:
    levels = np.repeat((np.arange(BEATS) + 1) / LEVEL, FRAMES_PER_BEAT).astype(np.float32)
    return np.column_stack([levels, levels])


def sequencer(sound=None, marks=REST + FOCUS):
    audio = song_audio()
    beats = (np.arange(BEATS) * BEAT_S).tolist()
    sound = unit_vectors(0) if sound is None else sound
    return Sequencer(audio, RATE, Analysis(beats, 0, sound.tolist()), marks)


def beats_played(out: np.ndarray) -> list[int]:
    """The beat each beat-long stretch of the output is, read past the fade at its start."""
    middles = np.arange(FADE + 1, len(out), FRAMES_PER_BEAT)
    return [round(out[m, 0] * LEVEL) - 1 for m in middles]


def decisions(seq):
    found = []
    for _ in range(BEATS * BEATS):
        if seq.decisions.empty():
            break
        found.append(seq.decisions.get())
    return found


def test_the_music_starts_at_the_first_beat_marked_so_and_plays_the_song_in_its_order():
    seq = sequencer()
    out = seq.render(8 * FRAMES_PER_BEAT - 1)  # up to the last frame of the part
    assert beats_played(out) == list(range(8))
    assert np.array_equal(out, song_audio()[: len(out)])  # nothing added or faded
    assert [(d.kind, d.mark, d.to_s) for d in decisions(seq)] == [(START, c.REST, 0.0)]


def test_when_a_part_runs_out_the_music_goes_on_at_the_same_place_of_a_bar_of_the_state():
    seq = sequencer()
    played = beats_played(seq.render(12 * FRAMES_PER_BEAT))
    assert played[:8] == list(range(8))
    assert played[8] in (16, 20)  # rest, first in its bar, a bar ahead; not 0, played lately
    assert played[9:] == [played[8] + 1, played[8] + 2, played[8] + 3]
    jump = decisions(seq)[1]
    assert (jump.kind, jump.mark, jump.from_s) == (CONTINUE, c.REST, 4.0)
    assert jump.to_s == played[8] * BEAT_S


def test_the_place_whose_beats_before_it_sound_most_like_those_just_played_wins():
    sound = unit_vectors(0)
    sound[16:20] = sound[4:8]  # before beat 20 sounds as before beat 8
    seq = sequencer(sound=sound)
    assert beats_played(seq.render(9 * FRAMES_PER_BEAT))[8] == 20


def test_a_switch_is_followed_at_the_next_beat_at_the_same_place_in_the_bar():
    seq = sequencer()
    before = seq.render(2 * FRAMES_PER_BEAT + FRAMES_PER_BEAT // 3)  # a third into beat 2
    seq.want(c.FOCUS)
    played = beats_played(np.vstack([before, seq.render(3 * FRAMES_PER_BEAT)]))
    assert played[:3] == [0, 1, 2]  # beat 2 is played to its end
    assert played[3] in (11, 27, 31, 35)  # focus, fourth in its bar, a bar of focus ahead
    assert played[4] == played[3] + 1
    switch = decisions(seq)[1]
    assert (switch.kind, switch.mark, switch.out_s) == (SWITCH, c.FOCUS, 3 * BEAT_S)
    assert switch.to_s == played[3] * BEAT_S


def render_in(seq, sizes, total, switch_at):
    """Render ``total`` frames in pieces of ``sizes`` (again and again), the state switched to
    focus after ``switch_at`` frames."""
    pieces, made = [], 0
    for size in sizes * total:
        if made == total:
            break
        size = min(size, total - made, switch_at - made if made < switch_at else total)
        pieces.append(seq.render(size))
        made += size
        if made == switch_at:
            seq.want(c.FOCUS)
    return np.vstack(pieces)


def test_rendering_in_pieces_of_any_size_gives_the_same_music():
    total, switch_at = 30 * FRAMES_PER_BEAT, 5 * FRAMES_PER_BEAT + 123
    whole, pieces = sequencer(), sequencer()
    one = render_in(whole, [total], total, switch_at)
    many = render_in(pieces, [1, 7, 333, 512, 4096], total, switch_at)
    assert np.array_equal(one, many)
    assert decisions(whole) == decisions(pieces)
    assert [d.kind for d in decisions(sequencer())] == [START]


def test_a_jump_fades_the_old_place_out_as_the_new_one_comes_in():
    seq = sequencer()
    out = seq.render(9 * FRAMES_PER_BEAT)
    start = 8 * FRAMES_PER_BEAT  # beat 7 ended and the music jumped
    target = beats_played(out)[8]
    old, new = 9 / LEVEL, (target + 1) / LEVEL  # the old place went on with beat 8
    half = start + FADE // 2
    x = (FADE // 2) / FADE * np.pi / 2
    assert out[half, 0] == pytest.approx(new * np.sin(x) + old * np.cos(x), abs=1e-6)
    assert out[start + FADE, 0] == pytest.approx(new)
    steps = np.abs(np.diff(out[start - 1 : start + FADE + 1, 0]))
    assert steps.max() < abs(new - old) / 4  # no click: it changes over the fade


def test_a_long_state_goes_through_all_its_parts():
    seq = sequencer()
    played = beats_played(seq.render(round(60 / BEAT_S) * FRAMES_PER_BEAT))
    assert set(played) == set(range(8)) | set(range(16, 24))
    assert all(beat in range(8) or beat in range(16, 24) for beat in played)


def test_the_music_goes_on_past_the_end_of_the_song():
    seq = sequencer()
    seq.want(c.FOCUS)
    seq.render(FRAMES_PER_BEAT)  # its first beat, at rest
    played = beats_played(seq.render(40 * FRAMES_PER_BEAT))
    assert all(beat in range(8, 16) or beat in range(24, 40) for beat in played)
    assert 39 in played  # it reached the end and came back


def test_a_song_without_both_marks_cannot_be_played():
    with pytest.raises(AdamusError, match="marked focus"):
        sequencer(marks=REST)
    with pytest.raises(AdamusError, match="marked rest"):
        sequencer(marks=FOCUS)


def test_a_place_with_a_bar_of_the_part_ahead_wins_over_a_better_match_that_runs_out():
    rest = [Mark(0.0, 4.0, c.REST), Mark(8.0, 9.0, c.REST), Mark(12.0, 16.0, c.REST)]
    focus = [Mark(4.0, 8.0, c.FOCUS), Mark(9.0, 12.0, c.FOCUS), Mark(16.0, 20.0, c.FOCUS)]
    sound = unit_vectors(0)
    sound[12:16] = sound[4:8]  # before beat 16, two beats of rest, it sounds as before beat 8
    seq = sequencer(sound=sound, marks=rest + focus)
    assert beats_played(seq.render(9 * FRAMES_PER_BEAT))[8] in (24, 28)


def test_the_music_switches_at_most_once_a_bar_however_the_state_wavers():
    seq = sequencer()
    seq.render(FRAMES_PER_BEAT // 2)
    seq.want(c.FOCUS)
    seq.render(FRAMES_PER_BEAT)  # switched at the end of beat 0
    seq.want(c.REST)  # back at once, and then again and again
    played = []
    for index in range(12):
        seq.want(c.REST if index % 2 == 0 else c.FOCUS)
        played += beats_played(seq.render(FRAMES_PER_BEAT))
    switches = [d for d in decisions(seq) if d.kind == SWITCH]
    beats_between = np.diff([round(d.out_s / BEAT_S) for d in switches])
    assert len(switches) >= 2 and beats_between.min() >= 4


def test_a_switch_where_the_song_itself_goes_on_into_the_other_state_follows_the_song():
    seq = sequencer()
    before = seq.render(7 * FRAMES_PER_BEAT + 10)  # into beat 7, the last of the rest
    seq.want(c.FOCUS)
    played = beats_played(np.vstack([before, seq.render(6 * FRAMES_PER_BEAT)]))
    assert played == list(range(13))  # beat 8, the first of focus, follows as in the song
    switch = decisions(seq)[1]
    assert (switch.kind, switch.mark, switch.from_s, switch.to_s) == (SWITCH, c.FOCUS, 4.0, 4.0)
    seq.want(c.REST)  # at once again: held until a bar has passed since that switch
    assert beats_played(seq.render(FRAMES_PER_BEAT))[0] == 13
