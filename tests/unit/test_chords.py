"""chords: the loop of chords the generator is held under both forms of the music."""

from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from adamus.realtime.chords import NOTES, ChordLoop, voicing
from tests.support.finished_sessions import config_with

BY_ID = {style.id: style for style in config_with().styles}
FOLK, ROCK, KOTO, OUD = BY_ID["folk"], BY_ID["rock"], BY_ID["koto"], BY_ID["oud"]
START = 1000.0  # on the monotonic clock


def bar_s(style):
    return 4 * 60.0 / style.tempo_bpm


def on(*notes):
    return [f"noteon {note};" for note in notes]


def off(*notes):
    return [f"noteoff {note};" for note in notes]


class TestVoicing:
    @pytest.mark.parametrize(
        ("chord", "notes"),
        [("C", (48, 60, 64, 67)), ("Am", (45, 57, 60, 64)), ("G7", (43, 59, 62, 65, 67)),
         ("Fmaj7", (41, 57, 60, 64, 65)), ("Em7", (40, 59, 62, 64, 67)), ("A5", (45, 57, 64)),
         ("D", (50, 57, 62, 66)), ("B", (47, 59, 63, 66))],
    )  # fmt: skip
    def test_the_root_in_the_bass_and_the_chord_above_it(self, chord, notes):
        assert voicing(chord) == notes

    @pytest.mark.parametrize(("chord", "notes"), [("Am", (45, 57, 64)), ("C", (48, 60, 67)),
                                                   ("G", (43, 62, 67))])  # fmt: skip
    def test_a_power_chord_is_root_fifth_and_octave(self, chord, notes):
        assert voicing(chord, power=True) == notes

    @given(st.sampled_from("ABCDEFG"), st.sampled_from(["", "m", "7", "m7", "maj7", "5"]))
    def test_every_chord_has_its_root_in_the_bass_and_above_it(self, root, kind):
        bass, *upper = voicing(root + kind)
        assert 40 <= bass <= 51 and all(57 <= note <= 68 for note in upper)
        assert bass % 12 == NOTES[root] and NOTES[root] in {note % 12 for note in upper}


class TestChordLoop:
    def test_the_first_chord_is_struck_at_once_and_held_through_its_bar(self):
        chords = ChordLoop(FOLK, START)
        assert chords.due(START, focused=False) == on(48, 60, 64, 67)
        assert chords.due(START + bar_s(FOLK) - 0.001, focused=False) == []

    def test_the_next_chord_is_struck_after_the_gap_once_the_last_is_let_go(self):
        chords = ChordLoop(FOLK, START)
        chords.due(START, focused=False)
        bar = START + bar_s(FOLK)
        assert chords.due(bar, focused=False) == off(48, 60, 64, 67)
        assert chords.wait(bar) == pytest.approx(FOLK.chord_gap_s)
        assert chords.due(bar + FOLK.chord_gap_s / 2, focused=False) == []
        assert chords.due(bar + FOLK.chord_gap_s, focused=False) == on(41, 57, 60, 65)  # F

    def test_without_a_gap_a_chord_is_let_go_and_the_next_struck_at_once(self):
        chords = ChordLoop(ROCK, START)
        chords.due(START, focused=False)
        assert chords.due(START + bar_s(ROCK), focused=False) == off(48, 60, 64, 67) + on(
            43, 59, 62, 67
        )

    def test_the_loop_goes_round_again(self):
        chords = ChordLoop(ROCK, START)
        struck = [chords.due(START + bar * bar_s(ROCK), focused=False) for bar in range(6)]
        assert struck[4] == off(41, 57, 60, 65) + on(48, 60, 64, 67)  # F, then C again
        assert struck[5] == off(48, 60, 64, 67) + on(43, 59, 62, 67)

    def test_a_chord_like_the_one_held_is_not_struck_again(self):
        chords = ChordLoop(KOTO, START)  # C5 F5 G5 C5: the loop ends and starts on C5
        for bar in range(4):
            chords.due(START + bar * bar_s(KOTO), focused=False)
            chords.due(START + bar * bar_s(KOTO) + KOTO.chord_gap_s, focused=False)
        assert chords.due(START + 4 * bar_s(KOTO), focused=False) == []
        drone = ChordLoop(OUD, START)
        assert drone.due(START, focused=False) == on(45, 57, 64)
        assert [drone.due(START + bar * bar_s(OUD), focused=True) for bar in range(1, 9)] == [
            []
        ] * 8

    def test_in_focus_a_style_with_power_chords_plays_them_at_once_and_its_chords_again_after(self):
        chords = ChordLoop(ROCK, START)
        chords.due(START, focused=False)
        assert chords.due(START + 0.5, focused=True) == off(48, 60, 64, 67) + on(48, 60, 67)
        assert chords.due(START + bar_s(ROCK), focused=True) == off(48, 60, 67) + on(43, 62, 67)
        assert chords.due(START + bar_s(ROCK) + 0.5, focused=False) == off(43, 62, 67) + on(
            43, 59, 62, 67
        )
        folk = ChordLoop(FOLK, START)
        folk.due(START, focused=False)
        assert folk.due(START + 0.5, focused=True) == []  # its chords are the same in focus

    def test_a_change_during_the_gap_strikes_only_the_last_chord_wanted(self):
        chords = ChordLoop(replace(ROCK, chord_gap_s=0.2), START)
        chords.due(START + 0.2, focused=False)
        assert chords.due(START + 1.0, focused=True) == off(48, 60, 64, 67)
        assert chords.due(START + 1.1, focused=False) == []  # nothing held: nothing to let go
        assert chords.due(START + 1.3, focused=False) == on(48, 60, 64, 67)

    def test_it_waits_for_the_next_bar_or_for_the_chord_to_be_struck(self):
        chords = ChordLoop(FOLK, START)
        assert chords.wait(START) == pytest.approx(bar_s(FOLK))
        chords.due(START, focused=False)
        assert chords.wait(START + 1.0) == pytest.approx(bar_s(FOLK) - 1.0)
        assert chords.wait(START + 5 * bar_s(FOLK) + 0.25) == pytest.approx(bar_s(FOLK) - 0.25)

    def test_at_the_end_every_note_held_is_let_go(self):
        chords = ChordLoop(FOLK, START)
        assert chords.release() == []
        chords.due(START, focused=False)
        assert chords.release() == off(48, 60, 64, 67)
        assert chords.release() == []
