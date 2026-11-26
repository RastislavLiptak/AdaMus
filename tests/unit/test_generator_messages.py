"""generator_messages: the FUDI text messages sent to Pure Data."""

from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from adamus.realtime import generator_messages
from tests.support import stand_in_pure_data
from tests.support.finished_sessions import config_with

GENERATOR = config_with().generator
STYLES = config_with().styles
BY_ID = {style.id: style for style in STYLES}
FOLK, COUNTRY, ROCK = BY_ID["folk"], BY_ID["country"], BY_ID["rock"]
IDLE, FOCUS = FOLK.rest, FOLK.focus


def test_message_is_atoms_joined_by_spaces_and_ended_by_a_semicolon():
    assert generator_messages.fudi("bypass", 1) == "bypass 1;"
    assert generator_messages.fudi("reset") == "reset;"
    assert generator_messages.fudi("rec", "open", "/a/b.wav") == "rec open /a/b.wav;"


def test_recording_messages():
    assert (generator_messages.REC_START, generator_messages.REC_STOP) == (
        "rec start;",
        "rec stop;",
    )


class TestStepMessages:
    def test_slot_0_is_the_quiet_form_and_slot_1_the_full_form_of_the_music(self):
        assert generator_messages.step_messages(FOLK, 0.25, 0.75, focused=False)[:2] == [
            f"prompt 0 {IDLE} 0.250;",
            f"prompt 1 {FOCUS} 0.750;",
        ]

    @pytest.mark.parametrize(
        ("weight", "text"),
        [(0.0, "0.000"), (1.0, "1.000"), (0.12345, "0.123"), (0.9996, "1.000"), (0.0004, "0.000")],
    )
    def test_weights_have_three_decimals(self, weight, text):
        focus = generator_messages.step_messages(FOLK, 1.0 - weight, weight, focused=False)[1]
        assert focus == f"prompt 1 {FOCUS} {text};"

    def test_forms_of_the_music_come_from_the_style(self):
        style = replace(FOLK, rest="calm pads", focus="techno")
        assert generator_messages.step_messages(style, 1.0, 0.0, focused=False)[:2] == [
            "prompt 0 calm pads 1.000;",
            "prompt 1 techno 0.000;",
        ]

    @pytest.mark.parametrize(
        ("w_focus", "volume"),
        [(0.0, "volume -4.00;"), (1.0, "volume 2.00;"), (0.25, "volume -2.50;")],
    )
    def test_the_volume_lies_between_the_two_forms_as_the_weights_do(self, w_focus, volume):
        assert (
            generator_messages.step_messages(ROCK, 1.0 - w_focus, w_focus, focused=False)[2]
            == volume
        )

    @pytest.mark.parametrize(
        ("style", "focused", "drumless"),
        [(COUNTRY, False, 1), (COUNTRY, True, 0), (FOLK, True, 1), (ROCK, True, 1)],
    )  # fmt: skip
    def test_drums_may_play_in_focus_in_a_style_that_has_them_and_never_else(
        self, style, focused, drumless
    ):
        assert (
            generator_messages.step_messages(style, 0.3, 0.7, focused)[3] == f"drumless {drumless};"
        )

    @pytest.mark.parametrize("weights", [(-0.001, 0.5), (0.5, 1.001), (float("nan"), 0.5)])
    def test_weight_outside_zero_to_one_is_refused(self, weights):
        with pytest.raises(AssertionError):
            generator_messages.step_messages(FOLK, *weights, focused=False)

    @given(st.floats(0, 1))
    def test_every_message_is_one_message_that_pd_reads_back_as_the_same_weight(self, weight):
        for message in generator_messages.step_messages(ROCK, 1.0 - weight, weight, focused=False):
            assert message.endswith(";") and message.count(";") == 1
        focus = generator_messages.step_messages(ROCK, 1.0 - weight, weight, focused=False)[1]
        sent = float(focus.rstrip(";").split()[-1])
        assert abs(sent - weight) <= 0.0005 + 1e-12


class TestStyleMessages:
    def test_how_the_generator_plays_the_style(self):
        assert generator_messages.style_messages(GENERATOR, ROCK) == [
            "cfgmusiccoca 5;", "cfgnotes 5;", "temperature 0.9;", "topk 40;",
        ]  # fmt: skip

    def test_every_text_of_every_style_is_made_ready_once(self):
        twice = [FOLK, replace(COUNTRY, rest=FOLK.rest)]  # a text two styles share
        assert generator_messages.prompt_messages(twice) == [
            f"prompt 0 {FOLK.rest} 1.000;", f"prompt 0 {FOLK.focus} 1.000;",
            f"prompt 0 {COUNTRY.focus} 1.000;",
        ]  # fmt: skip
        assert len(generator_messages.prompt_messages(STYLES)) == 2 * len(STYLES)


class TestSessionMessages:
    def test_start_of_a_session_in_the_prescribed_order(self):
        audio = Path("/tmp/sessions/20261110-150412_S01/audio.wav")
        assert generator_messages.session_start_messages(GENERATOR, COUNTRY, audio) == [
            "reset;",
            "cfgmusiccoca 5;", "cfgnotes 5;", "temperature 1;", "topk 100;",
            f"prompt 0 {COUNTRY.rest} 1.000;",
            f"prompt 1 {COUNTRY.focus} 0.000;",
            f"volume {COUNTRY.rest_volume_db:.2f};",
            "drumless 1;",
            "rec open /tmp/sessions/20261110-150412_S01/audio.wav;",
            "rec start;",
            "bypass 0;",
        ]  # fmt: skip

    def test_end_of_a_session_in_the_prescribed_order(self):
        assert generator_messages.session_end_messages() == ["bypass 1;", "rec stop;"]


class TestPathAtom:
    """A path travels as one atom, wherever it lies and whatever it is called."""

    def test_plain_absolute_path_is_passed_through(self):
        assert generator_messages.path_atom(Path("/a/b-c_d/Škola/e.wav")) == "/a/b-c_d/Škola/e.wav"

    def test_relative_path_is_made_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert generator_messages.path_atom(Path("sessions/audio.wav")) == str(
            tmp_path.resolve() / "sessions" / "audio.wav"
        )

    @pytest.mark.parametrize(
        ("path", "atom"),
        [
            ("/with space/a.wav", "/with\\ space/a.wav"),
            ("/My Music/New Folder/a.wav", "/My\\ Music/New\\ Folder/a.wav"),
            ("/semi;colon/a.wav", "/semi\\;colon/a.wav"),
            ("/comma,/a.wav", "/comma\\,/a.wav"),
            ("/dollar$1/a.wav", "/dollar\\$1/a.wav"),
            ("/back\\slash/a.wav", "/back\\\\slash/a.wav"),
            ("/tab\there/a.wav", "/tab\\\there/a.wav"),
        ],
    )
    def test_characters_pd_would_act_on_get_a_backslash_in_front(self, path, atom):
        assert generator_messages.path_atom(Path(path)) == atom

    @given(st.text(st.characters(codec="utf-8", exclude_characters="/\x00"), min_size=1))
    @example("..")  # the path is sent as it is, not shortened
    def test_pd_reads_any_folder_name_back_as_one_atom(self, name):
        """The stand-in for Pd splits messages the way Pd does."""
        path = Path("/sessions") / name / "audio.wav"
        messages, rest = stand_in_pure_data.parse(
            generator_messages.fudi("rec", "open", generator_messages.path_atom(path))
        )
        assert messages == [["rec", "open", str(path.absolute())]] and rest == ""

    def test_session_records_into_a_folder_with_a_space_in_its_name(self):
        messages = generator_messages.session_start_messages(
            GENERATOR, FOLK, Path("/my sessions/S01/audio.wav")
        )
        assert "rec open /my\\ sessions/S01/audio.wav;" in messages
