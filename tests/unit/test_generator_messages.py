"""generator_messages: the FUDI text messages sent to Pure Data."""

from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from adamus.realtime import generator_messages
from tests.support import fake_pure_data
from tests.support.finished_sessions import config_with

CONFIG = config_with().generator
FOCUS = "fast loud piano concerto with full symphony orchestra brass and timpani"


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
        assert generator_messages.step_messages(CONFIG, 0.25, 0.75) == [
            "prompt 0 quiet slow solo piano 0.250;",
            f"prompt 1 {FOCUS} 0.750;",
        ]

    @pytest.mark.parametrize(
        ("weight", "text"),
        [(0.0, "0.000"), (1.0, "1.000"), (0.12345, "0.123"), (0.9996, "1.000"), (0.0004, "0.000")],
    )
    def test_weights_have_three_decimals(self, weight, text):
        _idle, focus = generator_messages.step_messages(CONFIG, 1.0 - weight, weight)
        assert focus == f"prompt 1 {FOCUS} {text};"

    def test_forms_of_the_music_come_from_the_configuration(self):
        config = config_with(generator={"prompt_idle": "calm pads", "prompt_focus": "techno"})
        assert generator_messages.step_messages(config.generator, 1.0, 0.0) == [
            "prompt 0 calm pads 1.000;",
            "prompt 1 techno 0.000;",
        ]

    @pytest.mark.parametrize("weights", [(-0.001, 0.5), (0.5, 1.001), (float("nan"), 0.5)])
    def test_weight_outside_zero_to_one_is_refused(self, weights):
        with pytest.raises(AssertionError):
            generator_messages.step_messages(CONFIG, *weights)

    @given(st.floats(0, 1))
    def test_every_message_is_one_message_that_pd_reads_back_as_the_same_weight(self, weight):
        for message in generator_messages.step_messages(CONFIG, 1.0 - weight, weight):
            assert message.endswith(";") and message.count(";") == 1
        focus = generator_messages.step_messages(CONFIG, 1.0 - weight, weight)[1]
        sent = float(focus.rstrip(";").split()[-1])
        assert abs(sent - weight) <= 0.0005 + 1e-12


class TestSessionMessages:
    def test_start_of_a_session_in_the_prescribed_order(self):
        audio = Path("/tmp/sessions/20261110-150412_S01/audio.wav")
        assert generator_messages.session_start_messages(CONFIG, audio) == [
            "reset;",
            "prompt 0 quiet slow solo piano 1.000;",
            f"prompt 1 {FOCUS} 0.000;",
            "rec open /tmp/sessions/20261110-150412_S01/audio.wav;",
            "rec start;",
            "bypass 0;",
        ]

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
        messages, rest = fake_pure_data.parse(
            generator_messages.fudi("rec", "open", generator_messages.path_atom(path))
        )
        assert messages == [["rec", "open", str(path.absolute())]] and rest == ""

    def test_session_records_into_a_folder_with_a_space_in_its_name(self):
        messages = generator_messages.session_start_messages(
            CONFIG, Path("/my sessions/S01/audio.wav")
        )
        assert messages[3] == "rec open /my\\ sessions/S01/audio.wav;"
