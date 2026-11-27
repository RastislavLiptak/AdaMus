"""audio_outputs: the audio outputs of this Mac, read from macOS."""

import subprocess
import sys

import pytest

from adamus import audio_outputs
from adamus.errors import AdamusError
from tests.support.project_paths import ROOT
from tests.support.temporary_output import ANNOUNCED, NAME
from tests.support.waiting import wait_until

SEEN_WITHIN_S = 2.0


def names():
    return audio_outputs.outputs()


def test_every_output_has_a_name_and_the_system_plays_on_one_of_them():
    outputs = audio_outputs.outputs()
    assert all(outputs)
    system = audio_outputs.system_output()
    assert system is None or system in outputs


def test_what_macos_refuses_is_said_with_its_code():
    with pytest.raises(AdamusError) as error:
        audio_outputs.answered(560947818)
    assert str(error.value) == "macOS did not tell the audio devices (Core Audio error 560947818)"


@pytest.mark.slow
def test_an_output_another_program_adds_or_removes_is_seen_at_the_next_look():
    """As headphones are plugged in while the program runs, and taken out again."""
    names()  # looked at before, as the program has
    with subprocess.Popen(
        [sys.executable, "-m", "tests.support.temporary_output"],
        cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    ) as other:  # fmt: skip
        assert other.stdin is not None and other.stdout is not None
        try:
            assert other.stdout.readline().strip() == ANNOUNCED
            wait_until(lambda: NAME in names(), "the added output was not seen", SEEN_WITHIN_S)
        finally:
            other.stdin.close()  # it removes the output
    wait_until(lambda: NAME not in names(), "the removed output was still seen", SEEN_WITHIN_S)
