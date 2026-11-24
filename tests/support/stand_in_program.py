"""The program as the app runs it, on the stand-in computer of the tests: what the tests of the
app (src/ui) start in its place, so that they talk to the real program and its real API.

    python -m tests.support.stand_in_program <folder> [paced]

It runs in ``folder``, over the sessions and the data a test put there, with the configuration
of the repository, the generator and the focus models of the tests; its headset is the
stand-in one, which is on; paced, a session with it takes seconds of real time, so that a
person, or a test acting as one, can stop it. It speaks to the app as the program does
(app_link.py) and ends when its input closes.
"""

import os
import sys
from pathlib import Path

from adamus.app_link import AppMessages, messages_only_on_output
from adamus.config import read_yaml, write_yaml
from adamus.main import Program
from tests.support.project_paths import CONFIG_FILE
from tests.support.stand_in_generator_files import install_generator_files
from tests.support.stand_in_machine import StandInMachine, place_focus_model

GENERATOR = "generator"  # the folder of the stand-in generator, in the folder of the run
PACED = "paced"
BY_HAND_POLL_S = 0.025  # of real time per poll, paced: a session with the headset takes ~30 s


def main() -> int:
    folder = Path(sys.argv[1])
    raw = read_yaml(CONFIG_FILE)
    raw["generator"]["folder"] = GENERATOR
    os.chdir(folder)
    install_generator_files(Path(GENERATOR), raw["generator"]["model"])
    place_focus_model(Path())
    write_yaml(raw, Path("config.yaml"))
    machine = StandInMachine(Path("sessions"), Path("data"))
    machine.paced = sys.argv[2:] == [PACED]
    machine.poll_takes_s = BY_HAND_POLL_S
    messages = AppMessages(messages_only_on_output())
    return Program(Path("config.yaml"), machine, messages, sys.stdin).run()


if __name__ == "__main__":
    sys.exit(main())
