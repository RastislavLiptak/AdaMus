"""main.main: the one command of the program runs the program and nothing else."""

import io
import signal
import sys
from pathlib import Path

import pytest

from adamus import main
from adamus.app_link import AppMessages
from adamus.errors import AdamusError
from adamus.machine import Machine

MESSAGES = io.StringIO()  # where the stand-in output of the messages goes


@pytest.fixture
def program(monkeypatch):
    """A stand-in for the program that notes how it was started and returns code 7; the
    messages go to MESSAGES, not to the output of the test (main moves that aside)."""
    started = []

    class StandInProgram:
        def __init__(self, config_path, machine, messages, app_input):
            started.append((config_path, type(machine), type(messages), app_input))

        def run(self):
            return 7

    monkeypatch.setattr(main, "Program", StandInProgram)
    monkeypatch.setattr(main, "messages_only_on_output", lambda: MESSAGES)
    monkeypatch.setattr(main.sys, "argv", ["adamus"])
    return started


@pytest.fixture(autouse=True)
def signal_handlers_as_before():
    before = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGHUP)}
    yield
    for number, handler in before.items():
        signal.signal(number, handler)


def test_the_program_starts_with_the_configuration_of_the_current_folder_and_the_app_input(
    program,
):
    assert main.main() == 7  # the exit code is the one of the program
    assert program == [(Path("config.yaml"), Machine, AppMessages, sys.stdin)]


@pytest.mark.parametrize("arguments", [["--help"], ["run"], ["--config", "lab.yaml"]])
def test_there_are_no_arguments_to_give(program, monkeypatch, capsys, arguments):
    monkeypatch.setattr(main.sys, "argv", ["adamus", *arguments])
    assert main.main() == 1
    assert program == []  # nothing was started
    assert capsys.readouterr().out.splitlines() == [
        "usage: adamus (no arguments; it reads config.yaml in the current folder)"
    ]


def test_kill_and_a_closed_terminal_take_the_shutdown_path_of_ctrl_c(program):
    """Otherwise Pd, which runs in its own session, would be left playing."""
    main.main()
    assert signal.getsignal(signal.SIGTERM) is signal.default_int_handler
    assert signal.getsignal(signal.SIGHUP) is signal.default_int_handler


@pytest.mark.parametrize(
    "problem", [AdamusError("the generator did not start", "  see the log"), OSError("disk full")]
)
def test_a_problem_the_program_could_not_handle_is_one_message_and_code_1(
    program, monkeypatch, capsys, problem
):
    def failing(self):
        raise problem

    monkeypatch.setattr(main.Program, "run", failing)
    assert main.main() == 1
    captured = capsys.readouterr()
    assert captured.err.splitlines()[0] == f"error: {str(problem).splitlines()[0]}"
    assert captured.out == ""


def test_a_fault_of_the_program_is_not_hidden(program, monkeypatch):
    def failing(self):
        raise ZeroDivisionError

    monkeypatch.setattr(main.Program, "run", failing)
    with pytest.raises(ZeroDivisionError):
        main.main()
