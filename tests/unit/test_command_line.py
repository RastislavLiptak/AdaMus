"""cli.main: the one command of the program starts the console and nothing else."""

import signal
from pathlib import Path

import pytest

from adamus import cli
from adamus.errors import AdamusError


@pytest.fixture
def console(monkeypatch):
    """A stand-in for the console that notes how it was started and returns code 7."""
    started = []

    class StandInConsole:
        def __init__(self, config_path):
            started.append(config_path)

        def run(self):
            return 7

    monkeypatch.setattr(cli, "ConsoleApp", StandInConsole)
    monkeypatch.setattr(cli.sys, "argv", ["adamus"])
    return started


@pytest.fixture(autouse=True)
def signal_handlers_as_before():
    before = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGHUP)}
    yield
    for number, handler in before.items():
        signal.signal(number, handler)


def test_console_starts_with_the_configuration_of_the_current_folder(console):
    assert cli.main() == 7  # the exit code is the one of the console
    assert console == [Path("config.yaml")]


@pytest.mark.parametrize("arguments", [["--help"], ["run"], ["--config", "lab.yaml"]])
def test_there_are_no_arguments_to_give(console, monkeypatch, capsys, arguments):
    monkeypatch.setattr(cli.sys, "argv", ["adamus", *arguments])
    assert cli.main() == 1
    assert console == []  # nothing was started
    assert capsys.readouterr().out.splitlines() == [
        "usage: adamus (no arguments; it reads config.yaml in the current folder)"
    ]


def test_kill_and_a_closed_terminal_take_the_shutdown_path_of_ctrl_c(console):
    """Otherwise Pd, which runs in its own session, would be left playing."""
    cli.main()
    assert signal.getsignal(signal.SIGTERM) is signal.default_int_handler
    assert signal.getsignal(signal.SIGHUP) is signal.default_int_handler


@pytest.mark.parametrize(
    "problem", [AdamusError("the generator did not start", "  see the log"), OSError("disk full")]
)
def test_problem_the_console_could_not_handle_is_one_message_and_code_1(
    console, monkeypatch, capsys, problem
):
    def failing(self):
        raise problem

    monkeypatch.setattr(cli.ConsoleApp, "run", failing)
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert captured.err.splitlines()[0] == f"error: {str(problem).splitlines()[0]}"
    assert captured.out == ""


def test_fault_of_the_program_is_not_hidden(console, monkeypatch):
    def failing(self):
        raise ZeroDivisionError

    monkeypatch.setattr(cli.ConsoleApp, "run", failing)
    with pytest.raises(ZeroDivisionError):
        cli.main()
