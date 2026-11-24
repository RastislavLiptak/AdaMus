"""The checkers of the editor find nothing in the program, its tests and its tools.

ruff: the rules it is configured with in pyproject.toml (among them the length and the
branches of a function and its arguments), and the layout it gives the code.

swift-format, which comes with Xcode: the rules and the layout of the code of the app, as
src/ui/.swift-format gives them.

pyright: the types agree. Among other things, the lines of a session folder are typed
(session_format), so a key renamed by the writer or by a reader alone is found here; and a
stand-in of a test that lacks what the device it replaces offers is found where the test hands
it over.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.project_paths import ROOT

PYRIGHT = Path(sys.executable).parent / "pyright"


def checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)


@pytest.mark.parametrize("command", [["check"], ["format", "--check"]], ids=["rules", "layout"])
def test_ruff_finds_nothing(command):
    ruff = checked([sys.executable, "-m", "ruff", *command, "src", "tests", "scripts"])
    assert ruff.returncode == 0, ruff.stdout + ruff.stderr


def test_pyright_finds_no_problem():
    pyright = checked([str(PYRIGHT)])
    assert pyright.returncode == 0, pyright.stdout + pyright.stderr


def test_swift_format_finds_nothing_in_the_app():
    found = checked(["xcrun", "swift-format", "lint", "--strict", "--recursive", "src/ui"])
    assert found.returncode == 0, found.stdout + found.stderr
