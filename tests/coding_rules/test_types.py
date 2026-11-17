"""The types of the source agree: strict mypy finds no problem.

Among other things, the lines of a session folder are typed (session_files),
so a key renamed by the writer or by a reader alone is found here.
"""

import subprocess
import sys

from tests.support.project_paths import ROOT


def test_mypy_finds_no_problem_in_the_source():
    checked = subprocess.run(
        [sys.executable, "-m", "mypy"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_the_stand_ins_of_the_tests_offer_what_the_session_needs_of_the_devices():
    """Otherwise a test could pass with a stand-in the real device does not resemble."""
    checked = subprocess.run(
        [sys.executable, "-m", "mypy", "--follow-imports=silent",
         str(ROOT / "tests" / "support" / "stand_ins_typed.py")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert checked.returncode == 0, checked.stdout + checked.stderr
