"""The app (src/ui): its own tests, which Xcode builds and runs, pass. Its tests of the program
talk to the real program, run here on the stand-in computer over a session played for them
(tests/support/stand_in_program.py): this test starts it, and gives the tests of the app what it
printed and where its API answers, so that the app and the program cannot drift apart."""

import json
import os
import queue
import subprocess
import sys
import threading
import time

import pytest

from adamus.app_link import READY
from adamus.server.api_models import Dashboard, RunState, SourceCheck
from tests.support.played_sessions import play_session
from tests.support.project_paths import ROOT
from tests.support.stand_in_program import PACED
from tests.support.xcode_builds import one_build_at_a_time

PROTOCOL = [["REST", 20], ["FOCUS", 20], ["REST", 20], ["FREE", 0]]
XCODE_S = 900  # the longest a build and run of the tests of the app may take
READY_S = 120  # the longest the program may take to say where its API answers
STOP_S = 30  # the longest the program may take to end once its input closes
# Where Xcode keeps what it built, between runs, so that a run builds only what changed.
BUILT = ROOT / ".build" / "xcode-tests"
UNITS, WINDOW = "AdaMusTests", "AdaMusUITests"  # the targets of the tests of the app
PRODUCTS = BUILT / "Build" / "Products" / "Debug"
# The apps the tests of the window run: macOS asks whether they may use the project's folder.
WINDOW_APPS = [PRODUCTS / "AdaMus.app", PRODUCTS / f"{WINDOW}-Runner.app"]
SHOWN_OUTPUT = 6000  # characters of the end of what xcodebuild printed, when it failed


def xcode(action, target, environment=None, results=None):
    """An action of Xcode on the tests of ``target``; what they did into ``results``."""
    kept = ["-resultBundlePath", str(results)] if results is not None else []
    return subprocess.run(
        ["xcodebuild", action, "-project", "src/ui/AdaMus.xcodeproj", "-scheme", "AdaMus",
         "-destination", "platform=macOS,arch=arm64", "-derivedDataPath", str(BUILT),
         f"-only-testing:{target}", "-quiet", *kept],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=XCODE_S, check=False,
    )  # fmt: skip


def counted(results):
    """How many of the tests in ``results`` passed, failed and were skipped: a test that finds
    nothing to test skips itself, and Xcode counts a run of skipped tests as one that passed."""
    summary = json.loads(subprocess.run(
        ["xcrun", "xcresulttool", "get", "test-results", "summary", "--path", str(results)],
        capture_output=True, text=True, check=True,
    ).stdout)  # fmt: skip
    return summary["passedTests"], summary["failedTests"], summary["skippedTests"]


def lines_into(output, said):
    """Every line of ``output`` into ``said``, then an empty one at its end, where it is closed."""
    with output:
        for line in output:
            said.put(line)
    said.put("")


def started_program(folder, *options):
    """The program of the tests, and every line it printed until its API answers: within
    READY_S, so that a program that never says it fails the test, not hangs it."""
    program = subprocess.Popen(
        [sys.executable, "-m", "tests.support.stand_in_program", str(folder), *options], cwd=ROOT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )  # fmt: skip
    output = program.stdout
    assert output is not None
    said: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lines_into, args=(output, said), daemon=True).start()
    deadline = time.monotonic() + READY_S
    printed: list[str] = []
    while not printed or READY not in printed[-1]:  # it says where its API answers once ready
        try:
            line = said.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty:
            program.kill()
            program.wait()
            message = f"the program said nothing more within {READY_S} s: {printed}"
            raise AssertionError(message) from None
        assert line, f"the program ended before it was ready: {printed}"
        printed.append(line)
    return program, printed


def rare_answers():
    """Answers the program gives only now and then, each optional value of them set to its own
    name: the app must read every one under the name the program gives it."""
    return json.dumps({
        "dashboard": Dashboard(
            name="name", problem="problem", pending="pending", summary=[], failures=["failure"],
            caution="caution",
            highlights=[], sections=[],
        ).model_dump(mode="json"),
        "source_check": SourceCheck(warning="warning", problem="problem").model_dump(mode="json"),
        "run_state": RunState(
            run=1, phase="idle", session_id="session_id", problem="problem"
        ).model_dump(mode="json"),
    })  # fmt: skip


def run_on_a_program(folder, target, *options, given=None):
    """Run the Swift tests of ``target`` against a program of their own over ``folder``: they
    get what it printed, where its API answers, and ``given``. Whether they passed, how many
    passed, failed and were skipped, and how the program ended once its input closed, as the
    end of the app closes it."""
    results = folder / f"{target}.xcresult"
    program, printed = started_program(folder, *options)
    try:
        ready = json.loads(printed[-1])[READY]
        told = {
            "MESSAGES": "".join(printed), "ADDRESS": ready["address"], "TOKEN": ready["token"],
            **(given or {}),
        }  # fmt: skip
        environment = {
            **os.environ, **{f"TEST_RUNNER_ADAMUS_{name}": value for name, value in told.items()}
        }  # fmt: skip
        with one_build_at_a_time():  # the build only: the tests that follow build nothing
            done = xcode("build-for-testing", target)
            if done.returncode == 0 and target == WINDOW:  # macOS asks once (sign_app.sh)
                subprocess.run(["sh", "scripts/sign_app.sh", *WINDOW_APPS], cwd=ROOT, check=True)
        if done.returncode == 0:
            done = xcode("test-without-building", target, environment, results)
    finally:
        assert program.stdin is not None
        program.stdin.close()
        ended = program.wait(STOP_S)
    return done, counted(results) if results.exists() else None, ended


def test_the_tests_of_the_app_pass(tmp_path):
    """Without the app: the tests compile its code, so that no window opens."""
    folder = tmp_path / "run"
    play_session(folder, "morning_focus", PROTOCOL, recording="short")
    done, counts, ended = run_on_a_program(folder, UNITS, given={"ANSWERS": rare_answers()})
    assert done.returncode == 0, (done.stdout + done.stderr)[-SHOWN_OUTPUT:]
    assert counts is not None and counts[0] > 0 and counts[1:] == (0, 0), counts  # none skipped
    assert ended == 0


@pytest.mark.window
def test_the_window_does_what_the_operator_asks(tmp_path):
    """The window used as a person would, with the mouse and the keyboard: the app started by
    the tests of the window (src/ui/AdaMusUITests), on a program whose sessions with the headset
    take seconds, to be stopped by hand."""
    folder = tmp_path / "run"
    play_session(folder, "morning_focus", PROTOCOL, recording="short")
    done, counts, ended = run_on_a_program(folder, WINDOW, PACED)
    assert done.returncode == 0, (done.stdout + done.stderr)[-SHOWN_OUTPUT:]
    assert counts is not None and counts[0] > 0 and counts[1:] == (0, 0), counts  # none skipped
    assert ended == 0
