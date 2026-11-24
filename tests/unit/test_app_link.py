"""app_link: the messages to the app, and the program's ear for the app's end."""

import io
import json
import os
import select
import subprocess
import sys
import threading
import time

from adamus.app_link import AppMessages, watch_input
from adamus.preflight import checks
from adamus.preflight.checks import Check

WAIT_S = 10

# A program that keeps its output for the messages, then prints in every way there is: from
# Python, straight to the file descriptor, and from C. Only the message reaches the output.
# The programs these tests start import the program as this process found it (the one
# installed, or the changed copy a mutation test puts in front of it), not the one in src.
PRINTS_EVERY_WAY = """
import ctypes, os, sys
from adamus.app_link import AppMessages, messages_only_on_output
messages = AppMessages(messages_only_on_output())
print("from python")
os.write(1, b"from the descriptor\\n")
libc = ctypes.CDLL(None)
libc.puts(b"from C")
libc.fflush(None)
messages.ready("http://127.0.0.1:1", "secret")
"""


def test_the_checklist_is_one_line_and_a_failed_check_says_how_to_fix_it():
    output = io.StringIO()
    AppMessages(output).checks([
        Check(checks.CONFIGURATION, checks.PASSED, "config.yaml"),
        Check(checks.AUDIO_OUTPUT, checks.FAILED, "44100 Hz"),
        Check(checks.GENERATOR_START, checks.SKIPPED, checks.SKIPPED),
        Check(checks.STORAGE, checks.FAILED, "read-only"),
    ])  # fmt: skip
    (line,) = output.getvalue().splitlines()
    assert json.loads(line) == {"checks": [
        {"name": checks.CONFIGURATION, "status": "passed", "detail": "config.yaml", "fix": None},
        {"name": checks.AUDIO_OUTPUT, "status": "failed", "detail": "44100 Hz",
         "fix": checks.HOW_TO_FIX[checks.AUDIO_OUTPUT]},
        {"name": checks.GENERATOR_START, "status": "skipped", "detail": "skipped", "fix": None},
        {"name": checks.STORAGE, "status": "failed", "detail": "read-only",
         "fix": checks.HOW_TO_FIX[checks.STORAGE]},
    ]}  # fmt: skip


def test_ready_says_where_the_api_answers_and_its_token():
    output = io.StringIO()
    AppMessages(output).ready("http://127.0.0.1:8000", "secret")
    assert output.getvalue() == (
        '{"ready": {"address": "http://127.0.0.1:8000", "token": "secret"}}\n'
    )


class WritesSlowly(io.StringIO):
    """An output that lets other threads run between two writes, as a pipe that is full may:
    a message is written in pieces, which another one must not come between."""

    def write(self, text: str) -> int:
        time.sleep(0)  # the other threads get their turn
        return super().write(text)


def test_messages_from_threads_side_by_side_stay_whole_lines():
    output = WritesSlowly()
    messages = AppMessages(output)
    threads = [
        threading.Thread(target=lambda: [messages.ready("a" * 500, "b") for _ in range(50)])
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    lines = output.getvalue().splitlines()
    assert len(lines) == 200
    expected = {"ready": {"address": "a" * 500, "token": "b"}}
    assert all(json.loads(line) == expected for line in lines)


# A program that says it is ready, then waits for the end of the app, as the program does:
# holding its messages, which would be written out were they let go.
READY_THEN_WAITS = """
import sys
from adamus.app_link import AppMessages, messages_only_on_output
messages = AppMessages(messages_only_on_output())
messages.ready("http://127.0.0.1:1", "secret")
sys.stdin.read()
"""


def test_a_message_reaches_the_app_at_once_not_when_the_program_ends():
    program = subprocess.Popen(
        [sys.executable, "-c", READY_THEN_WAITS], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        bufsize=0,
    )  # fmt: skip
    assert program.stdin is not None and program.stdout is not None
    try:
        said, _, _ = select.select([program.stdout], [], [], WAIT_S)
        assert said, "the message stayed in the program until it would end"
        assert json.loads(program.stdout.readline()) == {
            "ready": {"address": "http://127.0.0.1:1", "token": "secret"}
        }
    finally:
        program.stdin.close()  # as the end of the app does
        program.wait(WAIT_S)
        program.stdout.close()


def test_only_messages_reach_the_output_whatever_else_prints():
    done = subprocess.run(
        [sys.executable, "-c", PRINTS_EVERY_WAY], capture_output=True, text=True, timeout=WAIT_S,
        check=True,
    )  # fmt: skip
    assert done.stdout == '{"ready": {"address": "http://127.0.0.1:1", "token": "secret"}}\n'
    assert sorted(done.stderr.splitlines()) == ["from C", "from python", "from the descriptor"]


def test_the_end_of_the_input_is_told_once_and_not_before():
    reading, writing = os.pipe()
    ended = threading.Event()
    with os.fdopen(reading) as source, os.fdopen(writing, "w") as app:
        thread = watch_input(source, ended.set)
        app.write("anything the app might send\n")
        app.flush()
        assert not ended.wait(0.2)
        app.close()
        assert ended.wait(WAIT_S)
        thread.join(WAIT_S)
        assert not thread.is_alive()
