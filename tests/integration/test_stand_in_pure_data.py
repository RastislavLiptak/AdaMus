"""The stand-in for Pd: like the real Pd with the patch, it leaves nothing behind when the
test that started it is gone, even before anyone connected."""

import os
import subprocess
import sys

from tests.support.stand_in_generator_files import STAND_IN_PD
from tests.support.waiting import wait_until

# Starts the stand-in, waits until it has started, says its pid and ends without waiting.
STARTER = """
import subprocess, sys, time
from pathlib import Path
log = Path(sys.argv[3])
pd = subprocess.Popen(
    [sys.executable, sys.argv[1], sys.argv[2]],
    stdout=subprocess.DEVNULL, stderr=log.open("w"),
)
for _ in range(2000):
    if log.read_text():
        break
    time.sleep(0.01)
print(pd.pid)
"""


def gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_a_stand_in_nobody_connected_to_ends_with_the_test_that_started_it(tmp_path):
    patch = tmp_path / "patch.pd"
    patch.write_text("#X msg 10 10 listen 0 127.0.0.1;")
    started = subprocess.run(
        [sys.executable, "-c", STARTER, str(STAND_IN_PD), str(patch), str(tmp_path / "pd.log")],
        capture_output=True, text=True, check=True, timeout=30,
    )  # fmt: skip
    pid = int(started.stdout)
    try:
        wait_until(lambda: gone(pid), "the stand-in is still running", 10)
    finally:
        if not gone(pid):
            os.kill(pid, 9)
