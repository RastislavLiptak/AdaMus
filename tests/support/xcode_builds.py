"""The builds of Xcode the tests make, one at a time: two side by side would take more memory
than the computer has room for beside the other tests."""

import fcntl
from collections.abc import Generator
from contextlib import contextmanager

from tests.support.project_paths import ROOT

LOCK = ROOT / ".build" / "xcode-builds.lock"


@contextmanager
def one_build_at_a_time() -> Generator[None]:
    LOCK.parent.mkdir(exist_ok=True)
    with LOCK.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # released as the file closes
        yield
