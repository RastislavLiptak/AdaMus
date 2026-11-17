"""The command ``adamus``: starts the interactive console. There is nothing else to choose."""

import signal
import sys
from pathlib import Path

from adamus import constants as c
from adamus.console.console_app import ConsoleApp
from adamus.errors import AdamusError


def main() -> int:
    """Run the console with the configuration of the current folder; return the exit code."""
    if sys.argv[1:]:
        print(f"usage: adamus (no arguments; it reads {c.FILE_CONFIG} in the current folder)")
        return c.EXIT_ERROR
    # kill and a closed terminal end the program like [CTRL]+[C]: through the
    # normal shutdown, which stops Pd. Otherwise Pd would keep playing.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    signal.signal(signal.SIGHUP, signal.default_int_handler)
    try:
        return ConsoleApp(Path(c.FILE_CONFIG)).run()
    except (AdamusError, OSError) as problem:
        print(f"error: {problem}", file=sys.stderr)
        return c.EXIT_ERROR
