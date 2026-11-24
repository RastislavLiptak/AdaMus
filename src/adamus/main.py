"""The command ``adamus``, which AdaMus.app runs: it checks the computer, then answers the app
through its API and runs the sessions the app asks for, until the app has gone. It takes no
arguments.

It tells the app how the checks go and where the API answers (app_link.py). It ends when the
app ends, as when it is killed, or [CTRL]+[C] or kill end it: through the normal shutdown,
which lets a running session end as it should and stops Pd.
"""

import os
import signal
import sys
import threading
from pathlib import Path
from typing import TextIO

from adamus import constants as c
from adamus.activity_log import ActivityLog
from adamus.app_link import AppMessages, messages_only_on_output, watch_input
from adamus.errors import AdamusError
from adamus.machine import Machine
from adamus.preflight.checks import Preflight
from adamus.server.api_server import ApiServer

EXIT_OK, EXIT_ERROR = 0, 1  # what the program ends with


def main() -> int:
    """Run the program with the configuration of the current folder; return the exit code."""
    if sys.argv[1:]:
        print(f"usage: adamus (no arguments; it reads {c.FILE_CONFIG} in the current folder)")
        return EXIT_ERROR
    # kill and a closed terminal end the program like [CTRL]+[C]: through the normal shutdown,
    # which stops Pd. Otherwise Pd would keep playing.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    signal.signal(signal.SIGHUP, signal.default_int_handler)
    messages = AppMessages(messages_only_on_output())
    try:
        return Program(Path(c.FILE_CONFIG), Machine(), messages, sys.stdin).run()
    except (AdamusError, OSError) as problem:
        print(f"error: {problem}", file=sys.stderr)
        return EXIT_ERROR


class Program:
    """The program from its start to its end. The app ends it by closing its ``input``."""

    def __init__(
        self, config_path: Path, machine: Machine, messages: AppMessages, app_input: TextIO
    ) -> None:
        self._config_path = config_path
        self._machine = machine
        self._messages = messages
        self._app_input = app_input
        self.activity = ActivityLog()
        self._gone = threading.Event()  # the app has ended
        self._checking = True  # while the checks run, the app's end interrupts them
        # Held while the checks are said to be over, and while the app's end interrupts them:
        # an interruption then always lands before the checks are over, inside run's try.
        self._interrupting = threading.Lock()

    def run(self) -> int:
        """Run until the program ends; return its exit code."""
        preflight: Preflight | None = None
        server: ApiServer | None = None
        watch_input(self._app_input, self._app_ended)
        try:
            preflight = Preflight(
                self._config_path, self._machine, self._messages.checks, self.activity
            )
            ready = preflight.run()
            with self._interrupting:
                self._checking = False
            if not ready:  # the checklist told the app what failed and how to fix it
                return EXIT_ERROR
            server = ApiServer(preflight, self._machine, self.activity, self._gone.is_set)
            self._messages.ready(*server.open())
            server.serve()
            return EXIT_OK
        except KeyboardInterrupt:  # [CTRL]+[C] in any state ends the program normally
            return EXIT_OK
        finally:
            self._checking = False  # over, also when interrupted (the lock is not taken here, as
            # an interruption that comes now has been sent while the checks ran, and is handled)
            if server is not None:
                server.close()
            if preflight is not None and preflight.generator is not None:
                preflight.generator.stop()

    def _app_ended(self) -> None:
        """The app has gone: a session that runs ends as when it is stopped, and the program
        ends after it. A check may take minutes (a training, an installation), so while they
        run, the program is interrupted as [CTRL]+[C] would (through the same shutdown).
        Called in the thread that watches the input."""
        self._gone.set()
        with self._interrupting:
            if self._checking:
                os.kill(os.getpid(), signal.SIGINT)
