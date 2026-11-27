"""The API of the program as a whole: its server and the sessions it is asked for.

The server answers in a thread of its own on a free port of the local computer; the main
thread runs the sessions until the app has gone.
"""

import socket
import threading
import time
from collections.abc import Callable

import uvicorn

from adamus import constants as c
from adamus.activity_log import ActivityLog
from adamus.errors import AdamusError
from adamus.machine import Machine
from adamus.server.api import Services, create_app
from adamus.server.new_session import NewSession, PreflightResults
from adamus.server.session_catalogue import SessionCatalogue
from adamus.server.session_runner import SessionRunner

ANY_FREE_PORT = 0
SERVER_START_S = 10.0  # the longest the server may take to listen
SERVER_LOOK_S = 0.01  # between two looks whether it does


class ApiServer:
    """``app_gone`` says whether the app the API answers has ended."""

    def __init__(
        self,
        preflight: PreflightResults,
        machine: Machine,
        activity: ActivityLog,
        app_gone: Callable[[], bool],
    ) -> None:
        self._preflight = preflight
        self._machine = machine
        self._activity = activity
        self._runner = SessionRunner(machine, activity, app_gone)
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._listening: socket.socket | None = None  # the port it answers on, held until it ends

    def open(self) -> tuple[str, str]:
        """Start the server; return the address it answers at and the token its calls carry."""
        services = Services(
            catalogue=SessionCatalogue(self._machine.sessions_root, self._machine.datasets_root),
            new_session=NewSession(self._preflight, self._machine, self._activity),
            runner=self._runner,
            library=self._machine.library,
        )
        return self._start_server(services), services.token

    def serve(self) -> None:
        """Run the sessions the app asks for until it has gone."""
        self._runner.serve()

    def close(self) -> None:
        """Stop the server, also one that only partly started."""
        if self._server is not None and self._thread is not None:
            self._server.should_exit = True
            self._thread.join(c.STOP_WAIT_S)
        if self._listening is not None:
            self._listening.close()

    def _start_server(self, services: Services) -> str:
        listening = self._listening = socket.socket()
        listening.bind((c.LOCAL_HOST, ANY_FREE_PORT))
        port = listening.getsockname()[1]
        config = uvicorn.Config(create_app(services), log_level="warning", access_log=False)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, kwargs={"sockets": [listening]}, name="server", daemon=True
        )
        self._thread.start()
        for _ in range(round(SERVER_START_S / SERVER_LOOK_S)):
            if self._server.started:
                return f"http://{c.LOCAL_HOST}:{port}"
            time.sleep(SERVER_LOOK_S)
        raise AdamusError(f"the server of the API did not start within {SERVER_START_S:g} s")
