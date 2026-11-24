"""The music generator: Pure Data with the mrt2~ object, driven by FUDI text messages.

AdaMus generates no audio itself. It starts Pd once per program run with a
small patch and from then on only sends it messages over TCP. The patch makes
Pd quit when that connection closes, so Pd never outlives the program.
"""

import logging
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from adamus import constants as c
from adamus.config import Generator, Timeouts
from adamus.errors import AdamusError
from adamus.realtime.generator_messages import fudi, path_atom
from adamus.session_format import LOG_SEPARATOR, log_time

log = logging.getLogger(__name__)

PATCH_TEMPLATE = Path(__file__).parent / "pure_data_patch.pd"
PATCH_PORT = "{port}"  # replaced in the template by the port Pd listens on,
PATCH_HOST = "{host}"  # on this computer only: on another address, any computer could send it
PRINTED_BY_PD = "Pure Data printed:"  # heads the lines of Pd in the report of a failed start
RETRY_S = 0.05  # pause between connection attempts: Pd listens soon after it starts
POLL_S = 0.05  # pause between looks at the stderr of Pd
ASSETS_LOADED = "mrt2~: assets loaded."
MODEL_LOADED = "mrt2~: model loaded."
FATAL_LINE_ENDINGS = ("mrt2~: failed to load model", "mrt2~: failed to init assets")
# mrt2~ plays silence whenever the next frame is not ready in time, and by
# itself it keeps hardly more than one frame ready. This is the most it can
# keep: it bridges the moments when the computer is busy with something else,
# and delays the music by as much.
AUDIO_BUFFER_SAMPLES = 8192


# --- The Pd process -----------------------------------------------------------

# How far a start got, as the generator reports it.
REPORT_PD_STARTED, REPORT_PD_CONNECTED = "PD_STARTED", "PD_CONNECTED"
REPORT_MODEL_LOADED = "MODEL_LOADED"


def free_port() -> int:
    """A local port no program listens on, for Pd to listen on: the system names one."""
    with socket.socket() as probe:
        probe.bind((c.LOCAL_HOST, 0))
        return probe.getsockname()[1]


class Report(Protocol):
    """Told each step of the start, with what there is to know about it."""

    def __call__(self, kind: str, /, **fields: Any) -> None: ...


class MusicGenerator:
    """Pure Data running the patch: started once, sent messages, stopped once."""

    def __init__(
        self,
        config: Generator,
        timeouts: Timeouts,
        report: Report,
    ) -> None:
        self.config = config
        self.timeouts = timeouts
        self._report = report  # told how far the start got: REPORT_PD_STARTED, ...
        self.command: list[str] = []  # how Pd was started
        self.pd_version: str | None = None
        self.port = 0  # that Pd listens on, chosen when it is started
        self._patch: Path | None = None  # the patch Pd runs: the template with the port filled in
        self._process: subprocess.Popen[str] | None = None
        self._socket: socket.socket | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lock = threading.Lock()
        self._printed: list[tuple[float, str]] = []  # when and what Pd printed for no session yet
        self._session_log: Path | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        """Start Pd, connect, put the generator to sleep, then load assets and model.

        A start that fails reports what Pd printed along with the cause.
        """
        try:
            self._start_pd()
            self._connect()
            self._report(REPORT_PD_CONNECTED)
            self.send(fudi("bypass", 1))  # asleep before it has a model, which it would play
            deadline = time.monotonic() + self.timeouts.model_load_s
            self._load("assets", self.config.assets, ASSETS_LOADED, deadline)
            self._load("model", self.config.model_file, MODEL_LOADED, deadline)
            self._report(REPORT_MODEL_LOADED)
            self.send(fudi("buffersize", AUDIO_BUFFER_SAMPLES))
        except AdamusError as problem:
            self.stop()
            printed = [f"  {line.rstrip()}" for _, line in self._printed]
            heading = [PRINTED_BY_PD] if printed else []
            raise AdamusError(*problem.args, *heading, *printed) from problem
        except BaseException:
            self.stop()
            raise
        version = subprocess.run(
            [self.command[0], "-version"], capture_output=True, text=True, check=False
        )
        self.pd_version = (version.stderr or version.stdout).strip() or None

    def send(self, message: str) -> None:
        if self._socket is None:
            raise AdamusError("the generator is not running")
        try:
            self._socket.sendall(message.encode("utf-8"))
        except OSError as problem:
            raise AdamusError(f"cannot send to Pd: {problem}") from problem

    def copy_stderr_to(self, session_log: Path | None) -> None:
        """While a session runs, what Pd prints goes to its pd.log.

        What Pd printed since the session before, or since it was started,
        goes there first.
        """
        with self._stderr_lock:
            self._session_log = session_log
            if session_log is not None:
                self._store(self._printed)
                self._printed.clear()

    def stop(self) -> None:
        """End Pd: it quits when the connection closes; SIGTERM ends one that was not connected."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._patch is not None:
            self._patch.unlink(missing_ok=True)
            self._patch = None
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(c.STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(c.STOP_WAIT_S)  # it ends when the stderr of Pd closes
        if self._process.stderr is not None:
            self._process.stderr.close()
        self._process = None

    def _start_pd(self) -> None:
        self.port = free_port()
        template = PATCH_TEMPLATE.read_text(encoding="utf-8").replace(PATCH_HOST, c.LOCAL_HOST)
        with tempfile.NamedTemporaryFile(
            "w", prefix="adamus_", suffix=".pd", delete=False, encoding="utf-8"
        ) as patch:
            patch.write(template.replace(PATCH_PORT, str(self.port)))
        self._patch = Path(patch.name)
        self.command = [
            str(self.config.pd_binary.resolve()),
            "-nogui", "-stderr", "-noadc",
            "-r", str(c.AUDIO_RATE),
            "-path", str(self.config.externals.resolve()),
            "-send", "pd dsp 1",
            str(self._patch),
        ]  # fmt: skip
        try:
            # Its own session keeps a terminal [CTRL]+[C] away from Pd, so that
            # the session can still send its closing messages.
            self._process = subprocess.Popen(
                self.command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, errors="replace", start_new_session=True,
            )  # fmt: skip
        except OSError as problem:
            raise AdamusError(f"cannot start Pd ({self.command[0]}): {problem}") from problem
        self._stderr_thread = threading.Thread(
            target=self._copy_stderr, name="pd-stderr", daemon=True
        )
        self._stderr_thread.start()
        self._report(REPORT_PD_STARTED, pid=self._process.pid)

    def _copy_stderr(self) -> None:
        """Thread: every stderr line of Pd goes to the running session, or waits for the next,
        with the time it came."""
        assert self._process is not None and self._process.stderr is not None
        for line in self._process.stderr:
            with self._stderr_lock:
                if self._session_log is None:
                    self._printed.append((time.time(), line))
                else:
                    self._store([(time.time(), line)])

    def _store(self, lines: list[tuple[float, str]]) -> None:
        """The lines go to pd.log, each after the time Pd printed it."""
        assert self._session_log is not None
        with self._session_log.open("a", encoding="utf-8") as session_log:
            for printed, line in lines:
                print(log_time(printed), line.rstrip(), sep=LOG_SEPARATOR, file=session_log)

    def _connect(self) -> None:
        attempts = max(1, round(self.timeouts.pd_connect_s / RETRY_S))
        for _ in range(attempts):
            if not self.running:
                raise AdamusError("Pd exited before accepting a connection")
            try:
                self._socket = socket.create_connection((c.LOCAL_HOST, self.port))
                return
            except OSError:
                time.sleep(RETRY_S)
        raise AdamusError(
            f"cannot connect to Pd on port {self.port} within {self.timeouts.pd_connect_s:g} s"
        )

    def _load(self, what: str, path: Path, loaded_line: str, deadline: float) -> None:
        """Send ``assets <path>`` or ``model <path>`` and wait until mrt2~ confirms it.

        Only lines printed after the message count: a default model loaded when
        the object was created must not be mistaken for the requested one.
        ``deadline`` is the moment by which assets and model must both be loaded.
        """
        seen = len(self._printed)
        self.send(fudi(what, path_atom(path)))
        for _ in range(max(1, round(self.timeouts.model_load_s / POLL_S))):
            if time.monotonic() >= deadline:
                break
            for _, line in self._printed[seen:]:
                if loaded_line in line:
                    return
                if line.rstrip().endswith(FATAL_LINE_ENDINGS):
                    raise AdamusError(f"the generator reported: {line.strip()}")
            if not self.running:
                raise AdamusError(f"Pd exited while loading the {what}")
            time.sleep(POLL_S)
        raise AdamusError(f"mrt2~ did not load the {what} within {self.timeouts.model_load_s:g} s")
