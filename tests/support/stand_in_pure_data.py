"""A stand-in for the ``pd`` binary used by the tests.

It accepts the same command line as Pure Data, listens for FUDI messages on
the port the patch tells ``netreceive`` to listen on, reports on stderr like ``mrt2~``
and records a WAV file like ``writesf~``. The audio is a tone with a click
track whose rate follows the weight of prompt slot 1, so a music-response
metric has something to correlate with.

Like Pd with the patch, it quits when its client disconnects. Unlike Pd, it
also ends when the test process that started it is gone, so that an
interrupted test run leaves no stand-ins behind.

Behaviour switches (environment): STAND_IN_PD_NO_LISTEN=1 never opens the port,
STAND_IN_PD_SILENT_MODEL=1 never answers a ``model`` message, STAND_IN_PD_DEFAULT_MODEL=1
reports a default model as loaded when the patch opens, like ``mrt2~`` does
when ``mrt2_base`` is installed, STAND_IN_PD_ASSETS_S=<seconds> takes that long to
load the assets, STAND_IN_PD_DIES_ON_MODEL=1 exits when it is told to load a model,
STAND_IN_PD_HANGS=1 neither quits when its client disconnects nor ends when it is terminated,
STAND_IN_PD_CHATTY=1 also prints to stdout and writes bytes that are not text to stderr,
STAND_IN_PD_LAST_WORDS=<lines> prints that many lines when it quits or is terminated,
STAND_IN_PD_NO_VERSION=1 fails when asked for its version,
STAND_IN_PD_OUTPUTS=<names separated by |> are the audio outputs it lists (pd -listdev),
STAND_IN_PD_CANNOT_PLAY=1 says it could not open the output it was given,
STAND_IN_PD_SILENT_PROMPTS=1 never says the music of a prompt is ready.
"""

import os
import re
import signal
import socket
import sys
import time
import wave
from pathlib import Path

import numpy as np

RATE = 48000
RECEIVE_WAIT_S = 0.05  # how long a read of the connection waits before the audio goes on
RECEIVE_BYTES = 4096
WAIT_FOR_PARENT_S = 1  # how often a stand-in that is not connected looks for its test
# The audio: a quiet tone with a loud click track whose rate follows the weight of slot 1.
TONE_HZ, TONE_LEVEL = 220, 0.02
CLICK_LEVEL, CLICK_SAMPLES = 0.6, 200
MORE_CLICKS = 7  # per second at full focus: 1 to 8 clicks per second
FULL_SCALE = 32767  # of 16-bit samples
OUTPUTS = ("Stand-in TV", "Stand-in Speakers")  # it lists, unless told others
NAMES_APART = "|"  # in STAND_IN_PD_OUTPUTS


def log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


class Recorder:
    def __init__(self) -> None:
        self.path = ""  # the file the next recording goes to
        self.wave: wave.Wave_write | None = None
        self.started_at = 0.0
        self.frames = 0
        self.focus = 0.0
        self.bypass = False  # like mrt2~: it plays as soon as it has a model, unless told not to

    def open(self, path: str) -> None:
        self.path = path

    def start(self) -> None:
        self.wave = wave.open(self.path, "wb")  # noqa: SIM115 - closed by stop()
        self.wave.setnchannels(2)
        self.wave.setsampwidth(2)
        self.wave.setframerate(RATE)
        self.started_at = time.time()
        self.frames = 0

    def advance(self) -> None:
        """Write audio up to the current wall-clock time."""
        if self.wave is None:
            return
        target = int((time.time() - self.started_at) * RATE)
        count = target - self.frames
        if count <= 0:
            return
        index = np.arange(self.frames, target)
        if self.bypass:
            mono = np.zeros(count)
        else:
            period = int(RATE / (1 + MORE_CLICKS * self.focus))
            mono = TONE_LEVEL * np.sin(2 * np.pi * TONE_HZ * index / RATE)
            mono = mono + CLICK_LEVEL * ((index % period) < CLICK_SAMPLES)
        pcm = (np.clip(mono, -1, 1) * FULL_SCALE).astype("<i2")
        self.wave.writeframes(np.column_stack([pcm, pcm]).tobytes())
        self.frames = target

    def stop(self) -> None:
        self.advance()
        if self.wave is not None:
            self.wave.close()
            self.wave = None


def answer_load(what: str, path: str) -> None:
    """Report like mrt2~ does when it is told to load assets or a model."""
    if what == "assets":
        log(f"mrt2~: loading assets from {path}")
        time.sleep(float(os.environ.get("STAND_IN_PD_ASSETS_S", 0)))
        log("mrt2~: assets loaded." if Path(path).is_dir() else "mrt2~: failed to init assets")
        return
    log(f"mrt2~: loading model {path}")
    if os.environ.get("STAND_IN_PD_DIES_ON_MODEL"):
        os._exit(1)
    if not os.environ.get("STAND_IN_PD_SILENT_MODEL"):
        log("mrt2~: model loaded." if Path(path).exists() else "mrt2~: failed to load model")


def parse(text: str) -> tuple[list[list[str]], str]:
    """Split text into messages of atoms, like Pd: a backslash makes the next character plain.

    Returns the complete messages and the text of a message that has not ended yet.
    """
    messages, atoms, atom, escaped, rest_from = [], [], "", False, 0
    for position, character in enumerate(text):
        if escaped:
            atom, escaped = atom + character, False
        elif character == "\\":
            escaped = True
        elif character == ";" or character.isspace():
            if atom:
                atoms.append(atom)
            atom = ""
            if character == ";":
                messages.append(atoms)
                atoms, rest_from = [], position + 1
        else:
            atom += character
    return messages, text[rest_from:]


# What mrt2~ takes without a word: the settings of a style, the volume, drums, the notes held.
SILENT_SETTINGS = ("cfgmusiccoca", "cfgnotes", "temperature", "topk", "volume", "drumless",
                   "noteon", "noteoff")  # fmt: skip


def said_to(head: str, args: list[str]) -> str:
    """What mrt2~ prints when it is reset, or told how much audio to keep ready."""
    if head == "reset":
        return "mrt2~: state reset."
    samples = int(args[0])
    return f"mrt2~: buffer size = {samples} samples ({samples * 1000 / RATE:.1f} ms @ 48 kHz)"


def handle(atoms: list[str], recorder: Recorder) -> None:
    if not atoms:
        return
    recorder.advance()
    head, args = atoms[0], atoms[1:]
    if head in ("assets", "model"):
        answer_load(head, args[0])
    elif head == "bypass":
        recorder.bypass = {"0": False, "1": True}[args[0]]  # nothing else is ever sent
    elif head in ("reset", "buffersize"):
        log(said_to(head, args))
    elif head == "prompt":
        if int(float(args[0])) == 1:
            recorder.focus = float(args[-1])
        if not os.environ.get("STAND_IN_PD_SILENT_PROMPTS"):  # as the engine says it, on stdout
            print("[MagentaRT] Combined Prompt (2) tokens: 1 2 3 4 5 6", flush=True)
    elif head == "rec":
        actions = {"open": recorder.open, "start": recorder.start, "stop": recorder.stop}
        actions[args[0]](*args[1:])
    elif head in SILENT_SETTINGS:
        float(args[0])  # a number, or the stand-in fails as Pd would not
    else:
        log(f"error: mrt2~: no method for '{head}'")


def log_from_signal_handler(text: str) -> None:
    """The handler may interrupt log() halfway: sys.stderr cannot be entered again then,
    so the lines go through a stream of their own."""
    with os.fdopen(os.dup(sys.stderr.fileno()), "w") as stream:
        for line in last_words():
            print(line, file=stream)
        print(text, file=stream)


def last_words() -> list[str]:
    lines = int(os.environ.get("STAND_IN_PD_LAST_WORDS", 0))
    return [f"stand-in pd: closing, line {k}" for k in range(lines)]


def ended_by_sigterm(*_) -> None:
    if os.environ.get("STAND_IN_PD_HANGS"):
        log_from_signal_handler("stand-in pd: ignoring SIGTERM")
        return
    log_from_signal_handler("stand-in pd: ended by SIGTERM")
    sys.exit(0)


def list_devices() -> None:
    """What Pd prints of the audio devices of the Mac, as Pd on the Mac prints it."""
    given = os.environ.get("STAND_IN_PD_OUTPUTS")
    outputs = given.split(NAMES_APART) if given is not None else OUTPUTS
    log("audio input devices:")
    log("0. Core Audio: Stand-in Microphone")
    log("audio output devices:")
    for number, name in enumerate(filter(None, outputs)):
        log(f"{number}. Core Audio: {name}")
    log("API number 4")
    log("")
    log("no MIDI input devices found")


def report_start() -> None:
    """What Pd and mrt2~ print when the patch opens."""
    if os.environ.get("STAND_IN_PD_CANNOT_PLAY"):
        number = sys.argv[sys.argv.index("-audiooutdev") + 1]
        log(f"error: audio output device number ({number}) out of range")
    if os.environ.get("STAND_IN_PD_CHATTY"):
        print("pd: a line on stdout", flush=True)
        sys.stderr.buffer.write(b"pd: not text \xff\xfe")
        sys.stderr.buffer.flush()
        log(" here")
    if os.environ.get("STAND_IN_PD_DEFAULT_MODEL"):
        log("mrt2~: loading default model from /models/mrt2_base.mlxfn")
        log("mrt2~: model loaded.")
    else:
        log("error: mrt2~: failed to init assets from default path")  # as real mrt2~ says
        log("mrt2~: no default model found — send 'model <path>' to load one")


def wait_to_be_terminated(started_by: int) -> None:
    while True:  # until the test that started it is gone (or SIGTERM ends it)
        if os.getppid() != started_by:
            break
        time.sleep(WAIT_FOR_PARENT_S)
    sys.exit(0)


def accepted(port: int, started_by: int) -> socket.socket | None:
    """The connection of the program; None when the test that started it is gone first."""
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(1)
    server.settimeout(WAIT_FOR_PARENT_S)
    while True:  # until the program connects, or the test that started it is gone
        try:
            connection, _ = server.accept()
            return connection
        except TimeoutError:
            if os.getppid() != started_by:
                return None


def serve(connection: socket.socket, started_by: int) -> None:
    """Answer the messages of the program, as Pd with the patch would."""
    connection.settimeout(RECEIVE_WAIT_S)
    recorder = Recorder()
    pending = ""
    while True:  # until the client or the test that started it is gone
        try:
            chunk = connection.recv(RECEIVE_BYTES)
        except TimeoutError:
            recorder.advance()
            if os.getppid() != started_by:
                return
            continue
        if not chunk:  # the client is gone: the patch tells Pd to quit
            recorder.stop()
            if os.environ.get("STAND_IN_PD_HANGS"):
                wait_to_be_terminated(started_by)
            for line in last_words():
                log(line)
            log("stand-in pd: quit, the connection closed")
            return
        messages, pending = parse(pending + chunk.decode())
        for atoms in messages:
            handle(atoms, recorder)


def main() -> int:
    if "-version" in sys.argv:
        if os.environ.get("STAND_IN_PD_NO_VERSION"):
            return 1
        log("Pd-0.57-0 (stand-in)")
        return 0
    if "-listdev" in sys.argv:
        list_devices()
        return 0
    patch = Path(sys.argv[-1]).read_text()
    found = re.search(r"listen (\d+)", patch)
    assert found is not None, "the patch listens on a port"
    port = int(found.group(1))
    signal.signal(signal.SIGTERM, ended_by_sigterm)
    started_by = os.getppid()
    report_start()
    if os.environ.get("STAND_IN_PD_NO_LISTEN"):
        wait_to_be_terminated(started_by)
    connection = accepted(port, started_by)
    if connection is not None:
        serve(connection, started_by)
    return 0


if __name__ == "__main__":
    sys.exit(main())
