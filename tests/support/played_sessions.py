"""Sessions played over a recording of the synthetic dataset, as the program leaves them: with
the music the generator would have recorded and their metrics, and their logs on the clock of
the session. Shared by the tests of the server and those of the interface."""

import re
from pathlib import Path

import numpy as np
import soundfile

from adamus import constants as c
from adamus.evaluation.metrics import analyze
from adamus.realtime.generator_messages import REC_START
from adamus.realtime.session import STEP_LOG
from adamus.session_format import log_time, read_log_line
from tests.support.finished_sessions import FinishedSession, config_with, synthetic_signals
from tests.support.synthetic_dataset import Protocol, write_recording

QUIET, LOUD = 0.05, 0.5  # of the stand-in music at rest and in focus
TONE_HZ = 440

STEP_NUMBER = re.escape(STEP_LOG.split("%d")[0]) + r"(\d+)"  # in a line a step logged


def write_music(session: FinishedSession) -> None:
    """A WAV for the time between "rec start;" and the end of the session, louder where the
    weights asked for focus."""
    start = next(e["t"] for e in session.events if e.get("msg") == REC_START)
    end = session.events[-1]["t"]
    times = np.arange(int((end - start) * c.AUDIO_RATE)) / c.AUDIO_RATE
    sent = [(t["t_sent"] - start, t["w_focus"]) for t in session.ticks if t["t_sent"] is not None]
    focus = np.interp(times, [s for s, _ in sent], [w for _, w in sent]) if sent else 0 * times
    wave = (QUIET + (LOUD - QUIET) * focus) * np.sin(2 * np.pi * TONE_HZ * times)
    soundfile.write(session.directory / c.FILE_AUDIO, np.column_stack([wave, wave]),
                    c.AUDIO_RATE, subtype="PCM_16")  # fmt: skip


def move_log_to_the_session_clock(session: FinishedSession) -> None:
    """The stand-in clock is not the time of day the log is written in, and the stand-in
    session runs faster than time: its lines move to the clock of the session, each step
    to when it was done, what came before the steps to the start and what came after them
    to the end, as in a real session. Pd says it records when the music begins."""
    path = session.directory / c.FILE_LOG
    times = [session.events[0]["t"]]
    read = map(read_log_line, path.read_text(encoding="utf-8").splitlines())
    lines = [said for _, said in filter(None, read)]
    with path.open("w", encoding="utf-8") as log:
        for said in lines:
            step = re.search(STEP_NUMBER, said)
            if step is not None:
                times.append(session.ticks[int(step.group(1))]["t_proc_end"])
            elif len(times) > 1:
                times.append(session.events[-1]["t"])
            print(log_time(times[-1]), said, file=log)
    began = next(e["t"] for e in session.events if e.get("msg") == REC_START)
    with (session.directory / c.FILE_PD_LOG).open("w", encoding="utf-8") as log:
        print(log_time(began), "mrt2~: recording.", file=log)


def play_session(root: Path, name: str, protocol: Protocol, recording: str) -> FinishedSession:
    """A session ``name`` over the recording data/synthetic/``recording`` with ``protocol``
    (its last block lasts until the stop), with music and metrics; its ``root`` noted."""
    folder = root / "data" / "synthetic" / recording
    if not folder.is_dir():
        write_recording(folder, synthetic_signals(protocol[:-1]), protocol[:-1])
    session = FinishedSession(root, config_with(protocol), recording=folder, name=name)
    write_music(session)
    move_log_to_the_session_clock(session)
    analyze(session.directory)
    return session
