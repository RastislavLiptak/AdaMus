"""Hand-made session folders, for tests of metrics and statistics on exact values."""

import json
from pathlib import Path

import numpy as np
from brainflow.board_shim import BoardShim

from adamus import constants as c
from adamus.config import write_yaml
from tests.support.finished_sessions import CONTEXT, FS, H, N
from tests.support.stand_in_devices import MUSE_2

ELECTRODES = len(c.DETECTOR_ELECTRODES)
BOARD = BoardShim.get_board_descr(MUSE_2)
# The times of a step: the wall clock is behind the EEG by this much, then it takes these.
EEG_BEHIND_S, PROCESSING_S, SENDING_S = 0.020, 0.001, 0.010
MAX_ABS_UV = 10.0  # of every electrode in every step
RAW_START_S = 100.0  # the time of the first sample in raw.csv


def tick(k, block=0, label="REST", full=True, valid=True, f=None, sent=True, score=0.0, t0=100.0):
    """One line of ticks.jsonl. Wall time = EEG time + 20 ms, processing 1 ms, sending 10 ms."""
    t = t0 + (CONTEXT + k * H) / FS
    was_sent = sent and valid and f is not None
    return {
        "k": k, "n_end": CONTEXT + k * H, "t_eeg": t - EEG_BEHIND_S, "t_proc_start": t,
        "t_proc_end": t + PROCESSING_S, "t_sent": t + SENDING_S if was_sent else None,
        "block_index": block, "block_label": label, "full_window": full,
        "max_abs_uv": [MAX_ABS_UV] * ELECTRODES, "valid": valid,
        "score": score if valid and f is not None else None, "f": f,
        "f_held": f is not None and not valid, "w_idle": 1.0 - (f or 0.0), "w_focus": f or 0.0,
    }  # fmt: skip


def event(kind, t, n=0, **fields):
    return {"t": t, "n": n, "type": kind, **fields}


def protocol_ticks(protocol, f_of_block, steps=None, valid=lambda k: True):
    """Ticks of a protocol of [label, seconds]; ``f_of_block[i]`` is f, None or a function of k."""
    starts = np.cumsum([0] + [seconds * FS for _, seconds in protocol])
    total = steps if steps is not None else (int(starts[-1]) - CONTEXT) // H + 1
    rows = []
    for k in range(total):
        n_end = CONTEXT + k * H
        block = min(int(np.searchsorted(starts, n_end - 1, side="right")) - 1, len(protocol) - 1)
        f = f_of_block[block]
        rows.append(
            tick(k, block=block, label=protocol[block][0], full=bool(n_end - N >= starts[block]),
                 valid=valid(k), f=f(k) if callable(f) else f)
        )  # fmt: skip
    return rows


def write_session_folder(directory, raw_config, ticks, events, meta=None, samples=None):
    """Write a session folder; raw.csv gets ``samples`` rows of zeros with timestamps."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_yaml(raw_config, directory / c.FILE_CONFIG)
    meta = {"fs": FS, "board_id": MUSE_2, "timestamp_row": BOARD["timestamp_channel"],
            "marker_row": BOARD["marker_channel"], "mode": "live", "playback_path": None,
            **(meta or {})}  # fmt: skip
    (directory / c.FILE_META).write_text(json.dumps(meta))
    write_lines(directory / c.FILE_TICKS, ticks)
    write_lines(directory / c.FILE_EVENTS, events)
    np.zeros((len(ticks), ELECTRODES, N), dtype="<f4").tofile(directory / c.FILE_FILTERED)
    if samples is None:
        samples = CONTEXT + (len(ticks) - 1) * H if ticks else 0
    raw = np.zeros((samples, BOARD["num_rows"]))
    raw[:, BOARD["timestamp_channel"]] = RAW_START_S + np.arange(samples) / FS
    np.savetxt(directory / c.FILE_RAW, raw, delimiter="\t", fmt="%.6f")
    return directory


def finished_session_folder(directory, raw_config, ticks, events=(), reason="user",
                            duration_s=100.0, **options):  # fmt: skip
    """The folder of a session that started at 100 s and ended after ``duration_s``, with
    ``events`` in between."""
    events = [
        event("SESSION_START", RAW_START_S),
        *events,
        event("SESSION_END", RAW_START_S + duration_s, reason=reason),
    ]
    return write_session_folder(directory, raw_config, ticks, events, **options)


def write_lines(path, rows):
    """A jsonl file: one line per row."""
    with Path(path).open("w", encoding="utf-8") as file:
        for row in rows:
            print(json.dumps(row), file=file)
