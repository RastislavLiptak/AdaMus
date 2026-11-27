"""Evaluation metrics of a session, computed after it ended.

Every metric is computed from the files of the session folder alone. A metric
that cannot be computed (no clean step, no audio, ...) is null.
"""

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile
from scipy.stats import mannwhitneyu, spearmanr

from adamus import constants as c
from adamus.errors import AdamusError
from adamus.evaluation.session_reader import NOT_COMPLETE, SessionFolder, load_session_folder
from adamus.session_format import EVENT_MUSIC_END

Result = dict[str, Any]  # a part of metrics.json
ONSET_HOP = 512  # audio samples between two values of the onset envelope
ONSET_FFT = 2048  # audio samples of one spectrum of the onset envelope (librosa's default)
AUDIO_BLOCK = 1 << 20  # audio frames read, or spectra computed, at a time: a session has
# no upper length, and the whole file at once would need gigabytes
MUSIC_MAX_LAG_S = 10.0  # the music is looked for up to this long after the control
MIN_CORRELATION_POINTS = 3
# A lag is tried only while this share of the steps still has its audio: over
# the few steps a long lag leaves of a short session, a correlation is high by chance.
MIN_SHARE_OF_STEPS = 0.8
PERCENTILE_KEY = f"p{c.PERCENTILE}"  # how the high percentile is called in metrics.json


def summary_of(values: list[float]) -> dict[str, float] | None:
    """Median, high percentile and maximum; None for no data."""
    if not values:
        return None
    return {
        "median": float(np.median(values)),
        PERCENTILE_KEY: float(np.percentile(values, c.PERCENTILE)),
        "max": float(np.max(values)),
    }


def sampling_rate(timestamps: np.ndarray) -> float | None:
    """Sampling rate actually received: (samples - 1) / (last timestamp - first timestamp)."""
    if len(timestamps) < 2 or timestamps[-1] <= timestamps[0]:
        return None
    return float((len(timestamps) - 1) / (timestamps[-1] - timestamps[0]))


def longest_gap(timestamps: np.ndarray) -> float | None:
    """The largest difference between neighbouring timestamps."""
    return float(np.diff(timestamps).max()) if len(timestamps) >= 2 else None


def valid_share(session: SessionFolder) -> list[Result]:
    """Valid full steps / full steps, for each block."""
    blocks = []
    for span in session.spans:
        full = [t for t in session.ticks if t["block_index"] == span.index and t["full_window"]]
        valid = sum(tick["valid"] for tick in full)
        ratio = valid / len(full) if full else None
        blocks.append({"index": span.index, "label": span.label,
                       "full": len(full), "valid": valid, "ratio": ratio})  # fmt: skip
    return blocks


def state_separation(session: SessionFolder) -> tuple[float | None, float | None]:
    """How well f separates FOCUS from REST, over valid full steps of the evaluation blocks.

    Returns the AUC, U / (n1 * n2) of the Mann-Whitney test of f (0.5 = chance,
    1 = complete separation), and the mean f in FOCUS minus the mean f in REST.
    """
    f: dict[str, list[float]] = {c.FOCUS: [], c.REST: []}
    for span in session.evaluation_spans:
        f[span.label] += session.block_f(span)
    focus, rest = f[c.FOCUS], f[c.REST]
    if not focus or not rest:
        return None, None
    test: Any = mannwhitneyu(focus, rest)  # the types scipy gives its results are incomplete
    u = test.statistic
    return float(u / (len(focus) * len(rest))), float(np.mean(focus) - np.mean(rest))


def response_times(session: SessionFolder) -> dict[str, Any]:
    """Seconds from the start of an evaluation block to the first valid step
    with f at least FOCUS_THRESHOLD (FOCUS block: rise) or below it (REST block: fall)."""
    times: dict[str, list[float | None]] = {c.FOCUS: [], c.REST: []}
    for span in session.evaluation_spans:
        reached = None
        for tick in session.controlled_ticks():
            assert tick["f"] is not None, "every controlled step has an f"
            focus_on = tick["f"] >= c.FOCUS_THRESHOLD
            if tick["block_index"] == span.index and focus_on == (span.label == c.FOCUS):
                reached = (tick["n_end"] - span.start) / session.fs
                break
        times[span.label].append(reached)
    medians = {}
    for label in (c.FOCUS, c.REST):
        known = [seconds for seconds in times[label] if seconds is not None]
        medians[label] = float(np.median(known)) if known else None
    return {
        "rise_s": times[c.FOCUS],
        "fall_s": times[c.REST],
        "rise_median_s": medians[c.FOCUS],
        "fall_median_s": medians[c.REST],
    }


def step_regularity(session: SessionFolder) -> dict[str, float] | None:
    """Differences of t_sent between neighbouring sent steps."""
    sent = [tick["t_sent"] for tick in session.ticks if tick["t_sent"] is not None]
    return summary_of(np.diff(sent).tolist())


def dropouts(session: SessionFolder) -> dict[str, Any] | None:
    """How often the output played a block of the music before it was ready; None when the
    music never began, or the session did not end it."""
    ended = session.event(EVENT_MUSIC_END)
    return None if ended is None else {"count": ended.get("dropouts")}


def best_lag(
    f: np.ndarray, at: np.ndarray, envelope: np.ndarray, frame_s: float, hop_s: float
) -> Result | None:
    """The lag at which the envelope follows f best, and their Spearman correlation there.

    ``at`` are the times of the steps, in seconds from the first value of the
    envelope, whose values are ``frame_s`` apart. The envelope is averaged over
    the step that follows each time, shifted by a lag of 0, 1, 2, ... steps up
    to MUSIC_MAX_LAG_S. Of two lags that are equally good, the shorter counts.
    """
    total = np.concatenate(([0.0], np.cumsum(envelope)))  # total[i] = sum of envelope[:i]
    enough = max(MIN_CORRELATION_POINTS, MIN_SHARE_OF_STEPS * len(f))
    best = None
    for lag_steps in range(round(MUSIC_MAX_LAG_S / hop_s) + 1):
        lag = lag_steps * hop_s
        first = np.ceil((at + lag) / frame_s).astype(int)
        last = np.ceil((at + lag + hop_s) / frame_s).astype(int)
        inside = (last <= len(envelope)) & (last > first)
        if inside.sum() < enough:
            continue
        strength = (total[last[inside]] - total[first[inside]]) / (last - first)[inside]
        if np.ptp(f[inside]) == 0 or np.ptp(strength) == 0:
            continue
        correlation: Any = spearmanr(f[inside], strength)  # as for the test above
        rho = float(correlation.statistic)
        if best is None or rho > best["rho"]:
            best = {"rho": rho, "lag_s": lag}
    return best


def music_response(session: SessionFolder, mono: np.ndarray, rate: int) -> Result | None:
    """How the music follows the control: the onset strength of the audio against f.

    A step counts from the moment the music was told its state; a step that kept the
    f of the one before told nothing and counts from the end of its processing.
    """
    began = session.music_began
    ticks = [tick for tick in session.ticks if tick["f"] is not None]
    if began is None or len(mono) == 0:
        return None
    f = np.array([tick["f"] for tick in ticks])
    at = np.array([tick["t_sent"] or tick["t_proc_end"] for tick in ticks]) - began
    envelope = librosa.onset.onset_strength(
        S=librosa.power_to_db(mel_spectrogram(mono, rate)), sr=rate, hop_length=ONSET_HOP
    )
    return best_lag(f, at, envelope, ONSET_HOP / rate, c.HOP_S)


def mel_spectrogram(mono: np.ndarray, rate: int) -> np.ndarray:
    """librosa's mel spectrogram that the onset envelope is computed from, a block of spectra at a
    time: the complex spectra of a long session at once would take gigabytes.

    Spectrum i covers samples [i * hop - n_fft / 2, i * hop + n_fft / 2), zeros outside
    the audio, as librosa centres them.
    """
    half = ONSET_FFT // 2
    count = 1 + len(mono) // ONSET_HOP
    blocks = []
    for first in range(0, count, AUDIO_BLOCK // ONSET_HOP):
        last = min(first + AUDIO_BLOCK // ONSET_HOP, count)  # spectra [first, last)
        start, end = first * ONSET_HOP - half, (last - 1) * ONSET_HOP + half
        inside = mono[max(start, 0) : min(end, len(mono))]
        segment = np.pad(inside, (max(-start, 0), max(end - len(mono), 0)))
        blocks.append(
            librosa.feature.melspectrogram(
                y=segment, sr=rate, n_fft=ONSET_FFT, hop_length=ONSET_HOP, center=False
            )
        )
    return np.concatenate(blocks, axis=-1)


def mono_of(pcm: np.ndarray) -> np.ndarray:
    """Whole-number samples [frames x channels] as one channel between -1 and 1."""
    full_scale = np.float32(np.iinfo(pcm.dtype).max + 1)
    mono: np.ndarray = pcm.mean(axis=1, dtype=np.float32) / full_scale
    return mono


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """The audio as one channel, and its rate: read a block at a time, never as two channels
    at once."""
    with soundfile.SoundFile(path) as audio:
        mono = np.empty(audio.frames, dtype=np.float32)
        for start in range(0, audio.frames, AUDIO_BLOCK):
            pcm = audio.read(AUDIO_BLOCK, dtype="int16", always_2d=True)
            mono[start : start + len(pcm)] = mono_of(pcm)
        return mono, audio.samplerate


def audio_metrics(session: SessionFolder) -> dict[str, Any]:
    """The metrics of the music: how often it dropped out, and how it followed the control;
    the second null when there is no audio.wav that can be read."""
    try:
        mono, rate = read_audio(session.directory / c.FILE_AUDIO)
    except soundfile.SoundFileError:  # also when the file is not there
        return {"audio_dropouts": dropouts(session), "music_response": None}
    return {
        "audio_dropouts": dropouts(session),
        "music_response": music_response(session, mono, rate),
    }


def completeness(session: SessionFolder, samples: int) -> Result:
    """Lines of ticks = records of filtered = the steps of the samples (Steps.count).

    A step needs the whole context of its filters, so the first one ends there.
    """
    value_bytes = np.dtype(c.FILTERED_DTYPE).itemsize
    record_bytes = value_bytes * len(c.DETECTOR_ELECTRODES) * session.steps.window
    records, rest = divmod((session.directory / c.FILE_FILTERED).stat().st_size, record_bytes)
    expected = session.steps.count(samples)
    passed = rest == 0 and len(session.ticks) == records == expected
    return {
        "ticks": len(session.ticks),
        "filtered_records": records,
        "expected": expected,
        "passed": passed,
    }


def received_timestamps(session: SessionFolder) -> np.ndarray:
    """The timestamp of every sample received; none when no sample was received."""
    raw = session.raw_samples()
    return raw[session.meta["timestamp_row"]] if raw.size else np.empty(0)


def compute_metrics(directory: Path) -> Result:
    session = load_session_folder(directory)
    timestamps = received_timestamps(session)
    try:
        completeness_result = completeness(session, len(timestamps))
    except OSError as problem:  # e.g. no filtered.f32
        raise AdamusError(f"{directory} {NOT_COMPLETE}: {problem}") from problem
    auc, f_difference = state_separation(session)
    response = response_times(session)
    audio = audio_metrics(session)
    lag = audio["music_response"]["lag_s"] if audio["music_response"] else None
    rise, fall = response["rise_median_s"], response["fall_median_s"]
    return {
        "session": session.directory.name,
        "sampling_rate_hz": sampling_rate(timestamps),
        "longest_sample_gap_s": longest_gap(timestamps),
        "valid_steps_per_block": valid_share(session),
        "processing_time_s": summary_of(
            [t["t_proc_end"] - t["t_proc_start"] for t in session.ticks]
        ),
        "state_separation_auc": auc,
        "mean_f_difference": f_difference,
        "response_time_s": response,
        "control_latency_s": summary_of(session.control_latencies()),
        "step_interval_s": step_regularity(session),
        **audio,
        # From a change of state to an audible change: response time plus the music lag.
        "total_response_s": None if lag is None else {
            "rise_s": None if rise is None else rise + lag,
            "fall_s": None if fall is None else fall + lag,
        },
        "record_completeness": completeness_result,
    }  # fmt: skip


def analyze(directory: Path) -> Path:
    """Compute all metrics of a session and write them to metrics.json: whole, at once, as
    the dashboard of the session may read it meanwhile."""
    path = Path(directory) / c.FILE_METRICS
    metrics = compute_metrics(directory)
    written = path.with_name(f".{path.name}.writing")
    with written.open("w", encoding="utf-8") as handle:
        print(json.dumps(metrics, indent=2), file=handle)
    written.replace(path)
    return path
