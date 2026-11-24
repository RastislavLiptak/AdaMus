"""What the headset delivered in a session, stored in the data folder as a recording of the
dataset c.OWN_RECORDINGS, so that it can be played again like any other recording."""

from pathlib import Path

from adamus import constants as c
from adamus.config import Block
from adamus.datasets import write_recording
from adamus.evaluation.session_reader import load_session_folder
from adamus.protocol import block_spans


def recorded_blocks(protocol: list[Block], fs: int, samples: int) -> list[Block]:
    """The blocks of ``protocol`` that the first ``samples`` samples reach, the last one cut
    where they end. Samples after a finite protocol stay with its last block, as in a session."""
    blocks = []
    for span, block in zip(block_spans(protocol, fs), protocol, strict=True):
        if span.start >= samples:
            break
        end = samples if span.end is None else min(span.end, samples)
        whole = span.end is not None and end == span.end
        blocks.append(block if whole else Block(block.label, (end - span.start) / fs))
    return blocks


def store_recording(session_folder: Path, datasets_root: Path) -> Path | None:
    """What the headset delivered in a session, as a recording of OWN_RECORDINGS named like
    the session folder; None when it delivered nothing."""
    session = load_session_folder(session_folder)
    raw = session.raw_samples()
    if raw.size == 0:
        return None
    rows = session.meta["electrode_rows"]
    samples = raw.shape[1]
    blocks = recorded_blocks(session.protocol, session.fs, samples)
    folder = Path(datasets_root) / c.OWN_RECORDINGS / session.directory.name
    return write_recording(folder, list(rows), raw[list(rows.values())], blocks)
