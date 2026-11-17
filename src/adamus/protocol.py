"""The protocol of a session: its REST, FOCUS and FREE blocks and where each begins and ends.

Boundaries are counted in samples, so a recording that is played back is cut
into the same blocks.
"""

from dataclasses import dataclass

from adamus.config import Block


@dataclass(frozen=True)
class BlockSpan:
    index: int
    label: str
    start: int  # first sample
    end: int | None  # one past the last sample; None = lasts until the session is stopped


def block_spans(protocol: list[Block], fs: int) -> list[BlockSpan]:
    """Block boundaries in samples: a block starts at fs * (sum of earlier durations)."""
    spans: list[BlockSpan] = []
    start, elapsed_s = 0, 0.0
    for index, block in enumerate(protocol):
        elapsed_s += block.duration_s
        end = round(fs * elapsed_s) if block.duration_s else None
        spans.append(BlockSpan(index, block.label, start, end))
        start = end if end is not None else start
    return spans


def locate_step(spans: list[BlockSpan], n_end: int, window: int) -> tuple[BlockSpan, bool]:
    """The block of the step ending at sample ``n_end`` and whether its window is "full".

    A step belongs to the block that holds its last sample. It is full when its
    whole window lies in that block. Steps after a finite protocol stay with the
    last block and are never full.
    """
    assert n_end >= window > 0
    holds = (span for span in spans if span.end is None or n_end - 1 < span.end)
    block = next(holds, spans[-1])
    full = n_end - window >= block.start and (block.end is None or n_end <= block.end)
    return block, full
