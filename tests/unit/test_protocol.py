"""protocol: where blocks start and end, and which block a step belongs to."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from adamus.config import Block
from adamus.protocol import BlockSpan, block_spans, locate_step

FS, WINDOW, H = 256, 1024, 128  # a longer window than the detector's: any window
BLOCK = 60 * FS
DEFAULT = [Block("REST", 60), Block("FOCUS", 60)] * 4 + [Block("FREE", 0)]
SPANS = block_spans(DEFAULT, FS)


class TestBlockSpans:
    def test_block_starts_at_fs_times_the_sum_of_earlier_durations(self):
        assert [span.start for span in SPANS] == [i * BLOCK for i in range(9)]
        assert [span.end for span in SPANS[:8]] == [(i + 1) * BLOCK for i in range(8)]

    def test_indices_and_labels(self):
        assert [span.index for span in SPANS] == list(range(9))
        assert [span.label for span in SPANS] == ["REST", "FOCUS"] * 4 + ["FREE"]

    def test_block_of_duration_zero_has_no_end(self):
        assert SPANS[8] == BlockSpan(8, "FREE", 8 * BLOCK, None)

    def test_fractional_durations_round_to_whole_samples(self):
        spans = block_spans([Block("REST", 1.5), Block("FOCUS", 0.25)], FS)
        assert [(span.start, span.end) for span in spans] == [(0, 384), (384, 448)]

    def test_rounding_does_not_accumulate(self):
        """Block 3 starts at fs * 0.9 s = 230.4, sample 230; not at 3 * round(76.8) = 231."""
        spans = block_spans([Block("REST", 0.3), Block("FOCUS", 0.3)] * 2, FS)
        assert [span.start for span in spans] == [0, 77, 154, 230]
        assert spans[-1].end == 307

    @given(
        st.lists(st.floats(0.1, 100), min_size=1, max_size=12), st.sampled_from([128, 256, 1000])
    )
    def test_blocks_tile_the_stream_without_gap_or_overlap(self, durations, fs):
        spans = block_spans([Block("REST", seconds) for seconds in durations], fs)
        assert spans[0].start == 0
        assert all(a.end == b.start for a, b in zip(spans, spans[1:], strict=False))
        assert all(span.end >= span.start for span in spans)
        assert abs(spans[-1].end - fs * sum(durations)) <= 0.5 + 1e-6


class TestLocateStep:
    def test_step_belongs_to_the_block_of_its_last_sample(self):
        assert locate_step(SPANS, BLOCK, WINDOW)[0].index == 0  # last sample BLOCK - 1
        assert locate_step(SPANS, BLOCK + 1, WINDOW)[0].index == 1  # last sample BLOCK

    def test_step_is_full_when_its_whole_window_lies_in_the_block(self):
        assert locate_step(SPANS, WINDOW, WINDOW) == (SPANS[0], True)
        assert locate_step(SPANS, BLOCK, WINDOW) == (SPANS[0], True)
        assert locate_step(SPANS, BLOCK + WINDOW - 1, WINDOW) == (SPANS[1], False)
        assert locate_step(SPANS, BLOCK + WINDOW, WINDOW) == (SPANS[1], True)

    @pytest.mark.parametrize(
        ("protocol", "expected"),
        [(DEFAULT, 113), ([Block("REST", 59), Block("FOCUS", 59)], 111)],
    )
    def test_number_of_full_steps_per_block(self, protocol, expected):
        """113 for blocks of 60 s, 111 for blocks of 59 s."""
        spans = block_spans(protocol, FS)
        timed = [span for span in spans if span.end is not None]
        full = dict.fromkeys(range(len(timed)), 0)
        for k in range((timed[-1].end - WINDOW) // H + 1):
            span, is_full = locate_step(spans, WINDOW + k * H, WINDOW)
            full[span.index] += is_full
        assert set(full.values()) == {expected}

    def test_open_ended_block_holds_everything_after_its_start(self):
        start = 8 * BLOCK
        assert locate_step(SPANS, start + H, WINDOW) == (SPANS[8], False)
        assert locate_step(SPANS, start + WINDOW, WINDOW) == (SPANS[8], True)
        assert locate_step(SPANS, start + 10_000_000, WINDOW) == (SPANS[8], True)

    def test_steps_after_a_finite_protocol_stay_in_the_last_block_but_are_not_full(self):
        spans = block_spans([Block("REST", 10), Block("FOCUS", 10), Block("REST", 10)], FS)
        end = 30 * FS
        assert locate_step(spans, end, WINDOW) == (spans[2], True)
        assert locate_step(spans, end + 1, WINDOW) == (spans[2], False)
        assert locate_step(spans, end + 100 * H, WINDOW) == (spans[2], False)

    @given(st.integers(WINDOW, 9 * BLOCK))
    def test_full_step_lies_inside_its_block(self, n_end):
        span, full = locate_step(SPANS, n_end, WINDOW)
        assert span.start <= n_end - 1
        assert span.end is None or n_end - 1 < span.end
        inside = n_end - WINDOW >= span.start and (span.end is None or n_end <= span.end)
        assert full == inside

    def test_step_shorter_than_a_window_is_refused(self):
        with pytest.raises(AssertionError):
            locate_step(SPANS, WINDOW - 1, WINDOW)
