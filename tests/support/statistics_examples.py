"""The statistics of a finished session, as the tests of their lines and their print use them."""

from pathlib import Path

from adamus.console.session_statistics import FocusStatistics, Statistics

FOCUS = FocusStatistics(
    switched_on=7, on_share=0.38, focus_blocks_on=2, focus_blocks=3, rest_blocks_off=3,
    rest_blocks=3, f_min=0.0, f_max=0.87, f_mean=0.34,
    latency_median_s=0.031, latency_high_s=0.058,
)  # fmt: skip


# Session S01 of 09:42, ended by [SPACE], with a clean signal 91 % of the time.
FIELDS = {
    "name": "S01", "directory": Path("sessions/20261110-150412_S01"), "duration_s": 582.0,
    "reason": "user", "verdict": "OK", "detail": None, "usable_share": 0.91, "focus": FOCUS,
}  # fmt: skip


def example(**changes) -> Statistics:
    """The statistics of FIELDS, with ``changes``."""
    return Statistics(**{**FIELDS, **changes})
