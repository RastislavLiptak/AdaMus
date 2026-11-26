"""The short sessions that the tests of realtime.session run with stand-in devices."""

from tests.support.finished_sessions import (
    CONTEXT,
    FS,
    FinishedSession,
    H,
    config_with,
    synthetic_matrix,
)

SHORT = [["REST", 10], ["FOCUS", 10], ["REST", 10], ["FOCUS", 10]]  # 40 s, 73 steps
STEPS = (40 * FS - CONTEXT) // H + 1
CONTROL = [["REST", 30], ["FOCUS", 30], ["REST", 30], ["FOCUS", 30]]
CONTROL_STEPS = (120 * FS - CONTEXT) // H + 1  # 233

# The two forms of the music, as the shipped configuration gives them.
STYLE = config_with().styles[0]  # the one a session of the configuration plays
IDLE_STYLE, FOCUS_STYLE = STYLE.rest, STYLE.focus


def short_session(tmp_path, matrix=None, **options):
    """A session over the protocol SHORT, by default over its synthetic signal."""
    matrix = synthetic_matrix(SHORT) if matrix is None else matrix
    return FinishedSession(tmp_path, config_with(SHORT), matrix, **options)
