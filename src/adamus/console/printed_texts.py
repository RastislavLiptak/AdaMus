"""What the console prints, and how it is laid out and coloured.

Every call of ``output.print`` is one line on the screen, so the code reads
like the screen looks. The two lists that change in place (checklist and
sources) are returned as one renderable each.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from rich.console import Console, Group, RenderableType
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from adamus import constants as c
from adamus.console.data_sources import Source
from adamus.console.session_statistics import OK, Statistics, statistics_lines, verdict_text


class CheckRow(Protocol):
    """A row of the checklist: a check of the preflight."""

    @property
    def name(self) -> str: ...

    @property
    def status(self) -> str: ...

    @property
    def detail(self) -> str: ...


MARK_AND_COLOUR = {
    c.CHECK_PENDING: ("", "bright_black"),
    c.CHECK_PASSED: ("✓", "green"),
    c.CHECK_FAILED: ("✗", "red"),
    c.CHECK_SKIPPED: ("✗", "red"),
    c.CHECK_NONE_FOUND: ("–", "yellow"),
}
LIST_ROWS = 10  # a list of sources or recordings scrolls when it is longer
INDENT = 2  # columns in front of checklist and statistics rows
MARK_WIDTH = 2  # the mark of a row and the space after it
CHECK_WIDTH = 23  # name of a preflight check
SOURCE_WIDTH = 28  # name of a data source or of a recording
LABEL_WIDTH = 16  # label of a statistics row


def _table(*widths: int) -> Table:
    """Columns of the given widths, then one for the rest of the line.

    A width includes the space that separates a column from the next one. A
    text longer than its column pushes the rest of its row to the right.
    """
    table = Table.grid(padding=(0, 1))
    for width in widths:
        table.add_column(min_width=width - 1, no_wrap=True)
    table.add_column()
    return table


def welcome(output: Console, config_path: Path) -> None:
    output.print(f"AdaMus · config: {config_path}")
    output.print()


def checklist(checks: Sequence[CheckRow]) -> RenderableType:
    """The preflight checklist: a mark, the name of the check and what it found."""
    table = _table(INDENT, MARK_WIDTH, CHECK_WIDTH)
    for check in checks:
        if check.status == c.CHECK_RUNNING:
            table.add_row("", Spinner("dots"), Text(check.name), Text(check.detail))
            continue
        mark, colour = MARK_AND_COLOUR[check.status]
        detail_colour = "" if check.status == c.CHECK_PASSED else colour
        table.add_row(
            "",
            Text(mark, style=colour),
            Text(check.name, style=colour),
            Text(check.detail, style=detail_colour),
        )
    return Group(Text("Preflight"), table)


def all_ready(output: Console) -> None:
    output.print()
    output.print(Text("All systems ready.", style="green"))
    output.print()


def errors(output: Console, failed_checks: Sequence[CheckRow], how_to_fix: dict[str, str]) -> None:
    """One paragraph per failed check: its name, what was found, what to do."""
    output.print()
    output.print(Text("Errors", style="bold red"))
    for check in failed_checks:
        output.print()
        output.print(Text(check.name, style="red"))
        for line in check.detail.splitlines():
            output.print(Text(f"  {line}"))
        if check.name in how_to_fix:
            output.print(Text(f"  {how_to_fix[check.name]}"))


def _scrolling_list(heading: str, rows: list[tuple[str, str]], selected: int) -> RenderableType:
    """At most LIST_ROWS rows of (name, kind); the list scrolls with the selected row."""
    hidden_above = max(0, selected - LIST_ROWS + 1)
    table = _table(MARK_WIDTH, SOURCE_WIDTH)
    for index in range(hidden_above, min(hidden_above + LIST_ROWS, len(rows))):
        chosen = index == selected
        name, kind = rows[index]
        table.add_row(
            Text(">" if chosen else " "), Text(name, style="reverse" if chosen else ""), Text(kind)
        )
    if not rows:
        table.add_row(Text(" "), Text("nothing available", style="yellow"), Text(""))
    return Group(Text(heading), table)


def source_list(sources: list[Source], selected: int) -> RenderableType:
    heading = "Select data source      [↑] [↓] move   [ENTER] select   [R] rescan headset"
    return _scrolling_list(heading, [(source.label, source.kind) for source in sources], selected)


def recording_list(recordings: list[Path], selected: int) -> RenderableType:
    """The recordings of the dataset that was just chosen."""
    heading = "Select recording        [↑] [↓] move   [ENTER] select   [Q] back"
    return _scrolling_list(heading, [(folder.name, "") for folder in recordings], selected)


def new_session_question(output: Console) -> None:
    output.print()
    output.print("Start a new session?    [Y] yes   [N] quit program")


def session_name_question(output: Console) -> None:
    """The typed name follows on the same line."""
    output.print()
    output.print("Session name (type it, then [ENTER]): ", end="")


def invalid_name(output: Console) -> None:
    output.print(Text("Use letters, digits, '_' and '-' only.", style="yellow"))


def ready(
    output: Console, name: str, source: Source, recording: Path | None, missing: list[str]
) -> None:
    """What is about to run, and which electrodes of the detector it lacks."""
    output.print()
    output.print(
        f"Session {name} · source: {_what(source, recording)} ({source.kind})", markup=False
    )
    if missing:
        output.print(
            Text(f"No {', '.join(missing)}: recorded as zeros, the detector uses the others.",
                 style="yellow")
        )  # fmt: skip
    output.print("READY       [SPACE] start   [Q] cancel")


def lacks_electrodes(output: Console, source: Source, recording: Path | None) -> None:
    """Why a source cannot run a session: it has no electrode of the detector."""
    needed = ", ".join(c.DETECTOR_ELECTRODES)
    output.print()
    output.print(
        Text(f"{_what(source, recording)} has none of {needed}. The detector needs at least one.",
             style="red")
    )  # fmt: skip
    output.print()


def _what(source: Source, recording: Path | None) -> str:
    return source.label if recording is None else f"{source.label} / {recording.name}"


def running(output: Console) -> None:
    """The only line printed until the session ends."""
    output.print("RUNNING     [SPACE] end session")


def cancelled(output: Console) -> None:
    output.print("Session cancelled.")
    output.print()


def could_not_start(output: Console, problem: Exception) -> None:
    output.print()
    output.print(Text(f"The session could not start: {problem}", style="red"))
    output.print()


NO_METRICS = "The metrics of the session could not be computed"
NO_STATISTICS = "The statistics of the session could not be shown"
NOT_STORED = "The recording of the session could not be stored in the data folder"


def not_read(output: Console, what: str, problem: Exception) -> None:
    output.print()
    output.print(Text(f"{what}: {problem}", style="red"))
    output.print()


def statistics(output: Console, result: Statistics) -> None:
    table = _table(INDENT, LABEL_WIDTH)
    verdict = verdict_text(result)
    for label, text in statistics_lines(result):
        value = Text(text)
        if text.startswith(verdict):  # the verdict is green or red, the rest of the line plain
            value.stylize("green" if result.verdict == OK else "red", 0, len(verdict))
        table.add_row("", label, value)
    output.print()
    output.print(f"Session {result.name} finished")
    output.print(table)
    output.print()
