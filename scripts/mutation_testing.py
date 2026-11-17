"""Mutation testing: does some test fail when the program is changed a little?

    python scripts/mutation_testing.py                        # the whole program
    python scripts/mutation_testing.py src/adamus/config.py   # only the named files

A mutant is one small change of one source file: a comparison turned around,
a number increased, a call left out. The tests that run the changed line are
run against the mutant. When they all pass, the mutant survived: no test
checks what that line does.

How a run stays correct and quiet:
- The repository is never changed. Every worker tests a private copy of the
  source folder, put in front of the installed one on the import path.
- Every mutant is tested by a fresh Python process. A process that crashes
  ends with an exit code instead of a crash report of the operating system.
- A process that takes too long, or grows past MEMORY_LIMIT, is stopped: a
  mutant may turn a loop into an endless one, or a small array into a huge
  one, and the computer must stay usable meanwhile. Such a mutant is reported
  as TIMEOUT or TOO_LARGE, not as caught. Like the program, the tool runs on
  macOS: it reads the memory of a process from the system's libproc.
- A mutant inside a function is tested first by the tests that run its line,
  the quickest first, so that one that is caught is caught soon. One that
  survives them is tested by every test of their files: a test that does not
  run the line may still check what it did. A mutant in code that runs when a
  module is imported may change anything, so the whole test suite is run
  against it, file after file: tests in another order would set up the
  fixtures of a file again and again.
"""

import argparse
import ast
import ctypes
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_TESTS = "--run-tests"  # internal: this process runs pytest against one mutant
# The slow tests are left out: one in real time per mutant would take too long, and the
# training tests read the whole dataset in every worker at once, more than the memory holds.
QUICK_TESTS = "not realtime and not training"
PYTEST_OPTIONS = ["-q", "-x", "-m", QUICK_TESTS, "-p", "no:cacheprovider"]
PYTEST_OPTIONS += ["-p", "no:faulthandler"]  # crashes are dealt with by run_tests_quietly
CRASH_SIGNALS = ("SIGSEGV", "SIGBUS", "SIGABRT", "SIGILL", "SIGFPE", "SIGTRAP")
NO_TESTS_COLLECTED = 5  # exit code of pytest
TEMPORARY = "temporary"  # the folder of the temporary files of the tests, in the work folder
TIMES_FILE = "MUTATION_TEST_TIMES"  # the variable naming where a run writes how long each test took
KILLED, SURVIVED, TIMEOUT, NOT_RUN = "killed", "survived", "timeout", "no test runs this line"
TOO_LARGE = "too large"
MEMORY_LIMIT = 2 * 1024**3  # bytes of memory a test process of one mutant may take
TIME_MARGIN_S = 60  # a mutant may take twice as long as the tests, and this much more
NOTES = {
    SURVIVED: "",
    NOT_RUN: "  (not tested)",
    TIMEOUT: "  (stopped: took too long)",
    TOO_LARGE: "  (stopped: too much memory)",
}  # the mutants listed: those no test caught, and those that were stopped
WATCH_S = 0.1  # how often a test process is looked at
SHOWN_OUTPUT = 3000  # characters of the end of what failing tests printed
UUID_BYTES = 16  # of struct rusage_info_v0
RUSAGE_INFO_V0 = 0  # the flavour of proc_pid_rusage below

Edit = Callable[[ast.AST], None]  # changes a node in place
Change = tuple[str, Edit]  # what changes, and the edit that does it
Counterparts = dict[type[ast.AST], type[ast.AST]]  # an operator and what it is replaced by

COMPARISONS: Counterparts = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In,
}  # fmt: skip
ARITHMETIC: Counterparts = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult,
    ast.FloorDiv: ast.Div, ast.Mod: ast.Mult,
}  # fmt: skip
LOGIC: Counterparts = {ast.And: ast.Or, ast.Or: ast.And}
SYMBOLS: dict[type[ast.AST], str] = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
    ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in", ast.Add: "+",
    ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%",
    ast.And: "and", ast.Or: "or",
}  # fmt: skip


@dataclass(frozen=True)
class Project:
    """What is mutated and what tests it."""

    root: Path  # tests are run from here
    source: Path  # the folder on the import path that holds the package
    package: str  # name of the package inside ``source``
    tests: tuple[str, ...]  # test paths, relative to ``root``


@dataclass(frozen=True)
class Mutant:
    file: Path  # relative to the source folder
    lines: range  # the lines of the statement that was changed
    change: str  # e.g. "< became <="
    text: str  # the whole file with the change made
    runs_at_import: bool  # not inside a function: it runs when the module is imported


ADAMUS = Project(
    root=Path(__file__).resolve().parents[1],
    source=Path(__file__).resolve().parents[1] / "src",
    package="adamus",
    tests=("tests/unit", "tests/integration", "tests/reference_values", "tests/project"),
)


# --- Making mutants -----------------------------------------------------------


def _is_logging(node: ast.AST) -> bool:
    function = getattr(node, "func", None)
    return (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "log"
    )


def _places(tree: ast.Module) -> dict[int, tuple[ast.stmt, bool]]:
    """For every node worth mutating: its statement, and whether it lies inside a function.

    Left out is what nobody is meant to test: docstrings, type annotations,
    decorators, assertions, log messages and the names of text encodings.
    """
    places: dict[int, tuple[ast.stmt, bool]] = {}
    pending: list[tuple[ast.AST, ast.stmt | None, bool]] = [(tree, None, False)]
    while True:  # until the whole tree has been visited
        if not pending:
            return places
        node, statement, in_function = pending.pop()
        if isinstance(node, ast.stmt):
            statement = node
        if statement is not None:
            places[id(node)] = (statement, in_function)
        is_docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        logs = _is_logging(node) or (isinstance(node, ast.Expr) and _is_logging(node.value))
        if isinstance(node, ast.Assert) or is_docstring or logs:
            del places[id(node)]
            continue
        for field, value in ast.iter_fields(node):
            if field in ("annotation", "returns", "decorator_list"):
                continue
            inside = in_function or (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda)
                and field == "body"
            )
            for child in value if isinstance(value, list) else [value]:
                is_encoding = isinstance(node, ast.keyword) and node.arg == "encoding"
                if isinstance(child, ast.AST) and not is_encoding:
                    pending.append((child, statement, inside))


def _set(field: str, make: Callable[[Any], Any]) -> Edit:
    """An edit that replaces one field of a node by something made from its old value."""
    return lambda node: setattr(node, field, make(getattr(node, field)))


def _swapped(node: ast.AST, field: str, index: int | None, table: Counterparts) -> list[Change]:
    """Replace the operator in ``field`` (or in its item ``index``) by its counterpart."""
    old = getattr(node, field) if index is None else getattr(node, field)[index]
    if type(old) not in table:
        return []
    new = table[type(old)]

    def edit(node: ast.AST) -> None:
        if index is None:
            setattr(node, field, new())
        else:
            getattr(node, field)[index] = new()

    return [(f"{SYMBOLS[type(old)]} became {SYMBOLS[new]}", edit)]


def _constant_changes(value: object) -> list[Change]:
    if isinstance(value, bool):
        return [(f"{value} became {not value}", _set("value", lambda old: not old))]
    if isinstance(value, int | float):
        return [(f"{value} became {value + 1}", _set("value", lambda old: old + 1))]
    if isinstance(value, str):
        return [(f"text {value!r} changed", _set("value", lambda old: old + "~"))]
    return []


def _changes(node: ast.AST) -> list[Change]:
    """The mutations of one node: what changes, and the edit that makes the change."""
    changes: list[Change] = []
    if isinstance(node, ast.Compare):
        for index in range(len(node.ops)):
            changes += _swapped(node, "ops", index, COMPARISONS)
    elif isinstance(node, ast.BinOp):
        changes = _swapped(node, "op", None, ARITHMETIC)
    elif isinstance(node, ast.BoolOp):
        changes = _swapped(node, "op", None, LOGIC)
    elif isinstance(node, ast.If | ast.While | ast.IfExp):
        changes = [("condition negated", _set("test", lambda test: ast.UnaryOp(ast.Not(), test)))]
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call | ast.Await):
        changes = [("call left out", _set("value", lambda _: ast.Constant(None)))]
    elif isinstance(node, ast.Return) and not _gives_none(node):
        changes = [("returns None", _set("value", lambda _: ast.Constant(None)))]
    elif isinstance(node, ast.Constant):
        changes = _constant_changes(node.value)
    return changes


def _gives_none(node: ast.Return) -> bool:
    """A bare return or "return None": returning None instead would change nothing."""
    return node.value is None or (isinstance(node.value, ast.Constant) and node.value.value is None)


def mutants_of(path: Path, source: Path) -> list[Mutant]:
    """Every mutant of one source file."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    places = _places(tree)
    mutants = []
    for position, node in enumerate(ast.walk(tree)):
        if id(node) not in places:
            continue
        statement, in_function = places[id(node)]
        for number, (name, _) in enumerate(_changes(node)):
            copy = ast.parse(text)  # a fresh tree: its nodes come in the same order
            target = list(ast.walk(copy))[position]
            _changes(target)[number][1](target)
            assert statement.end_lineno is not None
            mutants.append(
                Mutant(
                    file=path.relative_to(source),
                    lines=range(statement.lineno, statement.end_lineno + 1),
                    change=name,
                    text=ast.unparse(ast.fix_missing_locations(copy)),
                    runs_at_import=not in_function,
                )
            )
    return mutants


# --- Running tests ------------------------------------------------------------


class _TestTimes:
    """A pytest plugin: writes how long each test took, in seconds, to the file ``path``."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.seconds: dict[str, float] = {}

    def pytest_runtest_logreport(self, report: Any) -> None:
        self.seconds[report.nodeid] = self.seconds.get(report.nodeid, 0.0) + report.duration

    def pytest_sessionfinish(self) -> None:
        Path(self.path).write_text(json.dumps(self.seconds), encoding="utf-8")


def run_tests_quietly(pytest_arguments: list[str]) -> int:
    """Run pytest in this process; a crash ends the process with the number of its signal.

    Without this, every crash of a mutant would open a crash report window.
    """
    import pytest

    libc = ctypes.CDLL(None)
    libc.signal.argtypes = [ctypes.c_int, ctypes.c_void_p]
    libc.signal.restype = ctypes.c_void_p
    for name in CRASH_SIGNALS:
        libc.signal(getattr(signal, name), ctypes.cast(libc._exit, ctypes.c_void_p))
    times = os.environ.get(TIMES_FILE)
    return int(pytest.main(pytest_arguments, plugins=[_TestTimes(times)] if times else []))


def _environment(source: Path, work: Path, **more: str) -> dict[str, str]:
    """The import path starts with ``source``; no bytecode is cached for files that change.

    Temporary files go into ``work``, which goes with the run: a test process that is
    stopped, or a mutant that no longer deletes its files, cannot clean up after itself.
    """
    temporary = work / TEMPORARY
    temporary.mkdir(exist_ok=True)
    return {
        **os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(temporary), **more,
    }  # fmt: skip


def tests_of_each_line(
    project: Project, work: Path
) -> tuple[dict[Path, dict[int, set[str]]], dict[str, float], float]:
    """Run the tests once: which tests run which line, how long each test takes, and how
    long they all take."""
    import coverage

    data_file, times_file = work / "coverage", work / "times.json"
    command = [
        sys.executable, str(Path(__file__).resolve()), RUN_TESTS, "-q", "-m", QUICK_TESTS,
        "-p", "no:cacheprovider", f"--cov={project.package}", "--cov-context=test",
        "--cov-report=", "--cov-fail-under=0", *project.tests,
    ]  # fmt: skip
    started = time.monotonic()
    run = subprocess.run(
        command,
        cwd=project.root,
        env=_environment(
            project.source, work, COVERAGE_FILE=str(data_file), **{TIMES_FILE: str(times_file)}
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    seconds = time.monotonic() - started
    if run.returncode != 0:
        print(run.stdout[-SHOWN_OUTPUT:])
        print(run.stderr[-SHOWN_OUTPUT:])
        raise SystemExit("the tests must pass before mutation testing")
    data = coverage.CoverageData(str(data_file))
    data.read()
    tests: dict[Path, dict[int, set[str]]] = {}
    for measured in data.measured_files():
        if project.source.resolve() in Path(measured).resolve().parents:
            file = Path(measured).resolve().relative_to(project.source.resolve())
            tests[file] = {
                line: {context.rsplit("|", 1)[0] for context in contexts if context}
                for line, contexts in data.contexts_by_lineno(measured).items()
            }  # a context is a test and the phase it was in: "tests/x.py::test_y|run"
    times = json.loads(times_file.read_text(encoding="utf-8"))
    _check_found(project, sorted(times), work)
    return tests, times, seconds


def _check_found(project: Project, tests: list[str], work: Path) -> None:
    """Stop unless pytest finds every test by its name. Asked for by name, a test of a file
    that cannot be imported is not found either, so a mutant ends pytest the same way; that
    may count as caught only once every name is known to be right."""
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
    run = subprocess.run(
        [*command, "-m", QUICK_TESTS, *tests],
        cwd=project.root,
        env=_environment(project.source, work),
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        said = (run.stdout + run.stderr)[-SHOWN_OUTPUT:]
        raise SystemExit(f"pytest could not find the tests to run: {said}")


def try_mutant(project: Project, mutant: Mutant, copy: Path, tests: list[str], limit: float) -> str:
    """Put the mutant into a copy of the source folder and run ``tests`` against it."""
    target = copy / mutant.file
    original = target.read_text(encoding="utf-8")
    target.write_text(mutant.text, encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve()), RUN_TESTS, *PYTEST_OPTIONS, *tests]
    try:
        return run_watched(command, project.root, _environment(copy, copy.parent), limit)
    finally:
        target.write_text(original, encoding="utf-8")


def run_watched(command: list[str], cwd: Path, environment: dict[str, str], limit: float) -> str:
    """The verdict on one run of the tests; a run that takes too long or too much memory is
    stopped, with the verdict TIMEOUT or TOO_LARGE."""
    started = time.monotonic()
    with subprocess.Popen(
        command, cwd=cwd, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ) as process:
        while True:  # until the tests end, or take too long or too much memory
            if process.poll() is not None:
                return SURVIVED if process.returncode in (0, NO_TESTS_COLLECTED) else KILLED
            verdict = None
            if time.monotonic() - started > limit:
                verdict = TIMEOUT
            elif memory_of(process.pid) > MEMORY_LIMIT:
                verdict = TOO_LARGE
            if verdict is not None:
                process.kill()
                process.wait()
                return verdict
            time.sleep(WATCH_S)


class _Usage(ctypes.Structure):
    """struct rusage_info_v0 of macOS."""

    _fields_ = [("uuid", ctypes.c_uint8 * UUID_BYTES)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
        )  # fmt: skip
    ]


def memory_of(pid: int) -> int:
    """Bytes of memory the process takes, as the Activity Monitor counts them; 0 when gone."""
    usage = _Usage()
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    if libproc.proc_pid_rusage(pid, RUSAGE_INFO_V0, ctypes.byref(usage)) != 0:
        return 0
    return int(usage.phys_footprint)


def try_all(
    project: Project, mutants: list[Mutant], workers: int, work: Path
) -> list[tuple[Mutant, str]]:
    tests, times, seconds = tests_of_each_line(project, work)
    limit = 2 * seconds + TIME_MARGIN_S  # a mutant may turn a loop into an endless one
    print(f"The tests take {seconds:.0f} s. Testing {len(mutants)} mutants ...", flush=True)
    copies: queue.Queue[Path] = queue.Queue()
    for number in range(workers):
        copy = work / f"worker{number}"
        shutil.copytree(project.source, copy, ignore=shutil.ignore_patterns("__pycache__"))
        copies.put(copy)
    done = []

    def verdict(mutant: Mutant) -> tuple[Mutant, str]:
        lines = tests.get(mutant.file, {})
        own = {test for line in mutant.lines for test in lines.get(line, ())}
        rounds = [
            sorted(own, key=lambda test: (times.get(test, 0.0), test)),
            sorted({test.split("::")[0] for test in own}),  # if it survived the first
        ]
        if mutant.runs_at_import:
            rounds = [list(project.tests)]
        result = NOT_RUN
        if own or mutant.runs_at_import:
            copy = copies.get()
            try:
                for selected in rounds:
                    result = try_mutant(project, mutant, copy, selected, limit)
                    if result != SURVIVED:
                        break
            finally:
                copies.put(copy)
        done.append(result)
        if len(done) % 100 == 0:
            print(f"  {len(done)} of {len(mutants)}, {done.count(KILLED)} killed", flush=True)
        return mutant, result

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(verdict, mutants))


# --- Report -------------------------------------------------------------------


def report(project: Project, results: list[tuple[Mutant, str]]) -> None:
    verdicts = [result for _, result in results]
    caught = verdicts.count(KILLED) + verdicts.count(TIMEOUT) + verdicts.count(TOO_LARGE)
    print()
    print(f"Mutants     {len(results)}")
    print(f"Killed      {verdicts.count(KILLED)}")
    print(f"Timed out   {verdicts.count(TIMEOUT)}  (counted as killed)")
    print(f"Too large   {verdicts.count(TOO_LARGE)}  (counted as killed)")
    print(f"Survived    {verdicts.count(SURVIVED)}")
    print(f"Not tested  {verdicts.count(NOT_RUN)}  (no test runs the line)")
    if results:
        print(f"Score       {100 * caught / len(results):.1f} %")
    # A stopped mutant counts as caught, but a busy computer may have slowed down a test
    # that would have passed: it is listed too, to be looked at.
    listed = [(mutant, result) for mutant, result in results if result in NOTES]
    shown_file = None
    for mutant, result in sorted(listed, key=lambda item: (item[0].file, item[0].lines[0])):
        if mutant.file != shown_file:
            shown_file = mutant.file
            lines = (project.source / mutant.file).read_text(encoding="utf-8").splitlines()
            print()
            print(project.source.relative_to(project.root) / mutant.file)
        line = mutant.lines[0]
        print(f"  {line:4d}  {mutant.change}{NOTES[result]}")
        print(f"        {lines[line - 1].strip()}")


def run(project: Project, files: list[Path], workers: int) -> list[tuple[Mutant, str]]:
    mutants = [mutant for file in sorted(files) for mutant in mutants_of(file, project.source)]
    with tempfile.TemporaryDirectory() as work:
        results = try_all(project, mutants, workers, Path(work))
    report(project, results)
    return results


def default_workers(processors: int, memory: int) -> int:
    """As many workers as leave two processors free, and as fit, each at its MEMORY_LIMIT,
    into half the memory of the computer: the person using it keeps the other half."""
    return max(1, min(processors - 2, memory // (2 * MEMORY_LIMIT)))


def main() -> int:
    if sys.argv[1:2] == [RUN_TESTS]:
        return run_tests_quietly(sys.argv[2:])
    parser = argparse.ArgumentParser(description="Mutation testing of AdaMus.")
    parser.add_argument("files", nargs="*", type=Path, help="source files (default: all)")
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    parser.add_argument("--workers", type=int, default=default_workers(os.cpu_count() or 1, memory))
    arguments = parser.parse_args()
    package = ADAMUS.source / ADAMUS.package
    files = [path.resolve() for path in arguments.files] or list(package.rglob("*.py"))
    run(ADAMUS, files, arguments.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
