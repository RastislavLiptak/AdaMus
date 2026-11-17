"""Rules every file of the program keeps, checked on the syntax tree; so do its scripts
and the code the tests are built on.

They follow NASA's "Power of Ten" rules for safety-critical code, as far as
they make sense in Python: simple control flow, bounded loops, short
functions, no hidden state, and no numbers or names typed in twice.
"""

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "src" / "adamus"
SOURCE_FILES = sorted(SOURCE.rglob("*.py"))
FILES = sorted(
    [*SOURCE_FILES, *(REPO / "scripts").glob("*.py"), *(REPO / "tests" / "support").glob("*.py")]
)
TREES = {path: ast.parse(path.read_text(encoding="utf-8")) for path in FILES}
MAX_FUNCTION_LINES = 60  # one printed page
#: Numbers that may appear outside constants.py: counting, halving, unit conversion.
PLAIN_NUMBERS = {0, 1, 2, -1, 60, 100, 1000}


def functions(tree):
    return [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def named(path):
    return str(path.relative_to(REPO))


def docstrings(tree):
    """The ids of the docstrings: of the module, of every class and every function."""
    owners = [tree, *(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef | ast.FunctionDef))]
    return {
        id(owner.body[0].value)
        for owner in owners
        if owner.body
        and isinstance(owner.body[0], ast.Expr)
        and isinstance(owner.body[0].value, ast.Constant)
        and isinstance(owner.body[0].value.value, str)
    }


def has_line_break(value):
    if isinstance(value, str):
        return "\n" in value or "\r" in value
    return isinstance(value, bytes) and (b"\n" in value or b"\r" in value)


#: The only line breaks in texts: between the lines of an error message, and the key
#: [ENTER] as the terminal sends it. Each is named once.
NAMED_LINE_BREAKS = {
    "errors.py": "LINE_END = chr(10)",
    "keyboard.py": "CARRIAGE_RETURN, BACKSPACE_CHARACTER = chr(13), chr(8)",
}
LINE_BREAK_MADE = r"chr\(1[03]\)|os\.linesep"


@pytest.mark.parametrize("path", FILES, ids=named)
class TestEveryFile:
    def test_starts_by_saying_what_it_is_for(self, path):
        docstring = ast.get_docstring(TREES[path])
        assert docstring and len(docstring.split()) >= 5

    def test_functions_fit_on_one_page(self, path):
        for function in functions(TREES[path]):
            length = function.end_lineno - function.lineno + 1
            assert length <= MAX_FUNCTION_LINES, f"{function.name} has {length} lines"

    def test_no_function_calls_itself(self, path):
        for function in functions(TREES[path]):
            for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
                target = call.func
                by_name = isinstance(target, ast.Name) and target.id == function.name
                on_self = (
                    isinstance(target, ast.Attribute)
                    and target.attr == function.name
                    and isinstance(target.value, ast.Name)
                    and target.value.id in ("self", "cls")
                )
                assert not (by_name or on_self), f"{function.name} is recursive"

    def test_every_loop_has_a_bound_or_is_meant_to_run_until_stopped(self, path):
        """Loops run over a range or a collection.

        The exception is a loop that waits for something outside the program
        (a key, the next samples): it is written ``while True`` so that nobody
        takes it for a loop that ends by itself, and a comment on its line
        says what ends it.
        """
        lines = path.read_text(encoding="utf-8").splitlines()
        for loop in (node for node in ast.walk(TREES[path]) if isinstance(node, ast.While)):
            endless = isinstance(loop.test, ast.Constant) and loop.test.value is True
            assert endless, f"while loop with a condition at line {loop.lineno}"
            assert "#" in lines[loop.lineno - 1], f"line {loop.lineno} does not say what ends it"
            leaves = [
                n for n in ast.walk(loop) if isinstance(n, ast.Return | ast.Raise | ast.Break)
            ]
            assert leaves, f"the loop at line {loop.lineno} has no way out"

    def test_no_exception_is_swallowed_blindly(self, path):
        for handler in (n for n in ast.walk(TREES[path]) if isinstance(n, ast.ExceptHandler)):
            assert handler.type is not None, f"bare except at line {handler.lineno}"
            only_pass = len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass)
            assert not only_pass, f"exception ignored at line {handler.lineno}"

    def test_module_level_names_are_constants_types_or_the_logger(self, path):
        """No state lives at module level."""
        for node in TREES[path].body:
            if isinstance(node, ast.Assign | ast.AnnAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for name in (n.id for n in ast.walk(target) if isinstance(n, ast.Name)):
                        constant_like = name.strip("_").isupper() or name.startswith("__")
                        type_alias = name[0].isupper()
                        assert constant_like or type_alias or name == "log", (
                            f"{name} at module level"
                        )

    def test_every_module_level_name_is_assigned_once(self, path):
        """A second assignment replaces the first for every use, also those above it."""
        assigned = [
            name.id
            for node in TREES[path].body
            if isinstance(node, ast.Assign | ast.AnnAssign)
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            for name in ast.walk(target)
            if isinstance(name, ast.Name)
        ]
        twice = sorted({name for name in assigned if assigned.count(name) > 1})
        assert twice == [], f"assigned more than once: {twice}"

    def test_no_global_or_nonlocal_state(self, path):
        hidden = [n for n in ast.walk(TREES[path]) if isinstance(n, ast.Global | ast.Nonlocal)]
        assert hidden == []

    def test_every_number_has_a_name(self, path):
        """A number is either in the configuration or a named constant at the top of a module."""
        named = set()
        for statement in TREES[path].body:
            if isinstance(statement, ast.Assign | ast.AnnAssign):
                named.update(id(node) for node in ast.walk(statement))
        for node in ast.walk(TREES[path]):
            is_number = isinstance(node, ast.Constant) and type(node.value) in (int, float)
            if is_number and id(node) not in named:
                assert node.value in PLAIN_NUMBERS, (
                    f"{node.value} at line {node.lineno}: give it a name or put it in config.yaml"
                )

    def test_no_line_break_is_hidden_inside_a_text(self, path):
        """What is printed is written one line per call, so that the code looks like the output.

        Spelt in any way: an escape, a triple-quoted text, ``chr`` or ``os.linesep``.
        """
        skip = docstrings(TREES[path])
        breaks = [
            node.lineno
            for node in ast.walk(TREES[path])
            if isinstance(node, ast.Constant)
            and id(node) not in skip
            and has_line_break(node.value)
        ]
        assert breaks == [], f"a line break in a text at lines {breaks}"
        text = path.read_text(encoding="utf-8").replace(NAMED_LINE_BREAKS.get(path.name, "-"), "")
        assert re.findall(LINE_BREAK_MADE, text) == []


WRITTEN_FILES = sorted(
    path
    for path in {
        *FILES,
        *(REPO / "tests").rglob("*.py"),
        *(path for path in (REPO / "scripts").iterdir() if path.is_file()),
        *(REPO / name for name in ("README.md", "config.yaml", "pyproject.toml", ".gitignore")),
    }
    if path != Path(__file__).resolve()  # this file has to spell the names out
)
#: Module, metric, decision and requirement codes of the design document; a code also
#: counts inside a name, as in ``test_latency_is_x9_of_the_steps``.
CODE_NAME = (
    r"(?<![A-Za-z0-9])"
    r"(?:[Mm][1-6]|[APQLGRECaplqgrec][1-4]|[DZ]\d{1,2}|[Ss]tage [A-Ea-e])"
    r"(?![A-Za-z0-9])"
)


@pytest.mark.parametrize("path", WRITTEN_FILES, ids=lambda path: str(path.relative_to(REPO)))
def test_no_code_names_from_the_design_document(path):
    """Modules, steps and metrics are called by what they do, not by a letter and a number.

    The design document is gone; a reader of the program, its tests, its
    configuration and its README must not need it.
    """
    text = path.read_text(encoding="utf-8")
    assert re.findall(CODE_NAME, text) == []
    assert "specification" not in text.lower()


def test_every_constant_is_used():
    """A constant nobody reads is a second copy of something waiting to go stale."""
    names = [
        node.targets[0].id
        for node in TREES[SOURCE / "constants.py"].body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    ]
    assert len(names) > 50
    assert len(names) == len(set(names))
    used_in_constants = (SOURCE / "constants.py").read_text()
    elsewhere = "".join(path.read_text() for path in SOURCE_FILES if path.name != "constants.py")
    for name in names:
        read_elsewhere = re.search(rf"\bc\.{name}\b", elsewhere) is not None
        derived_from = len(re.findall(rf"\b{name}\b", used_in_constants)) > 1
        assert read_elsewhere or derived_from, f"{name} is never used"


#: Values derived from the constants once, where they are defined; typed again
#: elsewhere, one place could change without the other.
DERIVED_ONCE = {
    r"CONTEXT_WINDOWS \*|samples_of\(c\.(WINDOW_S|HOP_S)": "signal_processing.Steps",
    r"c\.(PRIOR_S|MEMORY_S) / c\.HOP_S": "constants.PRIOR_STEPS and MEMORY_STEPS",
}


EVERY_PYTHON_FILE = sorted(
    p for top in ("src", "scripts", "tests") for p in (REPO / top).rglob("*.py")
)


@pytest.mark.parametrize("path", EVERY_PYTHON_FILE, ids=lambda path: str(path.relative_to(REPO)))
def test_the_geometry_of_the_steps_is_derived_once(path):
    if path.name in ("signal_processing.py", "constants.py"):
        return
    text = path.read_text(encoding="utf-8")
    for pattern, where in DERIVED_ONCE.items():
        assert not re.search(pattern, text), f"use {where}"


def imported_modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            yield from (f"{node.module}.{alias.name}" for alias in node.names)


@pytest.mark.parametrize("path", SOURCE_FILES, ids=named)
def test_only_the_console_and_the_command_know_the_console(path):
    """The sessions, the detection and the evaluation work without a console, e.g. offline."""
    if path.parent.name == "console" or path.name == "cli.py":
        return
    console = [
        name
        for name in imported_modules(TREES[path])
        if name == "adamus.console" or name.startswith("adamus.console.")
    ]
    assert console == [], f"imports {console}"


#: What each part may use besides the libraries: the computation does not know the devices
#: or the sessions, so that training and evaluation run it as a session does; a session runs
#: without the evaluation and the console; a recording is read and written by itself.
BASICS = ("adamus.constants", "adamus.config", "adamus.errors")
MAY_USE = {
    "detection": (*BASICS, "adamus.detection", "adamus.protocol"),
    "evaluation": (*BASICS, "adamus.detection", "adamus.protocol", "adamus.session_files",
                   "adamus.evaluation", "adamus.realtime.generator_messages"),
    "realtime": (*BASICS, "adamus.detection", "adamus.protocol", "adamus.session_files",
                 "adamus.datasets", "adamus.realtime"),
    "datasets": BASICS,
}  # fmt: skip
PART_OF_MODULE = {"protocol.py": "detection", "datasets.py": "datasets"}


@pytest.mark.parametrize("path", SOURCE_FILES, ids=named)
def test_the_computation_does_not_depend_on_the_devices(path):
    part = PART_OF_MODULE.get(path.name, path.parent.name)
    if part not in MAY_USE:
        return
    used = [name for name in imported_modules(TREES[path]) if name.startswith("adamus.")]
    allowed = MAY_USE[part]
    other = [name for name in used if not any(name.startswith(module) for module in allowed)]
    assert other == [], f"{named(path)} uses {other}"


def test_no_package_folder_has_an_empty_init_file():
    """Only the folders of the project: not the installed libraries or the models."""
    for top in ("src", "scripts", "tests"):
        for init in (REPO / top).rglob("__init__.py"):
            assert init.read_text().strip(), f"{init} is empty"


def test_every_file_with_tests_is_named_test_and_every_test_file_has_tests():
    tests = REPO / "tests"
    for path in tests.rglob("*.py"):
        tree = ast.parse(path.read_text())
        has_tests = any(
            isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            for node in ast.walk(tree)
        )
        assert has_tests == path.name.startswith("test_"), path
