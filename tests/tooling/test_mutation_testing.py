"""scripts/mutation_testing.py: the mutants it makes, and a whole run over a tiny project."""

import ast
import os
import textwrap
import time
from pathlib import Path

import pytest

from scripts import mutation_testing
from scripts.mutation_testing import KILLED, NOT_RUN, SURVIVED, Project, mutants_of


def mutants(tmp_path, code):
    """The mutants of a module with the given code, as (line, change, mutated statement)."""
    (tmp_path / "pkg").mkdir(exist_ok=True)
    module = tmp_path / "pkg" / "module.py"
    module.write_text(textwrap.dedent(code))
    original = ast.unparse(ast.parse(module.read_text())).splitlines()
    result = []
    for mutant in mutants_of(module, tmp_path):
        assert mutant.file == Path("pkg/module.py")
        changed = [line for line in mutant.text.splitlines() if line not in original]
        result.append((mutant.lines[0], mutant.change, " ".join(changed).strip()))
    return sorted(result, key=lambda mutant: mutant[0])  # by line, otherwise as made


class TestMutants:
    def test_comparisons_are_turned_around(self, tmp_path):
        code = """
            def f(a, b):
                return a < b or a >= b or a == b or a is b or a not in b
            """
        changes = [change for _, change, _ in mutants(tmp_path, code)]
        assert changes == [
            "returns None", "or became and", "< became <=", ">= became >", "== became !=",
            "is became is not", "not in became in",
        ]  # fmt: skip

    def test_each_mutant_is_the_file_with_one_change(self, tmp_path):
        code = """
            def f(a, b):
                total = a + b
                return total
            """
        assert mutants(tmp_path, code) == [
            (3, "+ became -", "total = a - b"),
            (4, "returns None", "return None"),
        ]

    def test_arithmetic_numbers_truth_values_and_texts(self, tmp_path):
        code = """
            def f(a):
                return [a * 2, a / 2.5, a // 3, a % 4, True, "name"]
            """
        changes = [change for _, change, _ in mutants(tmp_path, code)][1:]
        assert changes == [
            "* became /", "/ became *", "// became /", "% became *", "True became False",
            "text 'name' changed", "2 became 3", "2.5 became 3.5", "3 became 4", "4 became 5",
        ]  # fmt: skip

    def test_conditions_are_negated_and_calls_left_out(self, tmp_path):
        code = """
            def f(items, log_line):
                if items:
                    items.clear()
                while items:
                    items.pop()
                return 1 if log_line else 0
            """
        wanted = ("condition negated", "call left out")
        assert [mutant for mutant in mutants(tmp_path, code) if mutant[1] in wanted] == [
            (3, "condition negated", "if not items:"),
            (4, "call left out", "None"),
            (5, "condition negated", "while not items:"),
            (6, "call left out", "None"),
            (7, "condition negated", "return 1 if not log_line else 0"),
        ]

    def test_a_return_that_already_gives_none_is_not_changed(self, tmp_path):
        """Turned into "return None" it would mean the same: a mutant nobody could kill."""
        code = """
            def f(ready):
                if ready:
                    return
                return None
            """
        assert [change for _, change, _ in mutants(tmp_path, code)] == ["condition negated"]

    def test_what_nobody_is_meant_to_test_is_left_alone(self, tmp_path):
        code = '''
            """A module docstring with the number 1 in it."""
            import logging
            from dataclasses import dataclass

            log = logging.getLogger("x")


            @dataclass(frozen=True)
            class Thing:
                """A class docstring."""

                size: int


            def f(path, count: int = 0) -> "list[int] | None":
                """A function docstring."""
                assert count >= 0, "count is never negative"
                log.warning("cannot read %s: %d", path, 1 + 1)
                return path.read_text(encoding="utf-8")
            '''
        assert mutants(tmp_path, code) == [
            (6, "text 'x' changed", "log = logging.getLogger('x~')"),
            (16, "0 became 1", "def f(path, count: int=1) -> 'list[int] | None':"),
            (20, "returns None", "return None"),
        ]

    def test_code_outside_functions_is_marked_as_running_at_import(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        module = tmp_path / "pkg" / "module.py"
        module.write_text(
            textwrap.dedent(
                """
                LIMIT = 10


                class Box:
                    size = 3

                    def grown(self, by=1):
                        return self.size + by + LIMIT
                """
            )
        )
        at_import = sorted(
            (m.lines[0], m.change, m.runs_at_import) for m in mutants_of(module, tmp_path)
        )
        assert at_import == [
            (2, "10 became 11", True),
            (6, "3 became 4", True),
            (8, "1 became 2", True),  # a default value is computed when the module is imported
            (9, "+ became -", False),
            (9, "+ became -", False),
            (9, "returns None", False),
        ]

    def test_statement_over_several_lines_is_known_by_all_of_them(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        module = tmp_path / "pkg" / "module.py"
        module.write_text("def f(a):\n    return [\n        a,\n        a + 1,\n    ]\n")
        addition = next(m for m in mutants_of(module, tmp_path) if m.change == "+ became -")
        assert list(addition.lines) == [2, 3, 4, 5]


@pytest.fixture
def tiny_project(tmp_path):
    """A package with one tested function, one half-tested and one that no test calls."""
    source = tmp_path / "src"
    (source / "tiny").mkdir(parents=True)
    (source / "tiny" / "numbers.py").write_text(
        textwrap.dedent(
            """
            LIMIT = 10


            def clipped(value):
                return min(value, LIMIT)


            def is_small(value):
                return value < 3


            def unused(value):
                return value + 1
            """
        )
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_numbers.py").write_text(
        textwrap.dedent(
            """
            from tiny.numbers import clipped, is_small


            def test_clipped():
                assert clipped(5) == 5 and clipped(50) == 10


            def test_is_small():
                assert is_small(1) and not is_small(7)  # says nothing about 3 itself
            """
        )
    )
    return Project(root=tmp_path, source=source, package="tiny", tests=("tests",))


def test_whole_run_tells_killed_mutants_from_survivors(tiny_project, capsys):
    results = mutation_testing.run(
        tiny_project, [tiny_project.source / "tiny" / "numbers.py"], workers=2
    )
    verdicts = {(mutant.lines[0], mutant.change): verdict for mutant, verdict in results}
    assert verdicts == {
        (2, "10 became 11"): KILLED,
        (6, "returns None"): KILLED,
        (10, "returns None"): KILLED,
        (10, "< became <="): SURVIVED,  # no test looks at the boundary
        (10, "3 became 4"): SURVIVED,
        (14, "returns None"): NOT_RUN,
        (14, "+ became -"): NOT_RUN,
        (14, "1 became 2"): NOT_RUN,
    }
    shown = capsys.readouterr().out
    assert "Mutants     8" in shown and "Killed      3" in shown and "Score       37.5 %" in shown
    assert "    10  < became <=" in shown and "        return value < 3" in shown
    assert "    14  + became -  (not tested)" in shown
    # The project itself was not touched.
    assert "return value < 3" in (tiny_project.source / "tiny" / "numbers.py").read_text()


def test_tests_that_fail_before_any_mutant_stop_the_run(tiny_project):
    (tiny_project.root / "tests" / "test_broken.py").write_text("def test_no():\n    assert 0\n")
    with pytest.raises(SystemExit, match="the tests must pass before mutation testing"):
        mutation_testing.run(tiny_project, [tiny_project.source / "tiny" / "numbers.py"], 1)


def test_mutant_that_crashes_python_is_killed_without_a_crash_report(tiny_project):
    """The test process ends with an exit code; the operating system sees no crash."""
    (tiny_project.source / "tiny" / "numbers.py").write_text(
        textwrap.dedent(
            """
            import ctypes


            def address():
                return 8


            def peek():
                return address() or ctypes.string_at(address())
            """
        )
    )
    (tiny_project.root / "tests" / "test_numbers.py").write_text(
        "from tiny.numbers import peek\n\n\ndef test_peek():\n    assert peek() == 8\n"
    )
    results = mutation_testing.run(
        tiny_project, [tiny_project.source / "tiny" / "numbers.py"], workers=1
    )
    verdicts = {(mutant.lines[0], mutant.change): verdict for mutant, verdict in results}
    assert verdicts[(10, "or became and")] == KILLED  # reads memory at address 8: a crash


def test_mutant_that_takes_ever_more_memory_is_stopped_and_counts_as_killed(
    tiny_project, monkeypatch, capsys
):
    """A changed number turns a loop of three into an endless one that keeps memory."""
    (tiny_project.source / "tiny" / "numbers.py").write_text(
        textwrap.dedent(
            """
            def collect(count):
                kept = []
                while len(kept) < count:
                    kept += [bytes(range(256)) * 4096]  # a megabyte
                    count += 0
                return len(kept)
            """
        )
    )
    (tiny_project.root / "tests" / "test_numbers.py").write_text(
        "from tiny.numbers import collect\n\n\ndef test_collect():\n    assert collect(3) == 3\n"
    )
    monkeypatch.setattr(mutation_testing, "MEMORY_LIMIT", 300 * 1024**2)
    started = time.monotonic()
    results = mutation_testing.run(
        tiny_project, [tiny_project.source / "tiny" / "numbers.py"], workers=1
    )
    verdicts = {(mutant.lines[0], mutant.change): verdict for mutant, verdict in results}
    assert verdicts[(6, "0 became 1")] == mutation_testing.TOO_LARGE
    assert time.monotonic() - started < 60  # stopped long before its time limit
    shown = capsys.readouterr().out
    assert "Too large   1  (counted as killed)" in shown
    assert "     6  0 became 1  (stopped: too much memory)" in shown  # to be looked at


def test_mutant_that_never_ends_is_stopped_and_shown(tiny_project, monkeypatch, capsys):
    """Stopped mutants count as caught, but a busy computer may have slowed a test that
    would have passed: they are listed, so that each can be looked at."""
    (tiny_project.source / "tiny" / "numbers.py").write_text(
        textwrap.dedent(
            """
            def count_to(count):
                done = 0
                while done < count:
                    done = done + 1
                return done
            """
        )
    )
    (tiny_project.root / "tests" / "test_numbers.py").write_text(
        "from tiny.numbers import count_to\n\n\ndef test_count_to():\n    assert count_to(3) == 3\n"
    )
    monkeypatch.setattr(mutation_testing, "TIME_MARGIN_S", 1)
    results = mutation_testing.run(
        tiny_project, [tiny_project.source / "tiny" / "numbers.py"], workers=1
    )
    verdicts = {(mutant.lines[0], mutant.change): verdict for mutant, verdict in results}
    assert verdicts[(5, "+ became -")] == mutation_testing.TIMEOUT
    shown = capsys.readouterr().out
    assert "Timed out   1  (counted as killed)" in shown
    assert "     5  + became -  (stopped: took too long)" in shown


def test_what_the_tests_leave_in_the_temporary_folder_goes_with_the_run(
    tiny_project, tmp_path, monkeypatch
):
    """A test process that is stopped, or a mutant that no longer deletes its files, cannot
    clean up after itself; the files would stay in the folder of the system for good."""
    system_temporary = tmp_path / "system"
    system_temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(system_temporary))
    (tiny_project.root / "tests" / "test_numbers.py").write_text(
        textwrap.dedent(
            """
            import tempfile

            from tiny.numbers import clipped


            def test_clipped():
                tempfile.mkstemp(prefix="left_behind_")
                assert clipped(5) == 5 and clipped(50) == 10
            """
        )
    )
    mutation_testing.run(tiny_project, [tiny_project.source / "tiny" / "numbers.py"], workers=1)
    assert not list(system_temporary.rglob("left_behind_*"))


def test_memory_of_a_process_and_of_one_that_is_not_there():
    held = bytes(range(256)) * (200 * 4096)  # 200 MB that are written, so they count
    assert mutation_testing.memory_of(os.getpid()) > len(held)
    assert mutation_testing.memory_of(999_999_999) == 0


def test_every_slow_kind_of_test_is_left_out():
    """Each mutant runs the tests in several workers at once: a test of a slow kind would
    take too long there, and the training tests would read the whole dataset in each."""
    import tomllib

    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    kinds = [marker.split(":")[0] for marker in project["tool"]["pytest"]["ini_options"]["markers"]]
    assert kinds == ["realtime", "training"]
    assert mutation_testing.QUICK_TESTS == " and ".join(f"not {kind}" for kind in kinds)
    assert mutation_testing.PYTEST_OPTIONS[2:4] == ["-m", mutation_testing.QUICK_TESTS]


@pytest.mark.parametrize(
    ("processors", "gigabytes", "workers"),
    [(10, 16, 4), (10, 64, 8), (4, 64, 2), (2, 16, 1), (10, 2, 1)],
)
def test_workers_leave_processors_and_half_the_memory_free(processors, gigabytes, workers):
    """Each worker may take up to 2 GB; a 16 GB Mac ran out of memory with eight of them."""
    assert mutation_testing.MEMORY_LIMIT == 2 * 1024**3
    assert mutation_testing.default_workers(processors, gigabytes * 1024**3) == workers


class TestWhichTestsRunAMutant:
    """A mutant inside a function is tried by the tests that run its line, the quickest
    first; one that survives them is tried by every test of their files."""

    @staticmethod
    def project_with(tiny_project, tests):
        (tiny_project.source / "tiny" / "numbers.py").write_text(
            "def double(value):\n    return value * 2\n"
        )
        (tiny_project.root / "tests" / "test_numbers.py").write_text(textwrap.dedent(tests))
        return mutation_testing.run(
            tiny_project, [tiny_project.source / "tiny" / "numbers.py"], workers=1
        )

    def test_the_quickest_test_of_the_line_runs_first_and_a_failure_ends_the_run(
        self, tiny_project
    ):
        """The slow test is first in its file; a mutant it does not catch must not wait for it."""
        results = self.project_with(
            tiny_project,
            """
            import time
            from pathlib import Path

            from tiny.numbers import double

            RAN = Path(__file__).with_name("ran.txt")


            def test_slow():
                RAN.open("a").write("slow\\n")
                double(1)
                time.sleep(0.5)


            def test_quick():
                RAN.open("a").write("quick\\n")
                assert double(3) == 6
            """,
        )
        assert {verdict for _, verdict in results} == {KILLED}
        ran = (tiny_project.root / "tests" / "ran.txt").read_text().split()
        assert ran.count("slow") == 1  # only while finding the tests of each line
        assert ran.count("quick") == 1 + len(results)

    def test_a_survivor_of_the_tests_of_its_line_is_tried_by_the_rest_of_their_file(
        self, tiny_project
    ):
        """The second test never runs the line, but it checks what the first one kept."""
        results = self.project_with(
            tiny_project,
            """
            from tiny.numbers import double

            KEPT = []


            def test_keeps_a_double():
                KEPT.append(double(3))


            def test_the_double_kept():
                assert KEPT == [6]
            """,
        )
        assert {verdict for _, verdict in results} == {KILLED}

    def test_a_test_is_found_by_its_name_whatever_its_parameters_hold(self, tiny_project):
        results = self.project_with(
            tiny_project,
            """
            import pytest

            from tiny.numbers import double


            @pytest.mark.parametrize("value", [3], ids=["a b: c/d [e] ::f"])
            def test_double(value):
                assert double(value) == 6
            """,
        )
        assert {verdict for _, verdict in results} == {KILLED}

    def test_a_mutant_that_breaks_the_import_of_a_test_file_is_caught(self, tiny_project):
        """Asked for by name, a test of a file that cannot be imported is not found at all:
        pytest then ends with the code of a wrong command, which the mutant caused."""
        (tiny_project.source / "tiny" / "numbers.py").write_text(
            "def checked(value):\n    if value < 0:\n        raise ValueError\n    return value\n"
        )
        (tiny_project.root / "tests" / "test_numbers.py").write_text(
            textwrap.dedent(
                """
                from tiny.numbers import checked

                ONE = checked(1)


                def test_checked():
                    assert checked(ONE) == 1
                """
            )
        )
        results = mutation_testing.run(
            tiny_project, [tiny_project.source / "tiny" / "numbers.py"], workers=1
        )
        verdicts = {(mutant.lines[0], mutant.change): verdict for mutant, verdict in results}
        assert verdicts[(2, "condition negated")] == KILLED


def test_tests_that_are_named_otherwise_in_the_next_run_stop_it_before_any_mutant(tiny_project):
    """Each mutant asks pytest for tests by the names they had while finding the tests of
    each line; a test pytest could not find would end the run with an error, as if caught."""
    (tiny_project.root / "tests" / "test_numbers.py").write_text(
        textwrap.dedent(
            """
            import time

            import pytest

            from tiny.numbers import clipped


            @pytest.mark.parametrize("value", [5], ids=[str(time.time_ns())])
            def test_clipped(value):
                assert clipped(value) == 5
            """
        )
    )
    with pytest.raises(SystemExit, match="^pytest could not find the tests to run"):
        mutation_testing.run(tiny_project, [tiny_project.source / "tiny" / "numbers.py"], 1)
