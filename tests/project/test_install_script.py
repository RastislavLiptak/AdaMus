"""scripts/install_generator.sh run for real, with stand-ins for the tools it downloads with."""

import re
import stat
import subprocess

import pytest

from adamus import constants as c
from adamus.machine import INSTALL_SCRIPT
from tests.support.project_paths import ROOT
from tests.support.stand_in_generator_files import install_generator_files

SCRIPT = ROOT / INSTALL_SCRIPT
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
PINNED = re.search(r'MRT_COMMIT="(\w+)"', SCRIPT.read_text())
assert PINNED is not None, "the script names the commit it checks"
CHECKED_COMMIT = PINNED[1]


def run(folder, cwd, tools=None):
    path = SYSTEM_PATH if tools is None else f"{tools}:{SYSTEM_PATH}"
    return subprocess.run(
        ["sh", str(SCRIPT), str(folder), "mrt2_small"],
        cwd=cwd, capture_output=True, text=True, env={"PATH": path}, check=False,
    )  # fmt: skip


def test_an_installed_generator_is_left_as_it_is(tmp_path):
    install_generator_files(tmp_path / "gen", "mrt2_small")
    done = run("gen", tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [f"Generator installed in {tmp_path}/gen"]


def test_an_absolute_folder_is_used_as_it_is(tmp_path):
    folder = tmp_path / "elsewhere" / "generator"
    install_generator_files(folder, "mrt2_small")
    work = tmp_path / "work"
    work.mkdir()
    done = run(folder, work)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [f"Generator installed in {folder}"]
    assert list(work.iterdir()) == []  # nothing below the folder it was started in


def tool(folder, name, body):
    script = folder / name
    script.write_text("#!/bin/sh\n" + body + "\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)


def test_a_failed_copy_of_pure_data_leaves_no_disk_image_attached(tmp_path):
    """The disk image of Pure Data holds no Pd.app here, so copying it fails."""
    tools, calls = tmp_path / "tools", tmp_path / "calls.log"
    tools.mkdir()
    record = f'echo "$0 $1" >> "{calls}"'
    tool(tools, "curl", record + "\n" + 'while [ $# -gt 0 ]; do [ "$1" = --output ] && '
         ': > "$2"; shift; done')  # fmt: skip
    tool(tools, "shasum", "cat > /dev/null")  # the download passes its check
    tool(tools, "unzip", record)
    tool(tools, "hdiutil", record + "\n" + 'while [ $# -gt 0 ]; do [ "$1" = -mountpoint ] && '
         'mkdir -p "$2"; shift; done')  # fmt: skip
    done = run(tmp_path / "gen", tmp_path, tools)
    assert done.returncode != 0
    used = [line.split()[1] for line in calls.read_text().splitlines() if "hdiutil" in line]
    assert used == ["attach", "detach"]


def test_a_download_of_pure_data_other_than_the_one_checked_is_refused(tmp_path):
    tools, calls = tmp_path / "tools", tmp_path / "calls.log"
    tools.mkdir()
    other = 'while [ $# -gt 0 ]; do [ "$1" = --output ] && echo other > "$2"; shift; done'
    tool(tools, "curl", other)
    tool(tools, "hdiutil", f'echo "$0" >> "{calls}"')
    done = run(tmp_path / "gen", tmp_path, tools)
    assert done.returncode != 0 and "is not the Pure Data" in done.stderr
    assert not calls.exists()  # never opened


def test_a_clone_of_the_external_other_than_the_commit_checked_is_refused(tmp_path):
    folder = tmp_path / "gen"
    install_generator_files(folder, "mrt2_small")
    (folder / c.GENERATOR_EXTERNAL_FILE).unlink()
    tools, calls = tmp_path / "tools", tmp_path / "calls.log"
    tools.mkdir()
    tool(tools, "git", '[ "$3" = rev-parse ] && echo 0000000; exit 0')
    tool(tools, "cmake", f'echo "$0" >> "{calls}"')
    done = run(folder, tmp_path, tools)
    assert done.returncode != 0 and "is not the commit that was checked" in done.stderr
    assert not calls.exists()  # never built


GIGABYTE = 1024**3


@pytest.mark.parametrize(("memory", "compilers"), [(16 * GIGABYTE, 4), (2 * GIGABYTE, 1)])
def test_the_external_is_built_with_one_compiler_per_four_gigabytes_of_memory(
    tmp_path, memory, compilers
):
    """Eight compilers at once ran a Mac with 16 GB out of memory; a small one gets one."""
    folder = tmp_path / "gen"
    install_generator_files(folder, "mrt2_small")
    (folder / c.GENERATOR_EXTERNAL_FILE).unlink()  # only the external is left to build
    tools, calls = tmp_path / "tools", tmp_path / "calls.log"
    tools.mkdir()
    tool(tools, "sysctl", f"echo {memory}")
    tool(tools, "git", f'[ "$3" = rev-parse ] && echo {CHECKED_COMMIT}; exit 0')
    built = "$2/examples/pd"  # cmake --build <folder> ...: the files of the external
    tool(tools, "cmake", f'echo "$@" >> "{calls}"\n[ "$1" = --build ] && mkdir -p "{built}" && '
         f': > "{built}/mrt2~.pd_darwin" && : > "{built}/mlx.metallib"; exit 0')  # fmt: skip
    done = run(folder, tmp_path, tools)
    assert done.returncode == 0, done.stderr
    (build,) = [line.split() for line in calls.read_text().splitlines() if "--build" in line]
    assert build[build.index("--parallel") + 1] == str(compilers)
    assert (folder / c.GENERATOR_EXTERNAL_FILE).is_file()
