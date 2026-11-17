"""scripts/install_generator.sh run for real, with stand-ins for the tools it downloads with."""

import stat
import subprocess

from tests.support.fake_generator_files import install_generator_files
from tests.support.project_paths import ROOT

SCRIPT = ROOT / "scripts" / "install_generator.sh"
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


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
    tool(tools, "unzip", record)
    tool(tools, "hdiutil", record + "\n" + 'while [ $# -gt 0 ]; do [ "$1" = -mountpoint ] && '
         'mkdir -p "$2"; shift; done')  # fmt: skip
    done = run(tmp_path / "gen", tmp_path, tools)
    assert done.returncode != 0
    used = [line.split()[1] for line in calls.read_text().splitlines() if "hdiutil" in line]
    assert used == ["attach", "detach"]
