"""scripts/build_app.sh: it builds the app into the project folder it is run in, with no way in
for the tests of the window, and signs it with a certificate of that folder, so that macOS
knows it again after a change and asks only once whether it may use the folder the project lies
in: macOS knows an app by its designated requirement, which here names the certificate, not the
content of the app.

It runs over a copy of the app in a project folder of the test, as the build of a clone would."""

import plistlib
import re
import shutil
import subprocess

import pytest

from tests.support.project_paths import ROOT
from tests.support.xcode_builds import one_build_at_a_time

# A build of the app from nothing: run by the full run only.
pytestmark = pytest.mark.slow

BUILD_S = 900  # the longest a build of the app may take, from nothing
APP = "AdaMus.app"
BUNDLE_ID = "local.adamus.AdaMus"
FOR_THE_TESTS = b"ADAMUS_PROGRAM_ADDRESS"  # how the tests of the window give it their program


def build(project):
    with one_build_at_a_time():
        subprocess.run(
            ["sh", str(ROOT / "scripts" / "build_app.sh")], cwd=project, check=True,
            capture_output=True, timeout=BUILD_S,
        )  # fmt: skip


def signature(app):
    """The designated requirement of ``app`` and the hash of its content."""
    said = subprocess.run(
        ["codesign", "-d", "-r-", "-vvv", str(app)], capture_output=True, text=True, check=True
    )
    shown = said.stdout + said.stderr
    requirement = re.search(r"^designated => (.+)$", shown, re.MULTILINE)
    content = re.search(r"^CDHash=(\w+)$", shown, re.MULTILINE)
    assert requirement and content, shown
    return requirement[1], content[1]


def test_the_app_is_built_into_its_project_and_known_again_after_a_change(tmp_path):
    project = tmp_path / "project"
    shutil.copytree(ROOT / "src" / "ui", project / "src" / "ui")
    build(project)
    app = project / APP
    first, first_content = signature(app)
    subprocess.run(["codesign", "--verify", "--strict", str(app)], check=True)
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleIdentifier"] == BUNDLE_ID and info["CFBundleIconName"] == "AppIcon"
    # the program it starts looks for the headset over Bluetooth: without saying why, macOS
    # ends the program at once, as the app answers for it
    assert info["NSBluetoothAlwaysUsageDescription"]
    built = (app / "Contents" / "MacOS" / "AdaMus").read_bytes()
    assert built.find(FOR_THE_TESTS) == -1, "the app lets the tests of the window in"

    metrics = project / "src" / "ui" / "AdaMus" / "Metrics.swift"
    used = "static let summaryColumn: CGFloat = "
    assert used in metrics.read_text()
    metrics.write_text(metrics.read_text().replace(used, used + "1 + "))  # a change of the app
    build(project)
    again, again_content = signature(app)
    assert again_content != first_content  # a new app by its content,
    assert again == first  # the same to macOS
    assert first.startswith(f'identifier "{BUNDLE_ID}" and certificate leaf = H"')

    keychains = subprocess.run(
        ["security", "list-keychains", "-d", "user"], capture_output=True, text=True, check=True
    ).stdout
    assert str(project) not in keychains  # the keychain of the project stays its own
