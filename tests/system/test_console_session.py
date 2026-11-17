"""The console in real time: preflight, a recording played by BrainFlow, [SPACE], statistics.

Only the audio device and the headset probe are stand-ins; Pd is replaced by
its stand-in binary. Keys are "pressed" when the console shows the matching text.
"""

import io
import time
from pathlib import Path

import pytest
import soundfile
from rich.console import Console

from adamus import constants as c
from adamus.config import read_yaml, write_yaml
from adamus.console import keyboard
from adamus.console.console_app import ConsoleApp
from adamus.console.keyboard import Keyboard
from adamus.console.machine import Machine
from adamus.session_files import REASON_USER
from tests.support.console_runs import place_focus_model
from tests.support.fake_generator_files import install_fake_generator
from tests.support.finished_sessions import read_lines, synthetic_signals
from tests.support.project_paths import CONFIG_FILE
from tests.support.stand_in_devices import ScriptedKeys
from tests.support.synthetic_dataset import write_recording
from tests.support.waiting import wait_until

pytestmark = pytest.mark.realtime


class MachineWithoutHeadset(Machine):
    def audio_output(self):
        return "Test output", 48000.0

    def headset_is_available(self, board):
        return False


def test_session_over_a_recording_from_source_selection_to_statistics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = read_yaml(CONFIG_FILE)
    raw["protocol"] = [["REST", 6], ["FOCUS", 6], ["REST", 6], ["FOCUS", 6]]
    raw["generator"].update(install_fake_generator(tmp_path / "generator"))
    place_focus_model(tmp_path)
    write_yaml(raw, tmp_path / "config.yaml")
    blocks = [tuple(block) for block in raw["protocol"]]
    signals = synthetic_signals(raw["protocol"])
    write_recording(Path("data") / "simulated" / "short", signals, blocks)
    output = io.StringIO()

    def when_shown(text, key, delay_s=0.0):
        def press():
            wait_until(lambda: text in output.getvalue(), f"{text!r} was never printed", 60)
            time.sleep(delay_s)
            return key

        return press

    keys = ScriptedKeys(
        when_shown("All systems ready.", keyboard.ENTER),  # the dataset
        keyboard.ENTER,  # its recording
        "y", "T", "0", "1", keyboard.ENTER,
        when_shown("READY", keyboard.SPACE),
        when_shown("RUNNING", keyboard.SPACE, delay_s=8),  # stop during the second block
        when_shown("finished", keyboard.ENTER), keyboard.ENTER, "n",
    )  # fmt: skip
    console = ConsoleApp(
        Path("config.yaml"),
        machine=MachineWithoutHeadset(),
        keys=Keyboard(keys),
        output=Console(file=output, width=120, force_terminal=False, highlight=False),
    )
    assert console.run() == 0

    text = output.getvalue()
    assert "All systems ready." in text
    assert "Session T01 · source: simulated / short (dataset)" in text
    assert "(ended by [SPACE])" in text
    assert "Result          OK · signal usable 100 % of the time" in text  # no calibration
    assert "Focus range" in text
    (folder,) = (tmp_path / "sessions").glob("*_T01")
    assert f"  Folder          sessions/{folder.name}" in text

    events = read_lines(folder / c.FILE_EVENTS)
    assert events[-1]["reason"] == REASON_USER
    assert 7 * 256 < events[-1]["n"] < 12 * 256  # about eight seconds of samples
    assert [e["label"] for e in events if e["type"] == "BLOCK_START"] == ["REST", "FOCUS"]
    assert [e["msg"] for e in events if e["type"] == "FUDI"][-2:] == ["bypass 1;", "rec stop;"]
    assert soundfile.info(folder / c.FILE_AUDIO).duration == pytest.approx(8, abs=2)
    assert read_yaml(folder / c.FILE_CONFIG)["protocol"] == raw["protocol"]

    # What led to the session: the start of the generator, as it reported it, and the keys.
    logged = [entry["type"] for entry in read_lines(folder / c.FILE_CONSOLE)]
    started = [kind for kind in logged if kind in ("PD_STARTED", "PD_CONNECTED", "MODEL_LOADED")]
    assert started == ["PD_STARTED", "PD_CONNECTED", "MODEL_LOADED"]
    assert logged.index("MODEL_LOADED") < logged.index("SESSION_STARTED")
    assert logged[-2:] == ["SESSION_ENDED", "KEY"]  # the [SPACE] the session read itself
    assert (folder / c.FILE_METRICS).is_file()
