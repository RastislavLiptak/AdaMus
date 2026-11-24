"""realtime.music_generator.MusicGenerator against a stand-in for the pd binary.

The stand-in (tests/support/stand_in_pure_data.py) speaks the same protocol and reports
like mrt2~, so everything but Pure Data itself is the real code.
"""

import os
import shutil
import signal
import socket
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
import soundfile

from adamus import constants as c
from adamus.errors import AdamusError
from adamus.realtime import music_generator
from adamus.realtime.generator_messages import (
    fudi,
    session_end_messages,
    session_start_messages,
    step_messages,
)
from adamus.realtime.music_generator import MusicGenerator
from adamus.session_format import read_log_line
from tests.support.finished_sessions import config_with
from tests.support.stand_in_pure_data import OUTPUTS
from tests.support.waiting import wait_until

MILLISECOND_S = 0.001  # the times of a log are cut to it
TV, SPEAKERS = OUTPUTS  # as the stand-in Pd lists them: numbers 0 and 1
WATCH_S = 0.02  # how often the output is looked for, in these tests


class Generators:
    """Builds generators on the stand-in installation, and keeps how far their starts got, as they
    reported it."""

    def __init__(self, stand_in_generator, tmp_path):
        self._stand_in_generator = stand_in_generator
        self.made: list[MusicGenerator] = []
        self.reported: list[dict[str, Any]] = []
        self.printed = tmp_path / "printed_outside_sessions.log"
        self.present = list(OUTPUTS)  # the outputs the Mac has, as the test changes them

    def __call__(self, generator=None, timeouts=None):
        config = config_with(
            generator={**self._stand_in_generator, **(generator or {})},
            timeouts={"pd_connect_s": 10, "model_load_s": 10, **(timeouts or {})},
        )
        made = MusicGenerator(config.generator, config.timeouts, self._report, self._outputs)
        self.made.append(made)
        return made

    def _report(self, kind, /, **fields):
        self.reported.append({"t": time.time(), "type": kind, **fields})

    def _outputs(self):
        if isinstance(self.present, Exception):
            raise self.present
        return list(self.present)


@pytest.fixture
def make_generator(stand_in_generator, tmp_path, monkeypatch):
    """Build generators on the stand-in installation; every one is stopped after the test."""
    monkeypatch.setattr(music_generator, "WATCH_S", WATCH_S)
    generators = Generators(stand_in_generator, tmp_path)
    yield generators
    for made in generators.made:
        made.stop()


def reported(make_generator):
    return make_generator.reported


def said(log_text):
    """What the lines of a pd.log say, without the times they begin with."""
    read = [read_log_line(line) for line in log_text.splitlines()]
    return "".join(said + "\n" for _, said in filter(None, read))


def pd_log(make_generator):
    """What Pd has printed outside sessions: a pretended session takes it, as a real one would."""
    generator = make_generator.made[-1]
    generator.copy_output_to(make_generator.printed)
    generator.copy_output_to(None)
    return said(make_generator.printed.read_text())


def wait_for(condition, seconds=5):
    wait_until(condition, "the expected state was not reached", seconds)


class TestStart:
    def test_command_line_of_pd(self, make_generator, stand_in_generator):
        generator = make_generator()
        assert generator.command == []
        generator.start(SPEAKERS)
        folder = stand_in_generator["folder"]
        patch = generator.command[-1]
        assert generator.command == [
            f"{folder}/Pd.app/Contents/Resources/bin/pd",
            "-nogui", "-stderr", "-noprefs", "-noadc", "-audiooutdev", "1", "-audiobuf", "10",
            "-r", "48000",
            "-path", f"{folder}/externals/mrt2~",
            "-send", "pd dsp 1",
            patch,
        ]  # fmt: skip

    def test_patch_is_a_temporary_file_that_is_gone_when_pd_is_stopped(self, make_generator):
        generator = make_generator()
        generator.start(SPEAKERS)
        patch = Path(generator.command[-1])
        assert patch.parent.resolve() == Path(tempfile.gettempdir()).resolve()
        assert patch.name.startswith("adamus_") and patch.suffix == ".pd" and patch.is_file()
        generator.stop()
        assert not patch.exists()

    def test_patch_of_a_start_that_failed_is_gone_too(self, make_generator, monkeypatch):
        monkeypatch.setenv("STAND_IN_PD_DIES_ON_MODEL", "1")
        generator = make_generator()
        with pytest.raises(AdamusError):
            generator.start(SPEAKERS)
        assert not Path(generator.command[-1]).exists()

    def test_patch_pd_runs_listens_on_a_port_the_system_found_free(self, make_generator):
        """A port fixed in advance may be used by another program (3000 by many servers)."""
        generator = make_generator()
        generator.start(SPEAKERS)
        patch = Path(generator.command[-1]).read_text()
        assert generator.port > 0
        listens = f"#X msg 290 180 listen {generator.port} {c.LOCAL_HOST};"  # this computer only
        assert listens in patch
        assert "{port}" not in patch and "{host}" not in patch
        assert patch.count("\n") == music_generator.PATCH_TEMPLATE.read_text().count("\n")

    def test_start_loads_assets_then_model_and_logs_its_progress(self, make_generator, tmp_path):
        generator = make_generator()
        generator.start(SPEAKERS)
        entries = reported(make_generator)
        assert [entry["type"] for entry in entries] == [
            "PD_STARTED",
            "PD_CONNECTED",
            "MODEL_LOADED",
        ]
        assert entries[0]["pid"] > 0 and entries[0]["output"] == SPEAKERS and generator.running
        log = pd_log(make_generator)
        assets = log.index(f"mrt2~: loading assets from {generator.config.assets.resolve()}")
        model = log.index(f"mrt2~: loading model {generator.config.model_file.resolve()}")
        assert (
            assets < log.index("mrt2~: assets loaded.") < model < log.index("mrt2~: model loaded.")
        )

    def test_generator_is_told_to_keep_as_much_audio_ready_as_it_can(self, make_generator):
        """With less, every moment the computer is busy elsewhere is a gap in the music."""
        make_generator().start(SPEAKERS)
        printed = pd_log(make_generator).splitlines()
        buffer = printed.index("mrt2~: buffer size = 8192 samples (170.7 ms @ 48 kHz)")
        assert printed[buffer - 1] == "mrt2~: model loaded."
        assert music_generator.AUDIO_BUFFER_SAMPLES == 8192  # the capacity of the ring of mrt2~
        assert music_generator.AUDIO_BUFFER_SAMPLES > c.MODEL_FRAME_S * c.AUDIO_RATE

    def test_pd_version_is_read_from_the_binary(self, make_generator):
        generator = make_generator()
        assert generator.pd_version is None
        generator.start(SPEAKERS)
        assert generator.pd_version == "Pd-0.57-0 (stand-in)"

    def test_a_pd_that_cannot_tell_its_version_still_runs(self, make_generator, monkeypatch):
        """The version only goes into meta.json; it is no reason to give up a working Pd."""
        monkeypatch.setenv("STAND_IN_PD_NO_VERSION", "1")
        generator = make_generator()
        generator.start(SPEAKERS)
        assert generator.running and generator.pd_version is None

    def test_the_generator_is_silent_before_it_has_a_model(self, make_generator, monkeypatch):
        """mrt2~ plays as soon as it has a model: measured with Pd and its output written to
        a file, it played for as long as ``bypass 1`` came after the model, and not at all when
        it came before."""
        generator = make_generator()
        sent = []
        send = generator.send
        monkeypatch.setattr(generator, "send", lambda text: (sent.append(text), send(text)))
        generator.start(SPEAKERS)
        heads = [message.split()[0] for message in sent]
        assert heads.index("bypass") < heads.index("model")
        assert sent[heads.index("bypass")] == fudi("bypass", 1)

    def test_prompts_a_session_starts_with_are_encoded_when_the_model_is_loaded(
        self, make_generator, monkeypatch, tmp_path
    ):
        """mrt2~ encodes a prompt in the background: sent only when a session began, the
        first 1.5-1.9 s of three sessions played its built-in piano style instead."""
        generator = make_generator()
        sent = []
        send = generator.send
        monkeypatch.setattr(generator, "send", lambda text: (sent.append(text), send(text)))
        generator.start(SPEAKERS)
        heads = [message.split()[0] for message in sent]
        assert heads.index("model") < heads.index("prompt")
        session_prompts = [
            message
            for message in session_start_messages(generator.config, tmp_path / c.FILE_AUDIO)
            if message.startswith("prompt")
        ]
        assert [message for message in sent if message.startswith("prompt")] == session_prompts

    def test_a_start_ends_once_the_music_of_each_prompt_is_ready(self, make_generator, monkeypatch):
        """Measured with Pd: a session started at once after the engine said so was quiet from
        its first half second (-43 dB, against -22 dB without the wait)."""
        generator = make_generator()
        prompts_sent = []
        send = generator.send

        def noted(text):
            if text.startswith("prompt"):
                prompts_sent.append(time.time())
            send(text)

        monkeypatch.setattr(generator, "send", noted)
        generator.start(SPEAKERS)
        ended = time.time()
        generator.copy_output_to(make_generator.printed)
        generator.copy_output_to(None)
        lines = filter(None, map(read_log_line, make_generator.printed.read_text().splitlines()))
        ready = [t for t, said in lines if said.startswith(music_generator.PROMPTS_READY)]
        assert len(prompts_sent) == len(ready) == 2
        cut = MILLISECOND_S  # the times of the log are cut to milliseconds
        assert prompts_sent[0] <= ready[0] + cut and ready[0] <= prompts_sent[1] + cut
        assert prompts_sent[1] <= ready[1] + cut <= ended + cut

    def test_music_of_the_prompts_that_never_gets_ready_fails_the_start(
        self, make_generator, monkeypatch
    ):
        monkeypatch.setenv("STAND_IN_PD_SILENT_PROMPTS", "1")
        generator = make_generator(timeouts={"model_load_s": 0.5})
        with pytest.raises(AdamusError) as error:
            generator.start(SPEAKERS)
        assert error.value.args[0] == (
            "mrt2~ did not make the music of the prompts ready within 0.5 s"
        )
        assert not generator.running

    def test_generator_sleeps_after_start(self, make_generator, tmp_path):
        """No audio until a session begins."""
        generator = make_generator()
        generator.start(SPEAKERS)
        folder = tmp_path / "S"
        folder.mkdir()
        generator.send(fudi("rec", "open", folder / "a.wav") + fudi("rec", "start"))
        time.sleep(0.5)
        generator.send(fudi("rec", "stop"))
        wait_for(lambda: (folder / "a.wav").stat().st_size > 44)
        audio, _ = soundfile.read(folder / "a.wav")
        assert not audio.any()

    def test_harmless_lines_of_mrt2_do_not_stop_the_start(self, make_generator, tmp_path):
        """mrt2~ complains about its default assets and model when the object is created."""
        generator = make_generator()
        generator.start(SPEAKERS)
        assert "failed to init assets from default path" in pd_log(make_generator)
        assert generator.running

    def test_missing_model_is_reported_in_the_words_of_the_generator(self, make_generator):
        generator = make_generator(generator={"model": "absent"})
        with pytest.raises(
            AdamusError, match="the generator reported: mrt2~: failed to load model"
        ):
            generator.start(SPEAKERS)
        assert not generator.running

    def test_missing_assets_are_reported(self, make_generator):
        generator = make_generator()
        shutil.rmtree(generator.config.assets)
        with pytest.raises(
            AdamusError, match="the generator reported: mrt2~: failed to init assets"
        ):
            generator.start(SPEAKERS)
        assert not generator.running

    def test_model_loaded_before_it_was_asked_for_does_not_count(self, make_generator, monkeypatch):
        """mrt2~ loads its default model on creation, if one is installed."""
        monkeypatch.setenv("STAND_IN_PD_DEFAULT_MODEL", "1")
        monkeypatch.setenv("STAND_IN_PD_SILENT_MODEL", "1")  # the requested model never loads
        generator = make_generator(timeouts={"model_load_s": 1.5})
        started = time.monotonic()
        with pytest.raises(AdamusError, match="mrt2~ did not load the model within 1.5 s"):
            generator.start(SPEAKERS)
        assert time.monotonic() - started >= 1.4

    def test_assets_and_model_share_one_time_limit(self, make_generator, monkeypatch):
        """Counted from the connection to Pd, however long Pd took to start."""
        monkeypatch.setenv("STAND_IN_PD_ASSETS_S", "1.0")
        monkeypatch.setenv("STAND_IN_PD_SILENT_MODEL", "1")
        generator = make_generator(timeouts={"model_load_s": 1.5})
        gave_up, stop = [], generator.stop

        def noted_stop():  # the start gives up by stopping Pd; that takes a while itself
            gave_up.append(time.time())
            stop()

        generator.stop = noted_stop
        with pytest.raises(AdamusError, match="mrt2~ did not load the model within 1.5 s"):
            generator.start(SPEAKERS)
        given_up = gave_up[0]
        (connected,) = [e["t"] for e in reported(make_generator)
                        if e["type"] == music_generator.REPORT_PD_CONNECTED]  # fmt: skip
        assert 1.4 <= given_up - connected < 2.0  # not 1.0 s + 1.5 s

    def test_assets_that_take_too_long(self, make_generator, monkeypatch):
        monkeypatch.setenv("STAND_IN_PD_ASSETS_S", "3")
        generator = make_generator(timeouts={"model_load_s": 0.5})
        with pytest.raises(AdamusError) as error:
            generator.start(SPEAKERS)
        assert error.value.args[0] == "mrt2~ did not load the assets within 0.5 s"
        assert not generator.running

    def test_pd_that_dies_while_it_loads_the_model(self, make_generator, monkeypatch):
        monkeypatch.setenv("STAND_IN_PD_DIES_ON_MODEL", "1")
        generator = make_generator()
        started = time.monotonic()
        with pytest.raises(AdamusError) as error:
            generator.start(SPEAKERS)
        assert error.value.args[0] == "Pd exited before it could load the model"
        assert time.monotonic() - started < 5  # noticed at once, not after the time limit

    def test_start_that_fails_reports_what_pd_printed(self, make_generator):
        """There is no session yet whose pd.log could hold it."""
        generator = make_generator(generator={"model": "absent"})
        with pytest.raises(AdamusError) as error:
            generator.start(SPEAKERS)
        cause, heading, *printed = error.value.args
        assert cause == "the generator reported: mrt2~: failed to load model"
        assert heading == "Pure Data printed:"
        assert printed[0] == "  error: mrt2~: failed to init assets from default path"
        assert "  mrt2~: assets loaded." in printed
        assert "  mrt2~: failed to load model" in printed
        assert str(error.value).splitlines() == [cause, heading, *printed]

    def test_ctrl_c_while_pd_is_starting_leaves_no_pd_behind(self, make_generator, monkeypatch):
        monkeypatch.setenv("STAND_IN_PD_SILENT_MODEL", "1")  # the start waits for the model

        def interrupted(seconds):
            monkeypatch.undo()  # once; stopping Pd sleeps too
            raise KeyboardInterrupt

        generator = make_generator()
        monkeypatch.setattr(music_generator.time, "sleep", interrupted)
        with pytest.raises(KeyboardInterrupt):
            generator.start(SPEAKERS)
        pid = reported(make_generator)[0]["pid"]
        assert not generator.running and not Path(generator.command[-1]).exists()
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    def test_start_that_fails_before_pd_printed_anything_reports_the_cause_alone(
        self, make_generator
    ):
        generator = make_generator(generator={"folder": "nowhere"})
        with pytest.raises(AdamusError) as error:
            generator.start(SPEAKERS)
        assert len(error.value.args) == 1

    def test_requested_model_is_awaited_after_a_default_one(
        self, make_generator, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("STAND_IN_PD_DEFAULT_MODEL", "1")
        make_generator().start(SPEAKERS)
        assert pd_log(make_generator).count("mrt2~: model loaded.") == 2
        assert reported(make_generator)[-1]["type"] == "MODEL_LOADED"

    def test_pd_that_does_not_listen(self, make_generator, monkeypatch):
        monkeypatch.setenv("STAND_IN_PD_NO_LISTEN", "1")
        generator = make_generator(timeouts={"pd_connect_s": 1.5})
        started = time.monotonic()
        with pytest.raises(AdamusError) as error:
            generator.start(SPEAKERS)
        port = generator.port
        assert error.value.args[0] == f"cannot connect to Pd on port {port} within 1.5 s"
        assert 1.4 <= time.monotonic() - started < 5
        assert not generator.running

    def test_a_port_another_program_listens_on_is_not_chosen(self, make_generator):
        with socket.socket() as other_program:
            other_program.bind((c.LOCAL_HOST, 0))
            other_program.listen()
            taken = other_program.getsockname()[1]
            assert music_generator.free_port() != taken
            generator = make_generator()
            generator.start(SPEAKERS)
            assert generator.port != taken

    def test_missing_pd_binary(self, make_generator):
        generator = make_generator(generator={"folder": "nowhere"})
        with pytest.raises(AdamusError) as error:
            generator.start(SPEAKERS)
        binary = Path("nowhere").resolve() / c.GENERATOR_PD_BINARY
        assert str(error.value) == (
            f"cannot start Pd ({binary}): [Errno 2] No such file or directory: '{binary}'"
        )

    def test_what_pd_prints_stays_off_the_screen_and_odd_bytes_do_not_stop_it(
        self, make_generator, monkeypatch, tmp_path, capfd
    ):
        monkeypatch.setenv("STAND_IN_PD_CHATTY", "1")
        generator = make_generator()
        generator.start(SPEAKERS)
        generator.stop()
        captured = capfd.readouterr()
        assert (captured.out, captured.err) == ("", "")
        assert "pd: not text \ufffd\ufffd here\n" in pd_log(make_generator)
        assert "mrt2~: model loaded." in pd_log(make_generator)

    def test_pd_that_exits_at_once_fails_without_waiting_for_the_timeout(self, make_generator):
        generator = make_generator()
        listing = f"audio output devices:\\n0. Core Audio: {SPEAKERS}\\n"
        generator.config.pd_binary.write_text(
            f'#!/bin/sh\ncase "$*" in *-listdev*) printf "{listing}" >&2 ;; esac\nexit 3\n'
        )
        started = time.monotonic()
        with pytest.raises(AdamusError, match="^Pd exited before accepting a connection$"):
            generator.start(SPEAKERS)
        assert time.monotonic() - started < 5

    def test_pd_runs_in_its_own_session_out_of_reach_of_a_terminal_ctrl_c(
        self, make_generator, tmp_path
    ):
        make_generator().start(SPEAKERS)
        pid = reported(make_generator)[0]["pid"]
        assert os.getsid(pid) == pid != os.getsid(0)


class TestAudioOutput:
    """Pd plays on the output it is given from its start to its end; another one starts it again."""

    def test_pd_lists_the_outputs_it_can_open_in_the_order_it_numbers_them(self, make_generator):
        generator = make_generator()
        assert music_generator.pd_outputs(generator.config.pd_binary, 10) == list(OUTPUTS)

    def test_pd_that_cannot_list_the_outputs(self, tmp_path):
        with pytest.raises(AdamusError) as error:
            music_generator.pd_outputs(tmp_path / "no_pd", 10)
        assert str(error.value).startswith(f"cannot start Pd ({tmp_path / 'no_pd'}): ")

    def test_pd_that_takes_too_long_to_list_them(self, tmp_path):
        slow = tmp_path / "slow_pd"
        slow.write_text("#!/bin/sh\nsleep 5\n")
        slow.chmod(0o755)
        with pytest.raises(AdamusError) as error:
            music_generator.pd_outputs(slow, 0.1)
        assert str(error.value) == "Pd did not list the audio outputs within 0.1 s"

    @pytest.mark.parametrize(("output", "number"), [(TV, "0"), (SPEAKERS, "1")])
    def test_pd_is_given_the_number_of_the_output_in_its_own_list(
        self, make_generator, output, number
    ):
        generator = make_generator()
        generator.start(output)
        given = generator.command.index("-audiooutdev") + 1
        assert generator.command[given] == number and generator.output == output

    def test_an_output_pd_does_not_list_is_refused_before_pd_starts(self, make_generator):
        generator = make_generator()
        with pytest.raises(AdamusError) as error:
            generator.start("Headphones")
        assert str(error.value) == "Headphones is not connected"
        assert not generator.running and reported(make_generator) == []

    def test_an_output_pd_could_not_open_fails_the_start(self, make_generator, monkeypatch):
        monkeypatch.setenv("STAND_IN_PD_CANNOT_PLAY", "1")
        generator = make_generator()
        with pytest.raises(AdamusError) as error:
            generator.start(SPEAKERS)
        assert str(error.value).splitlines()[0] == (
            f"Pd could not play on {SPEAKERS}: error: audio output device number (1) out of range"
        )
        assert not generator.running

    def test_another_output_starts_pd_again_and_reports_only_what_that_start_printed(
        self, make_generator, monkeypatch
    ):
        generator = make_generator()
        generator.start(SPEAKERS)
        generator.stop()
        monkeypatch.setenv("STAND_IN_PD_SILENT_MODEL", "1")
        with pytest.raises(AdamusError) as error:
            make_generator(timeouts={"model_load_s": 0.5}).start(TV)
        assert str(error.value).count("mrt2~: model loaded.") == 0
        monkeypatch.delenv("STAND_IN_PD_SILENT_MODEL")
        generator.start(TV)
        assert generator.running and generator.output == TV
        assert [e["output"] for e in reported(make_generator) if e["type"] == "PD_STARTED"] == [
            SPEAKERS, TV, TV,
        ]  # fmt: skip

    def test_an_output_that_goes_from_the_mac_is_reported_once(self, make_generator):
        generator = make_generator()
        generator.start(SPEAKERS)
        time.sleep(WATCH_S * 3)
        assert not generator.output_lost
        make_generator.present.remove(SPEAKERS)
        wait_for(lambda: generator.output_lost)
        time.sleep(WATCH_S * 3)
        lost = [e for e in reported(make_generator) if e["type"] == "AUDIO_OUTPUT_LOST"]
        assert [e["output"] for e in lost] == [SPEAKERS]

    def test_macos_that_cannot_tell_its_outputs_is_asked_again(self, make_generator):
        generator = make_generator()
        make_generator.present = AdamusError("macOS did not tell the audio devices")
        generator.start(SPEAKERS)
        time.sleep(WATCH_S * 3)
        assert not generator.output_lost
        make_generator.present = [TV]
        wait_for(lambda: generator.output_lost)

    def test_a_start_on_an_output_that_is_back_is_not_lost(self, make_generator):
        generator = make_generator()
        generator.start(SPEAKERS)
        make_generator.present = [TV]
        wait_for(lambda: generator.output_lost)
        make_generator.present = list(OUTPUTS)
        generator.stop()
        generator.start(SPEAKERS)
        time.sleep(WATCH_S * 3)
        assert not generator.output_lost

    def test_stop_ends_the_watch(self, make_generator):
        generator = make_generator()
        generator.start(SPEAKERS)
        generator.stop()
        make_generator.present = []
        time.sleep(WATCH_S * 3)
        assert not generator.output_lost


class TestSession:
    def record(self, generator, folder, seconds=0.6, weights=None):
        """Send what a session sends: start, optionally new weights, end."""
        folder.mkdir(parents=True, exist_ok=True)
        generator.copy_output_to(folder / c.FILE_PD_LOG)
        for message in session_start_messages(generator.config, folder / c.FILE_AUDIO):
            generator.send(message)
        time.sleep(seconds)
        if weights:
            generator.send(" ".join(step_messages(generator.config, *weights)))
            time.sleep(seconds)
        for message in session_end_messages():
            generator.send(message)
        wait_for(
            lambda: (
                (folder / c.FILE_AUDIO).exists()
                and "state reset" in (folder / c.FILE_PD_LOG).read_text()
            )
        )
        time.sleep(0.2)
        generator.copy_output_to(None)

    def test_audio_is_recorded_into_a_folder_with_spaces_and_other_odd_characters(
        self, make_generator, tmp_path
    ):
        generator = make_generator()
        generator.start(SPEAKERS)
        folder = tmp_path / "My Music; take $1, new"
        self.record(generator, folder)
        assert soundfile.info(folder / c.FILE_AUDIO).duration > 0.3

    def test_audio_is_recorded_between_rec_start_and_rec_stop(self, make_generator, tmp_path):
        generator = make_generator()
        generator.start(SPEAKERS)
        self.record(generator, tmp_path / "S01", weights=(0.25, 0.75))
        info = soundfile.info(tmp_path / "S01" / c.FILE_AUDIO)
        assert (info.samplerate, info.channels, info.subtype) == (48000, 2, "PCM_16")
        assert info.duration == pytest.approx(1.2, abs=0.4)

    def test_a_session_records_what_the_generator_plays(self, make_generator, tmp_path):
        generator = make_generator()
        generator.start(SPEAKERS)
        self.record(generator, tmp_path / "S01")
        audio, _ = soundfile.read(tmp_path / "S01" / c.FILE_AUDIO)
        assert audio.any()

    def test_every_line_of_the_session_log_begins_with_the_moment_pd_printed_it(
        self, make_generator, tmp_path
    ):
        """Also what Pd printed before the session took it: the timeline shows it then."""
        before = time.time()
        generator = make_generator()
        generator.start(SPEAKERS)
        time.sleep(0.3)
        taken = time.time()
        self.record(generator, tmp_path / "A")
        after = time.time()
        lines = (tmp_path / "A" / c.FILE_PD_LOG).read_text().splitlines()
        moments = [moment for moment, _ in filter(None, map(read_log_line, lines))]
        assert len(moments) == len(lines)
        assert moments == sorted(moments)
        assert before - MILLISECOND_S <= moments[0] < taken  # while Pd started
        assert moments[-1] <= after

    def test_session_log_holds_what_pd_printed_since_the_session_before(
        self, make_generator, tmp_path
    ):
        """The first session also holds how Pd started; nothing Pd prints is left out."""
        generator = make_generator()
        generator.start(SPEAKERS)
        self.record(generator, tmp_path / "A")
        generator.send("reset;")  # between the sessions; the stand-in answers with a line
        time.sleep(0.3)
        self.record(generator, tmp_path / "B")
        first = (tmp_path / "A" / c.FILE_PD_LOG).read_text()
        second = (tmp_path / "B" / c.FILE_PD_LOG).read_text()
        assert first.index("mrt2~: model loaded.") < first.index("mrt2~: state reset.")
        assert first.count("mrt2~: state reset.") == 1
        assert second.count("mrt2~: state reset.") == 2 and "model loaded" not in second
        assert pd_log(make_generator) == ""  # nothing is left waiting

    def test_one_generator_serves_many_sessions(self, make_generator, tmp_path):
        generator = make_generator()
        generator.start(SPEAKERS)
        for name in ("A", "B", "C"):
            self.record(generator, tmp_path / name, seconds=0.3)
            assert soundfile.info(tmp_path / name / c.FILE_AUDIO).frames > 0
        assert generator.running
        assert [e["type"] for e in reported(make_generator)].count("PD_STARTED") == 1


class TestStop:
    def test_stop_ends_pd(self, make_generator, tmp_path):
        generator = make_generator()
        generator.start(SPEAKERS)
        pid = reported(make_generator)[0]["pid"]
        generator.stop()
        assert not generator.running
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert pd_log(make_generator).splitlines()[-1] in (
            "stand-in pd: quit, the connection closed",
            "stand-in pd: ended by SIGTERM",
        )

    def test_what_pd_prints_while_it_quits_is_all_kept(self, make_generator, monkeypatch, tmp_path):
        """Many lines, so that they are still being copied when Pd has ended."""
        monkeypatch.setenv("STAND_IN_PD_LAST_WORDS", "3000")
        generator = make_generator()
        generator.start(SPEAKERS)
        generator.copy_output_to(tmp_path / "pd.log")
        generator.stop()
        last = said((tmp_path / "pd.log").read_text()).splitlines()[-1]
        assert last in ("stand-in pd: quit, the connection closed", "stand-in pd: ended by SIGTERM")

    def test_a_patch_removed_by_someone_else_does_not_keep_pd_running(self, make_generator):
        """The system may clear its folder of temporary files while a long run goes on."""
        generator = make_generator()
        generator.start(SPEAKERS)
        Path(generator.command[-1]).unlink()
        generator.stop()
        assert not generator.running

    def test_stop_ends_a_pd_that_was_never_connected_with_sigterm(
        self, make_generator, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("STAND_IN_PD_NO_LISTEN", "1")
        generator = make_generator(timeouts={"pd_connect_s": 0.5})
        with pytest.raises(AdamusError, match="^cannot connect to Pd"):
            generator.start(SPEAKERS)  # ends with stop()
        with pytest.raises(ProcessLookupError):
            os.kill(reported(make_generator)[0]["pid"], 0)
        assert pd_log(make_generator).splitlines()[-1] == "stand-in pd: ended by SIGTERM"

    def test_pd_that_hangs_is_killed(self, make_generator, tmp_path, monkeypatch):
        monkeypatch.setenv("STAND_IN_PD_HANGS", "1")
        monkeypatch.setattr(c, "STOP_WAIT_S", 0.5)
        generator = make_generator()
        generator.start(SPEAKERS)
        pid = reported(make_generator)[0]["pid"]
        started = time.monotonic()
        generator.stop()
        assert time.monotonic() - started >= 0.5  # it was given its time to end
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert pd_log(make_generator).splitlines()[-1] == "stand-in pd: ignoring SIGTERM"

    def test_pd_that_died_is_noticed_when_a_message_is_sent(self, make_generator, tmp_path):
        generator = make_generator()
        generator.start(SPEAKERS)
        os.kill(reported(make_generator)[0]["pid"], signal.SIGKILL)
        wait_for(lambda: not generator.running)
        with pytest.raises(AdamusError, match=r"^cannot send to Pd: \[Errno \d+\] .+$"):
            for _ in range(50):
                generator.send("reset;")
                time.sleep(0.02)

    def test_pd_quits_when_the_connection_closes(self, make_generator, tmp_path):
        """As the patch tells the real Pd: a program that is gone leaves no Pd behind."""
        generator = make_generator()
        generator.start(SPEAKERS)
        generator._socket.close()
        wait_for(lambda: not generator.running)
        assert pd_log(make_generator).splitlines()[-1] == "stand-in pd: quit, the connection closed"

    def test_stop_may_be_called_before_start_and_twice(self, make_generator):
        generator = make_generator()
        generator.stop()
        generator.start(SPEAKERS)
        generator.stop()
        generator.stop()
        assert not generator.running

    def test_sending_needs_a_running_generator(self, make_generator):
        generator = make_generator()
        with pytest.raises(AdamusError, match="^the generator is not running$"):
            generator.send("reset;")
        generator.start(SPEAKERS)
        generator.stop()
        with pytest.raises(AdamusError, match="^the generator is not running$"):
            generator.send("reset;")
