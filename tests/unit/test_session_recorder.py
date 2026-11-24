"""session_recorder: the files adamus writes into a session folder, and when Pd has finished
the one it writes."""

import datetime
import json
import logging
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile

import adamus
from adamus import constants as c
from adamus.config import load_config
from adamus.realtime import session_recorder
from adamus.realtime.session_recorder import SessionRecorder
from adamus.session_format import read_jsonl, read_log_line

MOMENT_S = 5.0  # a line of the log was written within this of now
STARTED = datetime.datetime(2026, 11, 10, 15, 4, 12).timestamp()


@pytest.fixture
def recorder(tmp_path):
    session = SessionRecorder(tmp_path / "sessions", "S01", STARTED)
    session.open()
    yield session
    session.close()


class TestFolder:
    def test_folder_is_named_by_local_start_time_and_session_name(self, recorder, tmp_path):
        assert recorder.directory == tmp_path / "sessions" / "20261110-150412_S01"
        assert recorder.directory.is_dir()

    def test_missing_parent_folders_are_created(self, tmp_path):
        session = SessionRecorder(tmp_path / "deep" / "er" / "sessions", "S01", STARTED)
        session.close()
        assert session.directory.parent == tmp_path / "deep" / "er" / "sessions"

    def test_existing_folder_is_never_reused(self, recorder, tmp_path):
        with pytest.raises(FileExistsError):
            SessionRecorder(tmp_path / "sessions", "S01", STARTED)

    def test_files_and_log_are_opened_only_when_the_session_runs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(logging.getLogger("adamus"), "level", logging.WARNING)
        handlers = list(logging.getLogger("adamus").handlers)
        session = SessionRecorder(tmp_path / "sessions", "S01", STARTED)
        assert list(session.directory.iterdir()) == []  # a session that never runs opens nothing
        session.close()
        session = SessionRecorder(tmp_path / "sessions", "S02", STARTED)
        level = logging.getLogger("adamus").level
        session.open()
        assert len(logging.getLogger("adamus").handlers) == len(handlers) + 1
        assert sorted(p.name for p in session.directory.iterdir()) == [
            "adamus.log", "events.jsonl", "filtered.f32", "ticks.jsonl",
        ]  # fmt: skip
        assert logging.getLogger("adamus").level == logging.DEBUG  # all of it into adamus.log
        session.close()
        assert logging.getLogger("adamus").handlers == handlers
        assert logging.getLogger("adamus").level == level  # as it was before the session

    def test_paths_of_the_files_written_by_brainflow_and_pd(self, recorder):
        assert recorder.raw_csv == recorder.directory / "raw.csv"
        assert recorder.audio_wav == recorder.directory / "audio.wav"
        assert recorder.pd_log == recorder.directory / "pd.log"


class TestConfigAndMeta:
    def test_config_copy_loads_back_as_the_same_configuration(self, recorder, raw_config):
        recorder.write_config(raw_config)
        assert load_config(recorder.directory / c.FILE_CONFIG).raw == raw_config

    def test_meta_holds_versions_session_facts_and_the_foreign_files(self, recorder):
        import platform
        from importlib.metadata import PackageNotFoundError, version

        recorder.write_meta({"board_id": 38, "playback_path": None})
        meta = json.loads((recorder.directory / c.FILE_META).read_text())
        assert "adamus" not in meta["versions"]  # the commit says which program it was
        assert meta["python"] == platform.python_version()
        versions = meta["versions"]
        assert versions["brainflow"] == version("brainflow") == "5.24.0"
        for library in ("numpy", "scipy", "librosa"):  # they decide the model and the metrics
            assert versions[library] == version(library)
        try:  # the generator's Python package is optional
            magenta_rt = version("magenta-rt")
        except PackageNotFoundError:
            magenta_rt = None
        assert versions["magenta-rt"] == magenta_rt
        assert "adamus_commit" in meta
        assert (meta["board_id"], meta["playback_path"]) == (38, None)


def test_library_that_is_not_installed_has_no_version(recorder, monkeypatch):
    """The generator's Python package is optional: tests and analysis run without it."""

    def only_brainflow(package):
        if package != "brainflow":
            raise session_recorder.metadata.PackageNotFoundError(package)
        return "5.24.0"

    monkeypatch.setattr(session_recorder.metadata, "version", only_brainflow)
    recorder.write_meta({})
    meta = json.loads((recorder.directory / c.FILE_META).read_text())
    versions = meta["versions"]
    assert (versions["brainflow"], versions["numpy"], versions["magenta-rt"]) == (
        "5.24.0",
        None,
        None,
    )


class TestCommit:
    """meta.json names the commit the program ran from, when it runs from a git checkout."""

    def meta(self, recorder):
        recorder.write_meta({})
        return json.loads((recorder.directory / c.FILE_META).read_text())

    def test_commit_is_asked_of_git_in_the_folder_of_the_package(self, recorder, monkeypatch):
        asked = []

        def git(command, **options):
            asked.append((command, options))
            return subprocess.CompletedProcess(command, 0, stdout="0123abc\n", stderr="")

        monkeypatch.setattr(session_recorder.subprocess, "run", git)
        assert self.meta(recorder)["adamus_commit"] == "0123abc"
        ((command, options),) = asked
        assert command == ["git", "-C", str(Path(adamus.__file__).parent), "rev-parse", "HEAD"]
        assert options == {"capture_output": True, "text": True, "check": False}

    @pytest.mark.parametrize("code", [1, 128])
    def test_no_commit_outside_a_git_checkout(self, recorder, monkeypatch, code):
        def git(command, **options):
            return subprocess.CompletedProcess(command, code, stdout="", stderr="fatal: not a git")

        monkeypatch.setattr(session_recorder.subprocess, "run", git)
        assert self.meta(recorder)["adamus_commit"] is None

    def test_no_commit_without_git(self, recorder, monkeypatch):
        def no_git(command, **options):
            raise FileNotFoundError("git")

        monkeypatch.setattr(session_recorder.subprocess, "run", no_git)
        assert self.meta(recorder)["adamus_commit"] is None

    def test_real_git_is_understood(self, recorder):
        git = subprocess.run(
            ["git", "-C", str(Path(adamus.__file__).parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        expected = git.stdout.strip() if git.returncode == 0 else None
        assert self.meta(recorder)["adamus_commit"] == expected

    def test_meta_is_readable_text_with_names_kept_as_typed(self, recorder):
        recorder.write_meta({"playback_path": "data/měření/eeg.csv"})
        text = (recorder.directory / c.FILE_META).read_text(encoding="utf-8")
        assert '\n  "playback_path": "data/měření/eeg.csv",\n' in text and text.endswith("}\n")


class TestEvents:
    def test_line_has_time_sample_count_type_and_fields_in_that_order(self, recorder):
        recorder.event("SESSION_START", 100.5, 0)
        recorder.event("BLOCK_START", 160.25, 15360, index=1, label="FOCUS", marker=2)
        assert (recorder.directory / c.FILE_EVENTS).read_text().splitlines() == [
            '{"t": 100.5, "n": 0, "type": "SESSION_START"}',
            '{"t": 160.25, "n": 15360, "type": "BLOCK_START", "index": 1, "label": "FOCUS", '
            '"marker": 2}',
        ]

    def test_event_is_on_disk_at_once(self, recorder):
        recorder.event("ERROR", 1.0, 0, where="loop")
        assert read_jsonl(recorder.directory / c.FILE_EVENTS) == [
            {"t": 1.0, "n": 0, "type": "ERROR", "where": "loop"}
        ]


class TestTicks:
    def test_line_k_and_filtered_record_k_belong_together(self, recorder):
        windows = [np.full((2, 8), k, dtype=np.float64) + [[0.25], [0.5]] for k in range(3)]
        for k, window in enumerate(windows):
            recorder.tick({"k": k, "E": 0.5}, window)
        assert read_jsonl(recorder.directory / c.FILE_TICKS) == [
            {"k": k, "E": 0.5} for k in range(3)
        ]
        stored = np.fromfile(recorder.directory / c.FILE_FILTERED, dtype="<f4")
        records = stored.reshape(3, 2, 8)  # record k = C x N values, channel after channel
        for k, window in enumerate(windows):
            assert np.array_equal(records[k], window.astype(np.float32))

    def test_filtered_record_is_little_endian_float32(self, recorder):
        recorder.tick({"k": 0}, np.array([[1.0, -2.5]]))
        raw = (recorder.directory / c.FILE_FILTERED).read_bytes()
        assert raw == np.array([1.0, -2.5], dtype="<f4").tobytes() and len(raw) == 8

    def test_every_step_is_on_disk_before_the_next_one(self, recorder):
        """A crash loses no more than the step in progress."""
        for k in range(3):
            recorder.tick({"k": k}, np.zeros((2, 4)))
            assert len(read_jsonl(recorder.directory / c.FILE_TICKS)) == k + 1
            assert (recorder.directory / c.FILE_FILTERED).stat().st_size == (k + 1) * 2 * 4 * 4

    def test_value_that_json_cannot_hold_is_refused(self, recorder):
        with pytest.raises(ValueError, match="Out of range float"):
            recorder.tick({"k": 0, "E": float("nan")}, np.zeros((1, 4)))


class TestLog:
    def test_debug_records_of_adamus_go_to_the_session_log_until_close(self, recorder):
        logging.getLogger("adamus.test").debug("fine detail")
        recorder.close()
        text = (recorder.directory / c.FILE_LOG).read_text()
        assert re.search(
            r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d DEBUG adamus.test: fine detail$",
            text,
            re.M,
        )
        read = read_log_line(text.splitlines()[-1])
        assert read is not None
        written, said = read
        assert abs(written - time.time()) < MOMENT_S and said == "DEBUG adamus.test: fine detail"
        logging.getLogger("adamus.test").debug("after close")
        assert "after close" not in (recorder.directory / c.FILE_LOG).read_text()

    def test_records_of_other_libraries_stay_out(self, recorder):
        logging.getLogger("other").warning("not ours")
        recorder.close()
        assert "not ours" not in (recorder.directory / c.FILE_LOG).read_text()


def test_outside_a_session_no_record_of_adamus_reaches_the_terminal():
    """Python prints a warning that no handler takes to stderr: in the terminal it would break
    into the checklist. A fresh interpreter, as pytest catches every record itself."""
    warn = "import logging, adamus; logging.getLogger('adamus.preflight').warning('lost')"
    run = subprocess.run([sys.executable, "-c", warn], capture_output=True, text=True, check=True)
    assert run.stderr == ""


class TestFinishedRecording:
    """Pd writes the length into the WAV header when it closes the file."""

    def test_a_closed_file_is_finished(self, tmp_path):
        path = tmp_path / "audio.wav"
        soundfile.write(path, np.zeros((480, 2), dtype=np.int16), 48000)
        assert session_recorder.recording_finished(path)

    def test_a_file_still_being_written_is_not(self, tmp_path):
        path = tmp_path / "audio.wav"
        writing = wave.open(str(path), "wb")  # noqa: SIM115 - closed below
        writing.setnchannels(2)
        writing.setsampwidth(2)
        writing.setframerate(48000)
        writing.writeframesraw(bytes(4 * 480))  # the header still counts no frame
        assert not session_recorder.recording_finished(path)
        writing.close()
        assert session_recorder.recording_finished(path)

    @pytest.mark.parametrize("start", [b"", b"RIF", b"RIFX\x00\x00\x00\x00"])
    def test_a_file_without_a_whole_header_is_not(self, tmp_path, start):
        (tmp_path / "audio.wav").write_bytes(start)
        assert not session_recorder.recording_finished(tmp_path / "audio.wav")

    def test_no_file_is_nothing_to_wait_for(self, tmp_path):
        assert session_recorder.recording_finished(tmp_path / "audio.wav")
