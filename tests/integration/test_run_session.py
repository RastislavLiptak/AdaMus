"""realtime.session.run_session and real_devices: a session started as the program starts it."""

# A test reaches into a session where nothing else shows what it checks: the clock a played
# recording keeps time by, and the record of a session that could not be run.
# pyright: reportPrivateUsage=false

import logging
import threading
import time
from pathlib import Path

import pytest

from adamus.realtime import session as session_module
from adamus.realtime.eeg_source import HeadsetSource, RecordingSource
from adamus.session_format import REASON_END_OF_STREAM
from tests.support.finished_sessions import (
    config_with,
    synthetic_matrix,
    synthetic_signals,
)
from tests.support.short_sessions import SHORT, short_session
from tests.support.stand_in_devices import (
    StandInClock,
    StandInMusic,
    StandInSource,
    StopWhen,
)
from tests.support.synthetic_dataset import write_recording


class TestRealDevices:
    """What a session runs on outside the tests."""

    def test_recording_is_played_at_the_rate_of_the_headset_on_the_clock_of_the_session(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        write_recording(Path("rec"), synthetic_signals(SHORT), [("REST", 40)])

        def no_headset(board):
            raise AssertionError("a recording is played: the headset is not opened")

        monkeypatch.setattr(session_module, "HeadsetSource", no_headset)
        music, stop = StandInMusic(StandInClock()), threading.Event()
        devices = session_module.real_devices(config_with(), Path("rec"), music, stop)
        assert devices.music is music and devices.stop is stop
        assert isinstance(devices.clock, session_module.Clock)
        source = devices.source
        assert isinstance(source, RecordingSource) and source._clock is devices.clock
        assert (source.board_id, source.fs) == (38, 256)
        # The record of the session names the file by its whole path.
        assert source.playback_file == (tmp_path / "rec" / "eeg.csv").resolve()
        assert source.playback_file.is_absolute()

    def test_without_a_recording_the_headset_of_the_configuration_is_read(self):
        muse_s = {"name": "Muse S", "id": 39}
        devices = session_module.real_devices(
            config_with(board=muse_s), None, StandInMusic(StandInClock()), threading.Event()
        )
        source = devices.source
        assert isinstance(source, HeadsetSource)
        assert source.playback_file is None
        assert (source.board_id, source.fs) == (39, 256)

    def test_clock_is_the_time_of_the_computer(self):
        clock = session_module.Clock()
        before = time.time()
        now = clock.now()
        clock.sleep(0.05)
        assert before <= now <= time.time() - 0.05
        assert clock.now() - now >= 0.05

    def test_durations_are_measured_on_the_monotonic_clock(self):
        clock = session_module.Clock()
        before = time.monotonic()
        started = clock.monotonic()
        clock.sleep(0.05)
        assert before <= started <= time.monotonic() - 0.05
        assert clock.monotonic() - started >= 0.05


class TestRunSession:
    def devices(self, matrix):
        self.source = StandInSource(matrix)
        clock = StandInClock()
        return session_module.Devices(self.source, StandInMusic(clock), StopWhen(), clock)

    def test_folder_is_given_before_the_session_runs_and_the_source_is_released(self, tmp_path):
        given = []

        def created(folder):
            given.append((folder, sorted(path.name for path in folder.iterdir())))

        devices = self.devices(synthetic_matrix(SHORT))
        result = session_module.run_session(
            config_with(SHORT), "S01", devices, tmp_path / "sessions", created
        )
        assert result.reason == REASON_END_OF_STREAM
        assert result.directory.parent == (tmp_path / "sessions").resolve()
        assert given == [(result.directory, [])]  # before anything was recorded
        assert self.source.closed

    def test_source_is_released_when_the_session_cannot_even_be_created(self, tmp_path):
        """E.g. two sessions of the same name started within one second."""
        given = []
        arguments = (config_with(SHORT), "S01")
        first = session_module.Session(*arguments, self.devices(synthetic_matrix(SHORT)), tmp_path)
        first._recorder.close()
        devices = self.devices(synthetic_matrix(SHORT))
        with pytest.raises(FileExistsError):
            session_module.run_session(*arguments, devices, tmp_path, given.append)
        assert self.source.closed
        assert given == []  # there is no folder of this session

    def test_a_session_that_never_ran_leaves_no_file_and_no_log_open(self, tmp_path):
        """E.g. a full disk as the activity log moves into the new folder."""

        def full_disk(folder):
            raise OSError("no space left")

        handlers = list(logging.getLogger("adamus").handlers)
        devices = self.devices(synthetic_matrix(SHORT))
        with pytest.raises(OSError, match="no space left"):
            session_module.run_session(config_with(SHORT), "S01", devices, tmp_path, full_disk)
        assert logging.getLogger("adamus").handlers == handlers
        assert self.source.closed
        (folder,) = tmp_path.iterdir()
        assert sorted(path.name for path in folder.iterdir()) == []  # nothing was recorded

    def test_folder_is_named_by_the_time_the_session_was_created(self, tmp_path):
        session = short_session(tmp_path)
        created = time.strftime(
            "%Y%m%d-%H%M%S", time.localtime(1000.0)
        )  # the stand-in clock's time
        assert session.directory.name == f"{created}_S01"
