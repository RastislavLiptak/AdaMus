"""A session with the headset, stored as a recording of the data folder: the blocks it holds,
what is stored, and that it plays again as the session ran."""

import numpy as np
import pytest

from adamus import constants as c
from adamus.config import Block, read_yaml
from adamus.console.data_sources import session_config
from adamus.datasets import electrodes_of_recording
from adamus.errors import AdamusError
from adamus.own_recordings import OWN_RECORDINGS, recorded_blocks, store_recording
from adamus.session_files import REASON_USER
from tests.support.console_runs import LIVE_PROTOCOL
from tests.support.finished_sessions import FS, FinishedSession, config_with, synthetic_matrix
from tests.support.stand_in_devices import FakeSource

# The protocols of the sessions: [SPACE] in the second FOCUS block, at 114 s; or after free
# play has begun, at 130 s, as in the configuration of the project, which ends with free play.
WITH_FREE_PLAY = [*LIVE_PROTOCOL, ["FREE", 0]]
SESSIONS = {"stopped in a block": (LIVE_PROTOCOL, 220, 0), "free play": (WITH_FREE_PLAY, 252, 20)}


@pytest.fixture(scope="module", params=list(SESSIONS))
def live_and_again(tmp_path_factory, request):
    protocol, stopped_at_step, free_play_s = SESSIONS[request.param]
    root = tmp_path_factory.mktemp("own")
    config = config_with(protocol)
    source = FakeSource(synthetic_matrix(protocol, tail_s=free_play_s), live=True)
    live = FinishedSession(root, config, source=source, stop_after_steps=stopped_at_step)
    stored = store_recording(live.directory, root / "data")
    again = FinishedSession(root, session_config(config, stored), recording=stored, name="S02")
    return live, again, stored


def test_the_last_block_of_the_recording_ends_where_the_session_was_stopped(live_and_again):
    live, again, stored = live_and_again
    assert live.result.reason == REASON_USER
    samples = len(np.loadtxt(stored / c.FILE_DATASET_EEG, delimiter=",", skiprows=1))
    protocol = read_yaml(stored / c.FILE_DATASET_PROTOCOL)["protocol"]
    timed = [["REST", 64], ["FOCUS", 20], ["REST", 20]]
    if protocol[-1][0] == "FREE":  # free play lasts as long as it was played
        assert protocol == [*timed, ["FOCUS", 20], ["FREE", samples / FS - 124]]
        assert 0 < samples / FS - 124 < 20
    else:
        assert protocol == [*timed, ["FOCUS", samples / FS - 104]]
        assert 0 < samples / FS - 104 < 20


def test_every_step_comes_again_in_its_block(live_and_again):
    live, again, _stored = live_and_again
    keys = ("k", "n_end", "block_index", "block_label", "full_window", "valid")
    assert [[t[key] for key in keys] for t in again.ticks] == [
        [t[key] for key in keys] for t in live.ticks
    ]


def test_the_focus_is_that_of_the_session(live_and_again):
    """The recording keeps microvolts to three decimals; f hardly notices."""
    live, again, _stored = live_and_again
    assert [t["f"] for t in again.ticks] == pytest.approx([t["f"] for t in live.ticks], abs=1e-3)
    assert any(t["f"] is not None and t["f"] > 0.5 for t in live.ticks)  # it did switch on


class TestRecordedBlocks:
    """The blocks a recording holds: those of the protocol, as far as the samples reach."""

    PROTOCOL = [Block("REST", 5.0), Block("FOCUS", 5.0), Block("REST", 5.0)]

    def test_all_of_them(self):
        assert recorded_blocks(self.PROTOCOL, FS, 15 * FS) == self.PROTOCOL

    def test_the_last_one_ends_with_the_samples(self):
        blocks = recorded_blocks(self.PROTOCOL, FS, 7 * FS + 64)
        assert blocks == [Block("REST", 5.0), Block("FOCUS", 2.25)]

    def test_a_block_that_had_not_begun_is_left_out(self):
        assert recorded_blocks(self.PROTOCOL, FS, 10 * FS) == self.PROTOCOL[:2]

    def test_samples_after_the_protocol_stay_with_its_last_block_as_in_the_session(self):
        assert recorded_blocks(self.PROTOCOL, FS, 20 * FS) == self.PROTOCOL

    def test_free_play_until_the_session_was_stopped_gets_its_length(self):
        blocks = recorded_blocks([Block("REST", 5.0), Block("FREE", 0.0)], FS, 8 * FS)
        assert blocks == [Block("REST", 5.0), Block("FREE", 3.0)]

    def test_nothing_was_recorded(self):
        assert recorded_blocks(self.PROTOCOL, FS, 0) == []

    def test_a_whole_block_keeps_the_length_it_was_given(self):
        """0.3 s are 76.8 samples: the samples of the block would give it 77 / 256 s."""
        protocol = [Block("REST", 0.3), Block("FOCUS", 5.0)]
        assert recorded_blocks(protocol, FS, 2 * FS)[0] == Block("REST", 0.3)


class TestStoreRecording:
    """What the headset delivered in a session, stored in the data folder."""

    PROTOCOL = [["REST", 10], ["FOCUS", 10]]

    def live_session(self, tmp_path, seconds, stop_after_steps=None):
        matrix = synthetic_matrix(self.PROTOCOL)[:, : seconds * FS]
        source = FakeSource(matrix, live=True)
        config = config_with(self.PROTOCOL)
        return FinishedSession(tmp_path, config, source=source, stop_after_steps=stop_after_steps)

    def test_the_samples_of_the_electrodes_and_the_blocks_actually_recorded(self, tmp_path):
        session = self.live_session(tmp_path, 13, stop_after_steps=15)  # [SPACE] in FOCUS
        folder = store_recording(session.directory, tmp_path / "data")
        assert folder == tmp_path / "data" / OWN_RECORDINGS / session.directory.name
        raw = np.loadtxt(session.directory / c.FILE_RAW, delimiter="\t", ndmin=2).T
        rows = session.source.electrode_rows
        assert electrodes_of_recording(folder / c.FILE_DATASET_EEG) == list(rows)
        stored = np.loadtxt(folder / c.FILE_DATASET_EEG, delimiter=",", skiprows=1, ndmin=2).T
        assert np.allclose(stored, raw[list(rows.values())], atol=0.0005)
        assert 10 * FS < raw.shape[1] < 13 * FS
        protocol = read_yaml(folder / c.FILE_DATASET_PROTOCOL)["protocol"]
        assert protocol == [["REST", 10.0], ["FOCUS", raw.shape[1] / FS - 10]]

    def test_a_raw_file_that_cannot_be_read_is_reported(self, tmp_path):
        session = self.live_session(tmp_path, 13, stop_after_steps=15)
        (session.directory / c.FILE_RAW).write_text("not samples\n")
        with pytest.raises(AdamusError, match="is not a complete session folder: "):
            store_recording(session.directory, tmp_path / "data")
        assert not (tmp_path / "data").exists()

    def test_a_session_that_received_nothing_leaves_no_recording(self, tmp_path):
        session = self.live_session(tmp_path, 0)
        assert store_recording(session.directory, tmp_path / "data") is None
        assert not (tmp_path / "data").exists()

    def test_a_session_whose_stream_never_started_leaves_no_recording(self, tmp_path):
        session = self.live_session(tmp_path, 0)
        (session.directory / c.FILE_RAW).unlink()  # BrainFlow had not written any
        assert store_recording(session.directory, tmp_path / "data") is None

    def test_an_incomplete_session_folder_is_reported(self, tmp_path):
        session = self.live_session(tmp_path, 13, stop_after_steps=15)
        (session.directory / c.FILE_META).unlink()
        with pytest.raises(AdamusError, match="is not a complete session folder"):
            store_recording(session.directory, tmp_path / "data")
