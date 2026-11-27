"""evaluation.metrics of the music: its dropouts, how it followed the focus and how the audio is
read, on hand-made sound with known answers."""

import librosa
import numpy as np
import pytest
import soundfile

from adamus import constants as c
from adamus.evaluation import metrics
from tests.support.handmade_session_folders import (
    event,
    session_builder,
    tick,
)

RATE = 48000


@pytest.fixture
def folder(tmp_path, raw_config):
    """Build a session folder of the protocol of the metric tests and load it."""
    return session_builder(tmp_path, raw_config)


class TestDropouts:
    """As the output told them: blocks it played before they were ready."""

    def test_counted_as_the_music_ended(self, folder):
        ended = folder([], [event("MUSIC_START", 100.0), event("MUSIC_END", 150.0, dropouts=3)])
        assert metrics.dropouts(ended) == {"count": 3}

    def test_none_when_the_music_never_ended_or_began(self, folder):
        assert metrics.dropouts(folder([], [event("MUSIC_START", 100.0)])) is None
        assert metrics.dropouts(folder([])) is None


def click_track(f_values, lag_s, seconds, step_s=0.5):
    """Audio whose number of clicks in each step follows f, delayed by ``lag_s``."""
    mono = np.zeros(int(seconds * RATE), dtype=np.float32)
    for i, f in enumerate(f_values):
        for j in range(1 + round(8 * f)):
            at = int(((i * step_s + lag_s) + j * step_s / 10) * RATE)
            if at + 200 < mono.size:
                mono[at : at + 200] += 0.8
    return mono


class TestAudio:
    F = [0.0] * 20 + [1.0] * 20 + [0.0] * 20 + [1.0] * 20 + [0.5] * 13

    @pytest.fixture
    def session(self, folder):
        ticks = [tick(k, block=2, f=f) for k, f in enumerate(self.F)]
        began = ticks[0]["t_sent"]  # the music starts with the first step
        assert began is not None
        music = [event("MUSIC_START", began), event("MUSIC_END", began + 50.0, dropouts=0)]
        return folder(ticks, music)

    @pytest.mark.parametrize("lag_s", [0.0, 1.5, 4.0, 10.0])
    def test_music_response_finds_the_lag_of_the_music_behind_the_control(self, session, lag_s):
        response = metrics.music_response(session, click_track(self.F, lag_s, 60.0), RATE)
        assert response is not None
        assert response["lag_s"] == lag_s
        assert response["rho"] > 0.8

    def test_step_that_sent_nothing_counts_from_the_end_of_its_processing(self, folder, session):
        """A step that kept the f of the one before (its signal was not valid) has no t_sent."""
        held = [tick(k, block=2, f=f, valid=k % 3 != 0) for k, f in enumerate(self.F)]
        assert [row["t_sent"] for row in held[:3]] == [None, held[1]["t_sent"], held[2]["t_sent"]]
        with_held_steps = folder(held, session.events[1:3])
        response = metrics.music_response(with_held_steps, click_track(self.F, 1.5, 60.0), RATE)
        assert response is not None
        assert response["lag_s"] == 1.5 and response["rho"] > 0.8

    def test_envelope_is_one_value_per_512_samples_of_audio(self):
        assert metrics.ONSET_HOP == 512

    def test_audio_with_one_channel_is_read_like_any_other(self, session):
        pcm = (click_track(self.F, 1.0, 50.0) * 30000).astype(np.int16)
        soundfile.write(session.directory / c.FILE_AUDIO, pcm, RATE, "PCM_16")
        result = metrics.compute_metrics(session.directory)
        assert result["music_response"]["lag_s"] == 1.0

    def test_music_response_is_low_for_music_that_ignores_the_control(self, session):
        unrelated = np.random.default_rng(1).random(len(self.F))
        response = metrics.music_response(session, click_track(unrelated, 0.0, 60.0), RATE)
        assert response is not None and response["rho"] < 0.5

    def test_music_response_needs_audio_with_onsets_and_steps_with_f(self, session, folder):
        assert metrics.music_response(session, np.zeros(0, dtype=np.float32), RATE) is None
        assert metrics.music_response(session, np.zeros(RATE * 50, dtype=np.float32), RATE) is None
        without_f = folder([tick(k) for k in range(10)], session.events[1:3])
        assert metrics.music_response(without_f, click_track(self.F, 0.0, 60.0), RATE) is None
        never_began = folder([tick(k, block=2, f=f) for k, f in enumerate(self.F)])
        assert metrics.music_response(never_began, click_track(self.F, 0.0, 60.0), RATE) is None

    def test_audio_file_is_read_and_analysed(self, session):
        pcm = (click_track(self.F, 1.0, 50.0) * 30000).astype(np.int16)
        soundfile.write(
            session.directory / c.FILE_AUDIO, np.column_stack([pcm, pcm]), RATE, "PCM_16"
        )
        result = metrics.compute_metrics(session.directory)
        assert result["audio_dropouts"] == {"count": 0}  # silence in the music is no dropout
        assert result["music_response"]["lag_s"] == 1.0
        assert set(result["total_response_s"]) == {"rise_s", "fall_s"}

    @pytest.mark.parametrize(
        ("rise", "fall", "expected"),
        [(3.5, 1.0, {"rise_s": 5.5, "fall_s": 3.0}), (None, 1.0, {"rise_s": None, "fall_s": 3.0}),
         (3.5, None, {"rise_s": 5.5, "fall_s": None})],
    )  # fmt: skip
    def test_total_response_is_the_response_time_plus_the_music_lag(
        self, session, monkeypatch, rise, fall, expected
    ):
        """To the listener, rise and fall each take the lag of the music longer."""
        times = {"rise_s": [rise], "fall_s": [fall], "rise_median_s": rise, "fall_median_s": fall}
        audio = {"audio_dropouts": None, "music_response": {"rho": 0.9, "lag_s": 2.0}}
        monkeypatch.setattr(metrics, "response_times", lambda session: times)
        monkeypatch.setattr(metrics, "audio_metrics", lambda session: audio)
        assert metrics.compute_metrics(session.directory)["total_response_s"] == expected

    @pytest.mark.parametrize("cut_to", [0, 30])  # nothing, or not even the whole header
    def test_audio_that_cannot_be_read_counts_as_none(self, session, cut_to):
        """A recording the disk took only in part: the other metrics of the session still count."""
        audio = session.directory / c.FILE_AUDIO
        soundfile.write(audio, np.zeros((RATE, 2), dtype=np.int16), RATE)
        audio.write_bytes(audio.read_bytes()[:cut_to])
        assert metrics.audio_metrics(session) == {
            "audio_dropouts": {"count": 0}, "music_response": None,
        }  # fmt: skip
        assert metrics.compute_metrics(session.directory)["record_completeness"]["passed"]

    def test_without_audio_how_the_music_followed_is_not_known(self, session):
        result = metrics.compute_metrics(session.directory)
        assert (result["music_response"], result["total_response_s"]) == (None, None)
        assert result["audio_dropouts"] == {"count": 0}  # as the output told them


def stepped_envelope(f, at, lag_s, frame_s, hop_s, seconds):
    """An envelope that holds f[k] during the step that follows at[k] + lag_s, and 0 elsewhere."""
    envelope = np.zeros(round(seconds / frame_s))
    for value, time in zip(f, at, strict=True):
        first = int(np.ceil((time + lag_s) / frame_s))
        last = int(np.ceil((time + lag_s + hop_s) / frame_s))
        envelope[first:last] = value
    return envelope


class TestBestLag:
    """The search for the lag, on envelopes whose answer is known exactly."""

    FRAME_S, HOP_S = 0.01, 0.5
    F = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.5, 0.4, 1.0, 0.0, 0.6, 0.35])
    AT = np.arange(12) * 0.5

    def lag_of(self, envelope, f=None, at=None, frame_s=FRAME_S, hop_s=HOP_S):
        f = self.F if f is None else f
        at = self.AT if at is None else at
        return metrics.best_lag(np.asarray(f), np.asarray(at), envelope, frame_s, hop_s)

    @pytest.mark.parametrize("lag_s", [0.0, 0.5, 2.5, 9.5, 10.0])
    def test_envelope_that_repeats_f_after_a_lag_is_found_at_that_lag(self, lag_s):
        envelope = stepped_envelope(self.F, self.AT, lag_s, self.FRAME_S, self.HOP_S, 30.0)
        assert self.lag_of(envelope) == {"rho": pytest.approx(1.0), "lag_s": lag_s}

    def test_lags_are_tried_up_to_ten_seconds_and_no_further(self):
        assert metrics.MUSIC_MAX_LAG_S == 10.0
        envelope = stepped_envelope(self.F, self.AT, 10.5, self.FRAME_S, self.HOP_S, 30.0)
        found = self.lag_of(envelope)
        assert found is not None
        assert found["lag_s"] <= 10.0 and found["rho"] < 0.99

    def test_of_two_equally_good_lags_the_shorter_counts(self):
        f = np.tile([0.0, 1.0, 2.0, 3.0], 6)  # repeats every 2 s
        at = np.arange(len(f)) * 0.5
        envelope = stepped_envelope(np.tile(f, 3), np.arange(3 * len(f)) * 0.5, 0.0, 0.01, 0.5, 36)
        assert self.lag_of(envelope, f, at) == {"rho": pytest.approx(1.0), "lag_s": 0.0}

    def test_envelope_is_averaged_over_each_step_not_summed(self):
        """Steps cover 46 or 47 frames of real audio; a sum would favour the longer ones."""
        frame_s = metrics.ONSET_HOP / RATE
        f = np.arange(40.0)
        at = np.arange(40) * 0.5
        envelope = stepped_envelope(1 + f / 1000, at, 0.0, frame_s, 0.5, 30.0)
        frames = {
            int(np.ceil((time + 0.5) / frame_s)) - int(np.ceil(time / frame_s)) for time in at
        }
        assert frames == {46, 47}
        found = self.lag_of(envelope, f, at, frame_s=frame_s)
        assert found is not None and found["rho"] == pytest.approx(1.0)

    def test_step_that_ends_with_the_last_frame_of_the_envelope_counts(self):
        f, at = self.F[:3], self.AT[:3]
        envelope = stepped_envelope(f, at, 0.0, self.FRAME_S, self.HOP_S, 1.5)
        assert len(envelope) == 150  # the third step ends exactly here
        assert self.lag_of(envelope, f, at) == {"rho": pytest.approx(1.0), "lag_s": 0.0}

    def test_three_steps_are_enough_and_two_are_not(self):
        assert metrics.MIN_CORRELATION_POINTS == 3
        envelope = stepped_envelope(self.F[:3], self.AT[:3], 0.0, self.FRAME_S, self.HOP_S, 9.0)
        found = self.lag_of(envelope, self.F[:3], self.AT[:3])
        assert found is not None and found["lag_s"] == 0.0
        assert self.lag_of(envelope, self.F[:2], self.AT[:2]) is None
        assert self.lag_of(envelope, self.F[:0], self.AT[:0]) is None

    def test_lag_is_tried_while_four_fifths_of_the_steps_still_have_their_audio(self):
        """Ten steps, five seconds of audio: lag 1.0 s leaves eight steps, lag 1.5 s seven."""
        f, at = self.F[:10], self.AT[:10]
        at_one_second = stepped_envelope(f, at, 1.0, self.FRAME_S, self.HOP_S, 5.0)
        assert self.lag_of(at_one_second, f, at) == {"rho": pytest.approx(1.0), "lag_s": 1.0}

    def test_long_lag_over_the_few_steps_it_leaves_does_not_win_by_chance(self):
        """What made the lag of a short session jump to ten seconds now and then."""
        f, at = self.F[:10], self.AT[:10]
        at_longer = stepped_envelope(f, at, 1.5, self.FRAME_S, self.HOP_S, 5.0)
        found = self.lag_of(at_longer, f, at)  # perfect over the seven steps lag 1.5 s leaves
        assert found is not None
        assert found["lag_s"] <= 1.0 and found["rho"] < 0.99

    def test_steps_shorter_than_a_frame_of_the_envelope_cover_nothing(self):
        """Most such steps hold no frame at all; they are left out, not divided by zero."""
        f = np.arange(40.0)
        at = np.arange(40) * 0.004
        assert self.lag_of(np.arange(2000.0), f, at, hop_s=0.004) is None

    def test_f_or_envelope_that_never_changes_has_no_correlation(self):
        changing = stepped_envelope(self.F, self.AT, 0.0, self.FRAME_S, self.HOP_S, 30.0)
        assert self.lag_of(changing, np.full(12, 0.5)) is None
        assert self.lag_of(np.ones(3000)) is None


class TestReadInBlocks:
    """A long session is read and analysed in blocks; the results are those of all at once."""

    def test_the_mel_spectrogram_is_librosas_of_the_whole_audio(self, monkeypatch):
        rng = np.random.default_rng(4)
        mono = (0.3 * rng.standard_normal(RATE * 3 + 777)).astype(np.float32)
        whole = librosa.feature.melspectrogram(y=mono, sr=RATE, hop_length=metrics.ONSET_HOP)
        monkeypatch.setattr(metrics, "AUDIO_BLOCK", 7 * metrics.ONSET_HOP)  # many small blocks
        in_blocks = metrics.mel_spectrogram(mono, RATE)
        assert in_blocks.shape == whole.shape
        assert np.allclose(in_blocks, whole, rtol=1e-5, atol=1e-9)

    def test_the_onset_envelope_is_librosas_of_the_whole_audio(self, monkeypatch):
        mono = click_track([0.1, 0.9, 0.4, 0.7] * 5, 0.0, 12.0)
        whole = librosa.onset.onset_strength(y=mono, sr=RATE, hop_length=metrics.ONSET_HOP)
        monkeypatch.setattr(metrics, "AUDIO_BLOCK", 5 * metrics.ONSET_HOP)
        spectrum = librosa.power_to_db(metrics.mel_spectrogram(mono, RATE))
        in_blocks = librosa.onset.onset_strength(S=spectrum, sr=RATE, hop_length=metrics.ONSET_HOP)
        assert np.allclose(in_blocks, whole, rtol=1e-4, atol=1e-4)

    def test_the_file_is_read_in_blocks_into_one_channel(self, tmp_path, monkeypatch):
        t = np.arange(RATE * 2)
        pcm = np.column_stack([(t % 1000) - 500, (t % 700) - 350]).astype(np.int16)
        soundfile.write(tmp_path / "audio.wav", pcm, RATE, "PCM_16")
        monkeypatch.setattr(metrics, "AUDIO_BLOCK", 9999)
        mono, rate = metrics.read_audio(tmp_path / "audio.wav")
        assert rate == RATE and mono.dtype == np.float32
        assert np.array_equal(mono, metrics.mono_of(pcm))


class TestMono:
    def test_channels_are_averaged_and_scaled_to_full_scale_one(self):
        pcm = np.array([[32767, 32767], [-32768, -32768], [100, 300], [0, 0]], dtype=np.int16)
        mono = metrics.mono_of(pcm)
        assert mono.dtype == np.float32
        assert mono.tolist() == [32767 / 32768, -1.0, 200 / 32768, 0.0]

    def test_one_channel_is_itself(self):
        pcm = np.array([[16384], [-16384]], dtype=np.int16)
        assert metrics.mono_of(pcm).tolist() == [0.5, -0.5]
