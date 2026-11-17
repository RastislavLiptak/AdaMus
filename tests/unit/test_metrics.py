"""evaluation.metrics: every metric on hand-made data with known answers."""

import json

import librosa
import numpy as np
import pytest
import soundfile

from adamus import constants as c
from adamus.errors import AdamusError
from adamus.evaluation import metrics
from adamus.evaluation.session_folder import load_session_folder
from tests.support.handmade_session_folders import (
    CONTEXT,
    ELECTRODES,
    FS,
    H,
    N,
    event,
    finished_session_folder,
    protocol_ticks,
    tick,
    write_session_folder,
)

RATE = 48000
PROTOCOL = [["REST", 10], ["FOCUS", 10], ["REST", 10], ["FOCUS", 10], ["REST", 10]]
NO_F = {0: None, 1: None, 2: None, 3: None, 4: None}


@pytest.fixture
def folder(tmp_path, raw_config):
    """Build a session folder from ticks and events and load it."""
    raw_config["protocol"] = PROTOCOL

    def build(ticks, events=(), **options):
        build.made += 1  # a folder of its own for every build of a test
        directory = tmp_path / str(build.made) / "20261110-150412_X"
        return load_session_folder(
            finished_session_folder(directory, raw_config, ticks, events, **options)
        )

    build.made = 0
    return build


def ticks_with(f_of_block, **options):
    return protocol_ticks(PROTOCOL, f_of_block, **options)


class TestSampling:
    def test_sampling_rate_rate_from_first_and_last_timestamp(self):
        assert metrics.sampling_rate(np.arange(513) / 256) == pytest.approx(256.0)
        assert metrics.sampling_rate(np.array([0.0, 1.0, 4.0])) == 0.5

    def test_longest_gap_longest_gap(self):
        assert metrics.longest_gap(np.array([0.0, 0.004, 0.008, 0.7, 0.704])) == pytest.approx(
            0.692
        )

    def test_two_samples_are_enough(self):
        assert metrics.sampling_rate(np.array([10.0, 10.5])) == 2.0
        assert metrics.longest_gap(np.array([10.0, 10.5])) == 0.5

    def test_only_the_first_and_the_last_timestamp_decide_the_rate(self):
        assert metrics.sampling_rate(np.array([5.0, 7.0, 6.0])) == 2.0
        assert metrics.sampling_rate(np.array([5.0, 5.0, 6.0])) == 2.0

    @pytest.mark.parametrize("stamps", [[], [1.0], [2.0, 2.0], [3.0, 2.0]])
    def test_sampling_rate_needs_two_samples_and_time_that_advances(self, stamps):
        assert metrics.sampling_rate(np.array(stamps)) is None

    @pytest.mark.parametrize("stamps", [[], [1.0]])
    def test_longest_gap_needs_two_samples(self, stamps):
        assert metrics.longest_gap(np.array(stamps)) is None

    def test_spread_is_median_95th_percentile_and_maximum(self):
        assert metrics.spread(list(range(1, 101))) == {
            "median": 50.5, "p95": pytest.approx(95.05), "max": 100.0,
        }  # fmt: skip
        assert metrics.spread([]) is None


class TestValidShare:
    def test_valid_share_counts_valid_among_full_steps_of_each_block(self, folder):
        session = folder(ticks_with(NO_F, valid=lambda k: k % 4 != 0))
        blocks = metrics.valid_share(session)
        assert [block["index"] for block in blocks] == [0, 1, 2, 3, 4]
        assert [block["label"] for block in blocks] == ["REST", "FOCUS", "REST", "FOCUS", "REST"]
        # The first step ends when two windows have arrived: block 0 has fewer full steps.
        assert [block["full"] for block in blocks] == [13, 17, 17, 17, 17]
        # Full steps of block 0 are k = 0..12; k = 0, 4, 8, 12 are invalid.
        assert blocks[0] == {"index": 0, "label": "REST", "full": 13, "valid": 9, "ratio": 9 / 13}

    def test_block_without_steps_has_no_ratio(self, folder):
        blocks = metrics.valid_share(folder(ticks_with(NO_F, steps=5)))
        assert blocks[1] == {"index": 1, "label": "FOCUS", "full": 0, "valid": 0, "ratio": None}


class TestStateSeparation:
    def test_separation_uses_every_rest_and_focus_block(self, folder):
        """There is no calibration: the first blocks count as much as the others."""
        session = folder(ticks_with({0: 0.1, 1: 0.9, 2: 0.2, 3: 0.9, 4: 0.1}))
        auc, difference = metrics.state_separation(session)
        assert auc == 1.0
        assert difference == pytest.approx(0.9 - (13 * 0.1 + 17 * 0.2 + 17 * 0.1) / 47)
        inverted_first = folder(ticks_with({0: 0.9, 1: 0.0, 2: 0.2, 3: 0.9, 4: 0.1}))
        assert metrics.state_separation(inverted_first)[0] < 1.0

    def test_auc_is_one_half_for_identical_distributions(self, folder):
        assert metrics.state_separation(folder(ticks_with({**NO_F, 2: 0.4, 3: 0.4, 4: 0.4}))) == (
            0.5,
            0.0,
        )

    def test_auc_is_zero_when_focus_is_always_lower(self, folder):
        auc, difference = metrics.state_separation(
            folder(ticks_with({**NO_F, 2: 0.8, 3: 0.1, 4: 0.9}))
        )
        assert auc == 0.0 and difference == pytest.approx(0.1 - 0.85)

    def test_auc_counts_pairs_where_focus_is_higher(self, folder):
        """FOCUS f alternates 0.2 / 0.8, REST is 0.5: about half of the pairs are won."""
        focus = lambda k: 0.2 if k % 2 else 0.8  # noqa: E731
        auc, _ = metrics.state_separation(folder(ticks_with({**NO_F, 2: 0.5, 3: focus, 4: 0.5})))
        assert auc == pytest.approx(9 / 17)  # 17 full steps, k = 56 to 72: 9 even, 8 odd

    def test_invalid_and_not_full_steps_do_not_count(self, folder):
        first_of_block_3 = (3 * 10 * FS - CONTEXT) // H + 1
        not_full = N // H - 1  # steps whose window still reaches into block 2
        ticks = ticks_with(
            {**NO_F, 2: 0.2, 3: lambda k: 0.0 if k < first_of_block_3 + not_full else 0.9, 4: 0.2},
            valid=lambda k: not first_of_block_3 + 5 <= k < first_of_block_3 + 8,
        )
        for row in ticks:
            if not row["valid"]:
                row["f"] = 0.0  # a held value
        assert metrics.state_separation(folder(ticks))[0] == 1.0

    def test_no_data(self, folder):
        assert metrics.state_separation(folder(ticks_with(NO_F))) == (None, None)
        assert metrics.state_separation(folder(ticks_with({**NO_F, 3: 0.9}))) == (None, None)


class TestResponseTimes:
    def test_response_time_is_counted_from_the_start_of_the_block(self, folder):
        first = {block: (block * 10 * FS - CONTEXT) // H + 1 for block in range(5)}

        def on_after(block, steps):
            return lambda k: 1.0 if k - first[block] >= steps else 0.0

        def off_after(block, steps):
            return lambda k: 0.0 if k - first[block] >= steps else 1.0

        session = folder(
            ticks_with({**NO_F, 2: off_after(2, 3), 3: on_after(3, 6), 4: off_after(4, 0)})
        )
        assert metrics.response_times(session) == {
            # blocks 0 and 1 have no f yet; the 7th step of block 3 ends 3.5 s after its start
            "rise_s": [None, 3.5],
            "fall_s": [None, 2.0, 0.5],
            "rise_median_s": 3.5,
            "fall_median_s": 1.25,
        }

    def test_f_of_exactly_one_half_counts_as_on(self, folder):
        session = folder(ticks_with({**NO_F, 2: 0.5, 3: 0.5, 4: 0.49}))
        times = metrics.response_times(session)
        assert times["rise_s"] == [None, 0.5] and times["fall_s"] == [None, None, 0.5]

    def test_block_that_never_crosses_the_threshold(self, folder):
        assert metrics.response_times(folder(ticks_with({**NO_F, 2: 0.9, 3: 0.1, 4: 0.9}))) == {
            "rise_s": [None, None], "fall_s": [None, None, None],
            "rise_median_s": None, "fall_median_s": None,
        }  # fmt: skip

    def test_invalid_steps_are_skipped(self, folder):
        first = (3 * 10 * FS - CONTEXT) // H + 1
        ticks = ticks_with(
            {**NO_F, 2: 0.1, 3: 0.9, 4: 0.1}, valid=lambda k: not first <= k < first + 4
        )
        assert metrics.response_times(folder(ticks))["rise_s"] == [None, 2.5]


class TestLoop:
    def test_control_latency_is_time_from_the_last_eeg_sample_to_the_sent_message(self, folder):
        session = folder(ticks_with({**NO_F, 2: 0.2, 3: 0.9, 4: 0.2}))
        latency = metrics.spread(session.control_latencies())
        assert latency == {
            "median": pytest.approx(0.030),
            "p95": pytest.approx(0.030),
            "max": pytest.approx(0.030),
        }

    def test_step_interval_is_the_gap_between_neighbouring_sent_steps(self, folder):
        every_step = metrics.step_regularity(folder(ticks_with({**NO_F, 2: 0.2, 3: 0.9, 4: 0.2})))
        assert every_step == {"median": pytest.approx(0.5), "p95": pytest.approx(0.5)}
        every_other = ticks_with({**NO_F, 2: 0.2, 3: 0.9, 4: 0.2}, valid=lambda k: k % 2 == 0)
        assert metrics.step_regularity(folder(every_other))["median"] == pytest.approx(1.0)

    def test_two_sent_steps_are_enough_for_an_interval(self, folder):
        ticks = [tick(k, block=2, f=0.2) for k in range(2)]
        assert metrics.step_regularity(folder(ticks)) == {
            "median": pytest.approx(0.5),
            "p95": pytest.approx(0.5),
        }
        assert metrics.step_regularity(folder(ticks[:1])) is None

    def test_nothing_sent(self, folder):
        session = folder(ticks_with(NO_F))
        assert metrics.step_regularity(session) is None
        assert metrics.spread(session.control_latencies()) is None


def silent_of(pcm):
    """Frames where every channel is zero."""
    return ~pcm.any(axis=1)


class TestDropouts:
    def audio(self, seconds=2.0):
        t = np.arange(int(seconds * RATE))
        mono = (1000 * np.sin(2 * np.pi * 440 * t / RATE)).astype(np.int16)
        mono[mono == 0] = 1  # no accidental digital silence
        return np.column_stack([mono, mono])

    def test_clean_audio(self):
        assert metrics.dropouts(silent_of(self.audio()), RATE) == {
            "count": 0,
            "total_s": 0.0,
            "first_sound_s": 0.0,
        }

    def test_silence_before_the_first_sound_is_not_a_dropout(self):
        pcm = self.audio()
        pcm[:24000] = 0
        found = metrics.dropouts(silent_of(pcm), RATE)
        assert found == {"count": 0, "total_s": 0.0, "first_sound_s": 0.5}

    def test_one_model_frame_of_silence_is_a_dropout(self):
        pcm = self.audio()
        pcm[30000 : 30000 + 1920] = 0
        found = metrics.dropouts(silent_of(pcm), RATE)
        assert found == {"count": 1, "total_s": 0.040, "first_sound_s": 0.0}

    def test_one_sample_less_is_not(self):
        pcm = self.audio()
        pcm[30000 : 30000 + 1919] = 0
        assert metrics.dropouts(silent_of(pcm), RATE)["count"] == 0

    def test_silence_in_one_channel_only_is_not_a_dropout(self):
        pcm = self.audio()
        pcm[30000:40000, 0] = 0
        assert metrics.dropouts(silent_of(pcm), RATE)["count"] == 0

    def test_several_dropouts_one_of_them_at_the_very_end(self):
        pcm = self.audio()
        pcm[10000:14800] = 0  # 100 ms
        pcm[-9600:] = 0  # 200 ms
        assert metrics.dropouts(silent_of(pcm), RATE) == {
            "count": 2, "total_s": pytest.approx(0.3), "first_sound_s": 0.0,
        }  # fmt: skip

    def test_audio_that_is_silent_throughout(self):
        silent = np.zeros((RATE, 2), dtype=np.int16)
        found = metrics.dropouts(silent_of(silent), RATE)
        assert found == {"count": 0, "total_s": 0.0, "first_sound_s": None}


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
        rec_start = ticks[0]["t_sent"]  # the audio starts with the first step
        recording = [
            event("FUDI", rec_start, msg="rec start;"),
            event("FUDI", rec_start + 50.0, msg="rec stop;"),
        ]
        return folder(ticks, recording)

    @pytest.mark.parametrize("lag_s", [0.0, 1.5, 4.0, 10.0])
    def test_music_response_finds_the_lag_of_the_music_behind_the_control(self, session, lag_s):
        response = metrics.music_response(session, click_track(self.F, lag_s, 60.0), RATE)
        assert response["lag_s"] == lag_s
        assert response["rho"] > 0.8

    def test_step_that_sent_nothing_counts_from_the_end_of_its_processing(self, folder, session):
        """A step that kept the f of the one before (its signal was not valid) has no t_sent."""
        held = [tick(k, block=2, f=f, valid=k % 3 != 0) for k, f in enumerate(self.F)]
        assert [row["t_sent"] for row in held[:3]] == [None, held[1]["t_sent"], held[2]["t_sent"]]
        with_held_steps = folder(held, session.events[1:3])
        response = metrics.music_response(with_held_steps, click_track(self.F, 1.5, 60.0), RATE)
        assert response["lag_s"] == 1.5 and response["rho"] > 0.8

    def test_envelope_is_one_value_per_512_samples_of_audio(self):
        assert metrics.ONSET_HOP == 512

    def test_audio_with_one_channel_is_read_like_any_other(self, session):
        pcm = (click_track(self.F, 1.0, 50.0) * 30000).astype(np.int16)
        soundfile.write(session.directory / c.FILE_AUDIO, pcm, RATE, "PCM_16")
        result = metrics.compute_metrics(session.directory)
        assert result["music_response"]["lag_s"] == 1.0
        assert result["audio_dropouts"]["first_sound_s"] == pytest.approx(1.0, abs=0.001)

    def test_music_response_is_low_for_music_that_ignores_the_control(self, session):
        unrelated = np.random.default_rng(1).random(len(self.F))
        assert metrics.music_response(session, click_track(unrelated, 0.0, 60.0), RATE)["rho"] < 0.5

    def test_music_response_needs_audio_with_onsets_and_steps_with_f(self, session, folder):
        assert metrics.music_response(session, np.zeros(0, dtype=np.float32), RATE) is None
        assert metrics.music_response(session, np.zeros(RATE * 50, dtype=np.float32), RATE) is None
        without_f = folder([tick(k) for k in range(10)], session.events[1:3])
        assert metrics.music_response(without_f, click_track(self.F, 0.0, 60.0), RATE) is None
        never_recorded = folder([tick(k, block=2, f=f) for k, f in enumerate(self.F)])
        assert metrics.music_response(never_recorded, click_track(self.F, 0.0, 60.0), RATE) is None

    def test_length_error_is_audio_length_minus_the_time_between_rec_start_and_stop(
        self, session, folder
    ):
        assert metrics.recording_length_error(session, 50.25) == pytest.approx(0.25)
        assert metrics.recording_length_error(session, 49.5) == pytest.approx(-0.5)
        assert metrics.recording_length_error(folder([]), 10.0) is None
        only_started = folder([], [event("FUDI", 100.0, msg="rec start;")])
        assert metrics.recording_length_error(only_started, 10.0) is None

    def test_audio_file_is_read_and_analysed(self, session):
        pcm = (click_track(self.F, 1.0, 50.0) * 30000).astype(np.int16)
        soundfile.write(
            session.directory / c.FILE_AUDIO, np.column_stack([pcm, pcm]), RATE, "PCM_16"
        )
        result = metrics.compute_metrics(session.directory)
        assert result["audio_dropouts"]["first_sound_s"] == pytest.approx(1.0, abs=0.001)
        assert (
            result["audio_dropouts"]["count"] > 0
        )  # the silence between clicks is longer than a frame
        assert result["recording_length_error_s"] == pytest.approx(0.0, abs=1e-6)
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
        audio = {
            "audio_dropouts": None,
            "recording_length_error_s": None,
            "music_response": {"rho": 0.9, "lag_s": 2.0},
        }
        monkeypatch.setattr(metrics, "response_times", lambda session: times)
        monkeypatch.setattr(metrics, "audio_metrics", lambda session: audio)
        assert metrics.compute_metrics(session.directory)["total_response_s"] == expected

    @pytest.mark.parametrize("cut_to", [0, 30])  # nothing, or not even the whole header
    def test_audio_that_cannot_be_read_counts_as_none(self, session, cut_to):
        """Pd stopped before it wrote the file: the other metrics of the session still count."""
        audio = session.directory / c.FILE_AUDIO
        soundfile.write(audio, np.zeros((RATE, 2), dtype=np.int16), RATE)
        audio.write_bytes(audio.read_bytes()[:cut_to])
        assert metrics.audio_metrics(session) == {
            "audio_dropouts": None, "recording_length_error_s": None, "music_response": None,
        }  # fmt: skip
        assert metrics.compute_metrics(session.directory)["record_completeness"]["passed"]

    def test_without_audio_the_generator_metrics_are_null(self, session):
        result = metrics.compute_metrics(session.directory)
        assert (
            result["audio_dropouts"],
            result["recording_length_error_s"],
            result["music_response"],
            result["total_response_s"],
        ) == (None, None, None, None)


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
        assert self.lag_of(envelope, f, at, frame_s=frame_s)["rho"] == pytest.approx(1.0)

    def test_step_that_ends_with_the_last_frame_of_the_envelope_counts(self):
        f, at = self.F[:3], self.AT[:3]
        envelope = stepped_envelope(f, at, 0.0, self.FRAME_S, self.HOP_S, 1.5)
        assert len(envelope) == 150  # the third step ends exactly here
        assert self.lag_of(envelope, f, at) == {"rho": pytest.approx(1.0), "lag_s": 0.0}

    def test_three_steps_are_enough_and_two_are_not(self):
        assert metrics.MIN_CORRELATION_POINTS == 3
        envelope = stepped_envelope(self.F[:3], self.AT[:3], 0.0, self.FRAME_S, self.HOP_S, 9.0)
        assert self.lag_of(envelope, self.F[:3], self.AT[:3])["lag_s"] == 0.0
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

    def test_the_file_is_read_in_blocks_into_one_channel_and_its_silence(
        self, tmp_path, monkeypatch
    ):
        t = np.arange(RATE * 2)
        pcm = np.column_stack([(t % 1000) - 500, (t % 700) - 350]).astype(np.int16)
        pcm[30000:32000] = 0
        soundfile.write(tmp_path / "audio.wav", pcm, RATE, "PCM_16")
        monkeypatch.setattr(metrics, "AUDIO_BLOCK", 9999)
        mono, silent, rate = metrics.read_audio(tmp_path / "audio.wav")
        assert rate == RATE and mono.dtype == np.float32
        assert np.array_equal(mono, metrics.mono_of(pcm))
        assert np.array_equal(silent, silent_of(pcm))


class TestMono:
    def test_channels_are_averaged_and_scaled_to_full_scale_one(self):
        pcm = np.array([[32767, 32767], [-32768, -32768], [100, 300], [0, 0]], dtype=np.int16)
        mono = metrics.mono_of(pcm)
        assert mono.dtype == np.float32
        assert mono.tolist() == [32767 / 32768, -1.0, 200 / 32768, 0.0]

    def test_one_channel_is_itself(self):
        pcm = np.array([[16384], [-16384]], dtype=np.int16)
        assert metrics.mono_of(pcm).tolist() == [0.5, -0.5]


class TestCompleteness:
    def test_completeness_complete_record(self, folder):
        session = folder([tick(k) for k in range(10)])
        assert metrics.completeness(session, CONTEXT + 9 * H) == {
            "ticks": 10, "filtered_records": 10, "expected": 10, "passed": True,
        }  # fmt: skip

    def test_samples_that_do_not_fill_a_hop_add_no_step(self, folder):
        assert metrics.completeness(folder([tick(k) for k in range(10)]), CONTEXT + 10 * H - 1)[
            "passed"
        ]

    def test_missing_steps_are_detected(self, folder):
        session = folder([tick(k) for k in range(10)])
        assert metrics.completeness(session, CONTEXT + 12 * H) == {
            "ticks": 10, "filtered_records": 10, "expected": 13, "passed": False,
        }  # fmt: skip

    def test_missing_filtered_record_is_detected(self, folder):
        session = folder([tick(k) for k in range(10)])
        path = session.directory / c.FILE_FILTERED
        path.write_bytes(path.read_bytes()[: 9 * 4 * ELECTRODES * N])
        result = metrics.completeness(session, CONTEXT + 9 * H)
        assert (result["filtered_records"], result["passed"]) == (9, False)

    def test_truncated_filtered_record_is_detected(self, folder):
        session = folder([tick(k) for k in range(10)])
        path = session.directory / c.FILE_FILTERED
        path.write_bytes(path.read_bytes() + b"\0")
        assert metrics.completeness(session, CONTEXT + 9 * H)["passed"] is False

    @pytest.mark.parametrize("samples", [0, 500, CONTEXT - 1])
    def test_session_shorter_than_the_signal_of_one_step_has_no_steps(self, folder, samples):
        assert metrics.completeness(folder([]), samples) == {
            "ticks": 0, "filtered_records": 0, "expected": 0, "passed": True,
        }  # fmt: skip


class TestComputeMetrics:
    def test_metrics_file_has_every_metric_and_is_plain_json(self, folder):
        session = folder(ticks_with({**NO_F, 2: 0.2, 3: 0.9, 4: 0.2}), [])
        path = metrics.analyze(session.directory)
        assert path == session.directory / "metrics.json"
        stored = json.loads(path.read_text())
        assert path.read_text() == json.dumps(stored, indent=2) + "\n"  # readable by people too
        assert list(stored) == [
            "session", "sampling_rate_hz", "longest_sample_gap_s", "valid_steps_per_block",
            "processing_time_s", "state_separation_auc", "mean_f_difference", "response_time_s",
            "control_latency_s", "step_interval_s", "audio_dropouts",
            "recording_length_error_s", "music_response", "total_response_s",
            "record_completeness",
        ]  # fmt: skip
        assert stored["session"] == "20261110-150412_X"
        assert stored["sampling_rate_hz"] == pytest.approx(256.0, rel=1e-4)
        assert stored["longest_sample_gap_s"] == pytest.approx(1 / 256, rel=1e-3)
        assert stored["processing_time_s"] == {
            "median": pytest.approx(0.001),
            "p95": pytest.approx(0.001),
            "max": pytest.approx(0.001),
        }
        assert stored["state_separation_auc"] == 1.0
        assert stored["record_completeness"]["passed"] is True

    def test_folder_that_is_not_a_session(self, tmp_path):
        with pytest.raises(AdamusError) as error:
            metrics.compute_metrics(tmp_path)
        assert str(error.value).startswith(
            f"{tmp_path} is not a complete session folder: cannot read {tmp_path / 'config.yaml'}: "
        )
        with pytest.raises(AdamusError, match="is not a complete session folder"):
            metrics.compute_metrics(tmp_path / "missing")

    @pytest.mark.parametrize("raw", ["empty", "missing"])  # BrainFlow made it, or not even that
    def test_session_in_which_no_sample_arrived(self, tmp_path, raw_config, raw):
        """It ended before its first sample, e.g. a headset that sends nothing: it has
        metrics all the same, which say that no sample and no step was recorded."""
        directory = write_session_folder(
            tmp_path / "20261110-150412_X", raw_config, [], [], samples=0
        )
        if raw == "missing":
            (directory / c.FILE_RAW).unlink()
        stored = metrics.compute_metrics(directory)
        assert stored["record_completeness"] == {
            "ticks": 0, "filtered_records": 0, "expected": 0, "passed": True,
        }  # fmt: skip
        assert (stored["sampling_rate_hz"], stored["longest_sample_gap_s"]) == (None, None)

    def test_raw_file_that_cannot_be_read_is_reported(self, folder):
        session = folder([tick(0)])
        (session.directory / c.FILE_RAW).write_text("not samples\n")
        with pytest.raises(AdamusError) as error:
            metrics.compute_metrics(session.directory)
        incomplete = f"{session.directory} is not a complete session folder"
        assert str(error.value) == f"{incomplete}: {error.value.__cause__}"

    def test_samples_that_are_not_numbers_do_not_cost_the_metrics(self, folder):
        """A recording may hold nan or inf; played back, they go into raw.csv as they are."""
        session = folder([tick(k) for k in range(3)])
        raw = session.directory / c.FILE_RAW
        table = np.loadtxt(raw, delimiter="\t", ndmin=2)
        table[5, 1], table[6, 2] = np.nan, np.inf
        np.savetxt(raw, table, delimiter="\t", fmt="%.6f")
        result = metrics.compute_metrics(session.directory)
        assert result["record_completeness"]["passed"] is True
        assert result["sampling_rate_hz"] == pytest.approx(256)

    def test_session_without_its_filtered_file(self, folder):
        session = folder([tick(0)])
        (session.directory / c.FILE_FILTERED).unlink()
        with pytest.raises(AdamusError) as error:
            metrics.compute_metrics(session.directory)
        incomplete = f"{session.directory} is not a complete session folder"
        assert str(error.value) == f"{incomplete}: {error.value.__cause__}"

    def test_damaged_ticks_file(self, folder):
        session = folder([tick(0)])
        (session.directory / c.FILE_TICKS).write_text('{"k": 0,\n')
        with pytest.raises(AdamusError, match="is not a complete session folder"):
            metrics.compute_metrics(session.directory)
