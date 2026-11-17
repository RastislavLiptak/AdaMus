"""constants.py: values that depend on each other stay consistent."""

from adamus import constants as c
from adamus.evaluation import metrics
from adamus.realtime import music_generator


def test_every_block_label_has_a_marker():
    assert c.BLOCK_MARKERS == {"REST": 1, "FOCUS": 2, "FREE": 3}
    assert 0 not in c.BLOCK_MARKERS.values()  # 0 means "no marker" in the stream


def test_durations_of_the_detector_are_whole_samples_at_its_rate():
    for seconds in (c.WINDOW_S, c.HOP_S):
        assert (seconds * c.DETECTOR_FS).is_integer()
    assert c.HOP_S <= c.WINDOW_S
    assert (c.PRIOR_STEPS, c.MEMORY_STEPS) == (600, 2400)
    assert (c.PRIOR_STEPS * c.HOP_S, c.MEMORY_STEPS * c.HOP_S) == (c.PRIOR_S, c.MEMORY_S)
    assert 0 < c.PRIOR_S <= c.MEMORY_S


def test_delayed_copies_are_distinct_and_fit_in_the_signal_before_the_window():
    delays = [round(seconds * c.DETECTOR_FS) for seconds in c.DELAYS_S]
    assert delays == sorted(set(delays)) == [0, 6, 13]
    assert max(c.DELAYS_S) < (c.CONTEXT_WINDOWS - 1) * c.WINDOW_S


def test_bands_ascend_inside_the_broadband_and_below_half_the_rate():
    edges = [edge for band in c.BANDS_HZ for edge in band]
    assert edges == sorted(edges)
    assert c.BROADBAND_HZ[0] <= edges[0] and edges[-1] <= c.BROADBAND_HZ[1]
    assert max(c.MAINS_HZ) < c.DETECTOR_FS / 2 and c.BROADBAND_HZ[1] < min(c.MAINS_HZ)


def test_eye_band_lies_below_the_broadband_and_the_noise_band_is_its_highest_band():
    assert c.EYE_BAND_HZ[0] < c.EYE_BAND_HZ[1] <= c.BROADBAND_HZ[0]
    assert c.NOISE_BAND_HZ == c.BANDS_HZ[-1] == (30.0, 45.0)
    assert 0 < c.NOISE_UV < c.ARTIFACT_UV


def test_artifact_electrodes_are_electrodes_of_the_detector():
    assert set(c.ARTIFACT_ELECTRODES) <= set(c.DETECTOR_ELECTRODES)
    assert len(set(c.DETECTOR_ELECTRODES)) == len(c.DETECTOR_ELECTRODES) == 4


def test_a_dropout_is_one_model_frame_of_silence():
    assert metrics.DROPOUT_SAMPLES == 1920  # 40 ms at 48 kHz
    assert metrics.DROPOUT_SAMPLES == c.MODEL_FRAME_S * c.AUDIO_RATE


def test_patch_is_the_seven_objects_and_nine_connections_with_one_port_to_fill_in():
    assert music_generator.PATCH_TEMPLATE.read_text().splitlines() == [
        "#N canvas 0 0 420 260 12;",
        "#X obj 30 20 netreceive {port};",
        "#X obj 30 60 route rec;",
        "#X obj 150 100 mrt2~;",
        "#X obj 30 150 writesf~ 2;",
        "#X obj 150 190 dac~;",
        "#X obj 290 60 sel 0;",  # the number of connections falls to 0 ...
        "#X msg 290 100 \\; pd quit;",  # ... and Pd quits
        "#X connect 0 0 1 0;",
        "#X connect 1 0 3 0;",
        "#X connect 1 1 2 0;",
        "#X connect 2 0 3 0;",
        "#X connect 2 1 3 1;",
        "#X connect 2 0 4 0;",
        "#X connect 2 1 4 1;",
        "#X connect 0 1 5 0;",
        "#X connect 5 0 6 0;",
    ]
    assert music_generator.PATCH_PORT == "{port}"


def test_fatal_lines_do_not_match_the_harmless_variants_of_mrt2():
    harmless = [
        "mrt2~: failed to load default model",
        "mrt2~: failed to init assets from default path",
    ]
    for line in harmless:
        assert not line.endswith(music_generator.FATAL_LINE_ENDINGS)
    assert "error: mrt2~: failed to load model".endswith(music_generator.FATAL_LINE_ENDINGS)
    assert "error: mrt2~: failed to init assets".endswith(music_generator.FATAL_LINE_ENDINGS)
