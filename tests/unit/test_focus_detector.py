"""focus_detector: log vectors, the running mean, the models and the detector."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from adamus import constants as c
from adamus.detection import focus_training
from adamus.detection.focus_detector import (
    FILE_KEYS,
    MODEL_KEYS,
    Detector,
    FocusModel,
    RunningMean,
    electrode_sets,
    features,
    load_models,
    log_vector,
    vector_length,
)
from adamus.errors import AdamusError

EVERY = c.DETECTOR_ELECTRODES


def spd(size=4, seed=0):
    """A random symmetric positive definite matrix."""
    a = np.random.default_rng(seed).normal(size=(size, size))
    return a @ a.T + size * np.eye(size)


def covariances(electrodes=4, seed=0):
    """The covariances of one step: every band and the eye band, then the delayed copies."""
    single = [spd(electrodes, seed + k) for k in range(len(c.BANDS_HZ) + 1)]
    return [*single, spd(3 * electrodes, seed + 99)]


def ridged(matrix):
    size = len(matrix)
    return matrix + 1e-5 * np.trace(matrix) / size * np.eye(size)


def log_of(matrix):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.log(values)) @ vectors.T


class TestFeatures:
    def test_name_every_value_the_models_were_trained_with(self):
        f = features()
        assert f["electrodes"] == ["AF7", "AF8", "TP9", "TP10"]
        assert (f["artifact_uv"], f["noise_uv"], f["noise_band_hz"]) == (400.0, 20.0, [30.0, 45.0])
        assert (f["fs"], f["window_s"], f["hop_s"], f["context_windows"]) == (256, 2.0, 0.5, 2)
        assert (f["eye_band_hz"], f["eye_filter_order"]) == ([0.5, 2.0], 2)
        assert (f["prior_s"], f["memory_s"]) == (300.0, 1200.0)
        assert len(f["bands_hz"]) == 7 and f["delays_s"] == [0.0, 0.025, 0.05]

    def test_are_plain_json(self):
        assert json.loads(json.dumps(features())) == features()


class TestElectrodeSets:
    def test_every_non_empty_set_once_in_the_detector_order(self):
        sets = electrode_sets()
        assert len(sets) == len(set(sets)) == 2 ** len(EVERY) - 1 == 15
        assert sets[0] == ("AF7",) and sets[-1] == EVERY
        assert all(list(s) == sorted(s, key=EVERY.index) for s in sets)

    @pytest.mark.parametrize(("electrodes", "length"), [(1, 14), (2, 45), (3, 93), (4, 158)])
    def test_vector_length_is_that_of_the_log_vector(self, electrodes, length):
        assert vector_length(electrodes) == length
        assert log_vector(covariances(electrodes)).shape == (length,)
        assert type(vector_length(electrodes)) is int


class TestLogVector:
    def test_diagonal_matrix_gives_the_logarithms_of_its_diagonal(self):
        vector = log_vector([np.diag([1.0, math.e, math.e**2])])
        assert vector == pytest.approx([0.0, 0.0, 0.0, 1.0, 0.0, 2.0], abs=1e-4)

    def test_upper_triangle_of_the_logarithm_off_diagonal_times_root_two(self):
        matrix = spd(3, seed=4)
        log = log_of(ridged(matrix))
        rows, columns = np.triu_indices(3)
        weights = np.where(rows == columns, 1.0, math.sqrt(2))
        assert log_vector([matrix]) == pytest.approx(log[rows, columns] * weights)

    def test_length_is_the_log_euclidean_distance_to_the_identity(self):
        matrix = spd(4, seed=5)
        assert np.linalg.norm(log_vector([matrix])) == pytest.approx(
            np.linalg.norm(log_of(ridged(matrix)))
        )

    def test_covariances_follow_one_another(self):
        a, b = spd(4, 1), spd(12, 2)
        assert np.array_equal(
            log_vector([a, b]), np.concatenate([log_vector([a]), log_vector([b])])
        )

    def test_a_stack_of_steps_gives_the_vector_of_every_step(self):
        steps = [covariances(seed=s) for s in range(3)]
        stacked = [np.array([step[k] for step in steps]) for k in range(len(steps[0]))]
        assert np.allclose(log_vector(stacked), np.array([log_vector(step) for step in steps]))

    def test_a_singular_matrix_is_ridged_and_the_floor_stops_minus_infinity(self):
        assert np.isfinite(log_vector([np.zeros((2, 2)), np.ones((2, 2))])).all()
        assert log_vector([np.diag([1.0, 0.0])])[2] == pytest.approx(math.log(5e-6), rel=1e-6)


class TestRunningMean:
    def test_starts_at_the_prior(self):
        running = RunningMean(np.array([1.0, 2.0]), 4, 10)
        assert np.array_equal(running.mean, [1.0, 2.0])

    def test_the_prior_counts_as_so_many_steps_of_the_person(self):
        """After n steps of the same vector v the mean is (p * prior + n * v) / (p + n)."""
        running = RunningMean(np.zeros(1), 4, 100)
        for n in range(1, 7):
            running.add(np.ones(1))
            assert running.mean[0] == pytest.approx(n / (4 + n))

    def test_memory_bounds_how_little_a_new_step_counts(self):
        running = RunningMean(np.zeros(1), 2, 4)
        shares = []
        for _ in range(6):
            before = running.mean[0]
            running.add(np.ones(1))
            shares.append((running.mean[0] - before) / (1 - before))
        assert shares == pytest.approx([1 / 3, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4])

    def test_the_prior_is_copied(self):
        prior = np.zeros(2)
        RunningMean(prior, 1, 2).add(np.ones(2))
        assert not prior.any()

    @pytest.mark.parametrize(("prior_steps", "memory_steps"), [(0, 10), (11, 10)])
    def test_prior_must_count_and_fit_into_the_memory(self, prior_steps, memory_steps):
        with pytest.raises(AssertionError):
            RunningMean(np.zeros(1), prior_steps, memory_steps)


def model_of(electrodes=EVERY, **changes):
    size = vector_length(len(electrodes))
    values: dict[str, Any] = {
        "prior": np.zeros(size), "mean": np.zeros(size), "scale": np.ones(size),
        "coef": np.zeros(size), "intercept": 0.0, "threshold": 0.0,
    }  # fmt: skip
    values.update(changes)
    return FocusModel(tuple(electrodes), **values)


class TestModel:
    def test_score_is_the_standardised_logistic_regression_minus_the_threshold(self):
        model = model_of(("AF7",), mean=np.full(14, 1.0), scale=np.full(14, 2.0),
                         coef=np.arange(14.0), intercept=0.5, threshold=1.5)  # fmt: skip
        vector = np.arange(14.0)
        assert model.score(vector) == pytest.approx(((vector - 1) / 2) @ np.arange(14.0) - 1.0)

    def test_score_is_a_plain_number(self):
        assert type(model_of().score(np.zeros(158))) is float


class TestShippedModels:
    def test_one_model_for_every_set_of_electrodes(self):
        models = load_models()
        assert sorted(models.by_electrodes) == sorted(electrode_sets())
        assert models.by_electrodes[EVERY].electrodes == EVERY
        assert models.by_electrodes[("AF8", "TP9")].coef.shape == (45,)

    def test_were_trained_with_these_features_on_named_people_of_the_muse_dataset(self):
        raw = load_models().raw
        assert raw["features"] == features()
        trained_on = raw["trained_on"]
        assert all(type(person) is str for person in trained_on["people"])
        assert trained_on["dataset"] == focus_training.TRAINING_DATASET
        assert not set(trained_on["left_out"]) & set(trained_on["people"])

    def test_every_model_was_trained_on_twenty_people_or_more(self):
        per_set = load_models().raw["trained_on"]["people per set"]
        assert sorted(per_set) == sorted("+".join(s) for s in electrode_sets())
        assert min(per_set.values()) >= 20

    def test_lie_in_their_folder_of_the_models(self):
        assert c.FOCUS_MODEL_PATH == c.FOCUS_DETECTOR_DIR / c.FOCUS_MODEL_FILE
        assert c.FOCUS_MODEL_PATH == Path("models/focus_detector/focus_model.json")


def written(tmp_path, raw):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def shipped():
    return json.loads(c.FOCUS_MODEL_PATH.read_text(encoding="utf-8"))


def with_entry(index, **changes):
    raw = shipped()
    raw["models"][index].update(changes)
    return raw


def load_error(tmp_path, raw):
    """Where ``raw`` was written, and why the program refuses to load it."""
    path = written(tmp_path, raw)
    with pytest.raises(AdamusError) as error:
        load_models(path)
    return path, str(error.value)


class TestLoadingErrors:
    def test_unreadable_file(self, tmp_path):
        missing = tmp_path / "none.json"
        with pytest.raises(AdamusError) as error:
            load_models(missing)
        assert str(error.value).startswith(f"cannot read the focus model {missing}: [Errno ")

    def test_not_json(self, tmp_path):
        (tmp_path / "model.json").write_text("{", encoding="utf-8")
        with pytest.raises(AdamusError, match="cannot read the focus model"):
            load_models(tmp_path / "model.json")

    @pytest.mark.parametrize("raw", [[1, 2], {"features": 1}, {**dict.fromkeys(FILE_KEYS), "x": 1}])
    def test_file_needs_exactly_its_keys(self, tmp_path, raw):
        path, error = load_error(tmp_path, raw)
        assert error == f"{path}: a focus model file needs exactly {', '.join(FILE_KEYS)}"

    def test_other_features(self, tmp_path):
        raw = shipped()
        raw["features"]["artifact_uv"] = 100.0
        path, error = load_error(tmp_path, raw)
        assert error == f"{path}: the model was trained on other features than these"

    @pytest.mark.parametrize(
        "trained_on", [[], {"people": []}, {"people": [1]}, {"people": "Person01"}]
    )
    def test_people_must_be_named(self, tmp_path, trained_on):
        raw = {**shipped(), "trained_on": trained_on}
        path, error = load_error(tmp_path, raw)
        assert error == f"{path}: trained_on must name the people the model was trained on"

    @pytest.mark.parametrize("models", [{}, [1], "x"])
    def test_models_must_be_a_list_of_models(self, tmp_path, models):
        path, error = load_error(tmp_path, {**shipped(), "models": models})
        assert error == f"{path}: models must be a list of models"

    def test_a_model_needs_exactly_its_keys(self, tmp_path):
        raw = shipped()
        del raw["models"][3]["threshold"]
        path, error = load_error(tmp_path, raw)
        assert error == f"{path}: a model needs exactly {', '.join(MODEL_KEYS)}"

    @pytest.mark.parametrize("names", [["Fp1"], ["AF8", "AF7"], "AF7", []])
    def test_unknown_set_of_electrodes(self, tmp_path, names):
        path, error = load_error(tmp_path, with_entry(0, electrodes=names))
        assert error == f"{path}: unknown set of electrodes {names!r}"

    @pytest.mark.parametrize("change", [{"coef": ["a"] * 14}, {"intercept": [1, 2]},
                                        {"threshold": None}])  # fmt: skip
    def test_numbers_expected(self, tmp_path, change):
        path, error = load_error(tmp_path, with_entry(0, **change))
        prefix = f"{path}: numbers expected: "
        assert error.startswith(prefix)
        assert error[len(prefix)].isalpha()  # the problem follows at once
        assert "float" in error

    @pytest.mark.parametrize("change", [
        {"coef": [0.0] * 13}, {"prior": [0.0] * 15}, {"mean": [[0.0] * 14]},
        {"scale": [1.0] * 13 + [0.0]}, {"scale": [1.0] * 13 + [-1.0]},
        {"coef": [0.0] * 13 + [math.inf]}, {"prior": [math.nan] * 14},
        {"intercept": math.inf}, {"threshold": math.nan},
    ])  # fmt: skip
    def test_numbers_that_do_not_fit(self, tmp_path, change):
        path, error = load_error(tmp_path, with_entry(0, **change))
        assert error == f"{path}: the numbers of the model of AF7 do not fit"

    def test_the_electrodes_of_a_model_that_does_not_fit_are_listed(self, tmp_path):
        raw = shipped()
        index = next(k for k, m in enumerate(raw["models"]) if m["electrodes"] == ["AF7", "TP9"])
        path, error = load_error(tmp_path, with_entry(index, intercept=math.inf))
        assert error == f"{path}: the numbers of the model of AF7, TP9 do not fit"

    def test_a_set_without_a_model(self, tmp_path):
        raw = shipped()
        del raw["models"][5]
        path, error = load_error(tmp_path, raw)
        assert error == f"{path}: there must be one model for every set of electrodes"

    def test_a_set_with_two_models(self, tmp_path):
        raw = shipped()
        raw["models"][5] = raw["models"][4]
        path = written(tmp_path, raw)
        with pytest.raises(AdamusError, match="one model for every set of electrodes"):
            load_models(path)
        raw = shipped()
        raw["models"].append(raw["models"][0])
        with pytest.raises(AdamusError, match="one model for every set of electrodes"):
            load_models(written(tmp_path, raw))


class TestDetector:
    def model(self, seed=0):
        rng = np.random.default_rng(seed)
        return model_of(prior=rng.normal(size=158), mean=rng.normal(size=158),
                        scale=rng.uniform(0.5, 2, 158), coef=rng.normal(size=158),
                        intercept=0.3, threshold=-0.2)  # fmt: skip

    def test_the_first_step_is_scored_against_the_prior(self):
        model = self.model()
        step = covariances(seed=1)
        score = Detector(model, 4, 10).step(step)
        assert score == pytest.approx(model.score(log_vector(step) - model.prior))

    def test_every_step_is_scored_against_the_mean_before_it_then_joins_it(self):
        model = self.model()
        detector = Detector(model, 4, 10)
        running = RunningMean(model.prior, 4, 10)
        for seed in range(6):
            vector = log_vector(covariances(seed=seed))
            assert detector.step(covariances(seed=seed)) == pytest.approx(
                model.score(vector - running.mean)
            )
            running.add(vector)

    def test_a_person_unlike_the_training_people_is_followed_after_a_while(self):
        """The same step again and again: its distance to the mean, so its score, settles."""
        model = model_of(coef=np.ones(158))
        detector = Detector(model, 4, 10)
        scores = [detector.step(covariances(seed=7)) for _ in range(200)]
        assert abs(scores[-1]) < 1e-6 < abs(scores[0])
