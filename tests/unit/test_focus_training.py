"""The training of the focus models: the Muse dataset, the regression, the threshold, the file."""

import io
import zipfile

import numpy as np
import pytest
from scipy.optimize import minimize

from adamus import constants as c
from adamus.detection import focus_training as train
from adamus.detection.focus_control import focus_of, next_level
from adamus.detection.focus_detector import (
    FocusModel,
    RunningMean,
    electrode_sets,
    load_models,
    vector_length,
)
from adamus.detection.signal_processing import clean, design_filters, process
from tests.support.focus_model_quality import auc, controlled, judged

START = 1_600_000_000  # Start_time_unix of the made-up experiment
EVERY = c.DETECTOR_ELECTRODES


def write_log(folder, blocks):
    """n_back_responses.csv: blocks of (session, block number, first onset s, letters)."""
    header = ["Start_time_unix", "Running[Block]", "Block", "Fixation.OnsetTime",
              "Stimulus100.RESP", "Stimulus100.OnsetTime", "Stimulus101.OnsetTime",
              "Other"]  # fmt: skip
    lines = [",".join(header)]
    for session, number, first_s, letters in blocks:
        for k in range(letters):
            onset = round((first_s + 2 * k) * 1000)
            column = [str(onset), ""] if number % 2 else ["", str(onset)]
            decoys = [str(onset - 900), "1"]  # a fixation before the letter, a key press
            lines.append(",".join([str(START), session, str(number), *decoys, *column, "x"]))
    lines.append(",".join([str(START), "NoMusic", "", "", "", "", "", "rest row"]))
    (folder / "n_back_responses.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tasks_and_breaks(t, focus_spans):
    """Four electrodes: slow rhythms weaker inside the focus spans, as in the real data."""
    rng = np.random.default_rng(int(t.size))
    slow = np.where(np.any([(t >= a) & (t < b) for a, b in focus_spans], axis=0), 0.5, 1.0)
    rows = [
        slow * 20 * np.sin(2 * np.pi * (6 + k) * t) + rng.normal(0, 3, t.size) for k in range(4)
    ]
    return np.stack(rows)


def write_eeg(folder, seconds, signal=None, holes=(), fs=256):
    """EEG_recording.csv: time, TP9, AF7, AF8, TP10, AUX, with slightly irregular times."""
    count = int(seconds * fs)
    times = START + np.arange(count) / fs + np.random.default_rng(1).uniform(0, 1e-4, count)
    for start, stop in holes:
        keep = (times < START + start) | (times >= START + stop)
        times = times[keep]
    if signal is None:
        rng = np.random.default_rng(0)
        signal = lambda t: np.stack(  # noqa: E731
            [10 * np.sin(2 * np.pi * (5 + k) * t) + rng.normal(0, 3, len(t)) for k in range(4)]
        )
    values = signal(times - START)  # in the detector's order AF7, AF8, TP9, TP10
    columns = values[[2, 0, 1, 3]]  # the file's order TP9, AF7, AF8, TP10
    table = np.column_stack([times, columns.T, np.zeros(len(times))])
    header = "timestamps,TP9,AF7,AF8,TP10,Right AUX"
    np.savetxt(
        folder / "EEG_recording.csv", table, delimiter=",", header=header, comments="", fmt="%.6f"
    )


BLOCK_STARTS = (70.0, 129.0, 188.0, 247.0)  # four blocks of 44 s, 15 s apart
RECORDING_S = 300.0


def made_up_person(folder, seed=0, dead=None):
    """A minute before the first block, then four blocks of 44 s with breaks between them."""
    folder.mkdir(parents=True)
    focus = [(start, start + 44.0) for start in BLOCK_STARTS]

    def signal(t):
        rows = tasks_and_breaks(t, focus)
        rows += np.random.default_rng(seed).normal(0, 2, rows.shape)  # every person differs
        if dead is not None:  # an electrode without contact: loud broadband noise
            rows[dead] = np.random.default_rng(seed).normal(0, 100, t.size)
        return rows

    write_eeg(folder, RECORDING_S, signal=signal)
    write_log(folder, [("NoMusic", k + 1, start, 22) for k, start in enumerate(BLOCK_STARTS)])
    return folder


class Log(io.StringIO):
    """A log that remembers how much had been written each time it was flushed."""

    def __init__(self):
        super().__init__()
        self.flushed = []

    def flush(self):
        self.flushed.append(len(self.getvalue()))
        super().flush()

    def every_line_flushed(self):
        text = self.getvalue()
        ends = [k + 1 for k, letter in enumerate(text) if letter == "\n"]
        return bool(ends) and set(ends) <= set(self.flushed)


class TestTheLog:
    def test_blocks_start_at_the_first_letter_and_end_two_seconds_after_the_last(self, tmp_path):
        write_log(tmp_path, [("PracticeSession", 1, 10.0, 3), ("NoMusic", 3, 100.0, 22),
                             ("NoMusic", 4, 159.0, 22)])  # fmt: skip
        assert train.blocks_of(tmp_path) == [
            (START + 100.0, START + 100.0 + 42 + 2.0),
            (START + 159.0, START + 159.0 + 42 + 2.0),
        ]

    def test_a_gap_of_forty_seconds_or_more_is_a_pause_between_sessions(self):
        blocks = [(0.0, 44.0), (84.0, 128.0), (167.5, 211.5)]  # gaps of 40 s and 39.5 s
        assert train.states_of(blocks) == [
            (0.0, 44.0, 1), (84.0, 128.0, 1), (129.0, 161.5, 0), (167.5, 211.5, 1),
        ]  # fmt: skip

    def test_states_are_the_blocks_and_the_breaks_between_them(self):
        blocks = [(0.0, 44.0), (59.0, 103.0), (400.0, 444.0)]
        assert train.states_of(blocks) == [
            (0.0, 44.0, 1),
            (45.0, 53.0, 0),  # a second after the block, to six seconds before the next
            (59.0, 103.0, 1),  # the 297 s before the next block are a pause between sessions
            (400.0, 444.0, 1),
        ]


class TestTheSignal:
    def test_signal_is_on_an_even_grid_in_the_detector_order(self, tmp_path):
        write_eeg(tmp_path, 4.0, signal=lambda t: np.stack([t, 2 * t, 3 * t, 4 * t]))
        signal, hole, first = train.signal_of(tmp_path)
        stamps = np.loadtxt(tmp_path / "EEG_recording.csv", delimiter=",", skiprows=1)[:, 0]
        assert first == stamps[0]  # from the first sample to the last, every 1/256 s
        assert signal.shape == (4, int((stamps[-1] - stamps[0]) * 256))
        t = np.arange(signal.shape[1]) / 256
        assert np.allclose(signal, np.stack([t, 2 * t, 3 * t, 4 * t]), atol=1e-3)
        assert not hole.any()

    def test_gaps_in_the_recording_are_marked(self, tmp_path):
        write_eeg(tmp_path, 4.0, holes=[(2.0, 2.3)])
        _signal, hole, _first = train.signal_of(tmp_path)
        assert 0.29 * 256 <= hole.sum() <= 0.31 * 256  # the whole gap, no more
        assert hole[int(2.01 * 256)] and hole[int(2.29 * 256)] and not hole[int(1.0 * 256)]

    def test_a_gap_after_the_first_sample_or_before_the_last_is_marked_too(self, tmp_path):
        write_eeg(tmp_path, 4.0, holes=[(0.001, 0.3), (3.7, 3.995)])
        _signal, hole, _first = train.signal_of(tmp_path)
        assert hole[0] and hole[-1] and not hole[int(2.0 * 256)]


@pytest.fixture(scope="module")
def people(tmp_path_factory):
    """Five made-up people; the last has an AF7 without contact."""
    root = tmp_path_factory.mktemp("dataset")
    folders = [made_up_person(root / "Experiment_1" / f"Person0{k}", seed=k) for k in range(4)]
    folders.append(made_up_person(root / "Experiment_2" / "Excluded_ID_1", seed=4, dead=0))
    read = {folder: train.person_steps(folder) for folder in folders}
    return root, list(read.values()), read


class TestUsable:
    """Enough clean steps of each state, and clean signal at least half of the time."""

    def person(self, rest, focus, unlabelled, invalid):
        states = np.array([0] * rest + [1] * focus + [-1] * (unlabelled + invalid))
        valid = np.array([True] * (rest + focus + unlabelled) + [False] * invalid)
        return {"valid": valid, "states": states}

    def test_thirty_steps_of_each_state_and_half_of_the_time_clean(self):
        assert train.usable(self.person(30, 30, 40, 100))

    @pytest.mark.parametrize(
        ("rest", "focus", "unlabelled", "invalid"),
        [(29, 30, 41, 100), (30, 29, 41, 100), (30, 30, 39, 101), (0, 300, 300, 0)],
    )
    def test_fewer_steps_of_a_state_or_less_clean_signal(self, rest, focus, unlabelled, invalid):
        assert not train.usable(self.person(rest, focus, unlabelled, invalid))


class TestOnePerson:
    def test_named_by_experiment_and_folder(self, people):
        _root, steps, _read = people
        assert [p["name"] for p in steps][-2:] == [
            "Experiment_1/Person03",
            "Experiment_2/Excluded_ID_1",
        ]

    def test_a_person_keeps_what_every_set_is_computed_from_not_the_sets(self, people):
        """The log vectors of all fifteen sets of all people would not fit into memory."""
        person = people[1][0]
        steps = len(person["states"])
        assert sorted(person) == sorted(["name", "states", "covariances", "max_abs_uv",
                                         "noise_uv", "carries_signal", "gaps"])  # fmt: skip
        assert [m.shape for m in person["covariances"]] == [(steps, 4, 4)] * 8 + [(steps, 12, 12)]
        assert person["max_abs_uv"].shape == person["noise_uv"].shape == (steps, 4)

    def test_log_vectors_and_validity_of_every_set_of_electrodes(self, people):
        person = people[1][0]
        for electrodes in electrode_sets():
            view = train.in_set(person, electrodes)
            assert (view["name"], view["electrodes"]) == (person["name"], electrodes)
            assert view["logs"].shape == (len(person["states"]), vector_length(len(electrodes)))
            assert view["valid"].shape == person["states"].shape and view["valid"].all()

    def test_validity_in_a_set_is_that_of_the_detector_step_by_step(self, people, tmp_path):
        made_up_person(tmp_path / "Experiment_1" / "Q", seed=1, dead=0)
        signal, _hole, _first = train.signal_of(tmp_path / "Experiment_1" / "Q")
        person, filters = people[1][-1], design_filters(256)  # the same person: a dead AF7
        for electrodes in [EVERY, ("AF8", "TP10"), ("AF7",), ("TP9",)]:
            view = train.in_set(person, electrodes)
            for k in (0, 40, len(view["valid"]) - 1):
                step = process(signal[:, k * 128 : k * 128 + 1024], 512, filters, EVERY)
                assert view["valid"][k] == clean(step, electrodes)

    def test_states_of_the_whole_windows(self, people):
        """Blocks of 44 s give up to (44 - 2) / 0.5 + 1 = 85 steps, breaks of 8 s up to 13."""
        states = people[1][0]["states"]
        assert 4 * 84 <= (states == 1).sum() <= 4 * 85
        assert 3 * 12 <= (states == 0).sum() <= 3 * 13
        assert set(np.unique(states)) == {-1, 0, 1}

    def test_a_step_every_hop_from_the_first_whole_context_to_the_end(self, people):
        root, steps, _read = people
        signal, _hole, _first = train.signal_of(root / "Experiment_1" / "Person01")
        assert len(steps[1]["states"]) == (signal.shape[1] - 1024) // 128 + 1

    def test_steps_over_a_gap_in_the_recording_are_not_clean(self, tmp_path):
        folder = tmp_path / "Experiment_1" / "Person09"
        folder.mkdir(parents=True)
        write_eeg(folder, 60.0, holes=[(30.0, 30.3)])
        write_log(folder, [("NoMusic", 1, 20.0, 5)])
        valid = train.in_set(train.person_steps(folder), EVERY)["valid"]
        ends = (1024 + 128 * np.arange(len(valid))) / 256  # the end of each step's window, s
        unclean = ends[~valid]
        assert len(unclean) >= 4  # every window that holds a part of the gap
        assert unclean.min() >= 30.0 and unclean.max() - 2.0 <= 30.3

    def test_an_electrode_without_contact_spoils_only_the_sets_with_it(self, people):
        dead = people[1][-1]
        assert not train.in_set(dead, ("AF7",))["valid"].any()
        assert not train.in_set(dead, EVERY)["valid"].any()
        behind = train.in_set(dead, ("AF8", "TP9", "TP10"))
        assert behind["valid"].all()
        assert not train.usable(train.in_set(dead, EVERY)) and train.usable(behind)

    def test_logs_are_those_the_detector_computes(self, people, tmp_path):
        from adamus.detection.focus_detector import log_vector

        made_up_person(tmp_path / "Experiment_1" / "Q", seed=1)
        signal, _hole, _first = train.signal_of(tmp_path / "Experiment_1" / "Q")
        filters, k = design_filters(256), 40
        context = signal[:, k * 128 : k * 128 + 1024]
        for electrodes in [EVERY, ("AF8", "TP10"), ("TP9",)]:
            expected = log_vector(with_only(context, electrodes, filters).covariances)
            assert np.allclose(train.in_set(people[1][1], electrodes)["logs"][k], expected)


def with_only(context, electrodes, filters):
    """The step a source with only ``electrodes`` gives: zeros where the others are."""
    zeros = context.copy()
    zeros[[EVERY.index(name) for name in EVERY if name not in electrodes]] = 0.0
    return process(zeros, 512, filters, electrodes)


class TestSubsetCovariances:
    def test_cut_out_like_the_detector_computes_them_for_fewer_electrodes(self):
        rng = np.random.default_rng(3)
        context = rng.normal(0, 10, (4, 1024))
        filters = design_filters(256)
        every = process(context, 512, filters, EVERY).covariances
        for electrodes in [("AF7", "TP10"), ("AF8", "TP9", "TP10"), ("TP10",)]:
            some = with_only(context, electrodes, filters).covariances
            for cut, own in zip(train.subset_covariances(every, electrodes), some, strict=True):
                assert np.allclose(cut, own)


def test_the_training_scores_a_step_as_the_detector_will(people):
    """The threshold is found on the training's scores: the detector must score the same."""
    steps = [train.in_set(p, EVERY) for p in people[1][:4]]
    entry = train.model_entry(steps, EVERY)
    numbers = {key: np.array(entry[key]) for key in ("prior", "mean", "scale", "coef")}
    model = FocusModel(EVERY, **numbers, intercept=entry["intercept"], threshold=entry["threshold"])
    x, _states = train.labelled(steps[0], numbers["prior"])
    fitted = {**numbers, "intercept": entry["intercept"]}
    expected = train.scores(fitted, x[:20]) - entry["threshold"]
    assert [model.score(vector) for vector in x[:20]] == pytest.approx(expected, abs=1e-9)


class TestCentring:
    def test_each_step_minus_the_mean_before_it_the_clean_ones_join(self):
        logs = np.arange(12.0).reshape(6, 2)
        valid = np.array([True, False, True, True, False, True])
        prior = np.array([1.0, -1.0])
        running = RunningMean(prior, c.PRIOR_STEPS, c.MEMORY_STEPS)
        expected = []
        for vector, is_clean in zip(logs, valid, strict=True):
            expected.append(vector - running.mean)
            if is_clean:
                running.add(vector)
        assert np.allclose(train.centred(logs, valid, prior), expected)

    def test_labelled_keeps_the_clean_steps_of_a_state(self, people):
        person = train.in_set(people[1][-1], ("AF8", "TP9"))
        prior = np.zeros(vector_length(2))
        x, y = train.labelled(person, prior)
        keep = person["valid"] & (person["states"] >= 0)
        assert len(x) == len(y) == keep.sum() and set(y) == {0, 1}
        assert np.allclose(x, train.centred(person["logs"], person["valid"], prior)[keep])

    def test_prior_is_the_mean_log_vector_of_the_clean_steps(self, people):
        steps = [train.in_set(p, EVERY) for p in people[1]]
        prior = train.prior_of(steps)
        rows = [p["logs"][p["valid"]] for p in steps]
        assert np.allclose(prior, np.vstack(rows).mean(axis=0))


class TestLogisticRegression:
    def data(self, seed=0, n0=300, n1=100):
        rng = np.random.default_rng(seed)
        x = np.vstack([rng.normal([0, 0, 5], [1, 2, 0.1], (n0, 3)),
                       rng.normal([1.5, 1, 5], [1, 2, 0.1], (n1, 3))])  # fmt: skip
        return x, np.array([0] * n0 + [1] * n1)

    def test_minimises_the_weighted_penalised_log_loss(self):
        """The same objective written out on its own, minimised without a gradient."""
        x, y = self.data()
        model = train.fit_logistic(x, y)
        z = (x - x.mean(axis=0)) / x.std(axis=0)
        weights = np.where(y == 1, len(y) / (2 * 100), len(y) / (2 * 300))

        def objective(theta):
            margin = z @ theta[:-1] + theta[-1]
            loss = np.logaddexp(0, -np.where(y == 1, margin, -margin))
            return train.C_REGULARISATION * np.sum(weights * loss) + 0.5 * theta[:-1] @ theta[:-1]

        direct = minimize(objective, np.zeros(4), method="Nelder-Mead",
                          options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000})  # fmt: skip
        assert np.allclose(np.append(model["coef"], model["intercept"]), direct.x, atol=1e-3)

    def test_the_loss_and_its_gradient(self):
        rng = np.random.default_rng(5)
        z, y = rng.normal(0, 1, (40, 3)), np.array([0, 1] * 20)
        weights, theta = rng.uniform(0.5, 2, 40), np.array([0.3, -0.7, 1.1, 0.2])
        value, gradient = train.penalised_loss(theta, z, y, weights)
        margin = z @ theta[:3] + theta[3]
        log_loss = np.where(y == 1, np.log1p(np.exp(-margin)), np.log1p(np.exp(margin)))
        expected = 0.1 * np.sum(weights * log_loss) + (0.09 + 0.49 + 1.21) / 2
        assert value == pytest.approx(expected, rel=1e-12)
        numeric = [
            (train.penalised_loss(theta + step, z, y, weights)[0]
             - train.penalised_loss(theta - step, z, y, weights)[0]) / 2e-6
            for step in np.eye(4) * 1e-6
        ]  # fmt: skip
        assert np.allclose(gradient, numeric, atol=1e-6)

    def test_reference_weights_of_a_fixed_example(self):
        """With C 0.1: the numbers once computed, so a change of the fit is noticed."""
        model = train.fit_logistic(*self.data())
        assert np.allclose(model["coef"], [1.24674526, 0.34143395, -0.00132067], atol=1e-6)
        assert model["intercept"] == pytest.approx(-0.41387802, abs=1e-6)

    def test_scores_follow_the_standardised_features(self):
        x, y = self.data()
        model = train.fit_logistic(x, y)
        manual = ((x - model["mean"]) / model["scale"]) @ model["coef"] + model["intercept"]
        assert np.allclose(train.scores(model, x), manual)
        assert auc(train.scores(model, x)[y == 1], train.scores(model, x)[y == 0]) > 0.8

    def test_a_feature_that_never_changes_does_not_break_the_standardisation(self):
        x, y = self.data()
        x[:, 2] = 7.0
        model = train.fit_logistic(x, y)
        assert model["scale"][2] == 1.0 and np.isfinite(model["coef"]).all()

    def test_classes_count_equally_however_many_steps_each_has(self):
        x, y = self.data(n0=900, n1=100)
        s = train.scores(train.fit_logistic(x, y), x)
        assert 0.6 < np.mean(s[y == 1] > 0) and np.mean(s[y == 0] <= 0) > 0.6


class TestThreshold:
    def test_balanced_threshold_lies_between_the_best_split(self):
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        states = np.array([0, 0, 1, 1, 0, 1])
        assert train.balanced_threshold(values, states) == 1.5  # rest 2/3 right, focus all caught

    def test_balanced_threshold_weighs_the_states_equally(self):
        values = np.arange(10.0)
        states = np.array([0] * 8 + [1] * 2)
        assert train.balanced_threshold(values, states) == 7.5

    def test_a_single_rest_step_counts_as_much_as_all_the_focus_steps(self):
        values = np.arange(5.0)
        states = np.array([1, 1, 1, 1, 0])  # the rest step scores highest
        assert train.balanced_threshold(values, states) == 4.5  # rest right, no focus caught

    def test_a_single_focus_step_counts_as_much_as_all_the_rest_steps(self):
        values = np.arange(5.0)
        states = np.array([0, 1, 0, 0, 0])  # all focus caught is worth three rest steps wrong
        assert train.balanced_threshold(values, states) == 0.5

    def test_halfway_to_the_next_value_however_far_it_is(self):
        values = np.array([0.0, 1.0, 5.0])
        states = np.array([0, 0, 1])
        assert train.balanced_threshold(values, states) == 3.0

    def test_above_the_highest_value_when_nothing_should_be_focus(self):
        assert train.balanced_threshold(np.array([1.0, 2.0]), np.array([0, 0])) == 2.5

    def test_every_person_is_scored_by_a_model_that_never_saw_them(self):
        people = [{"name": f"P{k}"} for k in range(10)]
        folds = train.held_out(people)
        assert len(folds) == train.GROUPS == 4
        scored = [person["name"] for _, group in folds for person in group]
        assert sorted(scored) == sorted(person["name"] for person in people)  # each once
        for fitted_on, group in folds:
            assert {p["name"] for p in fitted_on} | {p["name"] for p in group} == set(scored)
            assert not {p["name"] for p in fitted_on} & {p["name"] for p in group}

    def test_the_threshold_is_that_of_the_steps_scored_so(self, people, monkeypatch):
        """threshold_of scores the people as held_out divides them, and nothing else."""
        steps = [train.in_set(p, ("AF8", "TP9")) for p in people[1]]
        values, states = [], []
        for fitted_on, group in train.held_out(steps):
            model = train.fit_people(fitted_on)
            for person in group:
                x, y = train.labelled(person, model["prior"])
                values.append(train.scores(model, x))
                states.append(y)
        expected = train.balanced_threshold(np.concatenate(values), np.concatenate(states))
        assert train.threshold_of(steps) == expected


class TestTheQualityCheck:
    """The measures the check of the trained models uses (tests/support/focus_model_quality)."""

    def test_auc(self):
        assert auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
        assert auc(np.array([1.0]), np.array([1.0])) == 0.5
        assert auc(np.array([0.0, 3.0]), np.array([1.0, 2.0])) == 0.5

    def test_f_is_made_as_the_controller_makes_it(self):
        values = np.array([0.3, 1.0, -2.0, 0.5])
        valid = np.array([False, True, False, True])
        f = controlled(values, valid)
        level = next_level(0.0, 1.0)
        assert np.isnan(f[0]) and f[1] == pytest.approx(focus_of(level))
        assert f[2] == f[1]  # held
        assert f[3] == pytest.approx(focus_of(next_level(level, 0.5)))

    def test_judged_reports_both_states(self, people):
        steps = [train.in_set(p, EVERY) for p in people[1]]
        model = train.fit_people(steps[:4])
        result = judged(steps[0], model, 0.0)
        assert set(result) == {"auc", "auc of f", "focus caught", "rest right"}
        assert result["auc"] > 0.9  # the made-up slow rhythms are easy


class TestDownload:
    def archive(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("dataset-1.0.0/Experiment_1/Person01/EEG_recording.csv", "x")
        return buffer.getvalue()

    def serve(self, monkeypatch, content):
        asked = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *problem):
                return False

        def urlopen(url, timeout):
            asked.append(url)
            self.timeouts.append(timeout)
            return Response(content)

        self.timeouts = []

        monkeypatch.setattr(train.urllib.request, "urlopen", urlopen)
        return asked

    def test_an_unpacked_dataset_is_used_as_it_is(self, tmp_path, monkeypatch):
        (tmp_path / "dataset-1.0.0").mkdir()
        monkeypatch.setattr(
            train.urllib.request, "urlopen", lambda url, timeout: pytest.fail("download")
        )
        assert train.download(tmp_path, io.StringIO()) == tmp_path / "dataset-1.0.0"

    def test_otherwise_it_is_downloaded_and_unpacked(self, tmp_path, monkeypatch):
        asked = self.serve(monkeypatch, self.archive())
        log = Log()
        folder = train.download(tmp_path / "project" / "models" / "training_data", log)
        url = "https://physionet.org/content/brain-wearable-monitoring/get-zip/1.0.0/"
        assert asked == [url] == [train.DATASET_URL] and folder.name == "dataset-1.0.0"
        assert (folder / "Experiment_1" / "Person01" / "EEG_recording.csv").read_text() == "x"
        assert log.getvalue() == f"downloading {url} ...\n"
        assert log.every_line_flushed()
        assert self.timeouts == [60.0]  # a stalled connection ends the training, not the program
        models = tmp_path / "project" / "models"
        assert sorted(p.name for p in models.iterdir()) == ["training_data"]

    def test_what_an_interrupted_unpacking_left_is_not_taken_over(self, tmp_path, monkeypatch):
        self.serve(monkeypatch, self.archive())
        (tmp_path / "training_data.part" / "dataset-0.9").mkdir(parents=True)
        folder = train.download(tmp_path / "training_data", io.StringIO())
        assert folder.name == "dataset-1.0.0"
        assert sorted(p.name for p in (tmp_path / "training_data").iterdir()) == ["dataset-1.0.0"]
        assert sorted(p.name for p in tmp_path.iterdir()) == ["training_data"]

    def test_a_broken_download_leaves_nothing_that_looks_unpacked(self, tmp_path, monkeypatch):
        self.serve(monkeypatch, self.archive()[:-30])
        target = tmp_path / "training_data"
        with pytest.raises(zipfile.BadZipFile):
            train.download(target, io.StringIO())
        assert not target.exists()
        self.serve(monkeypatch, self.archive())  # the next start downloads it again
        assert train.download(target, io.StringIO()).name == "dataset-1.0.0"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["training_data"]


class TestTrain:
    def test_writes_a_model_for_every_set_that_the_detector_reads(
        self, people, monkeypatch, tmp_path
    ):
        root, _steps, read = people
        asked = []
        monkeypatch.setattr(train, "download", lambda target, log: asked.append(target) or root)
        monkeypatch.setattr(train, "person_steps", read.__getitem__)  # read once, by the fixture
        log = Log()
        target = train.train(tmp_path, log)
        assert target == tmp_path / c.FOCUS_MODEL_FILE
        assert asked == [tmp_path / "training_data"]
        models = load_models(target)
        assert sorted(models.by_electrodes) == sorted(electrode_sets())
        trained_on = models.raw["trained_on"]
        assert trained_on["left_out"] == ["Experiment_2/Excluded_ID_1"]
        assert len(trained_on["people"]) == 4
        assert trained_on["people per set"]["AF8+TP9+TP10"] == 5  # the dead AF7 is not in it
        lines = log.getvalue().splitlines()
        assert lines[0] == "Experiment_1/Person00 read"
        assert lines[5] == "AF7 model of 4 people"
        assert lines[9] == "AF7+AF8 model of 4 people"
        assert lines[-1] == f"written: {target}"
        assert log.every_line_flushed()  # the log shows how far a long training got
        assert trained_on["dataset"] == train.DATASET
