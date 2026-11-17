"""The focus models on people they had not seen, from the data they were trained on.

Every person of the downloaded dataset is read again (minutes), then seven groups
of people are each scored by models of the others, with thresholds of the others.
"""

import io

import pytest

from adamus import constants as c
from adamus.detection import focus_training as train
from tests.support.focus_model_quality import check

pytestmark = pytest.mark.training


@pytest.fixture(scope="module")
def people():
    data = train.download(c.FOCUS_DETECTOR_DIR / train.TRAINING_DATA, io.StringIO())
    folders = sorted(p for p in data.glob("Experiment_*/*") if p.is_dir())
    return [train.person_steps(folder) for folder in folders]


@pytest.mark.parametrize(
    ("electrodes", "auc_of_f", "balanced"),
    [(c.DETECTOR_ELECTRODES, 0.882, 0.811), (("AF8", "TP9", "TP10"), 0.860, 0.780)],
)
def test_new_people_are_told_apart_as_in_the_cross_validation(
    people, electrodes, auc_of_f, balanced
):
    views = (train.in_set(person, electrodes) for person in people)
    result = check([view for view in views if train.usable(view)])
    assert result["auc of f"] == pytest.approx(auc_of_f, abs=0.005)
    assert result["balanced"] == pytest.approx(balanced, abs=0.005)
