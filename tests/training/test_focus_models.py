"""The focus models, from the data they are trained on: the project's model is what the training
makes now, and the models tell people they had not seen apart as the cross-validation found.

Every person of the downloaded dataset is read once for all of it (a minute, side by side on
the processors). It is one test, each check reported on its own, so that one process of the
tests holds the people: they take gigabytes.
"""

import io
import json

import pytest

from adamus import constants as c
from adamus.detection import focus_training as train
from tests.support.focus_model_quality import check
from tests.support.project_paths import PROJECT_MODEL

# Minutes, and gigabytes: run by the full run only.
pytestmark = pytest.mark.slow

# The sets of electrodes the cross-validation was done on, and what it found for each: the AUC
# of f and the balanced accuracy, over people the models had not seen.
CROSS_VALIDATION = {c.DETECTOR_ELECTRODES: (0.882, 0.811), ("AF8", "TP9", "TP10"): (0.860, 0.780)}
FOUND_WITHIN = 0.005


def test_the_focus_models_from_their_training_data(subtests):
    data = train.download(PROJECT_MODEL.parent / train.TRAINING_DATA, io.StringIO())  # is there
    people = train.read_people(data)
    with subtests.test("the project model is what the training makes now"):
        # a change of the training that was not trained again would leave an old model in use
        made = json.loads(json.dumps(train.trained_models(people, io.StringIO())))
        assert made == json.loads(PROJECT_MODEL.read_text())
    for electrodes, (auc_of_f, balanced) in CROSS_VALIDATION.items():
        with subtests.test("new people are told apart as in the cross-validation", of=electrodes):
            result = check(train.usable_in_set(people, electrodes))
            assert result["auc of f"] == pytest.approx(auc_of_f, abs=FOUND_WITHIN)
            assert result["balanced"] == pytest.approx(balanced, abs=FOUND_WITHIN)
