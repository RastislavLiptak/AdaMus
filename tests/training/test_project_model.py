"""The project's focus models are what the training code makes now (minutes, gigabytes)."""

import io
import json

import pytest

from adamus.detection import focus_training as train
from tests.support.project_paths import PROJECT_MODEL

pytestmark = pytest.mark.training


def test_the_project_model_is_what_the_training_makes_now(tmp_path):
    """A change of the training that was not trained again would leave an old model in use."""
    data = PROJECT_MODEL.parent / train.TRAINING_DATA
    (tmp_path / train.TRAINING_DATA).symlink_to(data.resolve())  # the data, not downloaded again
    made = train.train(tmp_path, io.StringIO())
    assert json.loads(made.read_text()) == json.loads(PROJECT_MODEL.read_text())
