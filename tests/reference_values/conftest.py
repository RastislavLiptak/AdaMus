"""The reference values hold for one model only."""

import pytest

from tests.support.project_paths import require_reference_model


@pytest.fixture(autouse=True)
def the_model_of_the_reference_values():
    require_reference_model()
