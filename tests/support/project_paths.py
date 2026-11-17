"""Where the tests find the files of the project, whatever folder pytest was started in."""

import hashlib
from pathlib import Path

import pytest

from adamus import constants as c
from adamus.datasets import datasets_in
from adamus.own_recordings import OWN_RECORDINGS

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "config.yaml"
DATASETS = ROOT / c.DATASETS_DIR
PROJECT_MODEL = ROOT / c.FOCUS_MODEL_PATH  # trained by the program on its first start
MISSING_MODEL = "the focus models are not trained yet: start adamus once, it trains them"
# The reference values of the tests were computed with exactly this model file.
REFERENCE_MODEL_SHA256 = "60049ee220a08cb31309f56b24162ce6ca531329323eb44972510846ac1cb0ec"
OTHER_MODEL = (
    "is not the model the reference values were computed with: compute them again with it"
    " and put its sha256 into REFERENCE_MODEL_SHA256"
)


def require_trained_models(path: Path = PROJECT_MODEL) -> None:
    """Stop the test run when the program has not trained its models yet. The tests never
    train them themselves: that downloads gigabytes, and under a mutant it would do so into
    the project."""
    if not path.is_file():
        pytest.exit(f"{path}: {MISSING_MODEL}")


def require_reference_model(path: Path = PROJECT_MODEL) -> None:
    """Fail at once, saying why, when the models differ from those of the reference values;
    otherwise every reference test would fail on its own, looking like a fault of the
    program."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_MODEL_SHA256:
        pytest.fail(f"{path} {OTHER_MODEL}")


def shipped_datasets(root: Path = DATASETS) -> list[Path]:
    """The datasets that come with the project; the operator's own recordings are left out."""
    return [dataset for dataset in datasets_in(root) if dataset.name != OWN_RECORDINGS]


def what_the_program_keeps(root: Path = ROOT) -> list[tuple[str, int, int]]:
    """Every file of the project's sessions and data, and its models, with size and time of
    change: what a test must never write to."""
    files = [
        *(root / c.SESSIONS_DIR).rglob("*"),
        *(root / c.DATASETS_DIR).rglob("*"),
        root / c.FOCUS_MODEL_PATH,
    ]
    return sorted(
        (str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in files
        if path.is_file()
    )
