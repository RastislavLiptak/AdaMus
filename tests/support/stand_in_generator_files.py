"""A stand-in generator installation: the folder layout of the real one, with a stand-in pd."""

import stat
import sys
from pathlib import Path
from typing import Any

from adamus import constants as c

STAND_IN_PD = Path(__file__).parent / "stand_in_pure_data.py"


def install_stand_in_generator(folder: Path) -> dict[str, Any]:
    """Create the installation in ``folder``; return the ``generator`` section that uses it.

    The pd binary runs tests/support/stand_in_pure_data.py; the external, the assets and
    the model are empty files where the program looks for them.
    """
    install_generator_files(folder, "stand_in_model")
    with (folder / c.GENERATOR_PD_BINARY).open("w", encoding="utf-8") as script:
        print("#!/bin/sh", file=script)
        print(f'exec "{sys.executable}" "{STAND_IN_PD}" "$@"', file=script)
    return {"folder": str(folder), "model": "stand_in_model"}


def install_generator_files(folder: Path, model: str) -> None:
    """Empty stand-ins of every file of an installed generator; pd can be run."""
    files = [c.GENERATOR_PD_BINARY, c.GENERATOR_EXTERNAL_FILE,
             c.GENERATOR_MODEL_FILE.format(model=model),
             *(f"{c.GENERATOR_ASSETS}/{file}" for file in c.GENERATOR_ASSET_FILES)]  # fmt: skip
    for file in files:
        (folder / file).parent.mkdir(parents=True, exist_ok=True)
        (folder / file).write_bytes(b"")
    binary = folder / c.GENERATOR_PD_BINARY
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
