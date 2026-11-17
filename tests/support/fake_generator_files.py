"""A fake generator installation: the folder layout of the real one, with a stand-in pd."""

import socket
import stat
import sys
from pathlib import Path

from adamus import constants as c

FAKE_PD = Path(__file__).parent / "fake_pure_data.py"


def install_fake_generator(folder: Path) -> dict:
    """Create the installation in ``folder``; return the ``generator`` section that uses it.

    The pd binary runs tests/support/fake_pure_data.py; the external, the assets and
    the model are empty files where the program looks for them. The port is a
    free local one.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    install_generator_files(folder, "fake_model")
    with (folder / c.GENERATOR_PD_BINARY).open("w", encoding="utf-8") as script:
        print("#!/bin/sh", file=script)
        print(f'exec "{sys.executable}" "{FAKE_PD}" "$@"', file=script)
    return {"folder": str(folder), "model": "fake_model", "port": port}


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
