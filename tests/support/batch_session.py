"""One session without the console, for tests that need a whole run of the real program."""

from pathlib import Path

from adamus import constants as c
from adamus.config import load_config
from adamus.realtime.music_generator import MusicGenerator
from adamus.realtime.session import SessionResult, real_devices, run_session
from tests.support.stand_in_devices import StopKey


def run_batch_session(config_path: Path, name: str, recording: Path) -> SessionResult:
    """Start the generator, play ``recording`` through one session, stop the generator."""
    config = load_config(config_path)
    generator = MusicGenerator(config.generator, config.timeouts, lambda kind, /, **fields: None)
    try:
        generator.start()
        devices = real_devices(config, recording, generator, StopKey())  # nobody presses it
        return run_session(config, name, devices, c.SESSIONS_DIR, lambda folder: None)
    finally:
        generator.stop()
