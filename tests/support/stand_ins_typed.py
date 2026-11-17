"""The stand-ins of the tests offer what the session needs of the devices they replace;
mypy checks it here (tests/coding_rules/test_types.py)."""

from adamus.realtime.clock import TimeSource
from adamus.realtime.session import EegSource, MusicOutput, StopKeys
from tests.support.stand_in_devices import FakeClock, FakeGenerator, FakeSource, StopKey


def as_devices(
    source: FakeSource, generator: FakeGenerator, keys: StopKey, clock: FakeClock
) -> tuple[EegSource, MusicOutput, StopKeys, TimeSource]:
    """Every stand-in where the session takes the device: a member missing is a type error."""
    return source, generator, keys, clock
