"""The audio outputs of the Mac as macOS knows them at the moment: their names and rates, and
the one the system plays on. Read from Core Audio at each call, so that headphones plugged in
meanwhile are there."""

import ctypes
from ctypes import POINTER, byref, c_double, c_uint32, c_void_p, sizeof
from dataclasses import dataclass
from functools import cache
from typing import Any

from adamus.errors import AdamusError

CORE_AUDIO = "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
CORE_FOUNDATION = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
SYSTEM = 1  # the audio object of the whole system
NO_DEVICE = 0
UTF8 = 0x08000100  # the encoding of a name read out of Core Foundation
NAME_BYTES = 1024
FINE = 0  # what Core Audio answers when it could do what it was asked
# The properties asked for, and where they are looked at: the four letters Core Audio names them by.
DEFAULT_OUTPUT, DEVICES = "dOut", "dev#"
NAME, RATE, STREAMS = "lnam", "nsrt", "stm#"
EVERYWHERE, OUTPUT = "glob", "outp"
FIRST_ELEMENT = 0


@dataclass(frozen=True)
class AudioOutput:
    name: str
    rate: float  # samples a second, as the device is set to


class Address(ctypes.Structure):
    _fields_ = [("selector", c_uint32), ("scope", c_uint32), ("element", c_uint32)]


@cache
def core_audio() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    """Core Audio and Core Foundation, loaded once. A device that another program adds or
    removes is seen at the next call (measured on macOS 27: within a second)."""
    audio, foundation = ctypes.CDLL(CORE_AUDIO), ctypes.CDLL(CORE_FOUNDATION)
    common = [c_uint32, POINTER(Address), c_uint32, c_void_p]
    audio.AudioObjectGetPropertyData.argtypes = [*common, POINTER(c_uint32), c_void_p]
    audio.AudioObjectGetPropertyDataSize.argtypes = [*common, POINTER(c_uint32)]
    foundation.CFStringGetCString.argtypes = [c_void_p, ctypes.c_char_p, ctypes.c_long, c_uint32]
    foundation.CFRelease.argtypes = [c_void_p]
    return audio, foundation


def address(selector: str, scope: str) -> Address:
    codes = (int.from_bytes(code.encode(), "big") for code in (selector, scope))
    return Address(*codes, FIRST_ELEMENT)


def answered(status: int) -> None:
    if status != FINE:
        raise AdamusError(f"macOS did not tell the audio devices (Core Audio error {status})")


def ask(device: int, selector: str, value: Any, size: int, scope: str = EVERYWHERE) -> None:
    """Have Core Audio fill ``value`` (``size`` bytes) with a property of ``device``."""
    asked, given = address(selector, scope), c_uint32(size)
    get = core_audio()[0].AudioObjectGetPropertyData
    answered(get(device, byref(asked), 0, None, byref(given), value))


def size_of(device: int, selector: str, scope: str) -> int:
    asked, size = address(selector, scope), c_uint32(0)
    get_size = core_audio()[0].AudioObjectGetPropertyDataSize
    answered(get_size(device, byref(asked), 0, None, byref(size)))
    return size.value


def name_of(device: int) -> str:
    foundation, text = core_audio()[1], c_void_p()
    ask(device, NAME, byref(text), sizeof(text))
    buffer = ctypes.create_string_buffer(NAME_BYTES)
    copied = foundation.CFStringGetCString(text, buffer, NAME_BYTES, UTF8)
    foundation.CFRelease(text)
    return buffer.value.decode() if copied else ""


def output_of(device: int) -> AudioOutput:
    rate = c_double()
    ask(device, RATE, byref(rate), sizeof(rate))
    return AudioOutput(name_of(device), rate.value)


def outputs() -> list[AudioOutput]:
    """Every device that plays sound, in the order macOS lists them."""
    devices = (c_uint32 * (size_of(SYSTEM, DEVICES, EVERYWHERE) // sizeof(c_uint32)))()
    ask(SYSTEM, DEVICES, devices, sizeof(devices))
    return [output_of(device) for device in devices if size_of(device, STREAMS, OUTPUT)]


def system_output() -> AudioOutput | None:
    """The output the system plays on, as chosen in the sound settings; None when it has none."""
    device = c_uint32()
    ask(SYSTEM, DEFAULT_OUTPUT, byref(device), sizeof(device))
    return None if device.value == NO_DEVICE else output_of(device.value)
