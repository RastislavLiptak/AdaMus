"""An audio output that comes and goes as a test wants, as headphones do: an aggregate device
over the output the system plays on, made by another process and removed by it again.

    python -m tests.support.temporary_output

makes it, says so, and removes it once its input closes (also when the test that started it
ends any other way). A test starts it as its own process, so that the program under test learns
of it from macOS, as of any device another program adds.
"""

import ctypes
import sys
from ctypes import byref, c_int32, c_uint32, c_void_p, sizeof

from adamus import audio_outputs

NAME = "AdaMus test output"
UID = "cz.adamus.tests.output"  # an aggregate device is known by it
DEVICE_UID = "uid "  # the property of a device that holds its UID
SHARED = 0  # the aggregate is seen by every program, not by its maker alone
INTEGER = 3  # of 32 bits, as Core Foundation stores numbers
ANNOUNCED = "made"  # printed once the device is there


def libraries() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    audio, foundation = audio_outputs.core_audio()
    foundation.CFStringCreateWithCString.restype = c_void_p
    foundation.CFStringCreateWithCString.argtypes = [c_void_p, ctypes.c_char_p, c_uint32]
    foundation.CFNumberCreate.restype = c_void_p
    foundation.CFNumberCreate.argtypes = [c_void_p, ctypes.c_long, c_void_p]
    foundation.CFDictionaryCreate.restype = c_void_p
    foundation.CFDictionaryCreate.argtypes = (
        [c_void_p] * 2 + [c_void_p, ctypes.c_long] + [c_void_p] * 2
    )
    foundation.CFArrayCreate.restype = c_void_p
    foundation.CFArrayCreate.argtypes = [c_void_p, c_void_p, ctypes.c_long, c_void_p]
    audio.AudioHardwareCreateAggregateDevice.argtypes = [c_void_p, c_void_p]
    audio.AudioHardwareDestroyAggregateDevice.argtypes = [c_uint32]
    return audio, foundation


def make(audio: ctypes.CDLL, foundation: ctypes.CDLL) -> int:
    """The aggregate over the output the system plays on."""

    def text(value: str) -> int:
        return foundation.CFStringCreateWithCString(None, value.encode(), audio_outputs.UTF8)

    def callbacks(name: str) -> int:
        return ctypes.addressof(ctypes.c_char.in_dll(foundation, name))

    def dictionary(pairs: dict[str, int]) -> int:
        keys = (c_void_p * len(pairs))(*[text(key) for key in pairs])
        values = (c_void_p * len(pairs))(*pairs.values())
        key_calls, value_calls = (
            callbacks("kCFTypeDictionaryKeyCallBacks"),
            callbacks("kCFTypeDictionaryValueCallBacks"),
        )
        return foundation.CFDictionaryCreate(None, keys, values, len(pairs), key_calls, value_calls)

    system = c_uint32()
    audio_outputs.ask(
        audio_outputs.SYSTEM, audio_outputs.DEFAULT_OUTPUT, byref(system), sizeof(system)
    )
    uid = c_void_p()
    audio_outputs.ask(system.value, DEVICE_UID, byref(uid), sizeof(uid))
    sub = dictionary({"uid": uid.value or 0})
    subs = foundation.CFArrayCreate(
        None, (c_void_p * 1)(sub), 1, callbacks("kCFTypeArrayCallBacks")
    )
    shared = foundation.CFNumberCreate(None, INTEGER, byref(c_int32(SHARED)))
    description = dictionary(
        {"uid": text(UID), "name": text(NAME), "private": shared, "subdevices": subs}
    )
    device = c_uint32()
    audio_outputs.answered(audio.AudioHardwareCreateAggregateDevice(description, byref(device)))
    return device.value


def main() -> int:
    audio, foundation = libraries()
    device = make(audio, foundation)
    try:
        print(ANNOUNCED, flush=True)
        sys.stdin.read()  # until the test closes it
    finally:
        audio.AudioHardwareDestroyAggregateDevice(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
