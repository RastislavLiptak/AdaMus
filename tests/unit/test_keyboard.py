"""console.keyboard: keys read by a background thread and handed over through a queue."""

import threading

import pytest

from adamus.console import keyboard
from adamus.console.keyboard import Keyboard
from tests.support.stand_in_devices import ScriptedKeys
from tests.support.waiting import wait_until


def started(*keys):
    reader = Keyboard(ScriptedKeys(*keys))
    reader.start()
    return reader


def test_keys_arrive_in_the_order_pressed():
    reader = started("a", "B", " ")
    assert [reader.next_key(), reader.next_key(), reader.next_key()] == ["a", "B", keyboard.SPACE]


@pytest.mark.parametrize(
    ("typed", "key"),
    [("\r", keyboard.ENTER), ("\n", keyboard.ENTER), ("\x7f", keyboard.BACKSPACE),
     ("\x08", keyboard.BACKSPACE), ("q", "q")],
)  # fmt: skip
def test_variants_of_enter_and_backspace_are_one_key_each(typed, key):
    assert started(typed).next_key() == key


def test_ctrl_c_read_as_a_character_raises_keyboard_interrupt():
    reader = started("a")  # the script ends with [CTRL]+[C]
    assert reader.next_key() == "a"
    with pytest.raises(KeyboardInterrupt):
        reader.next_key()


def wait_until_read(reader, count):
    """The reading thread has taken ``count`` keys."""
    wait_until(lambda: reader._keys.qsize() >= count, "the keys were not read", 10)


class TestStopRequested:
    def test_only_space_stops_a_running_session(self):
        reader = started("q", "x", keyboard.ENTER, " ", "y")
        wait_until_read(reader, 6)  # five keys and the final [CTRL]+[C]
        assert reader.stop_requested() is True
        assert reader.next_key() == "y"  # keys pressed after [SPACE] are kept

    def test_other_keys_are_dropped_without_stopping(self):
        reader = Keyboard(ScriptedKeys("q", "x", interrupt_at_end=False))
        reader.start()
        wait_until_read(reader, 2)
        assert reader.stop_requested() is False
        assert reader._keys.qsize() == 0

    def test_no_key_does_not_stop_and_does_not_block(self):
        assert Keyboard(ScriptedKeys(interrupt_at_end=False)).stop_requested() is False

    def test_ctrl_c_raises_keyboard_interrupt(self):
        reader = started()
        wait_until_read(reader, 1)
        with pytest.raises(KeyboardInterrupt):
            reader.stop_requested()


@pytest.mark.parametrize("problem", [OSError("not a terminal"), RuntimeError("a fault")])
def test_keyboard_that_cannot_be_read_ends_the_program_like_ctrl_c(problem, caplog):
    """Otherwise the console would wait for a key that nobody can press; why is logged."""

    def unreadable():
        raise problem

    reader = Keyboard(unreadable)
    reader.start()
    with pytest.raises(KeyboardInterrupt):
        reader.next_key()
    reader.restore_terminal()  # nothing was changed, nothing to restore
    (logged,) = caplog.records
    assert logged.getMessage() == "the keyboard can no longer be read"
    assert logged.exc_info[1] is problem


class TestTerminal:
    """readchar switches the terminal to raw input; the settings are put back at the end."""

    def test_settings_are_remembered_at_the_start_and_restored_as_they_were(self, monkeypatch):
        calls = []
        monkeypatch.setattr(keyboard.sys, "stdin", type("Stdin", (), {"fileno": lambda self: 7})())
        monkeypatch.setattr(keyboard.termios, "tcgetattr", lambda fd: ["settings of", fd])
        monkeypatch.setattr(
            keyboard.termios,
            "tcsetattr",
            lambda fd, when, settings: calls.append((fd, when, settings)),
        )
        reader = Keyboard(ScriptedKeys(interrupt_at_end=False))
        reader.start()
        reader.restore_terminal()
        assert calls == [(7, keyboard.termios.TCSADRAIN, ["settings of", 7])]

    @pytest.mark.parametrize(
        "problem", [keyboard.termios.error(25, "not a terminal"), ValueError("closed"), OSError()]
    )
    def test_input_that_is_not_a_terminal_has_nothing_to_restore(self, monkeypatch, problem):
        def not_a_terminal(fd):
            raise problem

        calls = []
        monkeypatch.setattr(keyboard.termios, "tcgetattr", not_a_terminal)
        monkeypatch.setattr(keyboard.termios, "tcsetattr", lambda *arguments: calls.append(1))
        reader = started("a")
        assert reader.next_key() == "a"  # keys are read all the same
        reader.restore_terminal()
        assert calls == []

    @pytest.mark.parametrize(
        "problem",
        [keyboard.termios.error(5, "input/output error"), ValueError("closed"), OSError()],
    )
    def test_a_terminal_that_went_away_is_not_restored_and_nothing_fails(
        self, monkeypatch, problem
    ):
        """The window was closed while the program ran: the program must still end cleanly."""
        monkeypatch.setattr(keyboard.sys, "stdin", type("Stdin", (), {"fileno": lambda self: 7})())
        monkeypatch.setattr(keyboard.termios, "tcgetattr", lambda fd: ["settings of", fd])

        def gone(*arguments):
            raise problem

        monkeypatch.setattr(keyboard.termios, "tcsetattr", gone)
        reader = Keyboard(ScriptedKeys(interrupt_at_end=False))
        reader.start()
        reader.restore_terminal()  # returns

    def test_reading_thread_does_not_keep_the_program_alive(self):
        before = set(threading.enumerate())
        Keyboard(ScriptedKeys(interrupt_at_end=False)).start()
        (reader,) = set(threading.enumerate()) - before
        assert reader.daemon and reader.name == "keyboard"


def test_waiting_for_a_key_wakes_up_regularly_for_as_long_as_it_takes():
    """The wait is cut into short pieces so that the main thread can still take signals."""
    waits = []

    class KeyAfterALongWhile:
        def get(self, timeout):
            waits.append(timeout)
            if len(waits) < 500:
                raise keyboard.queue.Empty
            return "y"

    reader = Keyboard(ScriptedKeys(interrupt_at_end=False))
    reader._keys = KeyAfterALongWhile()
    assert reader.next_key() == "y"
    assert waits == [keyboard.KEY_WAIT_S] * 500


class TestDiscardPressedKeys:
    """Keys pressed while a session was starting must not end it."""

    def test_keys_pressed_so_far_are_forgotten(self):
        reader = Keyboard(ScriptedKeys(" ", "q", " ", interrupt_at_end=False))
        reader.start()
        wait_until_read(reader, 3)
        reader.discard_pressed_keys()
        assert reader.stop_requested() is False

    def test_keys_pressed_afterwards_count(self):
        reader = Keyboard(ScriptedKeys(interrupt_at_end=False))
        reader.discard_pressed_keys()  # nothing pressed yet: nothing happens
        reader._keys.put(" ")
        assert reader.stop_requested() is True

    def test_ctrl_c_among_them_still_ends_the_program(self):
        reader = started(" ")  # the script ends with [CTRL]+[C]
        wait_until_read(reader, 2)
        with pytest.raises(KeyboardInterrupt):
            reader.discard_pressed_keys()


@pytest.mark.parametrize(
    ("key", "name"),
    [(keyboard.SPACE, "SPACE"), (keyboard.UP, "UP"), (keyboard.DOWN, "DOWN"),
     (keyboard.ENTER, "ENTER"), (keyboard.BACKSPACE, "BACKSPACE"), ("Y", "Y")],
)  # fmt: skip
def test_names_of_keys_in_the_log(key, name):
    assert keyboard.key_name(key) == name
