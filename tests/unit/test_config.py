"""config.py: what a valid configuration contains and how an invalid one is reported."""

import copy
import math

import pytest

from adamus.config import (
    Block,
    load_config,
    parse_config,
    parse_protocol,
    read_yaml,
    write_yaml,
)
from adamus.errors import AdamusError
from tests.support.project_paths import CONFIG_FILE


class TestRepositoryConfiguration:
    """config.yaml in the repository carries the designed values."""

    def test_board(self, config):
        assert config.board.id == 38
        assert (config.board.name, config.board.serial_number) == ("Muse 2", "")

    def test_protocol_is_eight_blocks_and_free_play(self, config):
        assert config.protocol == [Block("REST", 60), Block("FOCUS", 60)] * 4 + [Block("FREE", 0)]

    def test_timeouts_and_polling(self, config):
        assert (config.timeouts.stream_s, config.poll_s) == (5, 0.05)

    def test_raw_mapping_is_kept_for_the_session_copy(self, config, raw_config):
        assert config.raw == raw_config


class TestFiles:
    def test_load_config_reads_a_file(self):
        assert load_config(CONFIG_FILE).board.id == 38

    def test_written_configuration_reads_back_equal(self, raw_config, tmp_path):
        write_yaml(raw_config, tmp_path / "copy.yaml")
        assert read_yaml(tmp_path / "copy.yaml") == raw_config
        assert load_config(tmp_path / "copy.yaml") == parse_config(raw_config)

    def test_file_is_written_in_the_order_given_and_as_readable_text(self, tmp_path):
        """The copy in a session folder reads like the configuration it was made from."""
        data = {"zeta": 1, "alpha": {"name": "měření"}, "protocol": [["REST", 60], ["FOCUS", 60]]}
        write_yaml(data, tmp_path / "copy.yaml")
        assert (tmp_path / "copy.yaml").read_text(encoding="utf-8") == (
            "zeta: 1\nalpha: {name: měření}\nprotocol:\n- [REST, 60]\n- [FOCUS, 60]\n"
        )

    def test_file_is_read_as_utf_8(self, tmp_path):
        (tmp_path / "names.yaml").write_bytes("name: měření\n".encode())
        assert read_yaml(tmp_path / "names.yaml") == {"name": "měření"}

    def test_missing_file(self, tmp_path):
        with pytest.raises(AdamusError) as error:
            load_config(tmp_path / "nope.yaml")
        missing = tmp_path / "nope.yaml"
        assert str(error.value) == (
            f"cannot read {missing}: [Errno 2] No such file or directory: '{missing}'"
        )

    def test_invalid_yaml(self, tmp_path):
        (tmp_path / "bad.yaml").write_text("board: [unclosed")
        with pytest.raises(AdamusError, match="cannot read"):
            load_config(tmp_path / "bad.yaml")

    def test_file_that_is_not_a_mapping(self, tmp_path):
        (tmp_path / "list.yaml").write_text("- 1\n- 2\n")
        with pytest.raises(AdamusError) as error:
            load_config(tmp_path / "list.yaml")
        assert str(error.value) == f"{tmp_path / 'list.yaml'} must contain a mapping"


def problems_of(raw):
    """The list of problems parse_config reports for an invalid mapping."""
    with pytest.raises(AdamusError) as error:
        parse_config(raw)
    lines = str(error.value).splitlines()
    assert lines[0] == "invalid configuration:"
    return [line.strip() for line in lines[1:]]


class TestStructure:
    @pytest.mark.parametrize("section", ["board", "timeouts"])
    def test_every_section_is_required(self, raw_config, section):
        del raw_config[section]
        assert problems_of(raw_config) == [f"{section}: missing or not a mapping"]

    def test_protocol_and_poll_s_are_required(self, raw_config):
        del raw_config["protocol"], raw_config["poll_s"]
        assert problems_of(raw_config) == [
            "protocol: expected a non-empty list of [label, duration_s]",
            "poll_s: expected a number, got None",
        ]

    @pytest.mark.parametrize(
        "path",
        [
            "board.name", "board.id", "board.serial_number", "timeouts.stream_s",
        ],
    )  # fmt: skip
    def test_every_key_is_required(self, raw_config, path):
        section, key = path.split(".")
        del raw_config[section][key]
        assert problems_of(raw_config) == [f"{path}: missing"]

    def test_unknown_keys_are_rejected_at_every_level(self, raw_config):
        raw_config["extra"] = 1
        raw_config["timeouts"]["retry_s"] = 30
        raw_config["board"]["colour"] = "black"
        assert problems_of(raw_config) == [
            "extra: unknown key",
            "board.colour: unknown key",
            "timeouts.retry_s: unknown key",
        ]

    def test_the_music_has_no_settings(self, raw_config):
        """The music is a song of the library, marked where it is added, not configured."""
        raw_config["music"] = {"song": "Imagine"}
        assert problems_of(raw_config) == ["music: unknown key"]

    @pytest.mark.parametrize("section", ["signal", "calibration", "detector"])
    def test_the_detector_has_no_settings(self, raw_config, section):
        """What the detector computes is fixed by its models, not by the configuration."""
        raw_config[section] = {"false_alarms": 0.1}
        assert problems_of(raw_config) == [f"{section}: unknown key"]

    def test_the_kept_copy_of_the_mapping_is_not_a_key_of_the_file(self, raw_config):
        raw_config["raw"] = {}
        assert problems_of(raw_config) == ["raw: unknown key"]

    def test_unknown_keys_that_are_not_texts_are_reported_too(self, raw_config):
        raw_config[7] = raw_config["zz"] = 1
        raw_config["timeouts"][3] = raw_config["timeouts"]["zz"] = 1
        assert problems_of(raw_config) == [
            "7: unknown key",
            "zz: unknown key",
            "timeouts.3: unknown key",
            "timeouts.zz: unknown key",
        ]

    def test_all_problems_are_reported_together(self, raw_config):
        raw_config["board"]["id"] = "x"
        raw_config["poll_s"] = "fast"
        del raw_config["timeouts"]["stream_s"]
        assert problems_of(raw_config) == [
            "board.id: wrong type, got 'x'",
            "timeouts.stream_s: missing",
            "poll_s: expected a number, got 'fast'",
        ]


class TestTypes:
    @pytest.mark.parametrize(
        ("section", "key", "value"),
        [
            ("board", "id", "38"), ("board", "id", True), ("board", "id", 38.0),
            ("board", "serial_number", 5), ("board", "name", 2),
            ("timeouts", "stream_s", "5"), ("timeouts", "stream_s", [5]),
        ],
    )  # fmt: skip
    def test_wrong_type_is_reported_with_the_value(self, raw_config, section, key, value):
        raw_config[section][key] = value
        assert problems_of(raw_config) == [f"{section}.{key}: wrong type, got {value!r}"]

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])  # YAML .nan, .inf, -.inf
    @pytest.mark.parametrize("key", ["stream_s"])
    def test_a_time_must_be_a_finite_number(self, raw_config, key, value):
        raw_config["timeouts"][key] = value
        assert problems_of(raw_config) == [
            f"timeouts.{key}: must be a finite number, got {value!r}"
        ]

    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_poll_and_block_durations_must_be_finite_numbers(self, raw_config, value):
        raw_config["poll_s"] = value
        raw_config["protocol"][2][1] = value
        assert problems_of(raw_config) == [
            "protocol[2]: expected [REST|FOCUS|FREE, seconds >= 0]",
            f"poll_s: expected a number, got {value!r}",
        ]

    def test_whole_numbers_are_accepted_where_a_number_is_expected(self, raw_config):
        raw_config["timeouts"] = {"stream_s": 5}
        assert parse_config(raw_config).timeouts.stream_s == 5


class TestRanges:
    @pytest.mark.parametrize(
        ("section", "key", "value", "problem"),
        [
            ("board", "name", "", "board.name: must not be empty"),
            ("board", "id", 123456, "board.id: BrainFlow knows no board 123456"),
            ("timeouts", "stream_s", 0, "timeouts.stream_s: must be positive"),
            ("timeouts", "stream_s", -1, "timeouts.stream_s: must be positive"),
            ("timeouts", "stream_s", 1e307, "timeouts.stream_s: must not be longer than a day"),
            ("timeouts", "stream_s", 10**400, "timeouts.stream_s: must not be longer than a day"),
            ("timeouts", "stream_s", 86401, "timeouts.stream_s: must not be longer than a day"),
        ],
    )  # fmt: skip
    def test_value_out_of_range(self, raw_config, section, key, value, problem):
        raw_config[section][key] = value
        assert problems_of(raw_config) == [problem]

    @pytest.mark.parametrize(
        ("section", "key", "value"),
        [("timeouts", "stream_s", 86400), ("board", "serial_number", "4F2A-01")],
    )  # fmt: skip
    def test_values_on_the_boundary_are_accepted(self, raw_config, section, key, value):
        raw_config[section][key] = value
        assert getattr(getattr(parse_config(raw_config), section), key) == value

    def test_timeouts_shorter_than_a_second_are_allowed(self, raw_config):
        raw_config["timeouts"] = {"stream_s": 0.5}
        raw_config["poll_s"] = 0.25
        assert parse_config(raw_config).timeouts.stream_s == 0.5

    def test_poll_s_must_be_positive(self, raw_config):
        raw_config["poll_s"] = 0
        assert problems_of(raw_config) == ["poll_s: must be positive"]


class TestProtocol:
    def parse(self, raw_config, protocol):
        raw_config["protocol"] = protocol
        return parse_config(raw_config).protocol

    def test_fractional_durations(self, raw_config):
        blocks = self.parse(raw_config, [["REST", 4.5], ["FOCUS", 5]])
        assert blocks == [Block("REST", 4.5), Block("FOCUS", 5)]

    @pytest.mark.parametrize(
        "protocol", [[["FOCUS", 60], ["REST", 60]], [["REST", 60]], [["FREE", 0]]]
    )
    def test_any_block_may_come_first(self, raw_config, protocol):
        """There is no calibration: the first block may be any."""
        assert len(self.parse(raw_config, protocol)) == len(protocol)

    def test_free_block_may_be_timed(self, raw_config):
        assert self.parse(raw_config, [["REST", 60], ["FOCUS", 60], ["FREE", 30]])[-1] == Block(
            "FREE", 30
        )

    @pytest.mark.parametrize(
        ("protocol", "problems"),
        [
            ([], ["protocol: expected a non-empty list of [label, duration_s]"]),
            ("REST", ["protocol: expected a non-empty list of [label, duration_s]"]),
            ([["REST", 60], ["FOCUS"]],
             ["protocol: expected a non-empty list of [label, duration_s]"]),
            ([["REST", 60], "FOCUS"],
             ["protocol: expected a non-empty list of [label, duration_s]"]),
            ([["REST", 60], ["FOCUS", 60], ["NAP", 60]],
             ["protocol[2]: expected [REST|FOCUS|FREE, seconds >= 0]"]),
            ([["REST", 60], ["FOCUS", 60], [["REST"], 60]],
             ["protocol[2]: expected [REST|FOCUS|FREE, seconds >= 0]"]),
            ([["REST", 60], ["FOCUS", -1]],
             ["protocol[1]: expected [REST|FOCUS|FREE, seconds >= 0]"]),
            ([["REST", 60], ["FOCUS", "60"]],
             ["protocol[1]: expected [REST|FOCUS|FREE, seconds >= 0]"]),
            ([["REST", 60], ["FOCUS", 0]],
             ["protocol[1]: only a final FREE block may have duration 0"]),
            ([["REST", 60], ["FOCUS", 60], ["FREE", 0], ["REST", 60]],
             ["protocol[2]: only a final FREE block may have duration 0"]),
            ([["REST", 60], ["FOCUS", 1e308]], ["protocol[1]: a block must last less than a day"]),
            ([["REST", 10**400]], ["protocol[0]: a block must last less than a day"]),
            ([["REST", 86400]], ["protocol[0]: a block must last less than a day"]),
        ],
    )  # fmt: skip
    def test_invalid_protocols(self, raw_config, protocol, problems):
        raw_config["protocol"] = protocol
        assert problems_of(raw_config) == problems


def test_section_must_be_a_mapping(raw_config):
    raw_config["timeouts"] = 5
    assert problems_of(raw_config) == ["timeouts: missing or not a mapping"]


def test_parsing_does_not_modify_its_input(raw_config):
    before = copy.deepcopy(raw_config)
    parse_config(raw_config)
    assert raw_config == before


class TestValuesAgainstEachOther:
    """A configuration that loads must also work: values are checked against each other."""

    def problems(self, raw_config, **sections):
        for name, values in sections.items():
            raw_config[name] = {**raw_config[name], **values} if type(values) is dict else values
        return problems_of(raw_config)

    @pytest.mark.parametrize("stream_s", [0.05, 0.01])
    def test_stream_timeout_must_be_longer_than_one_poll(self, raw_config, stream_s):
        assert self.problems(raw_config, timeouts={"stream_s": stream_s}) == [
            "timeouts.stream_s: must be longer than poll_s"
        ]

    def test_polling_must_not_be_slower_than_the_steps(self, raw_config):
        raw_config["poll_s"] = 0.51
        assert problems_of(raw_config) == ["poll_s: must not be longer than a step (0.5 s)"]
        raw_config["poll_s"] = 0.5
        assert parse_config(raw_config).poll_s == 0.5

    def test_board_of_another_rate_than_the_detector_is_refused(self, raw_config):
        """The model was trained at 256 Hz; a board of 1000 Hz cannot feed it."""
        raw_config["board"] = {"name": "Cyton", "id": 6, "serial_number": ""}
        assert problems_of(raw_config) == [
            "board.id: the detector needs 256 Hz, the board delivers 1000 Hz"
        ]

    def test_another_board_of_the_same_rate_is_accepted(self, raw_config):
        raw_config["board"] = {"name": "Muse S", "id": 39, "serial_number": ""}
        assert parse_config(raw_config).board.id == 39


def test_problem_of_several_lines_reads_as_one_line_each():
    problem = AdamusError("invalid configuration:", "  poll_s: must be positive")
    assert str(problem) == "invalid configuration:\n" + "  poll_s: must be positive"
    assert str(AdamusError("one line")) == "one line"


class TestProtocolAlone:
    """parse_protocol: the protocol of a session folder, read without the rest of its copy."""

    def test_a_valid_protocol_is_read_into_blocks(self):
        blocks = parse_protocol([["REST", 30], ["FOCUS", 30.5], ["FREE", 0]])
        assert [(b.label, b.duration_s) for b in blocks] == [
            ("REST", 30), ("FOCUS", 30.5), ("FREE", 0),
        ]  # fmt: skip

    @pytest.mark.parametrize("protocol", [None, [], [["NAP", 3]], [["FREE", 0], ["REST", 1]]])
    def test_an_invalid_protocol_says_what_is_wrong(self, protocol):
        with pytest.raises(AdamusError, match="invalid configuration:"):
            parse_protocol(protocol)
