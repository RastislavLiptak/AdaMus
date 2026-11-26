"""config.py: what a valid configuration contains and how an invalid one is reported."""

import copy
import math
from dataclasses import fields

import pytest

from adamus.config import (
    Block,
    Style,
    is_style_text,
    load_config,
    parse_config,
    parse_protocol,
    read_yaml,
    with_style,
    write_yaml,
)
from adamus.errors import AdamusError
from tests.support.project_paths import CONFIG_FILE

STYLE_KEYS = [field.name for field in fields(Style)]
CHORDS_PROBLEM = "chords: at least one, each a root A to G, then nothing, m, 7, m7, maj7 or 5"


class TestRepositoryConfiguration:
    """config.yaml in the repository carries the designed values."""

    def test_board(self, config):
        assert config.board.id == 38
        assert (config.board.name, config.board.serial_number) == ("Muse 2", "")

    def test_protocol_is_eight_blocks_and_free_play(self, config):
        assert config.protocol == [Block("REST", 60), Block("FOCUS", 60)] * 4 + [Block("FREE", 0)]

    def test_timeouts_and_polling(self, config):
        timeouts = config.timeouts
        assert (timeouts.stream_s, timeouts.pd_connect_s, timeouts.model_load_s) == (5, 30, 120)
        assert config.poll_s == 0.05

    def test_generator(self, config):
        generator = config.generator
        assert (generator.folder, generator.model) == ("models/music_generator", "mrt2_small")

    def test_the_styles_in_their_order(self, config):
        assert [(style.id, style.name) for style in config.styles] == [
            ("folk", "Folk"), ("rock", "Rock"), ("country", "Country"),
            ("koto", "Koto and Taiko"), ("oud", "Oud and Darbuka"),
        ]  # fmt: skip

    def test_the_rock_plays_power_chords_without_a_gap_and_some_styles_drums(self, config):
        by_id = {style.id: style for style in config.styles}
        assert by_id["rock"].focus_power_chords and by_id["rock"].chord_gap_s == 0
        assert not any(style.focus_power_chords for style in config.styles if style.id != "rock")
        assert {style.id for style in config.styles if style.focus_drums} == {
            "country", "koto", "oud"
        }  # fmt: skip

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
    @pytest.mark.parametrize("section", ["board", "generator", "timeouts"])
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
            "board.name", "board.id", "board.serial_number", "generator.folder", "generator.model",
            "generator.style_guidance", "generator.note_guidance", "timeouts.stream_s",
            "timeouts.pd_connect_s", "timeouts.model_load_s",
        ],
    )  # fmt: skip
    def test_every_key_is_required(self, raw_config, path):
        section, key = path.split(".")
        del raw_config[section][key]
        assert problems_of(raw_config) == [f"{path}: missing"]

    def test_unknown_keys_are_rejected_at_every_level(self, raw_config):
        raw_config["extra"] = 1
        raw_config["generator"]["volume"] = 0.9
        raw_config["board"]["colour"] = "black"
        assert problems_of(raw_config) == [
            "extra: unknown key",
            "board.colour: unknown key",
            "generator.volume: unknown key",
        ]

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
            ("generator", "folder", None),
            ("generator", "model", 2),
            ("timeouts", "stream_s", "5"), ("timeouts", "model_load_s", [120]),
        ],
    )  # fmt: skip
    def test_wrong_type_is_reported_with_the_value(self, raw_config, section, key, value):
        raw_config[section][key] = value
        assert problems_of(raw_config) == [f"{section}.{key}: wrong type, got {value!r}"]

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])  # YAML .nan, .inf, -.inf
    @pytest.mark.parametrize("key", ["stream_s", "pd_connect_s", "model_load_s"])
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
        raw_config["timeouts"] = {
            "stream_s": 5,
            "pd_connect_s": 30,
            "model_load_s": 120,
        }
        assert parse_config(raw_config).timeouts.model_load_s == 120


class TestRanges:
    @pytest.mark.parametrize(
        ("section", "key", "value", "problem"),
        [
            ("board", "name", "", "board.name: must not be empty"),
            ("board", "id", 123456, "board.id: BrainFlow knows no board 123456"),
            ("generator", "folder", "", "generator.folder: must name a folder"),
            ("generator", "model", "models/mrt2_small.mlxfn",
             "generator.model: must be the name of a model, not a path"),
            ("generator", "model", "", "generator.model: must be the name of a model, not a path"),
            ("timeouts", "stream_s", 0, "timeouts: every timeout must be positive"),
            ("timeouts", "pd_connect_s", -1, "timeouts: every timeout must be positive"),
            ("timeouts", "model_load_s", 0, "timeouts: every timeout must be positive"),
            ("timeouts", "model_load_s", 1e307, "timeouts: none may be longer than a day"),
            ("timeouts", "pd_connect_s", 10**400, "timeouts: none may be longer than a day"),
            ("timeouts", "stream_s", 86401, "timeouts: none may be longer than a day"),
        ],
    )  # fmt: skip
    def test_value_out_of_range(self, raw_config, section, key, value, problem):
        raw_config[section][key] = value
        assert problems_of(raw_config) == [problem]

    @pytest.mark.parametrize(
        ("section", "key", "value"),
        [("generator", "folder", "tools/generator-2"),
         ("generator", "folder", "../generator"),
         ("generator", "folder", "My Music/the generator; $1, new"),
         ("generator", "folder", "/opt/adamus.v2/generator"),
         ("generator", "model", "mrt2.base-1"),
         ("timeouts", "stream_s", 86400), ("timeouts", "model_load_s", 86400)],
    )  # fmt: skip
    def test_values_on_the_boundary_are_accepted(self, raw_config, section, key, value):
        raw_config[section][key] = value
        assert getattr(getattr(parse_config(raw_config), section), key) == value

    def test_timeouts_shorter_than_a_second_are_allowed(self, raw_config):
        raw_config["timeouts"] = {
            "stream_s": 0.5, "pd_connect_s": 0.5, "model_load_s": 0.5
        }  # fmt: skip
        assert parse_config(raw_config).timeouts.model_load_s == 0.5

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


class TestStyles:
    @pytest.mark.parametrize("styles", [None, [], "folk", {"id": "folk"}])
    def test_a_non_empty_list_of_styles_is_required(self, raw_config, styles):
        raw_config["styles"] = styles
        assert problems_of(raw_config) == ["styles: expected a non-empty list of styles"]

    @pytest.mark.parametrize("key", STYLE_KEYS)
    def test_every_key_of_a_style_is_required(self, raw_config, key):
        del raw_config["styles"][1][key]
        assert problems_of(raw_config) == [f"styles[1].{key}: missing"]

    def test_a_style_that_is_no_mapping_and_unknown_keys_are_reported(self, raw_config):
        raw_config["styles"][0] = "folk"
        raw_config["styles"][2]["colour"] = "red"
        assert problems_of(raw_config) == [
            "styles[0]: missing or not a mapping",
            "styles[2].colour: unknown key",
        ]

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("tempo_bpm", "100"),
            ("chords", "C F"),
            ("chords", ["C", 7]),
            ("focus_drums", "yes"),
            ("focus_power_chords", 1),
            ("top_k", 40.0),
            ("rest", None),
            ("rest_volume_db", True),
        ],
    )
    def test_wrong_type_is_reported_with_the_value(self, raw_config, key, value):
        raw_config["styles"][0][key] = value
        assert problems_of(raw_config) == [f"styles[0].{key}: wrong type, got {value!r}"]

    @pytest.mark.parametrize(
        ("changes", "problem"),
        [
            ({"id": "Folk"}, "id: lower-case letters only"),
            ({"id": ""}, "id: lower-case letters only"),
            ({"id": "folk2"}, "id: lower-case letters only"),
            ({"name": " "}, "name: must not be empty"),
            ({"rest": "Disco"}, "rest: not a valid style text"),
            ({"focus": "disco 80"}, "focus: not a valid style text"),
            (
                {"focus": "instrumental folk with soft acoustic guitar picking"},
                "rest, focus: must be two different styles",
            ),
            ({"tempo_bpm": 0}, "tempo_bpm: above 0, at most 300"),
            ({"tempo_bpm": 301}, "tempo_bpm: above 0, at most 300"),
            ({"chords": []}, CHORDS_PROBLEM),
            ({"chords": ["H"]}, CHORDS_PROBLEM),
            ({"chords": ["C", "Cm9"]}, CHORDS_PROBLEM),
            ({"chords": ["c"]}, CHORDS_PROBLEM),
            ({"chord_gap_s": -0.01}, "chord_gap_s: from 0, shorter than a bar"),
            ({"chord_gap_s": 2.4}, "chord_gap_s: from 0, shorter than a bar"),  # a bar at 100 BPM
            ({"rest_volume_db": 2.1}, ": volumes at most 2 dB"),
            ({"focus_volume_db": 6}, ": volumes at most 2 dB"),
            ({"temperature": 0}, "temperature: must be positive"),
            ({"top_k": 0}, "top_k: at least 1"),
        ],
    )
    def test_value_out_of_range(self, raw_config, changes, problem):
        raw_config["styles"][0].update(changes)
        separator = "" if problem.startswith(":") else "."
        assert problems_of(raw_config) == [f"styles[0]{separator}{problem}"]

    @pytest.mark.parametrize(
        "changes",
        [
            {"tempo_bpm": 0.5},
            {"chords": ["C", "Dm", "E7", "Fm7", "Gmaj7", "A5", "B"] * 4},
            {"chord_gap_s": 0},
            {"chord_gap_s": 2.39},
            {"rest_volume_db": -60, "focus_volume_db": 2},
            {"temperature": 0.01},
            {"top_k": 1},
            {"name": "Lo-fi, at Night", "about": ""},
        ],
    )
    def test_values_on_the_boundary_are_accepted(self, raw_config, changes):
        raw_config["styles"][0].update(changes)
        style = parse_config(raw_config).styles[0]
        assert all(getattr(style, key) == value for key, value in changes.items())

    @pytest.mark.parametrize(
        ("key", "problem"),
        [("id", "styles: every id must differ"), ("name", "styles: every name must differ")],
    )
    def test_two_styles_cannot_share_an_id_or_a_name(self, raw_config, key, problem):
        raw_config["styles"][3][key] = raw_config["styles"][1][key]
        assert problems_of(raw_config) == [problem]

    @pytest.mark.parametrize("key", ["style_guidance", "note_guidance"])
    def test_guidance_must_be_positive(self, raw_config, key):
        raw_config["generator"][key] = 0
        assert problems_of(raw_config) == ["generator: guidance must be positive"]

    def test_the_style_of_a_configuration_is_its_first(self, config):
        assert config.style == config.styles[0]

    def test_a_session_gets_the_style_chosen_as_its_only_one(self, config):
        played = with_style(config, "koto")
        assert played.styles == [style for style in config.styles if style.id == "koto"]
        assert played.style.id == "koto"
        assert played.raw["styles"] == [
            style for style in config.raw["styles"] if style["id"] == "koto"
        ]
        assert {**played.raw, "styles": config.raw["styles"]} == config.raw  # all else as it was
        assert len(config.styles) == 5  # the configuration of the program keeps every style

    def test_a_style_that_is_not_there_cannot_be_chosen(self, config):
        with pytest.raises(AdamusError, match="^jazz: no such style$"):
            with_style(config, "jazz")


def test_section_must_be_a_mapping(raw_config):
    raw_config["timeouts"] = 5
    assert problems_of(raw_config) == ["timeouts: missing or not a mapping"]


class TestNames:
    @pytest.mark.parametrize(
        "text", ["disco funk", "ambient pads with sub bass", "eighties disco", "lo-fi", "a", "x9"]
    )
    def test_valid_style_texts(self, text):
        assert is_style_text(text)

    @pytest.mark.parametrize(
        "text",
        ["disco 80", "Disco funk", "disco  funk", " disco", "disco ", "", "80s", "-lofi",
         "disco;funk", "disco, funk", "disco_funk", "disco\nfunk"],
    )  # fmt: skip
    def test_invalid_style_texts(self, text):
        """Pd would turn a numeric atom into a float and mrt2~ would drop it silently."""
        assert not is_style_text(text)


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


class TestGeneratorPaths:
    """Every path of the generator follows from generator.folder and generator.model."""

    def test_paths_inside_the_generator_folder(self, raw_config):
        raw_config["generator"].update(folder="tools/gen", model="mrt2_base")
        generator = parse_config(raw_config).generator
        assert str(generator.pd_binary) == "tools/gen/Pd.app/Contents/Resources/bin/pd"
        assert str(generator.externals) == "tools/gen/externals/mrt2~"
        assert str(generator.external_file) == "tools/gen/externals/mrt2~/mrt2~.pd_darwin"
        assert str(generator.assets) == "tools/gen/magenta-rt-v2/resources"
        assert (
            str(generator.model_file) == "tools/gen/magenta-rt-v2/models/mrt2_base/mrt2_base.mlxfn"
        )


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
