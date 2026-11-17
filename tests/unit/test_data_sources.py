"""console.data_sources: the sources a session can run from, and its configuration."""

import pytest

from adamus.config import Block, Board
from adamus.console.data_sources import (
    Source,
    available_sources,
    missing_electrodes,
    session_config,
)
from adamus.datasets import datasets_in
from adamus.errors import AdamusError

MUSE = Board("Muse 2", 38, "")


def test_the_headset_when_it_was_found_comes_first_then_the_datasets(data):
    assert available_sources(None, datasets_in(data)) == [
        Source("dataset", "alpha", dataset=data / "alpha"),
        Source("dataset", "zeta", dataset=data / "zeta"),
    ]
    sources = available_sources(MUSE, datasets_in(data))
    assert sources == [
        Source("headset", "Muse 2"),
        Source("dataset", "alpha", dataset=data / "alpha"),
        Source("dataset", "zeta", dataset=data / "zeta"),
    ]


class TestSessionConfig:
    def test_session_with_the_headset_runs_with_the_configuration_as_it_is(self, config):
        assert session_config(config, None) is config

    def test_recording_is_played_under_its_own_protocol(self, config, data):
        result = session_config(config, data / "zeta" / "one")
        assert result.protocol == [Block("REST", 5), Block("FOCUS", 5)]
        assert result.raw["protocol"] == [["REST", 5], ["FOCUS", 5]]
        assert (result.board, result.generator, result.poll_s) == (
            config.board,
            config.generator,
            config.poll_s,
        )

    def test_only_the_protocol_comes_from_the_recording(self, config, data):
        (data / "zeta" / "one" / "protocol.yaml").write_text(
            "protocol: [[REST, 5], [FOCUS, 5]]\npoll_s: 0.4"
        )
        assert session_config(config, data / "zeta" / "one").poll_s == config.poll_s

    @pytest.mark.parametrize(
        ("text", "problem"),
        [
            ("protocol: [[FOCUS, 5], [FREE, 0], [REST, 5]]", "only a final FREE block"),
            ("blocks: [[REST, 5], [FOCUS, 5]]", "protocol: expected a non-empty list"),
        ],
    )
    def test_recording_with_a_protocol_that_cannot_be_run_is_refused(
        self, config, data, text, problem
    ):
        protocol = data / "zeta" / "one" / "protocol.yaml"
        protocol.write_text(text)
        with pytest.raises(AdamusError, match=problem) as error:
            session_config(config, data / "zeta" / "one")
        assert str(error.value).splitlines()[0] == f"{protocol}: the recording cannot be played:"

    def test_the_configuration_of_the_console_is_not_modified(self, config, raw_config, data):
        session_config(config, data / "zeta" / "one")
        assert config.raw == raw_config


class TestMissingElectrodes:
    def test_headset_of_the_configuration_has_the_electrodes_of_the_detector(self, config):
        assert missing_electrodes(config, None) == []

    def test_headset_without_some_of_them_names_them_in_the_detector_order(
        self, config, monkeypatch
    ):
        from adamus.console import data_sources

        monkeypatch.setattr(data_sources, "electrodes_of_headset", lambda board: ["TP10", "AF8"])
        assert missing_electrodes(config, None) == ["AF7", "TP9"]

    def test_recording_is_asked_by_its_first_line(self, config, tmp_path):
        (tmp_path / "eeg.csv").write_text("Fp1,TP10,Fp2,AF8\n1,2,3,4\n")
        assert missing_electrodes(config, tmp_path) == ["AF7", "TP9"]
        (tmp_path / "eeg.csv").write_text("TP9,AF7,AF8,TP10,AUX\n1,2,3,4,5\n")
        assert missing_electrodes(config, tmp_path) == []
