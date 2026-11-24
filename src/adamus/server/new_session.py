"""The form for a new session: what it offers, what it checks, and the session it prepares.

The steps are those of a session from the start: a source (the headset, or a dataset
and one of its recordings), a name, and a check that the source has an electrode of
the detector. A source is known by its kind and its name, e.g. ``dataset:synthetic``.
The music plays on the audio output chosen in it: at first the one the system plays on.

The form looks for the sources itself: the datasets each time it is read, so that one put
into the data folder (or a recording a session stored there) is offered at once; the headset
when the operator asks, as the form does when it opens, because a look takes seconds.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from adamus import constants as c
from adamus import datasets
from adamus.activity_log import ACTION_HEADSET_LOOKED_FOR, ActivityLog
from adamus.audio_outputs import AudioOutput
from adamus.config import Config
from adamus.errors import AdamusError
from adamus.machine import Machine, RunningGenerator
from adamus.server import api_models
from adamus.session_format import NAME_MAX, SESSION_NAME, is_session_name
from adamus.session_sources import (
    HEADSET,
    Source,
    available_sources,
    missing_electrodes,
    session_config,
)

HEADSET_WORD = "headset"  # after the name of a headset in the form
NAME_HINT = f"Use at most {NAME_MAX} characters, without “/” or “:”."
SOURCES_HINT = f"Switch the headset on and look again, or put a dataset into {c.DATASETS_DIR}/."
OUTPUTS_HINT = "Connect headphones or speakers and look again."
ID_SEPARATOR = ":"  # between the kind of a source and its name


class PreflightResults(Protocol):
    """What the checks at the start obtained, as far as the sessions the app asks for need it."""

    @property
    def config(self) -> Config | None: ...

    @property
    def generator(self) -> RunningGenerator | None: ...


def wrong_rate(output: AudioOutput) -> str | None:
    """Why the music cannot play on ``output`` as it is set, if it cannot."""
    if output.rate == c.AUDIO_RATE:
        return None
    return (
        f"{output.name} plays {output.rate:g} Hz. Set it to {c.AUDIO_RATE} Hz in Audio MIDI Setup."
    )


@dataclass(frozen=True)
class PreparedSession:
    """Everything a session needs from the form, checked."""

    config: Config
    name: str
    source: Source
    recording: Path | None  # of a dataset; None for the headset
    output: str  # the audio output the music plays on


def missing_warning(missing: list[str]) -> str | None:
    """What the detector does without the electrodes the source lacks."""
    if not missing:
        return None
    return f"No {', '.join(missing)}: recorded as zeros, the detector uses the others."


def source_title(source: Source) -> str:
    """The source as the form shows it; a headset says it is one."""
    return f"{source.label} ({HEADSET_WORD})" if source.kind == HEADSET else source.label


def source_id(source: Source) -> str:
    return source.kind + ID_SEPARATOR + source.label


class NewSession:
    def __init__(
        self, preflight: PreflightResults, machine: Machine, activity: ActivityLog
    ) -> None:
        self._preflight = preflight
        self._machine = machine
        self._activity = activity
        self._headset_found = False  # by the last look for it

    def form(self) -> api_models.NewSessionForm:
        sources = [
            api_models.SourceChoice(
                id=source_id(source),
                title=source_title(source),
                live=source.kind == HEADSET,
                recordings=[r.name for r in datasets.recordings_of(source.dataset)]
                if source.dataset
                else [],
            )
            for source in self.sources()
        ]
        return api_models.NewSessionForm(
            sources=sources, sources_hint=SOURCES_HINT, name_pattern=SESSION_NAME,
            name_hint=NAME_HINT,
        )  # fmt: skip

    def sources(self) -> list[Source]:
        """The headset, when the last look found it, and the datasets in the data folder now."""
        headset = self._config().board
        found = datasets.datasets_in(self._machine.datasets_root)
        return available_sources(headset if self._headset_found else None, found)

    def look_for_headset(self) -> api_models.NewSessionForm:
        """Look for the headset of the configuration, which the operator may have switched on
        meanwhile; the form, with it if it was found."""
        headset, started = self._config().board, time.monotonic()
        self._headset_found = self._machine.headset_is_available(headset)
        self._activity.write(
            ACTION_HEADSET_LOOKED_FOR, headset=headset.name, found=self._headset_found,
            duration=time.monotonic() - started,
        )  # fmt: skip
        return self.form()

    def _config(self) -> Config:
        assert self._preflight.config is not None, "the preflight passed"
        return self._preflight.config

    def audio_outputs(self) -> api_models.AudioOutputs:
        """The outputs the Mac has now, and the one the system plays on."""
        outputs = self._machine.audio_outputs()
        try:
            system: str | None = self._machine.audio_output().name
        except AdamusError:
            system = None
        choices = [api_models.AudioOutputChoice(id=o.name, title=o.name) for o in outputs]
        return api_models.AudioOutputs(outputs=choices, system=system, hint=OUTPUTS_HINT)

    def _output(self, name: str) -> str:
        """The output ``name``, if the music can play on it; AdamusError says why not."""
        found = {output.name: output for output in self._machine.audio_outputs()}
        if name not in found:
            raise AdamusError(f"{name} is not connected.")
        problem = wrong_rate(found[name])
        if problem is not None:
            raise AdamusError(problem)
        return name

    def check(self, source: str, recording: str | None) -> api_models.SourceCheck:
        """Which electrodes of the detector the source lacks, and whether it can run at all."""
        try:
            prepared = self._config_and_recording(source, recording)
            missing = missing_electrodes(prepared[0], prepared[1])
        except (AdamusError, OSError) as problem:
            return api_models.SourceCheck(warning=None, problem=str(problem))
        if len(missing) == len(c.DETECTOR_ELECTRODES):
            return api_models.SourceCheck(warning=None, problem=self._lacks_all())
        return api_models.SourceCheck(warning=missing_warning(missing), problem=None)

    def prepare(self, request: api_models.SessionRequest) -> PreparedSession:
        """The session the form asks for; AdamusError says what is wrong with the request."""
        name = request.name.strip()
        if not is_session_name(name):
            raise AdamusError(NAME_HINT)
        checked = self.check(request.source_id, request.recording)
        if checked.problem is not None:
            raise AdamusError(checked.problem)
        output = self._output(request.output)
        config, recording = self._config_and_recording(request.source_id, request.recording)
        return PreparedSession(config, name, self._source(request.source_id), recording, output)

    def _source(self, source: str) -> Source:
        for found in self.sources():
            if source_id(found) == source:
                return found
        raise AdamusError(f"{source}: no such source")

    def _config_and_recording(
        self, source: str, recording: str | None
    ) -> tuple[Config, Path | None]:
        found = self._source(source)
        if found.kind == HEADSET:
            return self._config(), None
        assert found.dataset is not None
        folders = {folder.name: folder for folder in datasets.recordings_of(found.dataset)}
        if recording not in folders:
            raise AdamusError(f"{found.label}: no recording {recording}")
        played = folders[recording]
        return session_config(self._config(), played), played

    @staticmethod
    def _lacks_all() -> str:
        needed = ", ".join(c.DETECTOR_ELECTRODES)
        return f"The source has none of {needed}. The detector needs at least one."
