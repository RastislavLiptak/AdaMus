"""The API the app works through.

It answers on the local computer only. Every call carries the token the app was told when
the program started (a header, or for a sound and a cover, which a player and an image ask for
without a header, a query parameter), so that a program that was not given the token (of another
user, or a web page in a browser) cannot delete sessions or start one.

The songs of the library are shown, played and deleted through it too, and changed in drafts,
which go into the library once marked; a session keeps a copy of its song, which its overview
shows and plays as it was then.
"""

import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from adamus import constants as c
from adamus.errors import AdamusError
from adamus.evaluation.session_reader import load_session_folder
from adamus.server.api_models import (
    AudioOutputs,
    Dashboard,
    DraftKept,
    DraftOfFile,
    Electrode,
    Excerpt,
    HeadModel,
    NewSessionForm,
    RunState,
    SessionPage,
    SessionRename,
    SessionRequest,
    SessionSummary,
    Song,
    SongMark,
    SongMarks,
    SongPage,
    SourceCheck,
    Timeline,
)
from adamus.server.dashboard import UNREADABLE_PROBLEMS, dashboard_of, why_unreadable
from adamus.server.electrodes import electrodes_of, head_model
from adamus.server.excerpt import excerpt_of
from adamus.server.new_session import NAME_HINT, NewSession
from adamus.server.session_catalogue import SessionCatalogue
from adamus.server.session_runner import SessionRunner
from adamus.server.songs import marks_view, page_of, song_view
from adamus.server.timeline import SessionTracks, session_tracks
from adamus.session_format import is_session_name
from adamus.song_library import Library, audio_in, cover_in

TOKEN_HEADER = "X-AdaMus-Token"
ACCEPTED, NO_CONTENT = 202, 204  # HTTP status codes of success
NOT_FOUND, CONFLICT, INVALID = 404, 409, 422  # and of failure
MAX_PAGE = 1000  # sessions or songs on one page of their list, at most
BINARY = "application/octet-stream"
SOUND = "/sound"  # the end of the address of a sound: of a track, of a song
COVER = "/cover"  # the end of the address of the picture of a song
AUDIO = "audio/wav"  # what a session records
# What the player of the app is told a song file is, by its suffix; not guessed, as Python
# would call an .m4a file a bare stream (audio/mp4a-latm).
MEDIA_TYPES = {
    ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".aac": "audio/aac",
    ".m4b": "audio/mp4", ".flac": "audio/flac", ".wav": "audio/wav", ".aif": "audio/aiff",
    ".aiff": "audio/aiff", ".aifc": "audio/aiff", ".au": "audio/basic", ".caf": "audio/x-caf",
    ".jpg": "image/jpeg", ".png": "image/png",
}  # fmt: skip


class TrackCache:
    """The tracks of the session last shown: its timeline and its samples are read once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._folder: Path | None = None
        self._tracks: SessionTracks | None = None

    def tracks(self, folder: Path) -> SessionTracks:
        with self._lock:
            if self._folder != folder or self._tracks is None:
                self._tracks, self._folder = session_tracks(folder), folder
            return self._tracks

    def forget(self) -> None:
        with self._lock:
            self._folder, self._tracks = None, None


@dataclass(frozen=True)
class Services:
    """What the API works with; the tests put stand-ins in place of some."""

    catalogue: SessionCatalogue
    new_session: NewSession
    runner: SessionRunner
    library: Library
    token: str = field(default_factory=secrets.token_urlsafe)
    tracks: TrackCache = field(default_factory=TrackCache)


def services_of(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


Using = Annotated[Services, Depends(services_of)]
Offset = Annotated[int, Query(ge=0)]  # of the first session or song of a page, in the list
PageSize = Annotated[int, Query(gt=0, le=MAX_PAGE)]


def authorized(
    request: Request,
    services: Using,
    header: Annotated[str | None, Header(alias=TOKEN_HEADER)] = None,
    token: Annotated[str | None, Query()] = None,
) -> None:
    """Every call carries the token in its header; a call without it finds nothing. Only a
    sound and a cover carry it in their address, as what asks for them sends no header."""
    in_address = token if request.url.path.endswith((SOUND, COVER)) else None
    given = (header or in_address or "").encode()  # compare_digest takes text in ASCII only
    if not secrets.compare_digest(given, services.token.encode()):
        raise HTTPException(NOT_FOUND)


def create_api() -> FastAPI:
    """The routes of the API, without the services they work with."""
    # Without the pages FastAPI adds of itself (the description of the API and its viewers),
    # which the token would not guard.
    api = FastAPI(
        title=c.PROGRAM_NAME, dependencies=[Depends(authorized)],
        openapi_url=None, docs_url=None, redoc_url=None,
    )  # fmt: skip
    add_session_routes(api)
    add_change_routes(api)
    add_excerpt_route(api)
    add_electrode_routes(api)
    add_new_session_routes(api)
    add_song_routes(api)
    add_draft_routes(api)
    add_draft_file_routes(api)
    add_session_song_routes(api)
    add_run_routes(api)
    return api


def create_app(services: Services) -> FastAPI:
    """The API with the services it works with."""
    app = create_api()
    app.state.services = services
    return app


def folder_of(services: Services, session_id: str) -> Path:
    folder = services.catalogue.folder(session_id)
    if folder is None:
        raise HTTPException(NOT_FOUND, f"no session {session_id}")
    return folder


def tracks_of(services: Services, session_id: str) -> SessionTracks:
    folder = folder_of(services, session_id)
    try:
        return services.tracks.tracks(folder)
    except UNREADABLE_PROBLEMS as problem:
        raise HTTPException(INVALID, why_unreadable(folder, problem)) from problem


def is_busy_with(services: Services, session_id: str) -> bool:
    state = services.runner.state()
    return state.phase != "idle" and state.session_id == session_id


def refuse_if_running(services: Services, session_id: str) -> None:
    """A session that is not there, or is not finished yet, cannot be changed."""
    folder_of(services, session_id)
    if is_busy_with(services, session_id):
        raise HTTPException(CONFLICT, "the session is still running")


def add_session_routes(api: FastAPI) -> None:
    """The stored sessions: the list, the dashboard, the timeline."""

    @api.get("/api/sessions")
    def sessions(services: Using, offset: Offset, limit: PageSize) -> SessionPage:
        return services.catalogue.page(offset, limit)

    @api.get("/api/sessions/{session_id}/dashboard")
    def dashboard(services: Using, session_id: str) -> Dashboard:
        folder = folder_of(services, session_id)
        own = services.catalogue.own_recording(session_id)
        return dashboard_of(folder, own, is_busy_with(services, session_id))

    @api.get("/api/sessions/{session_id}/timeline")
    def timeline(services: Using, session_id: str) -> Timeline:
        return tracks_of(services, session_id).timeline

    @api.get("/api/sessions/{session_id}/tracks/{track_id}")
    def track(services: Using, session_id: str, track_id: str) -> Response:
        samples = tracks_of(services, session_id).samples
        if track_id not in samples:
            raise HTTPException(NOT_FOUND, f"no samples of {track_id}")
        return Response(samples[track_id].tobytes(), media_type=BINARY)

    @api.get("/api/sessions/{session_id}/tracks/{track_id}" + SOUND)
    def sound(services: Using, session_id: str, track_id: str) -> FileResponse:
        groups = tracks_of(services, session_id).timeline.groups
        kinds = {track.id: track.kind for group in groups for track in group.tracks}
        path = folder_of(services, session_id) / c.FILE_AUDIO  # the one sound a session has
        if kinds.get(track_id) != "audio" or not path.is_file():
            raise HTTPException(NOT_FOUND, f"no sound of {track_id}")
        return FileResponse(path, media_type=AUDIO)


def add_session_song_routes(api: FastAPI) -> None:
    """The song a stored session played, as it kept it: its sound, and its picture."""

    @api.get("/api/sessions/{session_id}/song" + SOUND)
    def session_song_sound(services: Using, session_id: str) -> FileResponse:
        song = folder_of(services, session_id) / c.SONG_DIR
        if not song.is_dir():
            raise HTTPException(NOT_FOUND, "the session kept no song")
        return file_of(audio_in(song))

    @api.get("/api/sessions/{session_id}/song" + COVER)
    def session_song_cover(services: Using, session_id: str) -> FileResponse:
        return cover_of(folder_of(services, session_id) / c.SONG_DIR)


def cover_of(song: Path) -> FileResponse:
    """The picture of the song in the folder ``song``; not found when it carries none."""
    cover = cover_in(song)
    if cover is None:
        raise HTTPException(NOT_FOUND, "the song carries no picture")
    return file_of(cover)


def file_of(path: Path) -> FileResponse:
    """A song, a recording or a picture, as what it is; one that is not there is not found."""
    if not path.is_file():
        raise HTTPException(NOT_FOUND, f"no {path.name}")
    media = MEDIA_TYPES.get(path.suffix.lower(), BINARY)
    return FileResponse(path, media_type=media)


def add_change_routes(api: FastAPI) -> None:
    """A stored session renamed or deleted, but not while it is still running."""

    @api.patch("/api/sessions/{session_id}")
    def rename(services: Using, session_id: str, request: SessionRename) -> SessionSummary:
        refuse_if_running(services, session_id)
        name = request.name.strip()
        if not is_session_name(name):
            raise HTTPException(INVALID, NAME_HINT)
        try:
            return services.catalogue.rename(session_id, name)
        except AdamusError as problem:
            raise HTTPException(CONFLICT, str(problem)) from problem
        except OSError as problem:  # without its path, as the app shows it
            why = f"The session could not be renamed: {problem.strerror}."
            raise HTTPException(INVALID, why) from problem

    @api.delete("/api/sessions/{session_id}", status_code=NO_CONTENT)
    def delete(services: Using, session_id: str) -> None:
        refuse_if_running(services, session_id)
        services.tracks.forget()
        services.catalogue.delete(session_id)


def add_electrode_routes(api: FastAPI) -> None:
    """The head, and the electrodes of a stored session on it."""

    @api.get("/api/head")
    def head() -> HeadModel:
        return head_model()

    @api.get("/api/sessions/{session_id}/electrodes")
    def electrodes(services: Using, session_id: str) -> list[Electrode]:
        folder = folder_of(services, session_id)
        try:
            return electrodes_of(load_session_folder(folder))
        except UNREADABLE_PROBLEMS as problem:
            raise HTTPException(INVALID, why_unreadable(folder, problem)) from problem


def add_excerpt_route(api: FastAPI) -> None:
    """A part of the timeline of a stored session, cut out of its data."""

    @api.get("/api/sessions/{session_id}/excerpt")
    def excerpt(services: Using, session_id: str, start_s: float, end_s: float) -> Excerpt:
        tracks = tracks_of(services, session_id)
        start, end = max(start_s, tracks.timeline.start_s), min(end_s, tracks.timeline.end_s)
        if end <= start:
            raise HTTPException(
                INVALID, "the part ends before it begins, or lies outside the session"
            )
        return Excerpt(text=excerpt_of(folder_of(services, session_id), tracks, start, end))


def why_busy(services: Services) -> str:
    """Why no new session can start yet, or the headset cannot be looked for: one runs, the
    last one is still being finished (its recording stored, its metrics computed), or the
    headset is being looked for."""
    phase = services.runner.state().phase
    if phase == "finishing":
        return "the last session is still being analysed: try again in a moment"
    if phase == "idle" and services.runner.looking_for_headset():
        return "the headset is being looked for: try again in a moment"
    return "a session is running"


def add_new_session_routes(api: FastAPI) -> None:
    """The form for a new session: its sources and their recordings, checked."""

    @api.get("/api/new-session")
    def new_session(services: Using) -> NewSessionForm:
        return services.new_session.form()

    @api.post("/api/new-session/refresh")  # the headset looked for again
    def refresh_sources(services: Using) -> NewSessionForm:
        form = services.runner.look_for_headset(services.new_session.look_for_headset)
        if form is None:
            raise HTTPException(CONFLICT, why_busy(services))
        return form

    @api.get("/api/sources/{source_id}/check")
    def check(services: Using, source_id: str, recording: str | None = None) -> SourceCheck:
        return services.new_session.check(source_id, recording)

    @api.get("/api/audio-outputs")  # read again each time: headphones come and go
    def audio_outputs(services: Using) -> AudioOutputs:
        try:
            return services.new_session.audio_outputs()
        except AdamusError as problem:
            raise HTTPException(INVALID, str(problem)) from problem


def song_folder(services: Services, song_id: str) -> Path:
    """The folder of a song of the library; a song that is not there is not found."""
    try:
        return services.library.folder(song_id)
    except AdamusError as problem:
        raise HTTPException(NOT_FOUND, str(problem)) from problem


def draft_folder(services: Services, draft_id: str) -> Path:
    """The folder of a draft; a draft that is not there is not found."""
    try:
        return services.library.draft_folder(draft_id)
    except AdamusError as problem:
        raise HTTPException(NOT_FOUND, str(problem)) from problem


def add_song_routes(api: FastAPI) -> None:
    """The library of songs: listed, played, deleted."""

    @api.get("/api/songs")
    def songs(services: Using, offset: Offset, limit: PageSize) -> SongPage:
        return page_of(services.library.songs(), offset, limit)

    @api.get("/api/marks")
    def marks() -> SongMarks:
        return marks_view()

    @api.delete("/api/songs/{song_id}", status_code=NO_CONTENT)
    def delete_song(services: Using, song_id: str) -> None:
        song_folder(services, song_id)
        services.library.delete(song_id)

    @api.get("/api/songs/{song_id}" + SOUND)
    def song_sound(services: Using, song_id: str) -> FileResponse:
        return file_of(audio_in(song_folder(services, song_id)))

    @api.get("/api/songs/{song_id}" + COVER)
    def song_cover(services: Using, song_id: str) -> FileResponse:
        return cover_of(song_folder(services, song_id))


def add_draft_routes(api: FastAPI) -> None:
    """Drafts, where a song is changed: of a file of this computer, which then goes into the
    library, or of a song of the library, which it then takes the place of."""

    @api.post("/api/drafts")
    def draft_of_file(services: Using, request: DraftOfFile) -> Song:
        try:
            return song_view(services.library.draft(Path(request.path)))
        except (AdamusError, OSError) as problem:
            raise HTTPException(INVALID, str(problem)) from problem

    @api.post("/api/songs/{song_id}/draft")
    def draft_of_song(services: Using, song_id: str) -> Song:
        song_folder(services, song_id)
        try:
            return song_view(services.library.draft_of(song_id))
        except (AdamusError, OSError) as problem:
            raise HTTPException(INVALID, str(problem)) from problem

    @api.post("/api/drafts/{draft_id}/marks")
    def mark(services: Using, draft_id: str, request: SongMark) -> Song:
        draft_folder(services, draft_id)
        library = services.library
        try:
            return song_view(library.mark(draft_id, request.start_s, request.end_s, request.mark))
        except AdamusError as problem:
            raise HTTPException(INVALID, str(problem)) from problem

    @api.post("/api/drafts/{draft_id}/keep")
    def keep(services: Using, draft_id: str, request: DraftKept) -> Song:
        draft_folder(services, draft_id)
        picture = None if request.picture is None else Path(request.picture)
        try:
            return song_view(
                services.library.keep(draft_id, request.title, request.artist, picture)
            )
        except (AdamusError, OSError) as problem:
            raise HTTPException(INVALID, str(problem)) from problem

    @api.delete("/api/drafts/{draft_id}", status_code=NO_CONTENT)
    def discard(services: Using, draft_id: str) -> None:
        draft_folder(services, draft_id)
        services.library.discard(draft_id)


def add_draft_file_routes(api: FastAPI) -> None:
    """A draft to be played, and its picture."""

    @api.get("/api/drafts/{draft_id}" + SOUND)
    def draft_sound(services: Using, draft_id: str) -> FileResponse:
        return file_of(audio_in(draft_folder(services, draft_id)))

    @api.get("/api/drafts/{draft_id}" + COVER)
    def draft_cover(services: Using, draft_id: str) -> FileResponse:
        return cover_of(draft_folder(services, draft_id))


def add_run_routes(api: FastAPI) -> None:
    """The session that runs: asked for, followed, stopped."""

    @api.get("/api/run")
    def run_state(services: Using) -> RunState:
        return services.runner.state()

    @api.post("/api/run", status_code=ACCEPTED)
    def run(services: Using, request: SessionRequest) -> RunState:
        try:
            prepared = services.new_session.prepare(request)
        except (AdamusError, OSError) as problem:
            raise HTTPException(INVALID, str(problem)) from problem
        state = services.runner.request(prepared)
        if state is None:
            raise HTTPException(CONFLICT, why_busy(services))
        return state

    @api.post("/api/run/stop", status_code=NO_CONTENT)
    def stop(services: Using) -> None:
        services.runner.stop()
