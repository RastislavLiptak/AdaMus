"""The API the app works through.

It answers on the local computer only. Every call carries the token the app was told when
the program started (a header, or for the player of a sound a query parameter), so that a
program that was not given the token (of another user, or a web page in a browser) cannot
delete sessions or start one.
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
    Dashboard,
    Electrode,
    Excerpt,
    HeadModel,
    NewSessionForm,
    RunState,
    SessionPage,
    SessionRequest,
    SourceCheck,
    Timeline,
)
from adamus.server.dashboard import UNREADABLE_PROBLEMS, dashboard_of, why_unreadable
from adamus.server.electrodes import electrodes_of, head_model
from adamus.server.excerpt import excerpt_of
from adamus.server.new_session import NewSession
from adamus.server.session_catalogue import SessionCatalogue
from adamus.server.session_runner import SessionRunner
from adamus.server.timeline import SessionTracks, session_tracks

TOKEN_HEADER = "X-AdaMus-Token"
ACCEPTED, NO_CONTENT = 202, 204  # HTTP status codes of success
NOT_FOUND, CONFLICT, INVALID = 404, 409, 422  # and of failure
MAX_PAGE = 1000  # sessions on one page of the list, at most
BINARY = "application/octet-stream"
SOUND = "/sound"  # the end of the address of the sound of a track
AUDIO = "audio/wav"


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
    token: str = field(default_factory=secrets.token_urlsafe)
    tracks: TrackCache = field(default_factory=TrackCache)


def services_of(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


Using = Annotated[Services, Depends(services_of)]
Offset = Annotated[int, Query(ge=0)]  # of the first session of a page, in the list
PageSize = Annotated[int, Query(gt=0, le=MAX_PAGE)]


def authorized(
    request: Request,
    services: Using,
    header: Annotated[str | None, Header(alias=TOKEN_HEADER)] = None,
    token: Annotated[str | None, Query()] = None,
) -> None:
    """Every call carries the token in its header; a call without it finds nothing. Only the
    sound carries it in its address, as the player that asks for it sends no header."""
    in_address = token if request.url.path.endswith(SOUND) else None
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
    add_excerpt_route(api)
    add_electrode_routes(api)
    add_new_session_routes(api)
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


def add_session_routes(api: FastAPI) -> None:
    """The stored sessions: the list, the dashboard, the timeline, the deletion."""

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

    @api.delete("/api/sessions/{session_id}", status_code=NO_CONTENT)
    def delete(services: Using, session_id: str) -> None:
        folder_of(services, session_id)
        if is_busy_with(services, session_id):
            raise HTTPException(CONFLICT, "the session is still running")
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
