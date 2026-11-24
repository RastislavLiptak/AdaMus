"""server.api: what the user interface can ask the program, and what it gets."""

# A test puts the runner into the phase that follows a session, which takes seconds otherwise.
# pyright: reportPrivateUsage=false

import json
import shutil
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from adamus import constants as c
from adamus.activity_log import ActivityLog
from adamus.audio_outputs import AudioOutput
from adamus.evaluation.session_reader import load_session_folder
from adamus.server.api import MAX_PAGE, TOKEN_HEADER, Services, create_api, create_app
from adamus.server.dashboard import PENDING
from adamus.server.electrodes import electrodes_of, head_model
from adamus.server.excerpt import excerpt_of
from adamus.server.new_session import NAME_HINT, NewSession
from adamus.server.session_catalogue import SessionCatalogue
from adamus.server.session_runner import SessionRunner
from adamus.server.timeline import session_tracks
from tests.support.finished_sessions import synthetic_signals
from tests.support.stand_in_devices import SPEAKERS, TV
from tests.support.stand_in_machine import StandInMachine
from tests.support.stand_in_preflight import Checked
from tests.support.synthetic_dataset import write_recording

SHORT = [("REST", 10), ("FOCUS", 10)]
TOKEN = "secret"


class Api:
    def __init__(self, folder, played_session):
        self.folder = folder
        self.session_id = played_session.directory.name
        shutil.copytree(played_session.directory, folder / "sessions" / self.session_id)
        write_recording(folder / "data" / "synthetic" / "short", synthetic_signals(SHORT), SHORT)
        self.checked = Checked()
        self.activity = ActivityLog()
        self.machine = machine = StandInMachine(folder / "sessions", folder / "data")
        self.runner = SessionRunner(self.checked, machine, self.activity, threading.Event().is_set)
        self.services = Services(
            catalogue=SessionCatalogue(folder / "sessions", folder / "data"),
            new_session=NewSession(self.checked, machine, self.activity), runner=self.runner,
            token=TOKEN,
        )  # fmt: skip
        self.client = TestClient(create_app(self.services), headers={TOKEN_HEADER: TOKEN})


@pytest.fixture
def api(tmp_path, played):  # noqa: F811
    return Api(tmp_path, played)


class TestToken:
    def test_the_api_answers_only_with_the_token(self, api):
        bare = TestClient(api.client.app)
        assert bare.get("/api/sessions?offset=0&limit=5").status_code == 404
        wrong = {TOKEN_HEADER: "guess"}
        assert bare.get("/api/sessions?offset=0&limit=5", headers=wrong).status_code == 404
        assert bare.get(f"/api/sessions?offset=0&limit=5&token={TOKEN}").status_code == 404
        assert api.client.get("/api/sessions?offset=0&limit=5").status_code == 200
        sound = f"/api/sessions/{api.session_id}/tracks/music/sound"  # a player sends no header
        assert bare.get(f"{sound}?token={TOKEN}").status_code == 200
        assert bare.get(f"{sound}?token=guess").status_code == 404

    @pytest.mark.parametrize("path", ["/openapi.json", "/docs", "/redoc"])
    def test_no_description_of_the_api_is_given_without_the_token(self, api, path):
        answer = TestClient(api.client.app).get(path)
        assert "openapi" not in answer.text.lower() and "/api/" not in answer.text

    def test_any_text_given_as_the_token_is_only_refused(self, api):
        assert TestClient(api.client.app).get("/api/run?token=é").status_code == 404

    def test_every_route_takes_the_token(self):
        routes = create_api().openapi()["paths"]
        assert routes
        for route in routes.values():
            for call in route.values():
                given = {(p["in"], p["name"]) for p in call["parameters"]}
                assert {("header", TOKEN_HEADER), ("query", "token")} <= given

    def test_nothing_but_the_api_is_served(self, api):
        assert api.client.get("/").status_code == 404

    def test_nothing_can_be_deleted_or_renamed_without_the_token(self, api):
        bare = TestClient(api.client.app)
        assert bare.delete(f"/api/sessions/{api.session_id}").status_code == 404
        renamed = bare.patch(f"/api/sessions/{api.session_id}", json={"name": "y"})
        assert renamed.status_code == 404
        assert api.services.catalogue.folder(api.session_id) is not None


class TestSessions:
    def test_a_page_of_sessions(self, api):
        page = api.client.get("/api/sessions", params={"offset": 0, "limit": 20}).json()
        assert page == {"total": 1, "sessions": [{"id": api.session_id, "name": "S01"}]}

    @pytest.mark.parametrize(
        "query", [{"offset": -1, "limit": 5}, {"offset": 0, "limit": 0},
                  {"offset": 0, "limit": MAX_PAGE + 1}, {"offset": 0}],
    )  # fmt: skip
    def test_a_page_must_lie_within_the_list(self, api, query):
        assert api.client.get("/api/sessions", params=query).status_code == 422

    @pytest.mark.parametrize("limit", [1, MAX_PAGE])
    def test_a_page_may_hold_one_session_or_as_many_as_a_page_holds(self, api, limit):
        page = api.client.get("/api/sessions", params={"offset": 0, "limit": limit})
        assert page.status_code == 200 and len(page.json()["sessions"]) == 1

    def test_the_dashboard_of_a_session(self, api):
        dashboard = api.client.get(f"/api/sessions/{api.session_id}/dashboard").json()
        assert dashboard["name"] == "S01" and dashboard["pending"] is None
        assert dashboard["summary"][0]["label"] == "Started"

    def test_the_head_and_the_electrodes_of_a_session_on_it(self, api):
        folder = api.folder / "sessions" / api.session_id
        assert api.client.get("/api/head").json() == head_model().model_dump()
        answer = api.client.get(f"/api/sessions/{api.session_id}/electrodes")
        expected = electrodes_of(load_session_folder(folder))
        assert answer.json() == [electrode.model_dump() for electrode in expected]

    def test_the_electrodes_of_a_session_that_cannot_be_read_say_why(self, api):
        (api.folder / "sessions" / api.session_id / c.FILE_META).write_text("{}")
        answer = api.client.get(f"/api/sessions/{api.session_id}/electrodes")
        assert answer.status_code == 422 and "electrode_rows" in answer.json()["detail"]

    def test_a_part_of_the_timeline_written_out(self, api):
        folder = api.folder / "sessions" / api.session_id
        answer = api.client.get(
            f"/api/sessions/{api.session_id}/excerpt", params={"start_s": 10, "end_s": 20}
        )
        assert answer.status_code == 200
        assert answer.json() == {"text": excerpt_of(folder, session_tracks(folder), 10.0, 20.0)}

    def test_a_part_is_written_out_as_far_as_the_session_reaches(self, api):
        folder = api.folder / "sessions" / api.session_id
        timeline = session_tracks(folder).timeline
        answer = api.client.get(
            f"/api/sessions/{api.session_id}/excerpt", params={"start_s": -1000, "end_s": 1e6}
        )
        whole = excerpt_of(folder, session_tracks(folder), timeline.start_s, timeline.end_s)
        assert answer.json()["text"] == whole

    @pytest.mark.parametrize(("start_s", "end_s"), [(20, 10), (10, 10), (1e6, 2e6)])
    def test_a_part_that_ends_before_it_begins_or_lies_outside_is_refused(
        self, api, start_s, end_s
    ):
        answer = api.client.get(
            f"/api/sessions/{api.session_id}/excerpt", params={"start_s": start_s, "end_s": end_s}
        )
        assert (answer.status_code, answer.json()["detail"]) == (
            422, "the part ends before it begins, or lies outside the session",
        )  # fmt: skip

    def test_a_session_that_is_not_there(self, api):
        part = api.client.get(
            "/api/sessions/20200101-000000_x/excerpt", params={"start_s": 0, "end_s": 1}
        )
        assert (part.status_code, part.json()["detail"]) == (404, "no session 20200101-000000_x")
        for path in ("dashboard", "timeline", "electrodes", "tracks/music/sound", "tracks/music"):
            answer = api.client.get(f"/api/sessions/20200101-000000_x/{path}")
            assert (answer.status_code, answer.json()["detail"]) == (
                404,
                "no session 20200101-000000_x",
            )
        deleted = api.client.delete("/api/sessions/20200101-000000_x")
        assert (deleted.status_code, deleted.json()["detail"]) == (
            404,
            "no session 20200101-000000_x",
        )
        renamed = api.client.patch("/api/sessions/20200101-000000_x", json={"name": "y"})
        assert (renamed.status_code, renamed.json()["detail"]) == (
            404,
            "no session 20200101-000000_x",
        )

    def test_each_session_shows_its_own_timeline_one_after_another(self, api, played):  # noqa: F811
        other = "20261110-160000_S02"
        shutil.copytree(played.directory, api.folder / "sessions" / other)
        (api.folder / "sessions" / other / c.FILE_AUDIO).unlink()  # a timeline of its own
        first = api.client.get(f"/api/sessions/{api.session_id}/timeline").json()
        second = api.client.get(f"/api/sessions/{other}/timeline").json()
        assert first != second
        assert api.client.get(f"/api/sessions/{api.session_id}/timeline").json() == first

    def test_the_timeline_and_the_samples_of_its_tracks(self, api):
        timeline = api.client.get(f"/api/sessions/{api.session_id}/timeline").json()
        expected = session_tracks(api.folder / "sessions" / api.session_id)
        assert timeline == expected.timeline.model_dump()
        response = api.client.get(f"/api/sessions/{api.session_id}/tracks/signal-AF7")
        assert response.headers["content-type"] == "application/octet-stream"
        received = np.frombuffer(response.content, dtype=np.float32)
        np.testing.assert_array_equal(received, expected.samples["signal-AF7"])
        missing = api.client.get(f"/api/sessions/{api.session_id}/tracks/protocol")
        assert (missing.status_code, missing.json()["detail"]) == (404, "no samples of protocol")

    def test_a_timeline_that_cannot_be_read_says_why_without_the_path(self, api):
        folder = api.folder / "sessions" / api.session_id
        (folder / c.FILE_CONFIG).write_text("x: 1\n")
        response = api.client.get(f"/api/sessions/{api.session_id}/timeline")
        detail = response.json()["detail"]
        assert response.status_code == 422
        assert detail.startswith("This session cannot be read: invalid configuration:")
        assert str(folder) not in detail

    def test_a_timeline_of_files_of_another_version_says_why(self, api):
        meta_path = api.folder / "sessions" / api.session_id / c.FILE_META
        meta = json.loads(meta_path.read_text())
        del meta["electrode_rows"]
        meta_path.write_text(json.dumps(meta))
        response = api.client.get(f"/api/sessions/{api.session_id}/timeline")
        assert response.status_code == 422
        assert response.json()["detail"].endswith("its files have no 'electrode_rows'")

    def test_the_sound_of_an_audio_track_can_be_played_from_any_point(self, api):
        """The audio element asks for parts of the file when the playhead jumps."""
        whole = (api.folder / "sessions" / api.session_id / c.FILE_AUDIO).read_bytes()
        part = api.client.get(
            f"/api/sessions/{api.session_id}/tracks/music/sound", headers={"Range": "bytes=100-199"}
        )
        assert part.status_code == 206 and part.content == whole[100:200]
        assert part.headers["content-type"] == "audio/wav"

    @pytest.mark.parametrize("track", ["signal-AF7", "protocol", "nothing"])
    def test_a_track_that_is_no_audio_has_no_sound(self, api, track):
        sound = api.client.get(f"/api/sessions/{api.session_id}/tracks/{track}/sound")
        assert (sound.status_code, sound.json()["detail"]) == (404, f"no sound of {track}")

    def test_deleting_a_session(self, api):
        api.client.get(f"/api/sessions/{api.session_id}/timeline")
        assert api.client.delete(f"/api/sessions/{api.session_id}").status_code == 204
        assert (
            api.client.get("/api/sessions", params={"offset": 0, "limit": 5}).json()["total"] == 0
        )
        assert api.client.get(f"/api/sessions/{api.session_id}/timeline").status_code == 404

    def test_renaming_a_session(self, api):
        api.client.get(f"/api/sessions/{api.session_id}/timeline")
        started = api.session_id.partition("_")[0]
        renamed = api.client.patch(f"/api/sessions/{api.session_id}", json={"name": " Karel 8 "})
        assert renamed.json() == {"id": f"{started}_Karel 8", "name": "Karel 8"}
        page = api.client.get("/api/sessions", params={"offset": 0, "limit": 5}).json()
        assert page["sessions"] == [renamed.json()]
        new = renamed.json()["id"]
        assert api.client.get(f"/api/sessions/{new}/timeline").status_code == 200
        assert api.client.get(f"/api/sessions/{new}/dashboard").status_code == 200
        assert api.client.get(f"/api/sessions/{api.session_id}/timeline").status_code == 404

    @pytest.mark.parametrize("name", ["", "   ", "a/b", "a:b", "x" * 51])
    def test_a_session_cannot_be_given_a_name_no_folder_can_have(self, api, name):
        refused = api.client.patch(f"/api/sessions/{api.session_id}", json={"name": name})
        assert (refused.status_code, refused.json()["detail"]) == (422, NAME_HINT)
        assert api.services.catalogue.folder(api.session_id) is not None

    def test_a_name_a_session_of_the_same_start_has_is_refused(self, api):
        taken = api.session_id.partition("_")[0] + "_taken"
        (api.folder / "sessions" / taken).mkdir()
        refused = api.client.patch(f"/api/sessions/{api.session_id}", json={"name": "taken"})
        assert (refused.status_code, refused.json()["detail"]) == (
            409,
            "A session that started at the same time is called taken.",
        )

    def test_a_folder_that_cannot_be_renamed_says_why_without_its_path(self, api, monkeypatch):
        def refused(session_id, name):
            raise PermissionError(13, "Permission denied", str(api.folder))

        monkeypatch.setattr(api.services.catalogue, "rename", refused)
        answer = api.client.patch(f"/api/sessions/{api.session_id}", json={"name": "y"})
        assert (answer.status_code, answer.json()["detail"]) == (
            422,
            "The session could not be renamed: Permission denied.",
        )

    def test_a_session_stored_again_where_a_deleted_one_was_is_read_anew(self, api, played):  # noqa: F811
        folder = api.folder / "sessions" / api.session_id
        assert api.client.get(f"/api/sessions/{api.session_id}/timeline").status_code == 200
        api.client.delete(f"/api/sessions/{api.session_id}")
        shutil.copytree(played.directory, folder)
        (folder / c.FILE_CONFIG).write_text("x: 1\n")  # not the session that was deleted
        assert api.client.get(f"/api/sessions/{api.session_id}/timeline").status_code == 422

    # starting: its folder is being made; finishing: its recording stored, its metrics computed
    @pytest.mark.parametrize("phase", ["starting", "running", "finishing"])
    def test_a_session_not_finished_yet_cannot_be_deleted_or_renamed(self, api, monkeypatch, phase):
        running = api.runner.state().model_copy(
            update={"phase": phase, "session_id": api.session_id}
        )
        monkeypatch.setattr(api.runner, "state", lambda: running)
        refused = api.client.delete(f"/api/sessions/{api.session_id}")
        assert (refused.status_code, refused.json()["detail"]) == (
            409,
            "the session is still running",
        )
        renamed = api.client.patch(f"/api/sessions/{api.session_id}", json={"name": "y"})
        assert (renamed.status_code, renamed.json()["detail"]) == (
            409,
            "the session is still running",
        )
        assert api.services.catalogue.folder(api.session_id) is not None
        dashboard = api.client.get(f"/api/sessions/{api.session_id}/dashboard").json()
        assert dashboard["pending"] == PENDING


class TestNewSession:
    def test_the_form(self, api):
        form = api.client.get("/api/new-session").json()
        assert [source["id"] for source in form["sources"]] == ["dataset:synthetic"]
        assert form["name_hint"] == NAME_HINT

    def test_the_recordings_and_the_check_of_a_source(self, api):
        sources = api.client.get("/api/new-session").json()["sources"]
        assert {s["id"]: s["recordings"] for s in sources}["dataset:synthetic"] == ["short"]
        check = api.client.get(
            "/api/sources/dataset:synthetic/check", params={"recording": "short"}
        )
        assert check.json() == {"warning": None, "problem": None}

    def test_a_session_is_asked_for_once(self, api):
        request = {
            "source_id": "dataset:synthetic",
            "recording": "short",
            "name": "S02",
            "output": SPEAKERS,
        }
        first = api.client.post("/api/run", json=request)
        assert first.status_code == 202
        assert first.json() == {"run": 1, "phase": "starting", "session_id": None, "problem": None}
        again = api.client.post("/api/run", json=request)
        assert (again.status_code, again.json()["detail"]) == (409, "a session is running")
        assert api.client.get("/api/run").json()["phase"] == "starting"
        assert api.client.post("/api/new-session/refresh").status_code == 409
        assert api.client.post("/api/run/stop").status_code == 204
        assert [action["type"] for action in api.activity.waiting] == ["STOP_REQUESTED"]

    @pytest.mark.parametrize("phase", ["running", "finishing"])
    def test_the_phase_of_a_run_is_told_as_it_is(self, api, phase):
        api.runner._set(phase)
        assert api.client.get("/api/run").json()["phase"] == phase

    def test_a_session_with_the_headset_is_asked_for_without_a_recording(self, api):
        """As the app asks: a value it does not have it leaves out."""
        api.client.post("/api/new-session/refresh")  # the headset found
        request = {"source_id": "headset:Muse 2", "name": "live", "output": SPEAKERS}
        assert api.client.post("/api/run", json=request).status_code == 202

    def test_while_the_last_session_is_finished_a_new_one_waits_for_it(self, api):
        api.runner._set("finishing")  # its recording stored, its metrics computed
        request = {
            "source_id": "dataset:synthetic",
            "recording": "short",
            "name": "S02",
            "output": SPEAKERS,
        }
        for answer in (api.client.post("/api/run", json=request),
                       api.client.post("/api/new-session/refresh")):  # fmt: skip
            assert answer.status_code == 409
            assert answer.json()["detail"] == (
                "the last session is still being analysed: try again in a moment"
            )

    def test_a_request_that_cannot_run_says_why(self, api):
        request = {
            "source_id": "dataset:synthetic",
            "recording": "short",
            "name": "a/b",
            "output": SPEAKERS,
        }
        response = api.client.post("/api/run", json=request)
        assert (response.status_code, response.json()["detail"]) == (422, NAME_HINT)
        assert api.client.get("/api/run").json()["run"] == 0

    def test_looking_for_the_headset(self, api):
        sources = api.client.post("/api/new-session/refresh").json()["sources"]
        assert [source["id"] for source in sources] == ["headset:Muse 2", "dataset:synthetic"]
        assert api.client.get("/api/new-session").json()["sources"] == sources

    def test_while_the_headset_is_looked_for_it_cannot_be_looked_for_or_used(self, api):
        api.client.post("/api/new-session/refresh")  # found
        api.runner._looking = True  # as while a look takes its seconds
        headset = {
            "source_id": "headset:Muse 2",
            "recording": None,
            "name": "S02",
            "output": SPEAKERS,
        }
        for answer in (api.client.post("/api/new-session/refresh"),
                       api.client.post("/api/run", json=headset)):  # fmt: skip
            assert answer.status_code == 409
            assert (
                answer.json()["detail"] == "the headset is being looked for: try again in a moment"
            )
        recording = {
            "source_id": "dataset:synthetic",
            "recording": "short",
            "name": "S02",
            "output": SPEAKERS,
        }
        assert api.client.post("/api/run", json=recording).status_code == 202


def recording_request(output=SPEAKERS):
    return {"source_id": "dataset:synthetic", "recording": "short", "name": "S02", "output": output}


class TestAudioOutput:
    """The music plays on the output asked for with the session: at first the system's."""

    def test_the_outputs_of_the_mac_and_the_one_the_system_plays_on(self, api):
        outputs = api.client.get("/api/audio-outputs").json()
        assert outputs["outputs"] == [{"id": TV, "title": TV}, {"id": SPEAKERS, "title": SPEAKERS}]
        assert outputs["system"] == SPEAKERS and outputs["hint"]

    def test_a_mac_that_plays_on_no_output(self, api):
        api.machine.audio, api.machine.outputs = None, []
        assert api.client.get("/api/audio-outputs").json()["system"] is None

    def test_an_output_that_is_not_connected_is_refused(self, api):
        answer = api.client.post("/api/run", json=recording_request("AirPods"))
        assert (answer.status_code, answer.json()["detail"]) == (422, "AirPods is not connected.")
        assert api.client.get("/api/run").json()["run"] == 0

    def test_an_output_put_away_after_it_was_chosen_starts_no_session(self, api):
        """Headphones chosen in the form, then put back in their case before Start."""
        api.machine.outputs = [api.machine.outputs[0]]  # the speakers the music plays on: gone
        answer = api.client.post("/api/run", json=recording_request(SPEAKERS))
        assert (answer.status_code, answer.json()["detail"]) == (
            422, f"{SPEAKERS} is not connected.",
        )  # fmt: skip
        assert api.client.get("/api/run").json()["run"] == 0
        assert api.checked.generator.outputs == [SPEAKERS]  # not started again

    def test_an_output_set_to_another_rate_says_how_to_set_it(self, api):
        api.machine.outputs[0] = AudioOutput(TV, 44100.0)
        answer = api.client.post("/api/run", json=recording_request(TV))
        assert (answer.status_code, answer.json()["detail"]) == (
            422, f"{TV} plays 44100 Hz. Set it to 48000 Hz in Audio MIDI Setup.",
        )  # fmt: skip
