"""server.api_server: the server on a free local port, its token, the sessions it runs."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from adamus.activity_log import ActivityLog
from adamus.errors import AdamusError
from adamus.server import api_server
from adamus.server.api import TOKEN_HEADER
from adamus.server.api_server import ApiServer
from tests.support.stand_in_machine import StandInMachine
from tests.support.stand_in_preflight import Checked


class Server:
    """An API server over an empty stand-in machine; ``gone`` is the app's end."""

    def __init__(self, folder):
        (folder / "data").mkdir(parents=True)
        self.gone = threading.Event()
        machine = StandInMachine(folder / "sessions", folder / "data")
        self.api = ApiServer(Checked(), machine, ActivityLog(), self.gone.is_set)


@pytest.fixture
def server(tmp_path):
    opened = Server(tmp_path)
    yield opened
    opened.api.close()


def get(url, token=None):
    request = urllib.request.Request(url, headers={TOKEN_HEADER: token} if token else {})
    with urllib.request.urlopen(request) as response:
        return response.read().decode()


def test_the_api_answers_on_a_local_port_to_calls_with_its_token(server):
    address, token = server.api.open()
    assert address.startswith("http://127.0.0.1:")
    page = json.loads(get(address + "/api/sessions?offset=0&limit=5", token))
    assert page == {"total": 0, "sessions": []}
    with pytest.raises(urllib.error.HTTPError) as refused:
        get(address + "/api/sessions?offset=0&limit=5")
    refused.value.close()  # the answer it holds


def test_every_server_has_a_token_of_its_own(tmp_path):
    first, second = Server(tmp_path / "1"), Server(tmp_path / "2")
    try:
        assert first.api.open()[1] != second.api.open()[1]
    finally:
        first.api.close()
        second.api.close()


def test_the_sessions_are_run_until_the_app_has_gone(server):
    server.api.open()
    server.gone.set()
    server.api.serve()  # returns at once


def test_closing_stops_the_server(server):
    address, token = server.api.open()
    server.api.close()
    with pytest.raises(urllib.error.URLError):
        get(address + "/api/run", token)


def test_a_server_never_opened_is_closed_without_ado(server):
    server.api.close()


def test_a_server_that_does_not_start_in_time_is_reported(server, monkeypatch):
    monkeypatch.setattr(api_server, "SERVER_START_S", 0.0)
    with pytest.raises(AdamusError, match="did not start within 0 s"):
        server.api.open()
