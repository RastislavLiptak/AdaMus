"""The stored sessions as the app lists them: newest first, a page at a time.

A session is known by its folder, which is also its id; the app shows only the
name the operator gave it. Deleting a session deletes its folder and the recording of
what the headset delivered in it.
"""

import shutil
from pathlib import Path

from adamus import constants as c
from adamus.server.api_models import SessionPage, SessionSummary
from adamus.session_format import is_session_folder, name_of


class SessionCatalogue:
    def __init__(self, sessions_root: Path, datasets_root: Path) -> None:
        self._sessions_root = Path(sessions_root)
        self._datasets_root = Path(datasets_root)

    def page(self, offset: int, limit: int) -> SessionPage:
        """``limit`` sessions from the ``offset``-th newest one on."""
        folders = self._folders()
        shown = folders[offset : offset + limit]
        return SessionPage(
            total=len(folders),
            sessions=[SessionSummary(id=folder.name, name=name_of(folder)) for folder in shown],
        )

    def folder(self, session_id: str) -> Path | None:
        """The folder of a session; None when there is no such session."""
        folder = self._sessions_root / session_id
        return folder if is_session_folder(session_id) and folder.is_dir() else None

    def own_recording(self, session_id: str) -> Path | None:
        """What the headset delivered in the session, as a recording of the data folder."""
        folder = self._datasets_root / c.OWN_RECORDINGS / session_id
        return folder if is_session_folder(session_id) and folder.is_dir() else None

    def delete(self, session_id: str) -> None:
        """Delete the session and its own recording, for good."""
        for folder in (self.folder(session_id), self.own_recording(session_id)):
            if folder is not None:
                shutil.rmtree(folder)

    def _folders(self) -> list[Path]:
        if not self._sessions_root.is_dir():
            return []
        found = [
            folder
            for folder in self._sessions_root.iterdir()
            if folder.is_dir() and is_session_folder(folder.name)
        ]
        # The folder name begins with the time the session started.
        return sorted(found, key=lambda folder: folder.name, reverse=True)
