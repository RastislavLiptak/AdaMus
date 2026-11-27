"""What of a song plays when: its marked parts, joined so that they go on as music does.

The song plays beat by beat in its own order while the beats are marked as the state wanted. At
the end of every beat the sequencer looks again: when the next beat is marked so, the song goes
on, also into the other state when it has just changed, as the song itself goes there; when it
is not marked so, because the part has run out or the state has changed, it jumps. It jumps to a
beat marked as wanted that lies at the same place in its bar, so the meter goes on; of those, to
the one whose four beats before it sound most like the four just played, and with a bar of the
part still ahead of it, so that it does not have to jump again at once. A beat played in the
last RECENT_S seconds counts less, so that a long state goes through all its parts rather than
around one. The music switches state at most once a bar, so that a state that wavers does not
throw it back and forth. A jump fades the old place out over JUMP_FADE_S while the new one comes
in; what two loud places together would put over full scale is cut at it.

It is driven by the output: each call renders the next frames, of any number, the same audio
whatever they are cut into. The state wanted is set from another thread and taken up at the next
beat.
"""

import queue
from dataclasses import dataclass

import numpy as np

from adamus import constants as c
from adamus.errors import AdamusError
from adamus.session_format import CONTINUE, START, SWITCH
from adamus.song_library import BEATS_PER_BAR, CHANNELS, MARKS, Analysis, Mark, beat_marks

JUMP_FADE_S = 0.06
RECENT_S = 30.0
CONTEXT_BEATS = 4  # how many beats before a place are compared with those before another
# What a beat played lately loses: more than any two places differ by, as the harmony of two
# beats agrees from 0 to 1 and their timbre from -1 to 1.
RECENT_PENALTY = 3.0


@dataclass(frozen=True)
class Decision:
    """Where the music went: at ``out_s`` (s of output so far) from the song at ``from_s`` to
    the song at ``to_s``, because it ``kind`` (started, a part ran out, the state switched)."""

    out_s: float
    kind: str
    mark: str
    from_s: float | None
    to_s: float


class Sequencer:
    """Renders a song at ``rate`` [frames x 2], its parts chosen and joined as described above."""

    def __init__(self, audio: np.ndarray, rate: int, analysis: Analysis, marks: list[Mark]) -> None:
        beats = np.array(analysis.beats)
        self._audio = audio
        self._rate = rate
        self._starts = np.round(beats * rate).astype(int)
        self._ends = np.append(self._starts[1:], len(audio))
        self._marks = beat_marks(beats, self._ends / rate, marks)
        self._phase = (np.arange(len(beats)) - analysis.first_downbeat) % BEATS_PER_BAR
        self._sound = np.array(analysis.sound)
        for state in MARKS:  # a session may switch to either at any time
            if state not in self._marks:
                raise AdamusError(f"no beat of the song is marked {state.lower()}")
        self._wanted = c.REST  # a session starts at rest
        self._since_switch = BEATS_PER_BAR  # beats played since the last switch
        self._beat = self._marks.index(c.REST)
        self._at = int(self._starts[self._beat])  # the next frame of the song to play
        self._fade_from: int | None = None  # the frame of the old place, while it fades out
        self._fade_left = 0
        self._fade_frames = round(JUMP_FADE_S * rate)
        self._rendered = 0  # frames
        self._last_played = np.full(len(beats), -np.inf)  # frame of the output each beat began at
        self.decisions: queue.SimpleQueue[Decision] = queue.SimpleQueue()
        self._decide(START, c.REST, None)

    def want(self, mark: str) -> None:
        """The state the music is to follow from the next beat on."""
        self._wanted = mark

    def render(self, frames: int) -> np.ndarray:
        out = np.zeros((frames, CHANNELS), dtype=np.float32)
        filled = 0
        for _ in range(frames):  # each turn fills at least a frame, or ends a beat
            if filled == frames:
                break
            end = int(self._ends[self._beat])
            count = min(frames - filled, end - self._at)
            out[filled : filled + count] = self._audio[self._at : self._at + count]
            self._fade(out[filled : filled + count])
            filled += count
            self._at += count
            self._rendered += count
            if self._at >= end:
                self._next_beat()
        return np.clip(out, -1.0, 1.0, out=out)  # two loud places faded together may go over

    def _fade(self, part: np.ndarray) -> None:
        """Mix the old place, fading out, into the start of the new one, fading in."""
        if self._fade_from is None:
            return
        count = min(len(part), self._fade_left)
        done = self._fade_frames - self._fade_left
        x = (done + np.arange(count)) / self._fade_frames * np.pi / 2
        old = self._audio[self._fade_from : self._fade_from + count]
        old = np.vstack([old, np.zeros((count - len(old), 2), dtype=old.dtype)])  # past the end
        part[:count] = part[:count] * np.sin(x)[:, None] + old * np.cos(x)[:, None]
        self._fade_from += count
        self._fade_left -= count
        if self._fade_left == 0:
            self._fade_from = None

    def _next_beat(self) -> None:
        """The beat after the one that has just ended: the next in the song if it is marked as
        the state followed, otherwise the best place to jump to. The state followed is the one
        wanted, unless the music switched less than a bar ago."""
        playing = self._marks[self._beat]
        assert playing is not None, "only a marked beat is ever played"
        wanted = playing if self._since_switch < BEATS_PER_BAR else self._wanted
        self._since_switch += 1
        kind = CONTINUE if playing == wanted else SWITCH
        if kind == SWITCH:
            self._since_switch = 0
        following = self._beat + 1
        if following < len(self._marks) and self._marks[following] == wanted:
            self._begin(following)  # the song goes on: into the other state, its own way
            if kind == SWITCH:
                self._decide(kind, wanted, self._at / self._rate)
            return
        target = self._best(following, wanted)
        self._fade_from, self._fade_left = self._at, self._fade_frames
        self._begin(target)
        self._decide(kind, wanted, self._fade_from / self._rate)

    def _begin(self, beat: int) -> None:
        self._beat = beat
        self._at = int(self._starts[beat])
        self._last_played[beat] = self._rendered

    def _best(self, following: int, wanted: str) -> int:
        """Where to go instead of beat ``following``: a beat marked ``wanted`` at its place in
        the bar, preferably with a bar of the part ahead; the best match, not played lately."""
        marked = np.array([mark == wanted for mark in self._marks])
        ahead = marked.copy()
        for step in range(1, BEATS_PER_BAR):
            ahead[:-step] &= marked[step:]
            ahead[-step:] = False
        place = self._phase == (self._phase[self._beat] + 1) % BEATS_PER_BAR
        for candidates in (ahead & place, marked & place, marked):
            if candidates.any():
                break
        beats = np.flatnonzero(candidates)
        score = np.array([self._match(following, int(beat)) for beat in beats])
        lately = self._rendered - self._last_played[beats] < RECENT_S * self._rate
        return int(beats[np.argmax(score - RECENT_PENALTY * lately)])

    def _match(self, a: int, b: int) -> float:
        """How alike the beats just before ``a`` and just before ``b`` sound."""
        last = len(self._sound) - 1
        pairs = [(min(max(a - m, 0), last), min(max(b - m, 0), last))
                 for m in range(1, CONTEXT_BEATS + 1)]  # fmt: skip
        return float(np.mean([self._sound[i] @ self._sound[j] for i, j in pairs]))

    def _decide(self, kind: str, mark: str, from_s: float | None) -> None:
        self.decisions.put(
            Decision(self._rendered / self._rate, kind, mark, from_s, self._at / self._rate)
        )
