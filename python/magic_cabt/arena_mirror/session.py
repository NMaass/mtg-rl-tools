"""The live mirroring session: follower -> tracker -> recorder + display.

This is the run loop behind ``python -m magic_cabt.arena_mirror live`` and,
via ``feed_entries``, the offline/e2e test path (feed pre-captured entries
instead of tailing a file).
"""

import sys
import time

from .follower import LogFollower
from .mirror import enrich_snapshot
from .tracker import StreamingNormalizer, ArenaMatchTracker

__all__ = ["MirrorSession"]

FLUSH_INTERVAL_SECONDS = 5.0


class MirrorSession(object):

    def __init__(self, recorder=None, display=None, card_db=None,
                 log_stream=None, verbose=True):
        self.recorder = recorder
        self.display = display
        self.card_db = card_db
        self.verbose = verbose
        self._log_stream = log_stream or sys.stderr
        self._display_started_for = None
        self._last_flush = time.monotonic()
        self._display_failed = False
        self.normalizer = StreamingNormalizer()
        self.tracker = ArenaMatchTracker(
            on_snapshot=self._on_snapshot,
            on_decision=self._on_decision,
            on_game_event=self._on_game_event,
        )

    # --- feeding ---

    def follow(self, log_path, from_start=False, poll_seconds=0.25):
        """Tail a live Player.log until interrupted."""
        follower = LogFollower(log_path, poll_seconds=poll_seconds,
                               from_start=from_start)
        self._say("following %s (from %s)" % (
            log_path, "start" if from_start else "current end"))
        try:
            for entry in follower.follow():
                self.feed_entry(entry)
        finally:
            self.flush()

    def feed_entries(self, entries):
        for entry in entries:
            self.feed_entry(entry)
        self.flush()

    def feed_entry(self, entry):
        raw_events, events = self.normalizer.feed(entry)
        for event in events:
            if self.recorder is not None:
                self.recorder.record_history_event(event)
            self.tracker.handle_event(event)
        now = time.monotonic()
        if self.recorder is not None and now - self._last_flush > FLUSH_INTERVAL_SECONDS:
            self.recorder.flush()
            self._last_flush = now

    def flush(self):
        if self.recorder is not None:
            self.recorder.flush()

    # --- tracker callbacks ---

    def _on_snapshot(self, snapshot, event):
        enrich_snapshot(snapshot, self.card_db)
        if self.recorder is not None:
            self.recorder.record_state(snapshot, event)
        self._send_to_display(snapshot)

    def _on_decision(self, record):
        if self.recorder is not None:
            self.recorder.record_decision(record)
        if self.verbose:
            select = record["observation"]["select"]
            chosen = []
            for index in record.get("select") or []:
                for option in select["option"]:
                    if option["index"] == index:
                        chosen.append(option["label"])
            self._say("decision #%d %s -> %s" % (
                record["sequence"], select["type"],
                "; ".join(chosen) if chosen else str(record.get("select"))))

    def _on_game_event(self, kind, event):
        if self.recorder is not None:
            if kind == "game_start":
                self.recorder.record_game(self.tracker.match_id,
                                          self.tracker.game_number)
            self.recorder.record_history_event(event)
        if kind == "game_start":
            self._display_started_for = None  # next snapshot re-opens board
            self._say("game start: match=%s game=%s" % (
                self.tracker.match_id, self.tracker.game_number))
        elif kind == "game_over":
            self._say("game over: match=%s" % (event.get("matchId"),))
            if self.display is not None and not self._display_failed:
                self._display_call(self.display.finish_game, None)

    # --- display plumbing ---

    def _send_to_display(self, snapshot):
        if self.display is None or self._display_failed:
            return
        key = (self.tracker.match_id, self.tracker.game_number)
        if self._display_started_for != key:
            players = [
                {"seat": player["seat"], "name": player["name"]}
                for player in snapshot.get("players") or []
            ]
            if not players:
                return  # wait for a snapshot that knows the players
            self._display_call(
                self.display.start_game, players,
                local_seat=snapshot.get("localSeat"),
                match_id=self.tracker.match_id,
                game_number=self.tracker.game_number)
            self._display_started_for = key
        self._display_call(self.display.send_state, snapshot)

    def _display_call(self, method, *args, **kwargs):
        try:
            method(*args, **kwargs)
        except Exception as error:
            # Recording must survive a dead/errored display window.
            self._display_failed = True
            self._say("display disabled after error: %s" % (error,))

    def _say(self, text):
        if self.verbose:
            self._log_stream.write("[arena-mirror] %s\n" % text)
            self._log_stream.flush()
