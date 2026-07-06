"""CABT-format replay bundle writer for live Arena sessions.

One bundle directory per run, written incrementally (append-per-event, so a
crash loses nothing already seen):

    decisions.jsonl      one CABT-style decision per line: observation with
                         indexed options + the selected indices — the same
                         observation.select shape the Java bridge emits
    mirror_states.jsonl  every board snapshot sent to the display; this is
                         also the replay playback stream
    game_history.jsonl   decision prompts/responses + lifecycle events, with
                         raw game-state payloads redacted (training-safe)
    summary.json         counts, matches, games; rewritten on each flush
    card_cache.json      grpId -> card info for replay on other machines
    raw_audit.jsonl      full unredacted normalized events (ONLY when the
                         recorder is created with raw_audit=True — a debug
                         artifact that may contain raw Arena payloads)
"""

import json
import os

from .metadata import MatchMetadataCollector

__all__ = ["MirrorRecorder"]

# History event types that carry raw Arena game-state payloads (players,
# zones, gameObjects, full payload). These are redacted before persisting to
# game_history.jsonl so no raw private-zone data lands in default output;
# mirror_states.jsonl already carries the sanitized board.
_RAW_STATE_HISTORY_TYPES = frozenset((
    "ARENA_GAME_STATE", "ARENA_QUEUED_GAME_STATE"))
_RAW_STATE_FIELDS = ("players", "zones", "gameObjects", "payload")


class MirrorRecorder(object):

    def __init__(self, output_dir, card_db=None, raw_audit=False):
        self.output_dir = output_dir
        self.card_db = card_db
        self.raw_audit = raw_audit
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self._decisions = open(
            os.path.join(output_dir, "decisions.jsonl"), "a", encoding="utf-8")
        self._states = open(
            os.path.join(output_dir, "mirror_states.jsonl"), "a", encoding="utf-8")
        self._history = open(
            os.path.join(output_dir, "game_history.jsonl"), "a", encoding="utf-8")
        self._audit = None
        if raw_audit:
            self._audit = open(
                os.path.join(output_dir, "raw_audit.jsonl"), "a",
                encoding="utf-8")
        self._grp_ids = set()
        self._meta = MatchMetadataCollector()
        self.counts = {
            "decisions": 0,
            "decisionsMatched": 0,
            "mirrorStates": 0,
            "historyEvents": 0,
            "games": [],
            "matchIds": [],
        }

    # --- event sinks (wired to ArenaMatchTracker callbacks) ---

    def record_decision(self, record):
        self._write(self._decisions, record)
        self.counts["decisions"] += 1
        if record.get("selectionMatched"):
            self.counts["decisionsMatched"] += 1
        for option in (record.get("observation") or {}).get("select", {}).get("option", []):
            grp_id = (option.get("payload") or {}).get("grpId")
            if grp_id:
                self._grp_ids.add(grp_id)

    def record_state(self, snapshot, event=None):
        entry = dict(snapshot)
        if event is not None:
            entry["timestamp"] = event.get("timestamp")
            entry["eventId"] = event.get("eventId")
            entry["matchId"] = event.get("matchId")
        self._write(self._states, entry)
        self.counts["mirrorStates"] += 1
        self._collect_grp_ids(snapshot)
        self._meta.note_snapshot(entry)

    def record_history_event(self, event):
        # learn match metadata (players, event, result) from the full event
        # before any redaction strips fields
        self._meta.observe(event)
        # the full, unredacted event only ever reaches the opt-in audit file
        if self._audit is not None:
            self._write(self._audit, event)
        if event.get("inHistory"):
            self._write(self._history, _redact_history_event(event))
            self.counts["historyEvents"] += 1
        match_id = event.get("matchId")
        if match_id and match_id not in self.counts["matchIds"]:
            self.counts["matchIds"].append(match_id)

    def record_game(self, match_id, game_number):
        game = {"matchId": match_id, "gameNumber": game_number}
        if game not in self.counts["games"]:
            self.counts["games"].append(game)

    # --- lifecycle ---

    def flush(self):
        for handle in (self._decisions, self._states, self._history,
                       self._audit):
            if handle is not None:
                handle.flush()
        summary = dict(self.counts)
        summary["schemaVersion"] = 2
        summary["format"] = "cabt-arena-mirror"
        try:
            summary.update(self._meta.finalize(self.card_db))
        except Exception:
            pass  # metadata is best-effort; never lose the counts summary
        with open(os.path.join(self.output_dir, "summary.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if self.card_db is not None and self._grp_ids:
            try:
                self.card_db.export_cache(
                    os.path.join(self.output_dir, "card_cache.json"),
                    grp_ids=sorted(self._grp_ids))
            except Exception:
                pass  # cache export is best-effort; live capture continues

    def close(self):
        self.flush()
        for handle in (self._decisions, self._states, self._history,
                       self._audit):
            if handle is not None:
                handle.close()

    # --- helpers ---

    def _write(self, handle, record):
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
        handle.write("\n")

    def _collect_grp_ids(self, snapshot):
        zones = snapshot.get("zones") or {}
        for objects in (zones.get("battlefield"), zones.get("stack"),
                        zones.get("exile"), zones.get("command")):
            for obj in objects or []:
                if obj.get("grpId"):
                    self._grp_ids.add(obj["grpId"])
        for seat_objects in (zones.get("hands") or {}).values():
            for obj in seat_objects:
                if obj.get("grpId"):
                    self._grp_ids.add(obj["grpId"])
        for seat_objects in (zones.get("graveyards") or {}).values():
            for obj in seat_objects:
                if obj.get("grpId"):
                    self._grp_ids.add(obj["grpId"])


def _redact_history_event(event):
    """Strip raw game-state payloads from a history event before persisting.

    ``ARENA_GAME_STATE`` events carry the full Arena payload (players, zones,
    gameObjects) which can hold hidden-zone data; the sanitized board already
    lives in mirror_states.jsonl, so game_history keeps only the lightweight
    envelope. Decision, connect, match, and game-over events pass through —
    they carry the local player's own actions or public info.
    """
    if event.get("type") not in _RAW_STATE_HISTORY_TYPES:
        return event
    redacted = {key: value for key, value in event.items()
                if key not in _RAW_STATE_FIELDS}
    redacted["redacted"] = True
    return redacted
