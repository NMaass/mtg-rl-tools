"""Play a recorded mirror bundle back into the XMage display."""

import json
import os
import sys
import time

__all__ = ["ReplayPlayer", "load_bundle"]


def load_bundle(bundle_dir):
    """Read a recorded bundle; returns (states, decisions, summary)."""
    states = _read_jsonl(os.path.join(bundle_dir, "mirror_states.jsonl"))
    decisions = _read_jsonl(os.path.join(bundle_dir, "decisions.jsonl"))
    summary_path = os.path.join(bundle_dir, "summary.json")
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
    return states, decisions, summary


class ReplayPlayer(object):
    """Streams recorded snapshots to a display, narrating decisions.

    ``speed``: wall-clock multiplier over recorded timestamps (2.0 = twice as
    fast). ``interval``: fixed seconds per state instead. ``step``: wait for
    Enter between states.
    """

    def __init__(self, display=None, speed=None, interval=None, step=False,
                 log_stream=None):
        self.display = display
        self.speed = speed
        self.interval = interval
        self.step = step
        self._log = log_stream or sys.stdout

    def play(self, bundle_dir):
        states, decisions, summary = load_bundle(bundle_dir)
        if not states:
            raise ValueError("no mirror_states.jsonl content in %s" % bundle_dir)
        decisions_by_state = self._index_decisions(decisions)
        self._say("replaying %d states, %d decisions from %s"
                  % (len(states), len(decisions), bundle_dir))

        game_key = None
        previous_time = None
        for state in states:
            key = (state.get("matchId"), state.get("gameId"))
            if key != game_key:
                game_key = key
                self._open_game(state)
            if self.display is not None:
                self.display.send_state(state)
            for decision in decisions_by_state.get(_state_key(state), []):
                self._narrate_decision(decision)
            self._pace(previous_time, state.get("timestamp"))
            previous_time = state.get("timestamp")
        self._say("replay finished")

    # --- internals ---

    def _open_game(self, state):
        players = [{"seat": player["seat"], "name": player["name"]}
                   for player in state.get("players") or []]
        if self.display is not None and players:
            self.display.start_game(players,
                                    local_seat=state.get("localSeat"),
                                    match_id=state.get("matchId"))
        self._say("=== game %s (match %s) ===" % (state.get("gameId"),
                                                  state.get("matchId")))

    def _index_decisions(self, decisions):
        by_state = {}
        for decision in decisions:
            current = (decision.get("observation") or {}).get("current") or {}
            key = (decision.get("matchId"), current.get("seq"))
            by_state.setdefault(key, []).append(decision)
        return by_state

    def _narrate_decision(self, decision):
        select = (decision.get("observation") or {}).get("select") or {}
        chosen = []
        for index in decision.get("select") or []:
            for option in select.get("option") or []:
                if option["index"] == index:
                    chosen.append(option["label"])
        text = "decision #%s %s -> %s" % (
            decision.get("sequence"), select.get("type"),
            "; ".join(chosen) if chosen else str(decision.get("select")))
        self._say(text)
        if self.display is not None:
            try:
                self.display.send_message(text)
            except Exception:
                pass

    def _pace(self, previous_iso, current_iso):
        if self.step:
            try:
                input("  [enter] for next state... ")
            except EOFError:
                pass
            return
        if self.interval is not None:
            time.sleep(self.interval)
            return
        if self.speed and previous_iso and current_iso:
            delta = _iso_delta_seconds(previous_iso, current_iso)
            if delta is not None and delta > 0:
                time.sleep(min(delta / self.speed, 10.0))

    def _say(self, text):
        self._log.write("%s\n" % text)
        self._log.flush()


def _state_key(state):
    return (state.get("matchId"), state.get("seq"))


def _iso_delta_seconds(previous_iso, current_iso):
    import datetime
    try:
        previous = datetime.datetime.fromisoformat(previous_iso)
        current = datetime.datetime.fromisoformat(current_iso)
    except (ValueError, TypeError):
        return None
    return (current - previous).total_seconds()


def _read_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
