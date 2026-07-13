"""Play a recorded mirror bundle back into the XMage display."""

import json
import os
import queue
import sys
import threading
import time

__all__ = ["ReplayPlayer", "ReplayController", "load_bundle",
           "describe_decision", "decision_is_pass", "option_is_pass"]


# Action types that are just yielding priority / declining to act — the noise a
# "jump to the next real play" control is meant to skip over.
_PASS_OPTION_TYPES = frozenset(("PASS",))


def describe_decision(decision):
    """One-line human summary of a recorded decision (e.g. for a log pane)."""
    select = (decision.get("observation") or {}).get("select") or {}
    chosen = _chosen_labels(decision, select)
    return "decision #%s %s -> %s" % (
        decision.get("sequence"), select.get("type"),
        "; ".join(chosen) if chosen else str(decision.get("select")))


def decision_is_pass(decision):
    """True when the decision is merely passing priority / taking no action.

    Used by the "next non-pass action" control: a priority pass, or an
    ActionsAvailable prompt answered with nothing, is skippable; casting a
    spell, attacking, blocking, mulliganing, choosing targets, etc. are not.
    """
    select = (decision.get("observation") or {}).get("select") or {}
    options = {opt["index"]: opt for opt in select.get("option") or []}
    indices = decision.get("select") or []
    if not indices:
        # An empty answer to "what do you want to do?" is an auto-pass; an
        # empty answer to attackers/blockers/targets is a real "none" choice.
        return (select.get("type") or "") == "ACTIONSAVAILABLEREQ"
    for index in indices:
        option = options.get(index)
        if option is None or option.get("type") not in _PASS_OPTION_TYPES:
            return False
    return True


def option_is_pass(option):
    """True when a single recorded option merely yields priority."""
    return (option or {}).get("type") in _PASS_OPTION_TYPES


def _chosen_labels(decision, select):
    labels = []
    options = {opt["index"]: opt for opt in select.get("option") or []}
    for index in decision.get("select") or []:
        option = options.get(index)
        if option is not None:
            labels.append(option["label"])
    return labels


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
        self.narrated = []

    def play(self, bundle_dir):
        states, decisions, summary = load_bundle(bundle_dir)
        if not states:
            raise ValueError("no mirror_states.jsonl content in %s" % bundle_dir)
        decisions_by_state = self._index_decisions(decisions)
        self.narrated = []  # decision sequence numbers, in playback order
        self._say("replaying %d states, %d decisions from %s"
                  % (len(states), len(decisions), bundle_dir))

        game_key = None
        match_key = None
        previous_time = None
        for state in states:
            key = state.get("gameInstance")
            if key != game_key:
                game_key = key
                self._open_game(state)
            # a decision with no board snapshot (e.g. concede) is attributed
            # to the end of its match; flush the previous match's leftovers
            # when the match changes so nothing is dropped or misordered
            if state.get("matchId") != match_key:
                self._flush_unplaced(decisions_by_state, match_key)
                match_key = state.get("matchId")
            if self.display is not None:
                self.display.send_state(state)
            # consume on first occurrence: several states can share a seq
            # (queued/diff states), and each decision narrates exactly once
            for decision in decisions_by_state.pop(_state_key(state), []):
                self._narrate_decision(decision)
            self._pace(previous_time, state.get("timestamp"))
            previous_time = state.get("timestamp")
        self._flush_unplaced(decisions_by_state, match_key)
        self._say("replay finished")

    def _flush_unplaced(self, decisions_by_state, match_id):
        for decision in decisions_by_state.pop(("match", match_id, None), []):
            self._narrate_decision(decision)

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
        # key on gameInstance, not matchId: seq restarts each game within a
        # match, so (matchId, seq) would attach a game-2 decision to a game-1
        # state. Board-less decisions (concede) key on (matchId, None) and are
        # flushed at match end.
        return index_decisions_by_state(decisions)

    def _narrate_decision(self, decision):
        self.narrated.append(decision.get("sequence"))
        text = describe_decision(decision)
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


def index_decisions_by_state(decisions):
    """Map each decision to the state key it narrates at.

    Same keying as playback: (gameInstance, seq) for board decisions, and
    (matchId, None) for board-less decisions (concede) flushed at match end.
    """
    by_state = {}
    for decision in decisions:
        current = (decision.get("observation") or {}).get("current") or {}
        if current.get("seq") is not None:
            key = ("game", current.get("gameInstance"), current.get("seq"))
        else:
            key = ("match", decision.get("matchId"), None)
        by_state.setdefault(key, []).append(decision)
    return by_state


class ReplayController(object):
    """A seekable, pausable replay engine for an interactive viewer.

    Unlike :class:`ReplayPlayer` (a straight-through narrator for the CLI),
    this drives playback from a background thread that a GUI steers with
    transport commands — play/pause, live speed, single-step either way, and
    jumps to the next decision or the next *non-pass* action. Each rendered
    frame is reported through ``on_progress`` so the UI can show a scrubber and
    a turn/phase readout.

    Callbacks fire on the worker thread; a Tk GUI marshals them to its own
    loop. ``on_progress(info)`` receives a dict with ``index``, ``total``,
    ``turn``, ``phase``, ``playing``, ``atEnd`` and the current decision text.
    """

    _SLICE = 0.05  # seconds between command polls while pacing a frame

    def __init__(self, bundle_dir, display=None, on_progress=None,
                 on_message=None, speed=4.0):
        self.states, decisions, self.summary = load_bundle(bundle_dir)
        self.display = display
        self._on_progress = on_progress
        self._on_message = on_message
        self._speed = max(0.25, float(speed))
        self._decisions = self._place_decisions(decisions)
        self.total = len(self.states)
        # state indices where the turn number advances — the scrubber draws a
        # tick at each so you can see the turn boundaries at a glance
        self.turn_marks = self._compute_turn_marks()

        self._index = 0
        self._playing = False
        self._stop = False
        self._commands = queue.Queue()
        self._thread = None
        self._opened_games = set()
        self._display_errors_reported = 0

    # --- lifecycle ---

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        self._commands.put(("stop", None))

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    # --- transport commands (thread-safe; callable from the GUI thread) ---

    def play(self):
        self._commands.put(("play", None))

    def pause(self):
        self._commands.put(("pause", None))

    def toggle(self):
        self._commands.put(("toggle", None))

    def set_speed(self, speed):
        self._commands.put(("speed", speed))

    def step(self, delta=1):
        self._commands.put(("step", delta))

    def seek(self, index):
        self._commands.put(("seek", index))

    def next_decision(self, meaningful=False):
        self._commands.put(("jump", meaningful))

    # --- worker ---

    def _run(self):
        if not self.states:
            self._report(at_end=True)
            return
        self._render(0)
        while not self._stop:
            if self._playing:
                self._play_step()
            else:
                self._wait_command()
        self._drain_narration_to_end()

    def _wait_command(self):
        try:
            command = self._commands.get(timeout=0.1)
        except queue.Empty:
            return
        self._apply(command)

    def _play_step(self):
        # advance to the next frame, pacing by recorded time / speed, but stay
        # responsive to commands by sleeping in small slices
        if self._index >= self.total - 1:
            self._playing = False
            self._report()
            return
        delay = self._frame_delay(self._index)
        waited = 0.0
        while waited < delay and self._playing and not self._stop:
            try:
                command = self._commands.get(timeout=min(self._SLICE, delay - waited))
                self._apply(command)
                if not self._playing:
                    return
                delay = self._frame_delay(self._index)  # speed may have changed
                waited = 0.0
                continue
            except queue.Empty:
                waited += self._SLICE
        if self._playing and not self._stop:
            self._render(self._index + 1)
            if self._index >= self.total - 1:
                self._playing = False
                self._report(at_end=True)

    def _apply(self, command):
        name, arg = command
        if name == "play":
            self._playing = self._index < self.total - 1
        elif name == "pause":
            self._playing = False
        elif name == "toggle":
            self._playing = (not self._playing) and self._index < self.total - 1
        elif name == "speed":
            try:
                self._speed = max(0.25, float(arg))
            except (TypeError, ValueError):
                pass
        elif name == "step":
            self._playing = False
            self._render(self._clamp(self._index + (arg or 1)))
        elif name == "seek":
            self._playing = False
            self._render(self._clamp(arg))
        elif name == "jump":
            self._playing = False
            self._jump(meaningful=bool(arg))
        elif name == "stop":
            self._stop = True
        self._report()

    def _jump(self, meaningful):
        for i in range(self._index + 1, self.total):
            decisions = self._decisions.get(i)
            if not decisions:
                continue
            if not meaningful or any(not decision_is_pass(d) for d in decisions):
                self._render(i)
                return
        # nothing ahead: go to the end so the user sees the final board
        self._render(self.total - 1)

    # --- rendering ---

    def _render(self, index):
        index = self._clamp(index)
        self._index = index
        state = self.states[index]
        self._ensure_game_open(state)
        if self.display is not None:
            try:
                self.display.send_state(state)
            except Exception as error:
                self._report_display_error("board update failed", error)
        for decision in self._decisions.get(index, []):
            self._narrate(decision)
        self._report()

    def _ensure_game_open(self, state):
        key = state.get("gameInstance")
        if key in self._opened_games or self.display is None:
            return
        players = [{"seat": p["seat"], "name": p["name"]}
                   for p in state.get("players") or []]
        if not players:
            return
        try:
            self.display.start_game(players, local_seat=state.get("localSeat"),
                                    match_id=state.get("matchId"))
        except Exception as error:
            self._report_display_error("board could not open the game", error)
        self._opened_games.add(key)

    def _report_display_error(self, what, error):
        """Playback survives a broken board, but never hides it: the first
        few failures are narrated so a blank display is diagnosable."""
        self._display_errors_reported += 1
        if self._on_message is not None and self._display_errors_reported <= 3:
            self._on_message("[display] %s: %s" % (what, error))

    def _narrate(self, decision):
        text = describe_decision(decision)
        if self._on_message is not None:
            self._on_message(text)
        if self.display is not None:
            try:
                self.display.send_message(text)
            except Exception:
                pass

    def _report(self, at_end=None):
        if self._on_progress is None:
            return
        state = self.states[self._index] if self.states else {}
        decisions = self._decisions.get(self._index, [])
        info = {
            "index": self._index,
            "total": self.total,
            "playing": self._playing,
            "atEnd": self._index >= self.total - 1 if at_end is None else at_end,
            "turn": state.get("turnNumber"),
            "phase": _phase_label(state),
            "activeName": _active_name(state),
            "decision": describe_decision(decisions[-1]) if decisions else None,
        }
        self._on_progress(info)

    def _drain_narration_to_end(self):
        pass

    # --- setup helpers ---

    def _place_decisions(self, decisions):
        """state index -> [decisions], assigned at each key's first state."""
        by_key = index_decisions_by_state(decisions)
        placed = {}
        consumed = set()
        last_index_for_match = {}
        for i, state in enumerate(self.states):
            last_index_for_match[state.get("matchId")] = i
            key = _state_key(state)
            if key in consumed:
                continue
            consumed.add(key)
            for decision in by_key.get(key, []):
                placed.setdefault(i, []).append(decision)
        # board-less (concede) decisions attach to their match's last state
        for key, decisions_list in by_key.items():
            if key[0] != "match":
                continue
            match_id = key[1]
            index = last_index_for_match.get(match_id, self.total - 1)
            for decision in decisions_list:
                placed.setdefault(index, []).append(decision)
        return placed

    def _compute_turn_marks(self):
        marks = []
        last = None
        for i, state in enumerate(self.states):
            turn = state.get("turnNumber")
            seat = state.get("activeSeat")
            key = (turn, seat)
            if turn is not None and key != last:
                last = key
                marks.append({"index": i, "turn": turn,
                              "active": _active_name(state)})
        return marks

    def _frame_delay(self, index):
        if index >= self.total - 1:
            return 0.0
        delta = _iso_delta_seconds(self.states[index].get("timestamp"),
                                   self.states[index + 1].get("timestamp"))
        if delta is None or delta <= 0:
            return 0.12 / self._speed
        return min(delta / self._speed, 4.0)

    def _clamp(self, index):
        if index < 0:
            return 0
        if index > self.total - 1:
            return self.total - 1
        return index


def _phase_label(state):
    phase = (state.get("phase") or "").replace("Phase_", "")
    step = (state.get("step") or "").replace("Step_", "")
    if step and step != phase:
        return "%s / %s" % (phase, step) if phase else step
    return phase or None


def _active_name(state):
    active = state.get("activeSeat")
    for player in state.get("players") or []:
        if player.get("seat") == active:
            return player.get("name")
    return None


def _state_key(state):
    return ("game", state.get("gameInstance"), state.get("seq"))


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
