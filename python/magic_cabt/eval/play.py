"""Local agent-vs-agent tournament runner.

    python -m magic_cabt.eval.play \
        --agent0 random --agent1 first \
        --games 20 --seed 1 \
        --deck0 examples/basic_deck.txt --deck1 examples/basic_deck.txt \
        --out target/eval/random-vs-first

Starts real XMage games through ``CabtBridge``, routes each decision to the
acting seat's agent, writes a per-game replay (the same self-play JSONL frames
``examples/run_selfplay.py`` produces, so ``magic_cabt.training.records`` reads
them unchanged), and writes a ``summary.json``.

Individual game failures are caught and tallied so a crash on game 7 does not
lose games 8..N -- unless ``--fail-fast`` is given.

Benchmark runs fail on illegal agent selections by default. Repair mode exists
only as an explicit smoke/debug option via ``--illegal-action-mode repair``.

The module splits cleanly into pure logic (``play_game`` / ``summarize``) and
IO (``run_tournament`` / ``main``) so the aggregation and routing can be
unit-tested against a mock bridge without launching Java.
"""

import argparse
import json
import os
import sys

from magic_cabt.agents import (
    clamp_selection,
    is_legal_selection,
    make_agent,
)
from magic_cabt.protocol import CabtBridge, CabtGameError, load_decklist

__all__ = [
    "ILLEGAL_ACTION_FAIL",
    "ILLEGAL_ACTION_REPAIR",
    "InvalidAgentSelectionError",
    "play_game",
    "run_tournament",
    "summarize",
    "build_parser",
    "main",
]

DEFAULT_PLAYER_NAMES = ("P0", "P1")
ILLEGAL_ACTION_FAIL = "fail"
ILLEGAL_ACTION_REPAIR = "repair"
_ILLEGAL_ACTION_MODES = (ILLEGAL_ACTION_FAIL, ILLEGAL_ACTION_REPAIR)


class InvalidAgentSelectionError(ValueError):
    """Raised when an agent emits an illegal option-index selection."""

    def __init__(self, seat, selection, select, invalid_by_seat):
        self.seat = seat
        self.selection = list(selection) if isinstance(selection, list) else selection
        self.select_type = select.get("type") if isinstance(select, dict) else None
        self.invalid_by_seat = list(invalid_by_seat)
        super(InvalidAgentSelectionError, self).__init__(
            "illegal selection %r from seat %r for prompt %r" %
            (self.selection, seat, self.select_type))


def play_game(bridge, agents, deck0, deck1, seed=None, max_turns=None,
              player_names=DEFAULT_PLAYER_NAMES, record_writer=None,
              illegal_action_mode=ILLEGAL_ACTION_FAIL):
    """Play one game to completion on an already-open ``bridge``.

    ``agents`` is a 2-tuple indexed by seat. Each pending decision is routed
    to ``agents[select.playerIndex]``. An illegal agent selection fails the
    game by default so benchmark metrics cannot hide broken policies. Pass
    ``illegal_action_mode="repair"`` only for smoke/debug runs that should keep
    advancing after a bad policy output.

    ``record_writer``, if given, is called once per decision with a self-play
    replay frame and once at game end with the trailing ``{result}`` frame.

    Returns an outcome dict (see ``summarize`` for the fields it consumes).
    """
    if illegal_action_mode not in _ILLEGAL_ACTION_MODES:
        raise ValueError("unknown illegal_action_mode: %r" %
                         (illegal_action_mode,))
    response = bridge.game_start(
        deck0, deck1, player_names=list(player_names),
        seed=seed, max_turns=max_turns,
    )
    decisions = 0
    invalid_by_seat = [0, 0]
    game_id = _game_id_of(record_writer)

    while not bridge.finished:
        observation = response["observation"]
        select = observation["select"]
        seat = select["playerIndex"]
        selection = agents[seat].select(observation)
        if not is_legal_selection(selection, select):
            if 0 <= seat < 2:
                invalid_by_seat[seat] += 1
            if illegal_action_mode == ILLEGAL_ACTION_FAIL:
                raise InvalidAgentSelectionError(
                    seat, selection, select, invalid_by_seat)
            selection = clamp_selection(selection, select)
            if not is_legal_selection(selection, select):
                raise InvalidAgentSelectionError(
                    seat, selection, select, invalid_by_seat)
        if record_writer is not None:
            record_writer.write_frame({
                "gameId": game_id,
                "sequence": response.get("sequence"),
                "player": seat,
                "observation": observation,
                "selected": selection,
            })
        response = bridge.game_select(selection)
        decisions += 1

    result = bridge.result or {}
    if record_writer is not None:
        record_writer.write_frame({"gameId": game_id, "result": result})

    return {
        "completed": True,
        "winnerSeat": _winner_seat(result, player_names),
        "decisions": decisions,
        "invalidBySeat": invalid_by_seat,
        "invalidSelections": invalid_by_seat[0] + invalid_by_seat[1],
        "failClosed": False,
        "error": None,
        "result": result,
    }


def _winner_seat(result, player_names):
    """Map an engine winner string to a seat index (``None`` for a draw).

    XMage reports the winner as a sentence embedding the player name
    (``"Player P0 is the winner"``); with seat-unique names a substring match
    is unambiguous. A missing/no-match winner is treated as a draw.
    """
    winner = (result or {}).get("winner")
    if not isinstance(winner, str) or not winner:
        return None
    for seat, name in enumerate(player_names):
        if name and name in winner:
            return seat
    return None


def _game_id_of(record_writer):
    return getattr(record_writer, "game_id", None) if record_writer else None


def summarize(outcomes, agent_specs=None):
    """Aggregate per-game outcomes into a tournament summary dict."""
    attempted = len(outcomes)
    completed = sum(1 for o in outcomes if o.get("completed"))
    crashes = attempted - completed
    wins = [0, 0]
    draws = 0
    total_decisions = 0
    invalid = [0, 0]
    fail_closed = 0
    errors = []

    for outcome in outcomes:
        total_decisions += outcome.get("decisions", 0) or 0
        by_seat = outcome.get("invalidBySeat") or [0, 0]
        invalid[0] += by_seat[0]
        invalid[1] += by_seat[1]
        if outcome.get("failClosed"):
            fail_closed += 1
        if outcome.get("error"):
            errors.append(outcome["error"])
        if outcome.get("completed"):
            seat = outcome.get("winnerSeat")
            if seat in (0, 1):
                wins[seat] += 1
            else:
                draws += 1

    summary = {
        "gamesAttempted": attempted,
        "gamesCompleted": completed,
        "crashes": crashes,
        "failClosed": fail_closed,
        "winsBySeat": {"0": wins[0], "1": wins[1]},
        "lossesBySeat": {"0": wins[1], "1": wins[0]},
        "draws": draws,
        "totalDecisions": total_decisions,
        "averageDecisionsPerGame": (
            total_decisions / float(completed) if completed else 0.0),
        "invalidSelections": {
            "total": invalid[0] + invalid[1],
            "seat0": invalid[0],
            "seat1": invalid[1],
        },
        "errors": errors,
    }
    if agent_specs is not None:
        summary["agents"] = {"seat0": agent_specs[0], "seat1": agent_specs[1]}
    return summary


class _GameRecordWriter(object):
    """Per-game JSONL replay writer (one self-play frame per line)."""

    def __init__(self, handle, game_id):
        self._handle = handle
        self.game_id = game_id

    def write_frame(self, frame):
        self._handle.write(json.dumps(frame) + "\n")


def run_tournament(agent_specs, deck0, deck1, games=1, seed=None, max_turns=None,
                   bridge=None, bridge_factory=None, out_dir=None,
                   fail_fast=False, player_names=DEFAULT_PLAYER_NAMES, log=None,
                   illegal_action_mode=ILLEGAL_ACTION_FAIL):
    """Run ``games`` games between the two agent specs; return the summary.

    ``agent_specs`` is a 2-tuple of agent names (``"random"`` / ``"first"``).
    Fresh agents are built per game with a per-game seed so results are
    reproducible and stochastic agents don't share state.

    Supply either an already-open ``bridge`` (kept open across games; used by
    tests with a mock) or a ``bridge_factory`` callable that opens one
    (closed when done). ``out_dir``, if set, receives ``game-NNNN.jsonl``
    replays and ``summary.json``.
    """
    if illegal_action_mode not in _ILLEGAL_ACTION_MODES:
        raise ValueError("unknown illegal_action_mode: %r" %
                         (illegal_action_mode,))
    if bridge is None and bridge_factory is None:
        raise ValueError("run_tournament needs a bridge or a bridge_factory")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    owns_bridge = bridge is None
    active_bridge = bridge if bridge is not None else bridge_factory()
    outcomes = []
    try:
        for game_index in range(games):
            game_seed = None if seed is None else seed + game_index
            agents = (make_agent(agent_specs[0], seed=game_seed),
                      make_agent(agent_specs[1], seed=(
                          None if game_seed is None else game_seed + 1)))
            outcome = _play_one(
                active_bridge, agents, deck0, deck1, game_index, game_seed,
                max_turns, player_names, out_dir, log, illegal_action_mode)
            outcomes.append(outcome)
            if not outcome.get("completed") and fail_fast:
                if log:
                    log("stopping early (--fail-fast) after game %d" % game_index)
                break
    finally:
        if owns_bridge:
            active_bridge.close()

    summary = summarize(outcomes, agent_specs)
    if out_dir:
        with open(os.path.join(out_dir, "summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def _play_one(bridge, agents, deck0, deck1, game_index, game_seed, max_turns,
              player_names, out_dir, log, illegal_action_mode):
    game_id = "game-%04d" % game_index
    handle = None
    writer = None
    if out_dir:
        handle = open(os.path.join(out_dir, game_id + ".jsonl"), "w")
        writer = _GameRecordWriter(handle, game_id)
    try:
        outcome = play_game(
            bridge, agents, deck0, deck1, seed=game_seed, max_turns=max_turns,
            player_names=player_names, record_writer=writer,
            illegal_action_mode=illegal_action_mode)
        if log:
            log("%s: winner seat=%s decisions=%d invalid=%d"
                % (game_id, outcome.get("winnerSeat"),
                   outcome.get("decisions", 0),
                   outcome.get("invalidSelections", 0)))
        return outcome
    except InvalidAgentSelectionError as error:
        if log:
            log("%s: ILLEGAL-ACTION %s" % (game_id, error))
        _recover_bridge(bridge)
        return _failed_outcome(
            str(error), fail_closed=False, invalid_by_seat=error.invalid_by_seat)
    except CabtGameError as error:
        # Engine fail-closed (bad decision, engine error): tally and move on.
        if log:
            log("%s: FAIL-CLOSED %s" % (game_id, error))
        _recover_bridge(bridge)
        return _failed_outcome(str(error), fail_closed=True)
    except Exception as error:  # noqa: BLE001 - keep the tournament running
        if log:
            log("%s: ERROR %s" % (game_id, error))
        _recover_bridge(bridge)
        return _failed_outcome(str(error), fail_closed=False)
    finally:
        if handle is not None:
            handle.close()


def _recover_bridge(bridge):
    """Best-effort: end the broken game so the next one can start."""
    try:
        bridge.game_finish()
    except Exception:  # noqa: BLE001 - recovery must never raise
        pass


def _failed_outcome(message, fail_closed, invalid_by_seat=None):
    if invalid_by_seat is None:
        invalid_by_seat = [0, 0]
    return {
        "completed": False,
        "winnerSeat": None,
        "decisions": 0,
        "invalidBySeat": list(invalid_by_seat),
        "invalidSelections": invalid_by_seat[0] + invalid_by_seat[1],
        "failClosed": fail_closed,
        "error": message,
        "result": None,
    }


# --- CLI --------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic_cabt.eval.play",
        description="Run local agent-vs-agent XMage games and summarize them.",
    )
    parser.add_argument("--agent0", default="random",
                        help="seat 0 agent (random|first)")
    parser.add_argument("--agent1", default="first",
                        help="seat 1 agent (random|first)")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--deck0", required=True)
    parser.add_argument("--deck1", required=True)
    parser.add_argument("--out", default=None,
                        help="output dir for per-game replays and summary.json")
    parser.add_argument("--classpath", default=None,
                        help="Java classpath (default: $MAGIC_CABT_CLASSPATH)")
    parser.add_argument("--fail-fast", action="store_true",
                        help="stop at the first game that crashes")
    parser.add_argument("--illegal-action-mode", default=ILLEGAL_ACTION_FAIL,
                        choices=_ILLEGAL_ACTION_MODES,
                        help="fail benchmark games on illegal agent output by "
                             "default; use repair only for smoke/debug runs")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    deck0 = load_decklist(args.deck0)
    deck1 = load_decklist(args.deck1)
    summary = run_tournament(
        (args.agent0, args.agent1), deck0, deck1,
        games=args.games, seed=args.seed, max_turns=args.max_turns,
        bridge_factory=lambda: CabtBridge(classpath=args.classpath),
        out_dir=args.out, fail_fast=args.fail_fast, log=_stderr_log,
        illegal_action_mode=args.illegal_action_mode,
    )
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _stderr_log(message):
    sys.stderr.write(message + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
