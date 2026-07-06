"""Terminal human-vs-agent play.

    python -m magic_cabt.play.human_vs_agent \
        --human-seat 0 --agent1 heuristic \
        --deck0 examples/basic_deck.txt --deck1 examples/basic_deck.txt

A human takes one seat; the other seat is a configured agent. Each turn the
CLI prints the phase/step/player summary, lists the legal options with their
indices, and reads a selection (one index, or several for multi-select
prompts). Input is validated before it is sent to the engine, so a typo is
re-prompted rather than fed to XMage.

The game is written in the same self-play replay format the tournament runner
uses (via ``magic_cabt.eval.play.play_game``), so a human game annotates and
validates through ``magic_cabt.training.records`` exactly like an agent game.

Stays strictly local and terminal-based: no server, no web UI.
"""

import argparse
import os
import sys

from magic_cabt.agents import Agent, make_agent, options_of, select_block, \
    is_legal_selection
from magic_cabt.eval.play import _GameRecordWriter, play_game
from magic_cabt.protocol import CabtBridge, load_decklist

__all__ = [
    "parse_selection",
    "HumanAgent",
    "render_prompt",
    "build_parser",
    "main",
]


def parse_selection(text, select):
    """Parse a human's raw input into a legal option-index list.

    Accepts space- or comma-separated indices (``"0"``, ``"1 3"``, ``"1,3"``)
    and an empty string (a valid answer only when ``minCount == 0``). Raises
    ``ValueError`` with a human-readable message when the input is not a set of
    integers or is not a legal selection for ``select`` -- the caller re-prompts
    on that error instead of sending bad indices to the engine.
    """
    tokens = [token for token in text.replace(",", " ").split() if token]
    indices = []
    for token in tokens:
        try:
            value = int(token)
        except ValueError:
            raise ValueError("%r is not a number" % token)
        indices.append(value)
    if not is_legal_selection(indices, select):
        raise ValueError(_illegal_reason(indices, select))
    return indices


def _illegal_reason(indices, select):
    option_count = len(options_of(select))
    min_count = select.get("minCount") or 0
    max_count = select.get("maxCount") or 0
    for value in indices:
        if value < 0 or value >= option_count:
            return ("index %d is out of range (0..%d)"
                    % (value, option_count - 1))
    if len(indices) != len(set(indices)):
        return "indices must be distinct"
    if len(indices) < min_count:
        return "select at least %d option(s)" % min_count
    if max_count > 0 and len(indices) > max_count:
        return "select at most %d option(s)" % max_count
    return "not a legal selection"


def render_prompt(observation, seat_name=None):
    """Return a human-readable string for a decision: state + legal options."""
    select = select_block(observation)
    current = observation.get("current") if isinstance(observation, dict) else None
    lines = []
    if isinstance(current, dict):
        lines.append(
            "turn %s | %s / %s"
            % (current.get("turnNumber"), current.get("phase"),
               current.get("step")))
        for player in current.get("players") or []:
            lines.append(
                "  seat %s %s: %s life, %s cards, %s in play"
                % (player.get("playerIndex"), player.get("name"),
                   player.get("life"), player.get("handCount"),
                   current.get("battlefieldSize")))
    who = seat_name if seat_name is not None else select.get("playerIndex")
    min_count = select.get("minCount")
    max_count = select.get("maxCount")
    lines.append("decision %s for %s (choose %s..%s):"
                 % (select.get("type"), who, min_count, max_count))
    for option in options_of(select):
        label = option.get("label")
        lines.append("  [%d] %s%s"
                     % (option.get("index"), option.get("type"),
                        (" - " + label) if label else ""))
    return "\n".join(lines)


class HumanAgent(Agent):
    """Prompt a human for a selection; re-ask until the input is legal."""

    name = "human"

    def __init__(self, input_fn=None, output_fn=None, seat_name=None, name=None):
        Agent.__init__(self, name)
        self._input_fn = input_fn if input_fn is not None else _default_input
        self._output_fn = output_fn if output_fn is not None else print
        self._seat_name = seat_name

    def select(self, observation):
        select = select_block(observation)
        self._output_fn(render_prompt(observation, self._seat_name))
        while True:
            raw = self._input_fn("your choice (indices, blank to pass): ")
            try:
                return parse_selection(raw, select)
            except ValueError as error:
                self._output_fn("  invalid input: %s" % error)


def _default_input(prompt):
    return input(prompt)


# --- CLI --------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic_cabt.play.human_vs_agent",
        description="Play a local terminal game as a human against an agent.",
    )
    parser.add_argument("--human-seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--agent0", default="heuristic",
                        help="seat 0 agent when the human is seat 1")
    parser.add_argument("--agent1", default="heuristic",
                        help="seat 1 agent when the human is seat 0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--deck0", required=True)
    parser.add_argument("--deck1", required=True)
    parser.add_argument("--out", default=None,
                        help="write the replay JSONL to this file")
    parser.add_argument("--classpath", default=None,
                        help="Java classpath (default: $MAGIC_CABT_CLASSPATH)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    deck0 = load_decklist(args.deck0)
    deck1 = load_decklist(args.deck1)

    human = HumanAgent(seat_name="you (seat %d)" % args.human_seat)
    agents = [None, None]
    agents[args.human_seat] = human
    other_seat = 1 - args.human_seat
    other_spec = args.agent1 if other_seat == 1 else args.agent0
    agents[other_seat] = make_agent(other_spec, seed=args.seed)

    writer = None
    handle = None
    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        handle = open(args.out, "w")
        writer = _GameRecordWriter(handle, "human-game")

    try:
        with CabtBridge(classpath=args.classpath) as bridge:
            outcome = play_game(
                bridge, tuple(agents), deck0, deck1,
                seed=args.seed, max_turns=args.max_turns,
                record_writer=writer)
    finally:
        if handle is not None:
            handle.close()

    winner = outcome.get("winnerSeat")
    if winner is None:
        print("game over: draw")
    elif winner == args.human_seat:
        print("game over: you win!")
    else:
        print("game over: agent (seat %d) wins" % other_seat)
    if args.out:
        print("replay written to %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
