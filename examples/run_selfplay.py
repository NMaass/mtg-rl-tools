"""Self-play smoke run: two random legal agents play a real XMage game.

    python examples/run_selfplay.py [--seed N] [--max-turns N] \
        [--deck0 FILE] [--deck1 FILE] [--out DIR]

Launches the Java bridge subprocess (set $MAGIC_CABT_CLASSPATH or pass
--classpath; scripts/run-cabt-adapter-tests.sh writes a ready classpath to
Mage.Server.Plugins/Mage.Player.AI/target/cabt-classpath.full.txt), starts a
game, routes every decision to the acting seat's agent, and writes a replay
(observations + selections + result) as JSONL plus a final board render.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "python"))

from magic_cabt import CabtBridge, load_decklist  # noqa: E402

from random_agent import agent as random_agent  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--classpath", default=None,
                        help="Java classpath (default: $MAGIC_CABT_CLASSPATH)")
    parser.add_argument("--deck0", default=os.path.join(here, "basic_deck.txt"))
    parser.add_argument("--deck1", default=os.path.join(here, "basic_deck.txt"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--out", default=os.path.join(here, os.pardir, "target",
                                                      "cabt-selfplay"))
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    agents = [random_agent, random_agent]
    replay_path = os.path.join(args.out, "replay.jsonl")
    os.makedirs(args.out, exist_ok=True)

    with CabtBridge(classpath=args.classpath) as bridge, \
            open(replay_path, "w") as replay:
        response = bridge.game_start(
            load_decklist(args.deck0), load_decklist(args.deck1),
            player_names=["Agent0", "Agent1"],
            seed=args.seed, max_turns=args.max_turns,
        )
        steps = 0
        while not bridge.finished:
            observation = response["observation"]
            seat = observation["select"]["playerIndex"]
            selection = agents[seat](observation)
            replay.write(json.dumps({
                "sequence": response["sequence"],
                "player": response["player"],
                "observation": observation,
                "selected": selection,
            }) + "\n")
            response = bridge.game_select(selection)
            steps += 1

        replay.write(json.dumps({"result": bridge.result}) + "\n")
        board = None
        try:
            board = bridge.visualize_data()
        except Exception:
            pass  # no active game once it is over on some paths

    print("game over after %d decisions" % steps)
    print("winner: %s" % (bridge.result or {}).get("winner"))
    if board:
        print(board)
    print("replay written to %s" % replay_path)


if __name__ == "__main__":
    main()
