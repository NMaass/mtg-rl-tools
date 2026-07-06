"""Self-play smoke run: two agents play a real XMage game.

    python examples/run_selfplay.py [--seed N] [--max-turns N] \
        [--deck0 FILE] [--deck1 FILE] [--out DIR] \
        [--agent0 random|first|heuristic] [--agent1 ...]

Thin wrapper over ``magic_cabt.eval.play``: it delegates game orchestration to
``run_tournament`` (one game) rather than duplicating the decision loop, and
writes the same per-game replay JSONL plus a ``summary.json``.

Launches the Java bridge subprocess (set ``$MAGIC_CABT_CLASSPATH`` or pass
--classpath; scripts/run-cabt-adapter-tests.sh writes a ready classpath to
Mage.Server.Plugins/Mage.Player.AI/target/cabt-classpath.full.txt).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "python"))

from magic_cabt import CabtBridge, load_decklist  # noqa: E402
from magic_cabt.eval.play import run_tournament  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--classpath", default=None,
                        help="Java classpath (default: $MAGIC_CABT_CLASSPATH)")
    parser.add_argument("--deck0", default=os.path.join(here, "basic_deck.txt"))
    parser.add_argument("--deck1", default=os.path.join(here, "basic_deck.txt"))
    parser.add_argument("--agent0", default="random")
    parser.add_argument("--agent1", default="random")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--out", default=os.path.join(here, os.pardir, "target",
                                                      "cabt-selfplay"))
    args = parser.parse_args()

    summary = run_tournament(
        (args.agent0, args.agent1),
        load_decklist(args.deck0), load_decklist(args.deck1),
        games=1, seed=args.seed, max_turns=args.max_turns,
        bridge_factory=lambda: CabtBridge(classpath=args.classpath),
        out_dir=args.out, log=lambda message: print(message),
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("replay written to %s" % os.path.join(args.out, "game-0000.jsonl"))


if __name__ == "__main__":
    main()
