"""CLI: python -m magic_cabt.heuristic compare|review|refine."""
import argparse
import json
import os
from pathlib import Path
from .experiments import load_pool, run_suite
from .jev import Budget


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--pool", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--agents", nargs="+", default=["heuristic", "jev", "first", "random"])
    compare.add_argument("--seeds", nargs="+", type=int, default=[11])
    compare.add_argument("--max-turns", type=int, default=80)
    compare.add_argument("--max-decisions", type=int, default=5000)
    compare.add_argument("--wall-seconds", type=float, default=90)
    compare.add_argument("--max-games", type=int)
    compare.add_argument("--jev-max-calls", type=int, default=20)
    compare.add_argument("--jev-cost-stop", type=float, default=1.0)
    review = commands.add_parser("review")
    review.add_argument("--run", required=True)
    review.add_argument("--out", required=True)
    refine = commands.add_parser("refine")
    refine.add_argument("--pool", required=True)
    refine.add_argument("--suite", required=True)
    refine.add_argument("--out", required=True)
    refine.add_argument("--candidates", type=int, default=2)
    refine.add_argument("--tuning-seeds", nargs="+", type=int, default=[101])
    refine.add_argument("--holdout-seeds", nargs="+", type=int, default=[201, 202])
    args = parser.parse_args(argv)
    if args.command == "compare":
        if args.max_games is not None and args.max_games < 1:
            parser.error("--max-games must be positive")
        summary, _ = run_suite(load_pool(args.pool), args.out, agents=args.agents, seeds=args.seeds,
                               api_key=os.environ.get("OPENROUTER_API_KEY"),
                               budget=Budget(args.jev_max_calls, args.jev_cost_stop),
                               max_turns=args.max_turns, max_decisions=args.max_decisions,
                               wall_seconds=args.wall_seconds, max_games=args.max_games,
                               progress=lambda game: print(json.dumps({k: game[k] for k in
                                   ("id", "agents", "decks", "status", "winnerSeat", "decisions", "elapsedSeconds")}), flush=True))
        print(json.dumps(summary, indent=2))
        return 0 if not summary["statuses"].get("error") and not summary["statuses"].get("timeout") else 1
    if args.command == "review":
        from .review import render
        render(Path(args.run), Path(args.out))
        return 0
    from .refinement import run_refinement
    result = run_refinement(load_pool(args.pool), args.suite, args.out, args.candidates,
                            args.tuning_seeds, args.holdout_seeds)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
