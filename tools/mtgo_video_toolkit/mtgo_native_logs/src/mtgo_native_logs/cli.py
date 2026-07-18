"""CLI for native MTGO logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .discovery import discover_logs
from .parser import NativeLogParser


def build_parser():
    parser = argparse.ArgumentParser(prog="mtgo-native-log")
    sub = parser.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("discover")
    discover.add_argument("--root", action="append", default=None)
    discover.add_argument("--out", required=True)
    discover.add_argument("--maximum-files", type=int, default=10000)

    parse = sub.add_parser("parse")
    parse.add_argument("--input", required=True)
    parse.add_argument("--out", required=True)
    parse.add_argument("--keep-player-names", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "discover":
        result = {"schemaVersion": 1,
                  "files": discover_logs(args.root, args.maximum_files)}
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
        print(json.dumps({"files": len(result["files"]), "out": str(target)},
                         indent=2))
        return 0
    if args.command == "parse":
        result = NativeLogParser(
            pseudonymize_players=not args.keep_player_names).parse_file(args.input)
        target = Path(args.out)
        target.mkdir(parents=True, exist_ok=True)
        _write_jsonl(target / "observed_actions.jsonl",
                     [row.to_dict() for row in result.actions])
        _write_jsonl(target / "canonical_events.jsonl",
                     [row.to_dict() for row in result.events])
        (target / "native_log_manifest.json").write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps({"actions": len(result.actions),
                          "events": len(result.events),
                          "out": str(target)}, indent=2))
        return 0
    return 2


def _write_jsonl(path: Path, values):
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
