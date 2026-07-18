"""Copy toolkit modules into a local repo without editing existing source files."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


MODULES = (
    "mtg_state_contract",
    "mtgo_video_acquisition",
    "mtgo_video_parser",
    "mtgo_native_logs",
    "xmage_state_follower",
    "mtgo_pipeline_coordinator",
)


def assemble(toolkit_root: str, repository: str,
             destination: str = "tools/mtgo_video_toolkit",
             replace: bool = False):
    source = Path(toolkit_root).resolve()
    repo = Path(repository).resolve()
    target = repo / destination
    if target.exists():
        if not replace:
            raise FileExistsError(
                f"destination exists: {target}; pass --replace to overwrite")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(
        ".venv", "__pycache__", ".pytest_cache", "*.pyc", "dist", "build"))
    record = {
        "toolkitSource": str(source),
        "repository": str(repo),
        "destination": str(target.relative_to(repo)).replace("\\", "/"),
        "modules": list(MODULES),
        "integrationMode": "vendored-no-existing-files-edited",
    }
    with (target / "ASSEMBLY_RECORD.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return record


def build_parser():
    parser = argparse.ArgumentParser(prog="mtgo-toolkit-assemble")
    parser.add_argument("--toolkit-root", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--destination", default="tools/mtgo_video_toolkit")
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = assemble(args.toolkit_root, args.repo, args.destination, args.replace)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
