"""CLI for local video-source manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import SourceEntry, SourceManifest
from .ytdlp import YtDlpClient

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


def build_parser():
    parser = argparse.ArgumentParser(prog="mtgo-video-source")
    sub = parser.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("discover")
    discover.add_argument("--url", required=True)
    discover.add_argument("--out", required=True)
    discover.add_argument("--yt-dlp", default=None)

    download = sub.add_parser("download")
    download.add_argument("--manifest", required=True)
    download.add_argument("--out", required=True)
    download.add_argument("--yt-dlp", default=None)
    download.add_argument("--maximum-height", type=int, default=1080)
    download.add_argument("--acknowledge-rights-and-terms", action="store_true")

    local = sub.add_parser("import-local")
    local.add_argument("--directory", required=True)
    local.add_argument("--out", required=True)
    local.add_argument("--permission-note", default="local/user-authorized")

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--yt-dlp", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        client = YtDlpClient(args.yt_dlp)
        result = {"ytDlp": client.executable, "available": client.available()}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["available"] else 1
    if args.command == "discover":
        manifest = YtDlpClient(args.yt_dlp).discover(args.url)
        manifest.save(args.out)
        print(json.dumps({"entries": len(manifest.entries), "out": args.out}, indent=2))
        return 0
    if args.command == "import-local":
        root = Path(args.directory).expanduser().resolve()
        entries = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                row = SourceEntry(title=path.stem,
                                  permission_note=args.permission_note)
                row.attach_local_file(str(path))
                entries.append(row)
        manifest = SourceManifest(entries, metadata={"localRoot": str(root)})
        manifest.save(args.out)
        print(json.dumps({"entries": len(entries), "out": args.out}, indent=2))
        return 0
    if args.command == "download":
        manifest = SourceManifest.load(args.manifest)
        client = YtDlpClient(args.yt_dlp)
        completed, failed = 0, []
        for index, entry in enumerate(manifest.entries):
            try:
                client.download(
                    entry, args.out,
                    rights_acknowledged=args.acknowledge_rights_and_terms,
                    maximum_height=args.maximum_height)
                completed += 1
            except Exception as error:
                failed.append({"index": index, "url": entry.webpage_url or
                               entry.source_url,
                               "error": f"{type(error).__name__}: {error}"})
        manifest.metadata["downloadFailures"] = failed
        manifest.save(args.manifest)
        print(json.dumps({"completed": completed, "failed": failed}, indent=2))
        return 0 if not failed else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
