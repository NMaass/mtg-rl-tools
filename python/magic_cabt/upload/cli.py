"""CLI for building, previewing, and uploading redacted training bundles."""

import argparse
import json
import sys

from .bundle import BundleValidationError, build_upload_envelope
from .client import post_envelope

__all__ = ["main"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="magic-cabt-upload",
        description="Build a redacted, opt-in training upload envelope from a "
                    "local Arena/XMage recording bundle.",
    )
    parser.add_argument("--bundle", required=True,
                        help="recording bundle directory")
    parser.add_argument("--endpoint", default=None,
                        help="ingest endpoint; omit with --dry-run to preview only")
    parser.add_argument("--api-key", default=None,
                        help="optional bearer token for the ingest endpoint")
    parser.add_argument("--contributor-id", default=None)
    parser.add_argument("--max-records", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true",
                        help="print envelope summary without uploading")
    parser.add_argument("--print-envelope", action="store_true",
                        help="print the full redacted envelope JSON")
    parser.add_argument("--yes", action="store_true",
                        help="confirm opt-in training consent (allowTraining only)")
    parser.add_argument("--allow-aggregate-stats", action="store_true",
                        help="additionally consent to public aggregate stats "
                             "(allowPublicAggregateStats; off by default)")
    args = parser.parse_args(argv)

    if not args.yes:
        sys.stderr.write("upload requires --yes to confirm opt-in training consent\n")
        return 2
    try:
        envelope = build_upload_envelope(
            args.bundle,
            contributor_id=args.contributor_id,
            consent=True,
            allow_aggregate_stats=args.allow_aggregate_stats,
            max_records=args.max_records,
        )
    except BundleValidationError as exc:
        sys.stderr.write("bundle invalid: %s\n" % exc)
        return 2

    if args.print_envelope:
        json.dump(envelope, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(json.dumps(_summary(envelope), indent=2, sort_keys=True) + "\n")

    if args.dry_run:
        return 0
    if not args.endpoint:
        sys.stderr.write("--endpoint is required unless --dry-run is set\n")
        return 2
    response = post_envelope(args.endpoint, envelope, api_key=args.api_key)
    sys.stdout.write(json.dumps(response, indent=2, sort_keys=True) + "\n")
    return 0


def _summary(envelope):
    return {
        "kind": envelope.get("kind"),
        "contributorId": envelope.get("contributorId"),
        "recordCount": (envelope.get("source") or {}).get("recordCount"),
        "truncated": (envelope.get("source") or {}).get("truncated"),
        "consent": envelope.get("consent"),
        "redactionReport": envelope.get("redactionReport"),
    }


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
