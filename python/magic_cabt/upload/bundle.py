"""Build and validate opt-in training upload envelopes.

The upload path is intentionally envelope-based instead of "send raw log": the
client validates a local Arena/XMage run directory, redacts records, attaches a
manifest and consent block, and sends a bounded JSON payload to the ingest API.
"""

import json
import os
import time

from magic_cabt.training.io import iter_decision_records
from magic_cabt.training.manifest import build_manifest

from .redact import redact_decision_record, redact_json_value

__all__ = [
    "REQUIRED_FILES",
    "MAX_DEFAULT_RECORDS",
    "validate_bundle_dir",
    "build_upload_envelope",
]

REQUIRED_FILES = ("decisions.jsonl", "summary.json")
OPTIONAL_FILES = ("manifest.json", "card_cache.json")
MAX_DEFAULT_RECORDS = 20000
CONSENT_VERSION = "training-v1"


class BundleValidationError(ValueError):
    pass


def validate_bundle_dir(bundle_dir):
    """Return a file map for a valid upload bundle directory."""
    if not os.path.isdir(bundle_dir):
        raise BundleValidationError("bundle directory does not exist: %s" % bundle_dir)
    files = {}
    for name in REQUIRED_FILES:
        path = os.path.join(bundle_dir, name)
        if not os.path.isfile(path):
            raise BundleValidationError("bundle is missing required file: %s" % name)
        files[name] = path
    for name in OPTIONAL_FILES:
        path = os.path.join(bundle_dir, name)
        if os.path.isfile(path):
            files[name] = path
    return files


def build_upload_envelope(bundle_dir, contributor_id=None, consent=True,
                          max_records=MAX_DEFAULT_RECORDS):
    """Build a redacted JSON upload envelope from a local bundle directory."""
    if not consent:
        raise BundleValidationError("upload requires explicit consent")
    files = validate_bundle_dir(bundle_dir)
    decisions = []
    for idx, record in enumerate(iter_decision_records(files["decisions.jsonl"])):
        if idx >= max_records:
            break
        decisions.append(redact_decision_record(record))
    if not decisions:
        raise BundleValidationError("bundle has no decision records")
    manifest = _load_json(files.get("manifest.json")) or build_manifest(decisions)
    summary = redact_json_value(_load_json(files["summary.json"]) or {})
    envelope = {
        "schemaVersion": 1,
        "kind": "magic_cabt_upload_bundle",
        "createdAtUnix": int(time.time()),
        "contributorId": contributor_id or "anonymous",
        "consent": {
            "consentVersion": CONSENT_VERSION,
            "allowTraining": True,
            "allowPublicAggregateStats": True,
            "allowRawResearchAccess": False,
        },
        "source": {
            "bundleName": os.path.basename(os.path.abspath(bundle_dir)),
            "recordCount": len(decisions),
            "truncated": len(decisions) >= max_records,
        },
        "manifest": redact_json_value(manifest),
        "summary": summary,
        "decisions": decisions,
        "redactionReport": {
            "mode": "default",
            "rawPlayerLogUploaded": False,
            "containsDecisionRecords": True,
            "maxRecords": max_records,
        },
    }
    return envelope


def _load_json(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
