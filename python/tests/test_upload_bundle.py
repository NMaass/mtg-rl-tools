import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.upload.bundle import build_upload_envelope, validate_bundle_dir
from magic_cabt.upload.cli import main as upload_main
from magic_cabt.upload.redact import redact_json_value


OPTIONS = [
    {"index": 0, "type": "PASS_PRIORITY", "label": "Pass priority"},
    {"index": 1, "type": "CAST_SPELL", "label": "Cast spell"},
]


def decision_record():
    return {
        "schemaVersion": 1,
        "source": "arena_human",
        "gameId": "match-secret-id",
        "sequenceNumber": 0,
        "playerIndex": 0,
        "observation": {
            "current": {"turnNumber": 1},
            "select": {
                "type": "PRIORITY",
                "minCount": 1,
                "maxCount": 1,
                "option": OPTIONS,
            },
        },
        "select": {
            "type": "PRIORITY",
            "minCount": 1,
            "maxCount": 1,
            "option": OPTIONS,
        },
        "selectedIndices": [0],
        "terminal": False,
        "metadata": {
            "captureConfidence": "mirror",
            "playerName": "Alice",
            "opponentName": "Bob",
            "rawLogPath": "C:/Users/Alice/AppData/Player.log",
        },
    }


class UploadBundleTest(unittest.TestCase):

    def _bundle(self):
        tmp = tempfile.TemporaryDirectory()
        with open(os.path.join(tmp.name, "decisions.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(decision_record()) + "\n")
        with open(os.path.join(tmp.name, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump({"playerName": "Alice", "accountId": "secret"}, handle)
        return tmp

    def test_validate_bundle_dir_requires_decisions_and_summary(self):
        with self._bundle() as directory:
            files = validate_bundle_dir(directory)
            self.assertIn("decisions.jsonl", files)
            self.assertIn("summary.json", files)

    def test_redaction_removes_sensitive_keys_and_pseudonymizes_names(self):
        redacted = redact_json_value({
            "accountId": "secret",
            "playerName": "Alice",
            "nested": {"opponentName": "Bob"},
        })
        self.assertEqual("<redacted>", redacted["accountId"])
        self.assertNotEqual("Alice", redacted["playerName"])
        self.assertTrue(redacted["playerName"].startswith("id_"))
        self.assertNotEqual("Bob", redacted["nested"]["opponentName"])

    def test_build_upload_envelope_redacts_and_builds_manifest(self):
        with self._bundle() as directory:
            envelope = build_upload_envelope(directory, contributor_id="tester")
        self.assertEqual("magic_cabt_upload_bundle", envelope["kind"])
        self.assertEqual("tester", envelope["contributorId"])
        self.assertEqual(1, envelope["source"]["recordCount"])
        self.assertFalse(envelope["redactionReport"]["rawPlayerLogUploaded"])
        self.assertIn("manifest", envelope)
        serialized = json.dumps(envelope)
        self.assertNotIn("Alice", serialized)
        self.assertNotIn("C:/Users", serialized)

    def test_upload_cli_dry_run_requires_consent_and_prints_summary(self):
        with self._bundle() as directory:
            stdout = sys.stdout
            stderr = sys.stderr
            try:
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
                self.assertEqual(0, upload_main([
                    "--bundle", directory,
                    "--dry-run",
                    "--yes",
                ]))
                text = sys.stdout.getvalue()
            finally:
                sys.stdout = stdout
                sys.stderr = stderr
        self.assertIn("recordCount", text)
        self.assertIn("training-v1", text)


if __name__ == "__main__":
    unittest.main()
