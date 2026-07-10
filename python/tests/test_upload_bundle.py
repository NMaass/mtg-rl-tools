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


def summary_fixture():
    return {
        "playerName": "Alice",
        "accountId": "secret",
        "matchIds": ["match-secret-1", "match-secret-2"],
        "eventName": "Historic",
        "title": "Alice vs Bob — Win (2-1) — Historic",
        "deckName": "Alice Secret Brew",
        "you": {"seat": 1, "name": "Alice", "colors": "WU",
                "archetype": "Azorius"},
        "opponent": {"seat": 2, "name": "Bob", "colors": "BR",
                     "archetype": "Rakdos"},
    }


class UploadBundleTest(unittest.TestCase):

    def _bundle(self):
        tmp = tempfile.TemporaryDirectory()
        with open(os.path.join(tmp.name, "decisions.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(decision_record()) + "\n")
        with open(os.path.join(tmp.name, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary_fixture(), handle)
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

    def test_envelope_pseudonymizes_game_and_match_ids(self):
        with self._bundle() as directory:
            envelope = build_upload_envelope(directory, contributor_id="tester")
        serialized = json.dumps(envelope)
        self.assertNotIn("match-secret-id", serialized)
        self.assertNotIn("match-secret-1", serialized)
        self.assertNotIn("match-secret-2", serialized)
        game_id = envelope["decisions"][0]["gameId"]
        self.assertTrue(game_id.startswith("id_"))
        match_ids = envelope["summary"]["matchIds"]
        self.assertEqual(2, len(match_ids))
        for match_id in match_ids:
            self.assertTrue(match_id.startswith("id_"))
        # Distinct raw ids stay distinct after pseudonymization.
        self.assertNotEqual(match_ids[0], match_ids[1])

    def test_envelope_redacts_you_name_and_title(self):
        with self._bundle() as directory:
            envelope = build_upload_envelope(directory, contributor_id="tester")
        serialized = json.dumps(envelope)
        self.assertNotIn("Alice", serialized)
        self.assertNotIn("Bob", serialized)
        self.assertNotIn("Alice vs Bob", serialized)
        summary = envelope["summary"]
        self.assertEqual("<redacted>", summary["title"])
        self.assertEqual("<redacted>", summary["deckName"])
        self.assertTrue(summary["you"]["name"].startswith("id_"))
        # Non-identifying event metadata is preserved.
        self.assertEqual("Historic", summary["eventName"])

    def test_opponent_object_keeps_structure_but_pseudonymizes_name(self):
        with self._bundle() as directory:
            envelope = build_upload_envelope(directory, contributor_id="tester")
        opponent = envelope["summary"]["opponent"]
        self.assertIsInstance(opponent, dict)
        self.assertEqual("BR", opponent["colors"])
        self.assertEqual("Rakdos", opponent["archetype"])
        self.assertEqual(2, opponent["seat"])
        self.assertNotEqual("Bob", opponent["name"])
        self.assertTrue(opponent["name"].startswith("id_"))

    def test_yes_alone_grants_training_consent_only(self):
        with self._bundle() as directory:
            envelope = build_upload_envelope(directory, contributor_id="tester")
        self.assertTrue(envelope["consent"]["allowTraining"])
        self.assertFalse(envelope["consent"]["allowPublicAggregateStats"])
        self.assertFalse(envelope["consent"]["allowRawResearchAccess"])
        with self._bundle() as directory:
            opted_in = build_upload_envelope(directory, contributor_id="tester",
                                             allow_aggregate_stats=True)
        self.assertTrue(opted_in["consent"]["allowPublicAggregateStats"])

    def test_upload_cli_yes_alone_disables_aggregate_stats(self):
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
                summary = json.loads(sys.stdout.getvalue())
            finally:
                sys.stdout = stdout
                sys.stderr = stderr
        self.assertTrue(summary["consent"]["allowTraining"])
        self.assertFalse(summary["consent"]["allowPublicAggregateStats"])

    def test_upload_cli_aggregate_stats_flag_opts_in(self):
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
                    "--allow-aggregate-stats",
                ]))
                summary = json.loads(sys.stdout.getvalue())
            finally:
                sys.stdout = stdout
                sys.stderr = stderr
        self.assertTrue(summary["consent"]["allowPublicAggregateStats"])

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
