import io
import json
import os
import tempfile
import unittest
from unittest import mock

import magic_cabt.training.build_manifest as manifest_cli
from magic_cabt.training.manifest import build_manifest, write_manifest


def option(index, kind, label, canonical=None, card=None):
    payload = {}
    if canonical is not None:
        payload["canonicalKey"] = canonical
    if card is not None:
        payload["card"] = card
    return {"index": index, "type": kind, "label": label, "payload": payload}


def record(seq, prompt="PRIORITY", options=None, selected=None,
           match="m1", game=1, instance="g1", next_state=True,
           terminal=False, reward=None, result=None, deck0="d0", deck1="d1"):
    options = options if options is not None else [
        option(0, "PASS", "Pass", "pass"),
        option(1, "CAST", "Cast Bear", "cast:bear", {
            "grpId": 101, "name": "Bear", "manaCost": "1G"}),
    ]
    return {
        "schemaVersion": 1,
        "source": "engine_selfplay",
        "gameId": match,
        "gameNumber": game,
        "sequenceNumber": seq,
        "playerIndex": 0,
        "observation": {
            "current": {
                "gameInstance": instance,
                "players": [{"seat": 0, "life": 20}],
                "zones": {"battlefield": [{
                    "name": "Forest", "zone": "battlefield",
                    "setCode": "TST", "cardNumber": "1"}]},
            }
        },
        "select": {"type": prompt, "minCount": 1, "maxCount": 1,
                   "option": options},
        "selectedIndices": selected if selected is not None else [0],
        "nextObservation": {"current": {"gameInstance": instance}}
            if next_state else None,
        "terminal": terminal,
        "reward": reward,
        "result": result,
        "metadata": {"captureConfidence": "exact",
                     "deck0Id": deck0, "deck1Id": deck1},
    }


class ManifestInventoryTest(unittest.TestCase):
    def test_inventory_counts_games_decisions_transitions_cards_and_decks(self):
        rows = [
            record(1),
            record(2, prompt="TARGET_SELECT", options=[
                option(0, "TARGET", "Bear A", "bear", {"grpId": 101}),
                option(1, "TARGET", "Bear B", "bear", {"grpId": 101}),
            ], next_state=False),
            record(1, match="m2", game=1, instance="g2", options=[
                option(0, "PASS", "Pass", "pass")
            ], terminal=True, next_state=False, reward=1.0,
                   result={"winner": 0}, deck0="d2", deck1="d3"),
        ]
        manifest = build_manifest(rows, name="fixture")
        inventory = manifest["inventory"]
        self.assertEqual(2, inventory["games"]["known"])
        self.assertEqual(2, inventory["decisions"]["roles"]["root"])
        self.assertEqual(1, inventory["decisions"]["roles"]["parameter"])
        self.assertEqual(1, inventory["decisions"]["strategicRootRecords"])
        self.assertEqual(1, inventory["decisions"]["deterministicRecords"])
        # Duplicate target options share one canonical semantic group.
        self.assertEqual(2, inventory["decisions"]["singleSemanticOptionRecords"])
        self.assertEqual(1, inventory["decisions"]["nontrivialSemanticOptionRecords"])
        self.assertEqual(1, inventory["decisions"]["trainableSingleChoiceRecords"])
        self.assertEqual(2, inventory["transitions"]["nonTerminalRecords"])
        self.assertEqual(1, inventory["transitions"]["nonTerminalWithNextObservation"])
        self.assertEqual(0.5, inventory["transitions"]["nonTerminalNextObservationRate"])
        self.assertEqual(1, inventory["transitions"]["terminalOutcomeRecords"])
        self.assertGreaterEqual(inventory["cards"]["uniqueObservedIdentifiers"], 2)
        self.assertEqual(3, inventory["cards"]["recordsWithObservedIdentifiers"])
        self.assertEqual(4, inventory["decks"]["uniqueIdentifiers"])
        self.assertEqual(3, inventory["decks"]["recordsWithIdentifiers"])
        self.assertNotIn("_inventorySets", manifest)

    def test_duplicate_public_fingerprints_are_reported(self):
        first = record(1)
        duplicate = record(2)
        manifest = build_manifest([first, duplicate])
        duplicates = manifest["inventory"]["duplicates"]
        self.assertEqual(1, duplicates["uniquePublicFingerprintsTracked"])
        self.assertEqual(1, duplicates["duplicateRecordsObserved"])
        self.assertFalse(duplicates["trackingTruncated"])

    def test_unknown_game_and_missing_options_are_visible(self):
        value = record(1)
        value.pop("gameId")
        value.pop("gameNumber")
        value["observation"]["current"].pop("gameInstance")
        value["select"]["option"] = []
        manifest = build_manifest([value])
        self.assertEqual(1, manifest["inventory"]["games"]["unknownRecords"])
        self.assertEqual(1, manifest["inventory"]["decisions"]["noOptionRecords"])
        self.assertIsNone(
            manifest["inventory"]["decisions"]["semanticOptionCount"]["mean"])

    def test_manifest_is_json_serializable_and_writable(self):
        manifest = build_manifest([record(1)])
        json.dumps(manifest)
        with tempfile.TemporaryDirectory() as scratch:
            path = os.path.join(scratch, "manifest.json")
            write_manifest(path, manifest)
            with open(path, encoding="utf-8") as handle:
                loaded = json.load(handle)
        self.assertEqual(manifest, loaded)


class BuildManifestCliTest(unittest.TestCase):
    def test_repeated_inputs_and_bundle_directories_are_combined(self):
        with tempfile.TemporaryDirectory() as scratch:
            first = os.path.join(scratch, "one.jsonl")
            bundle = os.path.join(scratch, "bundle")
            os.makedirs(bundle)
            second = os.path.join(bundle, "decisions.jsonl")
            for path, sequence in ((first, 1), (second, 2)):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps({"sequenceNumber": sequence}) + "\n")
            output = os.path.join(scratch, "manifest.json")
            records = {
                first: [{"source": "a", "select": {"option": []}}],
                second: [{"source": "b", "select": {"option": []}}],
            }
            with mock.patch.object(
                    manifest_cli, "iter_decision_records",
                    side_effect=lambda path, source_hint=None: iter(records[path])):
                self.assertEqual(0, manifest_cli.main([
                    "--input", first, "--input", bundle,
                    "--out", output, "--name", "corpus"]))
            with open(output, encoding="utf-8") as handle:
                manifest = json.load(handle)
        self.assertEqual("corpus", manifest["name"])
        self.assertEqual(2, manifest["records"])
        self.assertEqual(2, len(manifest["inputs"]))
        self.assertTrue(all(len(item["sha256"]) == 64
                            for item in manifest["inputs"]))

    def test_missing_input_returns_two(self):
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            self.assertEqual(2, manifest_cli.main([
                "--input", "/missing/data.jsonl"]))
        self.assertIn("dataset not found", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
