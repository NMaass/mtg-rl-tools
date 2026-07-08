import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.training.build_manifest import main as build_manifest_main
from magic_cabt.training.causal import causal_variables, factor_credit_trace
from magic_cabt.training.manifest import build_manifest


OPTIONS = [
    {"index": 0, "type": "PASS_PRIORITY", "label": "Pass priority"},
    {"index": 1, "type": "CAST_SPELL", "label": "Cast spell"},
]


def record(next_current=None):
    current = {
        "turnNumber": 4,
        "activePlayerId": "p0",
        "priorityPlayerId": "p0",
        "players": [
            {"playerIndex": 0, "playerId": "p0", "life": 18,
             "handCount": 4, "libraryCount": 31, "graveyardCount": 2},
            {"playerIndex": 1, "playerId": "p1", "life": 12,
             "handCount": 2, "libraryCount": 29, "graveyardCount": 4},
        ],
        "battlefield": [
            {"controllerId": "p0", "cardTypes": ["LAND"]},
            {"controllerId": "p0", "cardTypes": ["CREATURE"]},
            {"controllerId": "p1", "cardTypes": ["CREATURE"]},
        ],
        "stack": [],
    }
    row = {
        "schemaVersion": 1,
        "source": "engine_selfplay",
        "gameId": "g1",
        "sequenceNumber": 3,
        "playerIndex": 0,
        "observation": {
            "current": current,
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
        "selectedIndices": [1],
        "nextObservation": None,
        "terminal": False,
        "reward": None,
        "result": None,
        "metadata": {"captureConfidence": "exact"},
    }
    if next_current is not None:
        row["nextObservation"] = {"current": next_current}
    return row


class CausalFeatureTest(unittest.TestCase):

    def test_causal_variables_extract_public_strategic_factors(self):
        values = causal_variables(record())
        self.assertEqual(18, values["life_total"])
        self.assertEqual(6, values["life_diff"])
        self.assertEqual(4, values["hand_count"])
        self.assertEqual(2, values["hand_diff"])
        self.assertEqual(2, values["battlefield_diff"])
        self.assertEqual(1, values["creature_count"])
        self.assertEqual(0, values["creature_diff"])
        self.assertEqual(1, values["land_count"])
        self.assertEqual(1, values["is_active_player"])
        self.assertEqual(1, values["has_priority"])

    def test_factor_credit_trace_includes_delta_when_next_observation_exists(self):
        next_current = record()["observation"]["current"].copy()
        next_current["players"] = [p.copy() for p in next_current["players"]]
        next_current["players"][1]["life"] = 9
        trace = factor_credit_trace(record(next_current=next_current))
        self.assertEqual("PRIORITY", trace["promptType"])
        self.assertIsNotNone(trace["after"])
        self.assertEqual(3, trace["delta"]["life_diff"])

    def test_manifest_summarizes_dataset_and_factor_ranges(self):
        manifest = build_manifest([record()], name="fixture")
        self.assertEqual("fixture", manifest["name"])
        self.assertEqual(1, manifest["records"])
        self.assertEqual(1, manifest["validRecords"])
        self.assertEqual({"engine_selfplay": 1}, manifest["sources"])
        self.assertEqual({"PRIORITY": 1}, manifest["promptTypes"])
        self.assertEqual(2, manifest["optionCount"]["max"])
        self.assertEqual(18, manifest["causalFactors"]["life_total"]["min"])

    def test_manifest_cli_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "records.jsonl")
            out = os.path.join(tmp, "manifest.json")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record()) + "\n")
            self.assertEqual(0, build_manifest_main([
                "--input", source, "--out", out, "--name", "fixture"])
            )
            with open(out, encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual("fixture", manifest["name"])
            self.assertEqual(1, manifest["records"])


if __name__ == "__main__":
    unittest.main()
