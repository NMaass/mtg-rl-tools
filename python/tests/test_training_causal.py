import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.training.build_manifest import main as build_manifest_main
from magic_cabt.training.causal import causal_variables, factor_credit_trace
from magic_cabt.training.io import iter_decision_records
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
        self.assertEqual(1, values["battlefield_diff"])
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


def arena_record():
    """Arena-mirror snapshot shape: ``seat``/``zones``/``controllerSeat`` field names."""
    current = {
        "turnNumber": 4,
        "activeSeat": 0,
        "prioritySeat": 0,
        "players": [
            {"seat": 0, "name": "P0", "life": 18, "handCount": 4, "libraryCount": 31},
            {"seat": 1, "name": "P1", "life": 12, "handCount": 2, "libraryCount": 29},
        ],
        "zones": {
            "battlefield": [
                {"controllerSeat": 0, "cardTypes": ["LAND"]},
                {"controllerSeat": 0, "cardTypes": ["CREATURE"]},
                {"controllerSeat": 1, "cardTypes": ["CREATURE"]},
            ],
            "stack": [],
            "graveyards": {"0": [{"instanceId": 1}, {"instanceId": 2}], "1": [{"instanceId": 3}]},
        },
    }
    return {
        "schemaVersion": 1,
        "source": "arena_human",
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
        "metadata": {"captureConfidence": "mirror"},
    }


class ArenaCausalFeatureTest(unittest.TestCase):

    def test_arena_snapshot_extracts_factors_via_field_fallbacks(self):
        values = causal_variables(arena_record())
        self.assertEqual(18, values["life_total"])
        self.assertEqual(6, values["life_diff"])
        self.assertEqual(4, values["hand_count"])
        self.assertEqual(2, values["hand_diff"])
        self.assertEqual(2, values["battlefield_count"])
        self.assertEqual(1, values["battlefield_diff"])
        self.assertEqual(1, values["creature_count"])
        self.assertEqual(0, values["creature_diff"])
        self.assertEqual(1, values["land_count"])
        self.assertEqual(1, values["land_diff"])
        self.assertEqual(0, values["stack_count"])
        self.assertEqual(2, values["graveyard_count"])
        self.assertEqual(1, values["graveyard_diff"])
        self.assertEqual(4, values["turn_number"])
        self.assertEqual(1, values["is_active_player"])
        self.assertEqual(1, values["has_priority"])

    def test_arena_manifest_summarizes_factors(self):
        manifest = build_manifest([arena_record()], name="arena-fixture")
        self.assertEqual("arena-fixture", manifest["name"])
        self.assertEqual(1, manifest["records"])
        self.assertEqual(1, manifest["validRecords"])
        self.assertEqual({"arena_human": 1}, manifest["sources"])
        self.assertEqual(18, manifest["causalFactors"]["life_total"]["min"])
        self.assertEqual(2, manifest["causalFactors"]["battlefield_count"]["max"])
        self.assertEqual(2, manifest["causalFactors"]["graveyard_count"]["max"])


class ManifestTerminalCountTest(unittest.TestCase):

    def test_self_play_manifest_counts_terminal_record(self):
        """Streaming manifest builder must see terminal=True on the last record."""
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "selfplay.jsonl")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "sequence": 1, "player": 0,
                    "observation": {
                        "current": {
                            "turnNumber": 1,
                            "players": [
                                {"playerIndex": 0, "playerId": "p0", "life": 20},
                                {"playerIndex": 1, "playerId": "p1", "life": 20},
                            ],
                        },
                        "select": {
                            "type": "PRIORITY", "minCount": 1, "maxCount": 1,
                            "playerIndex": 0,
                            "option": [
                                {"index": 0, "type": "PASS_PRIORITY"},
                            ],
                        },
                    },
                    "selected": [0],
                }) + "\n")
                handle.write(json.dumps({"result": {"winner": 0}}) + "\n")
            records = list(iter_decision_records(source))
            self.assertTrue(records[0]["terminal"])
            manifest = build_manifest(iter_decision_records(source), name="sp")
            self.assertEqual(1, manifest["records"])
            self.assertEqual(1, manifest["terminalRecords"])
            self.assertEqual(1, manifest["rewardRecords"])
            self.assertEqual(1, manifest["resultRecords"])


if __name__ == "__main__":
    unittest.main()
