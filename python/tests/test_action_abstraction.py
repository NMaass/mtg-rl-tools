import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.models import estimate_parameter_count, get_model_config
from magic_cabt.training.action_abstraction import (
    abstract_record,
    action_bucket_distribution,
)
from magic_cabt.training.action_entropy import analyze_action_entropy, entropy
from magic_cabt.training.analyze_actions import main as analyze_actions_main


OPTIONS = [
    {"index": 0, "type": "PASS_PRIORITY", "label": "Pass priority", "payload": {}},
    {"index": 1, "type": "PLAY_LAND", "label": "Play Swamp", "payload": {}},
    {"index": 2, "type": "CAST_SPELL", "label": "Cast Death's Shadow",
     "payload": {"cardType": "CREATURE"}},
    {"index": 3, "type": "PROMPT_OBJECT", "label": "Target your creature",
     "payload": {"controllerIndex": 0}},
]


def record(selected, prompt="PRIORITY"):
    return {
        "schemaVersion": 1,
        "source": "engine_selfplay",
        "gameId": "g1",
        "sequenceNumber": selected,
        "playerIndex": 0,
        "observation": {},
        "select": {
            "type": prompt,
            "minCount": 1,
            "maxCount": 1,
            "option": OPTIONS,
        },
        "selectedIndices": [selected],
        "terminal": False,
        "metadata": {"captureConfidence": "exact"},
    }


class ActionAbstractionTest(unittest.TestCase):

    def test_small_profile_maps_common_buckets(self):
        self.assertEqual("PASS", abstract_record(record(0), "small")["chosenBucket"])
        self.assertEqual("PLAY_LAND", abstract_record(record(1), "small")["chosenBucket"])
        self.assertEqual("CAST_SPELL", abstract_record(record(2), "small")["chosenBucket"])
        self.assertEqual("TARGET_OR_CHOICE", abstract_record(record(3), "small")["chosenBucket"])

    def test_full_profile_keeps_target_side_information(self):
        summary = abstract_record(record(3, prompt="TARGET"), "full")
        self.assertEqual("PROMPT_TARGET_OWN_OBJECT", summary["chosenBucket"])

    def test_distribution_and_entropy_report_predictability(self):
        rows = [record(0), record(0), record(0), record(1), record(2)]
        dist = action_bucket_distribution(rows, profile="small")
        self.assertEqual(5, dist["total"])
        self.assertEqual(3, dist["buckets"]["PASS"])
        self.assertGreater(entropy(dist["buckets"]), 0.0)

        report = analyze_action_entropy(rows, profile="small")
        self.assertEqual(5, report["records"])
        self.assertEqual("PASS", report["global"]["mostCommon"])
        self.assertIn("PRIORITY", report["byPromptType"])

    def test_model_configs_have_different_capacity(self):
        small = get_model_config("small")
        full = get_model_config("full")
        self.assertEqual("small", small["actionProfile"])
        self.assertEqual("full", full["actionProfile"])
        self.assertLess(estimate_parameter_count(small),
                        estimate_parameter_count(full))

    def test_analyze_actions_cli_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "records.jsonl")
            out = os.path.join(tmp, "entropy.json")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record(0)) + "\n")
                handle.write(json.dumps(record(1)) + "\n")
            self.assertEqual(0, analyze_actions_main([
                "--input", source, "--profile", "small", "--out", out]))
            with open(out, encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual("small", report["profile"])
            self.assertEqual(2, report["records"])


if __name__ == "__main__":
    unittest.main()
