import json
import os
import tempfile
import unittest

from magic_cabt.analysis.compare import (
    _model_cell,
    _option_identity,
    build_comparison,
    render_comparison_html,
)
from magic_cabt.analysis.schema import decision_fingerprint


def decision(sequence, chosen=0, options=None):
    options = options or [
        {"index": 0, "type": "PASS_PRIORITY", "label": "Pass",
         "payload": {"canonicalKey": "pass"}},
        {"index": 1, "type": "CAST_SPELL", "label": "Cast Shock",
         "payload": {"canonicalKey": "cast:shock"}},
    ]
    return {
        "gameId": "game-1", "matchId": "match-1", "gameNumber": 1,
        "sequenceNumber": sequence, "selectedIndices": [chosen],
        "observation": {
            "current": {
                "matchId": "match-1", "gameNumber": 1,
                "gameInstance": "g-1", "seq": sequence,
                "turnNumber": 2, "phase": "MAIN1",
                "players": [{"seat": 1, "life": 20},
                            {"seat": 2, "life": 18}],
                "zones": {"battlefield": [{"name": "Bear"}]},
            },
            "select": {"type": "PRIORITY", "option": options},
        },
    }


def analysis(record, checkpoint, top_index, played_rank):
    options = record["observation"]["select"]["option"]
    top = options[top_index]
    payload = top.get("payload") or {}
    return {
        "schemaVersion": 1,
        "decisionFingerprint": decision_fingerprint(record),
        "createdAt": "2026-01-01T00:00:00+00:00",
        "model": {"modelId": checkpoint, "checkpointId": checkpoint},
        "analysis": {
            "topK": [{
                "optionIndex": top_index,
                "canonicalGroupKey": payload.get("canonicalKey"),
                "label": top["label"], "score": 2.0, "probability": 0.8,
            }],
            "chosenRank": played_rank, "chosenScore": 0.2, "value": 0.1,
        },
        "latencyMs": 4,
    }


class AnalysisComparisonTest(unittest.TestCase):
    def test_builds_disagreement_rows_and_standalone_html(self):
        with tempfile.TemporaryDirectory() as scratch:
            decisions = [decision(1, chosen=0), decision(2, chosen=1)]
            with open(os.path.join(scratch, "decisions.jsonl"), "w",
                      encoding="utf-8") as handle:
                for row in decisions:
                    handle.write(json.dumps(row) + "\n")
            records = [
                analysis(decisions[0], "ranker-sha", 0, 1),
                analysis(decisions[0], "jepa-sha", 1, 2),
                analysis(decisions[1], "ranker-sha", 1, 1),
                analysis(decisions[1], "jepa-sha", 1, 1),
            ]
            with open(os.path.join(scratch, "analysis.jsonl"), "w",
                      encoding="utf-8") as handle:
                for row in records:
                    handle.write(json.dumps(row) + "\n")
            models = [
                {"name": "Ranker", "checkpoint": "ranker.pt",
                 "checkpointSha256": "a",
                 "model": {"modelId": "ranker",
                           "checkpointId": "ranker-sha"}},
                {"name": "JEPA", "checkpoint": "jepa.pt",
                 "checkpointSha256": "b",
                 "model": {"modelId": "jepa",
                           "checkpointId": "jepa-sha"}},
            ]
            report = build_comparison(scratch, models, title="Fixture comparison")
            self.assertEqual(2, report["metrics"]["decisions"])
            self.assertEqual(1, report["metrics"]["agreement"]["disagree"])
            self.assertEqual(1, report["metrics"]["agreement"]["all-agree"])
            self.assertEqual("Cast Shock",
                             report["rows"][0]["models"][1]["topLabel"])
            rendered = render_comparison_html(report)
            self.assertIn("Fixture comparison", rendered)
            self.assertIn("application/json", rendered)

    def test_fungible_human_choice_uses_canonical_rank(self):
        options = [
            {"index": 0, "type": "TARGET", "label": "Token",
             "payload": {"canonicalKey": "target:soldier-token"}},
            {"index": 1, "type": "TARGET", "label": "Token",
             "payload": {"canonicalKey": "target:soldier-token"}},
        ]
        record = {"analysis": {"topK": [{
            "optionIndex": 0,
            "canonicalGroupKey": "target:soldier-token",
            "label": "Token", "probability": 0.6,
        }], "chosenRank": 2}}
        cell = _model_cell(record, options, 1)
        self.assertEqual(1, cell["playedRank"])
        self.assertEqual(2, cell["rawPlayedRank"])

    def test_unkeyed_equal_labels_remain_distinct(self):
        options = [
            {"index": 0, "type": "TARGET", "label": "Token", "payload": {}},
            {"index": 1, "type": "TARGET", "label": "Token", "payload": {}},
        ]
        self.assertNotEqual(_option_identity(options, 0),
                            _option_identity(options, 1))


if __name__ == "__main__":
    unittest.main()
