import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.research.benchmark import (
    benchmark_analyses,
    benchmark_matches,
    cluster_bootstrap_interval,
    decision_fingerprint,
)


def decision(game, sequence, chosen, source="arena_human"):
    options = [
        {"index": 0, "type": "PASS", "payload": {"canonicalKey": "pass"}},
        {"index": 1, "type": "TARGET", "payload": {"canonicalKey": "kill-threat"}},
        {"index": 2, "type": "TARGET", "payload": {"canonicalKey": "kill-threat"}},
        {"index": 3, "type": "FACE", "payload": {"canonicalKey": "face"}},
    ]
    return {
        "gameId": game,
        "sequenceNumber": sequence,
        "source": source,
        "observation": {
            "current": {
                "turnNumber": sequence,
                "players": [{"life": 20}, {"life": 20}],
            },
            "select": {"type": "PRIORITY", "option": options},
        },
        "select": {"type": "PRIORITY", "option": options},
        "selectedIndices": [chosen],
    }


def analysis(record, checkpoint, order, chosen_rank):
    options = record["select"]["option"]
    return {
        "decisionFingerprint": decision_fingerprint(record),
        "model": {"checkpointId": checkpoint},
        "analysis": {
            "chosenRank": chosen_rank,
            "topK": [
                {
                    "optionIndex": index,
                    "canonicalGroupKey": options[index]["payload"]["canonicalKey"],
                    "score": 1.0,
                }
                for index in order
            ],
        },
    }


class ResearchBenchmarkTest(unittest.TestCase):

    def test_cluster_interval_resamples_whole_games(self):
        result = cluster_bootstrap_interval(
            {"g1": [1, 1], "g2": [0, 0]}, 100, seed=2)
        self.assertEqual(0.5, result["estimate"])
        self.assertEqual(2, result["clusters"])
        self.assertEqual(4, result["observations"])

    def test_analysis_metrics_respect_target_fungibility(self):
        rows = [decision("g1", 1, 1), decision("g2", 2, 2)]
        # The exact chosen object is rank two, but another instance of the same
        # canonical action group is rank one.
        model_a = [analysis(row, "a", [2, 1, 3], 2) for row in rows]
        model_b = [analysis(row, "b", [3, 1, 2], 2) for row in rows]
        report = benchmark_analyses(
            rows, {"a": model_a, "b": model_b}, bootstrap_samples=100)
        self.assertEqual(
            1.0,
            report["models"]["a"]["metrics"]["canonicalTop1Accuracy"]["estimate"],
        )
        self.assertEqual(
            0.0,
            report["models"]["b"]["metrics"]["canonicalTop1Accuracy"]["estimate"],
        )
        self.assertEqual(
            0.5, report["models"]["a"]["metrics"]["rawMRR"]["estimate"])
        self.assertEqual(
            1.0, report["comparisons"][0]["canonicalTop1Delta"]["estimate"])

    def test_multiple_checkpoints_require_explicit_selection(self):
        row = decision("g", 1, 1)
        records = [
            analysis(row, "checkpoint-a", [1], 1),
            analysis(row, "checkpoint-b", [1], 1),
        ]
        with self.assertRaises(ValueError):
            benchmark_analyses(
                [row], {"mixed": records}, bootstrap_samples=10)
        report = benchmark_analyses(
            [row], {"mixed": records},
            checkpoint_by_name={"mixed": "checkpoint-b"},
            bootstrap_samples=10,
        )
        self.assertEqual(
            "checkpoint-b",
            report["models"]["mixed"]["selection"]["checkpointId"],
        )

    def test_long_format_match_rows_are_paired(self):
        rows = []
        for seed in range(4):
            rows.append({
                "agent": "a", "pairKey": "p%d" % seed,
                "clusterId": "c%d" % seed, "suite": "smoke", "score": 1,
            })
            rows.append({
                "agent": "b", "pairKey": "p%d" % seed,
                "clusterId": "c%d" % seed, "suite": "smoke", "score": 0,
            })
        report = benchmark_matches(rows, bootstrap_samples=100, seed=3)
        self.assertEqual(1.0, report["agents"]["a"]["meanScore"]["estimate"])
        self.assertEqual(1.0, report["comparisons"][0]["scoreDelta"]["estimate"])
        self.assertEqual(4, report["comparisons"][0]["commonPairedUnits"])

    def test_wide_format_match_rows_are_normalized(self):
        rows = [{
            "testAgent": "a",
            "referenceAgent": "b",
            "testScore": "win",
            "referenceScore": "loss",
            "pairKey": "p1",
            "suite": "x",
        }]
        report = benchmark_matches(rows, bootstrap_samples=10)
        self.assertEqual(2, report["normalizedRows"])
        self.assertEqual(1.0, report["comparisons"][0]["scoreDelta"]["estimate"])


if __name__ == "__main__":
    unittest.main()
