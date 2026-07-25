import json
import os
import tempfile
import unittest

from magic_cabt.research import cli as research_cli
from magic_cabt.research.tactical import (
    evaluate_scenarios,
    load_scenarios,
    validate_scenarios,
)


def option(index, kind, key, label=None):
    return {"index": index, "type": kind, "label": label or key,
            "payload": {"canonicalKey": key}}


def scenario(identifier="bolt-face", acceptable=None, prohibited=None,
             options=None, tags=None, history=None):
    options = options or [
        option(0, "PASS", "pass"),
        option(1, "CAST", "cast:bolt"),
    ]
    return {
        "schemaVersion": 1,
        "suite": "core-tactics-v1",
        "scenarioId": identifier,
        "tags": tags or ["tempo"],
        "history": history or [],
        "decision": {
            "observation": {"current": {"gameInstance": "g1"}},
            "select": {"type": "PRIORITY", "minCount": 1, "maxCount": 1,
                       "option": options},
            "selectedIndices": [0],
        },
        "acceptableActionKeys": acceptable or ["CAST|cast:bolt"],
        "prohibitedActionKeys": prohibited or [],
    }


class FixedScorer:
    model_info = {"modelId": "fixed", "checkpointId": "fixture"}

    def __init__(self, scores):
        self.scores = scores

    def score(self, _record):
        return list(self.scores)


class StatefulScorer(FixedScorer):
    def __init__(self, scores):
        super().__init__(scores)
        self.observed = 0
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.observed = 0

    def observe(self, _record):
        self.observed += 1

    def score(self, _record):
        if self.observed != 1:
            raise ValueError("history not observed")
        return super().score(_record)


class TacticalSuiteTest(unittest.TestCase):
    def test_validation_rejects_duplicate_ids_and_missing_actions(self):
        first = scenario()
        second = scenario(acceptable=["CAST|missing"])
        errors, _warnings = validate_scenarios([first, second])
        self.assertTrue(any("duplicate scenarioId" in error
                            for error in errors))
        self.assertTrue(any("expectations do not resolve" in error
                            for error in errors))

    def test_validation_rejects_private_observation_labels(self):
        value = scenario()
        value["decision"]["observation"]["oracleLabels"] = {"hand": [1]}
        errors, _warnings = validate_scenarios([value])
        self.assertTrue(any("oracle/private" in error for error in errors))

    def test_canonical_groups_are_ranked_once(self):
        value = scenario(options=[
            option(0, "TARGET", "bear", "Bear A"),
            option(1, "TARGET", "bear", "Bear B"),
            option(2, "TARGET", "opponent", "Opponent"),
        ], acceptable=["TARGET|bear"])
        report = evaluate_scenarios(
            [value], [("model", FixedScorer([0.1, 0.9, 0.8]))], top_k=2)
        row = report["models"]["model"]["scenarios"][0]
        self.assertEqual("TARGET|bear", row["topActionKey"])
        self.assertEqual([0, 1], row["topActions"][0]["concreteIndices"])
        self.assertTrue(row["top1"])

    def test_metrics_include_topk_mrr_prohibited_and_groups(self):
        values = [
            scenario("one", prohibited=["PASS|pass"], tags=["tempo"]),
            scenario("two", acceptable=["PASS|pass"], tags=["resource"]),
        ]
        report = evaluate_scenarios(
            values, [("model", FixedScorer([0.7, 0.6]))], top_k=2)
        metrics = report["models"]["model"]["metrics"]
        self.assertEqual(0.5, metrics["top1Rate"])
        self.assertEqual(1.0, metrics["topKRate"])
        self.assertEqual(0.75, metrics["mrr"])
        self.assertEqual(0.5, metrics["prohibitedTop1Rate"])
        self.assertEqual(
            1, report["models"]["model"]["byTag"]["tempo"]["scenarios"])
        self.assertEqual(
            2, report["models"]["model"]["byPromptType"]["PRIORITY"]["scenarios"])

    def test_stateful_scorer_receives_history_and_reset(self):
        value = scenario(history=[{
            "observation": {"current": {"seq": 1}},
            "select": {"type": "PRIORITY", "option": [
                option(0, "PASS", "pass"), option(1, "CAST", "cast:setup")]},
            "selectedIndices": [1],
        }])
        scorer = StatefulScorer([0.0, 1.0])
        report = evaluate_scenarios([value], [("stateful", scorer)])
        self.assertEqual(1, scorer.resets)
        self.assertEqual(
            1.0, report["models"]["stateful"]["metrics"]["top1Rate"])

    def test_scorer_errors_are_reported_without_aborting_suite(self):
        report = evaluate_scenarios(
            [scenario()], [("bad", FixedScorer([float("nan"), 0.0]))])
        metrics = report["models"]["bad"]["metrics"]
        self.assertEqual(0, metrics["scored"])
        self.assertEqual(1, metrics["errors"])
        self.assertEqual(0.0, metrics["coverage"])

    def test_load_jsonl_validates_and_preserves_source_lines(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = os.path.join(scratch, "suite.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(scenario()) + "\n")
            rows, warnings = load_scenarios(path)
        self.assertEqual([], warnings)
        self.assertEqual(1, rows[0]["__sourceLine"])

    def test_research_cli_scores_baseline_and_writes_provenance(self):
        value = scenario(acceptable=["PASS|pass"])
        with tempfile.TemporaryDirectory() as scratch:
            suite = os.path.join(scratch, "suite.jsonl")
            output = os.path.join(scratch, "report.json")
            with open(suite, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(value) + "\n")
            exit_code = research_cli.main([
                "benchmark-scenarios",
                "--suite", suite,
                "--model", "first=baseline:first-legal",
                "--out", output,
            ])
            self.assertEqual(0, exit_code)
            with open(output, encoding="utf-8") as handle:
                report = json.load(handle)
        self.assertRegex(report["input"]["sha256"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(
            "baseline:first-legal",
            report["models"]["first"]["input"]["spec"])
        self.assertEqual(
            1.0, report["models"]["first"]["metrics"]["top1Rate"])


if __name__ == "__main__":
    unittest.main()
