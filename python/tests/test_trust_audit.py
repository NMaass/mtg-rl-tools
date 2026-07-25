import json
import os
import tempfile
import unittest

from magic_cabt.research.trust_audit import audit, main


def decision(sequence=1, selected=0, match="m", game=1, instance="g-1",
             history=False, forbidden=False):
    observation = {
        "current": {
            "matchId": match,
            "gameNumber": game,
            "gameInstance": instance,
            "seq": sequence,
            "localSeat": 1,
            "players": [{"seat": 1, "life": 20}, {"seat": 2, "life": 20}],
        },
        "publicHistory": [],
    }
    if history:
        observation["history"] = [{"unverified": True}]
    if forbidden:
        observation["oracleLabels"] = {"opponentHand": [123]}
    return {
        "schemaVersion": 1,
        "source": "engine_selfplay",
        "gameId": match,
        "gameNumber": game,
        "sequenceNumber": sequence,
        "playerIndex": 0,
        "observation": observation,
        "select": {
            "type": "PRIORITY",
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"index": 0, "type": "PASS_PRIORITY", "label": "Pass",
                 "payload": {"canonicalKey": "pass"}},
                {"index": 1, "type": "CAST_SPELL", "label": "Cast Bear",
                 "payload": {"canonicalKey": "cast:bear"}},
            ],
        },
        "selectedIndices": [selected],
        "terminal": False,
        "reward": None,
        "result": None,
        "metadata": {"captureConfidence": "exact"},
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_bundle(root, decisions=None, transitions=None, macro_actions=None):
    os.makedirs(root, exist_ok=True)
    if decisions is not None:
        write_jsonl(os.path.join(root, "decisions.jsonl"), decisions)
    if transitions is not None:
        write_jsonl(os.path.join(root, "transitions.jsonl"), transitions)
    if macro_actions is not None:
        write_jsonl(os.path.join(root, "macro_actions.jsonl"), macro_actions)


def check(result, identifier, scope=None):
    found = [row for row in result["checks"]
             if row["id"] == identifier and
             (scope is None or row["scope"] == scope)]
    if not found:
        raise AssertionError("missing check %s scope=%s" % (identifier, scope))
    return found[0]


class DatasetAuditTest(unittest.TestCase):
    def test_valid_dataset_passes_core_gates(self):
        with tempfile.TemporaryDirectory() as scratch:
            write_bundle(scratch, decisions=[decision(1), decision(2, selected=1)])
            result = audit([scratch])
        self.assertEqual("pass", check(result, "dataset.decisions.schema")["status"])
        self.assertEqual("pass", check(result, "dataset.decisions.selected-index")["status"])
        self.assertEqual("pass", check(result, "dataset.hidden-information.observation")["status"])
        self.assertEqual("pass", check(result, "dataset.sequence.order")["status"])
        self.assertTrue(result["summary"]["trusted"])
        self.assertEqual(1, len(result["files"]))
        self.assertEqual(64, len(result["files"][0]["sha256"]))

    def test_invalid_selected_index_fails_before_compilation(self):
        with tempfile.TemporaryDirectory() as scratch:
            write_bundle(scratch, decisions=[decision(selected=9)])
            result = audit([scratch])
        self.assertEqual("fail", check(result, "dataset.decisions.selected-index")["status"])
        self.assertFalse(result["summary"]["trusted"])

    def test_forbidden_oracle_key_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            write_bundle(scratch, decisions=[decision(forbidden=True)])
            result = audit([scratch])
        row = check(result, "dataset.hidden-information.observation")
        self.assertEqual("fail", row["status"])
        self.assertIn("observation.oracleLabels", row["details"]["paths"])

    def test_strict_promotes_generic_history_warning(self):
        with tempfile.TemporaryDirectory() as scratch:
            write_bundle(scratch, decisions=[decision(history=True)])
            result = audit([scratch], strict=True)
        row = check(result, "dataset.history.visibility")
        self.assertEqual("fail", row["status"])
        self.assertEqual("warn", row["originalStatus"])

    def test_reopened_game_segment_fails(self):
        rows = [
            decision(1, match="a", game=1, instance="ga"),
            decision(1, match="b", game=1, instance="gb"),
            decision(2, match="a", game=1, instance="ga"),
        ]
        with tempfile.TemporaryDirectory() as scratch:
            write_bundle(scratch, decisions=rows)
            result = audit([scratch])
        row = check(result, "dataset.sequence.order")
        self.assertEqual("fail", row["status"])
        self.assertTrue(row["details"]["reopenedGames"])

    def test_transition_and_macro_quality_are_audited(self):
        transitions = [{
            "source": "decisions", "matchId": "m", "gameNumber": 1,
            "gameInstance": "g-1", "horizon": 0, "prev": {}, "next": {},
        }]
        macro = [{
            "recordType": "MacroActionRecord", "complete": False,
            "orphanedParameterGroup": True,
            "classificationConfidence": "heuristic",
        }]
        with tempfile.TemporaryDirectory() as scratch:
            write_bundle(scratch, decisions=[decision()], transitions=transitions,
                         macro_actions=macro)
            result = audit([scratch])
        self.assertEqual("fail", check(
            result, "dataset.transitions.structure")["status"])
        self.assertEqual("warn", check(
            result, "dataset.macro-actions.completeness")["status"])

    def test_arbitrary_transition_filename_is_classified_by_content(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = os.path.join(scratch, "search-output.jsonl")
            write_jsonl(path, [{
                "source": "search", "matchId": "m", "gameNumber": 1,
                "gameInstance": "g-1", "horizon": 1,
                "prev": {"turnNumber": 1}, "next": {"turnNumber": 1},
            }])
            result = audit([path])
        self.assertEqual("pass", check(
            result, "dataset.inputs.classification")["status"])
        self.assertEqual("pass", check(
            result, "dataset.transitions.structure")["status"])
        self.assertEqual("not-applicable", check(
            result, "dataset.decisions.present")["status"])
        self.assertTrue(result["summary"]["trusted"])

    def test_malformed_json_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = os.path.join(scratch, "broken.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{not-json}\n")
            result = audit([path])
        self.assertEqual("fail", check(
            result, "dataset.inputs.classification")["status"])
        self.assertFalse(result["summary"]["trusted"])

    def test_known_filename_with_wrong_shape_fails_classification(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = os.path.join(scratch, "decisions.jsonl")
            write_jsonl(path, [{"prev": {}, "next": {}, "horizon": 1}])
            result = audit([path])
        row = check(result, "dataset.inputs.classification")
        self.assertEqual("fail", row["status"])
        self.assertIn(
            "filename implies decisions",
            row["details"]["invalid"][0]["classificationError"])

    def test_cli_writes_atomic_json_report(self):
        with tempfile.TemporaryDirectory() as scratch:
            bundle = os.path.join(scratch, "bundle")
            write_bundle(bundle, decisions=[decision()])
            output = os.path.join(scratch, "audit", "report.json")
            exit_code = main(["--input", bundle, "--out", output])
            self.assertEqual(0, exit_code)
            self.assertTrue(os.path.isfile(output))
            self.assertFalse(os.path.exists(output + ".tmp"))
            with open(output, encoding="utf-8") as handle:
                result = json.load(handle)
        self.assertEqual("magic-training-trust-audit-v1", result["kind"])


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "requires torch")
class CheckpointAuditTest(unittest.TestCase):
    def metrics(self, overlap=False, complete=True):
        train_id = '["train",1,"g-train"]'
        eval_id = '["m",1,"g-1"]'
        if overlap:
            eval_id = train_id
        evaluation = {
            "examples": 2,
            "loss": 0.5,
            "policyTop1": 0.5,
            "policyTop3": 1.0,
            "policyMRR": 0.75,
        }
        if not complete:
            evaluation.pop("policyMRR")
        return {
            "kind": "magic-recurrent-information-state-training-v1",
            "bestEpoch": 1,
            "bestSelectionMetric": 0.5,
            "visibilityPolicy": "public-history-and-perspective-state-v1",
            "split": {
                "unit": "game",
                "trainGameIds": [train_id],
                "evalGameIds": [eval_id],
            },
            "history": [{"epoch": 1, "eval": evaluation}],
            "inputs": [],
        }

    def save_checkpoint(self, path, state=None, metrics=None):
        torch.save({
            "kind": "magic-recurrent-information-state-v1",
            "stateDict": state or {"weight": torch.ones(2)},
            "extra": {"metrics": metrics or self.metrics()},
        }, path)

    def test_finite_checkpoint_disjoint_split_and_evidence_pass(self):
        with tempfile.TemporaryDirectory() as scratch:
            bundle = os.path.join(scratch, "bundle")
            write_bundle(bundle, decisions=[decision()])
            path = os.path.join(scratch, "model.pt")
            self.save_checkpoint(path)
            result = audit([bundle], checkpoints=[path],
                           require_checkpoint_audit=True)
        scope = "checkpoint:model.pt"
        self.assertEqual("pass", check(
            result, "checkpoint.parameters", scope)["status"])
        self.assertEqual("pass", check(
            result, "checkpoint.split", scope)["status"])
        self.assertEqual("pass", check(
            result, "checkpoint.recurrent-diagnostics", scope)["status"])

    def test_nan_parameter_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            bundle = os.path.join(scratch, "bundle")
            write_bundle(bundle, decisions=[decision()])
            path = os.path.join(scratch, "nan.pt")
            self.save_checkpoint(
                path, state={"weight": torch.tensor([float("nan")])})
            result = audit([bundle], checkpoints=[path])
        self.assertEqual("fail", check(
            result, "checkpoint.parameters", "checkpoint:nan.pt")["status"])

    def test_overlapping_split_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            bundle = os.path.join(scratch, "bundle")
            write_bundle(bundle, decisions=[decision()])
            path = os.path.join(scratch, "overlap.pt")
            self.save_checkpoint(path, metrics=self.metrics(overlap=True))
            result = audit([bundle], checkpoints=[path])
        self.assertEqual("fail", check(
            result, "checkpoint.split", "checkpoint:overlap.pt")["status"])

    def test_missing_family_evidence_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            bundle = os.path.join(scratch, "bundle")
            write_bundle(bundle, decisions=[decision()])
            path = os.path.join(scratch, "missing.pt")
            self.save_checkpoint(path, metrics=self.metrics(complete=False))
            result = audit([bundle], checkpoints=[path])
        self.assertEqual("fail", check(
            result, "checkpoint.recurrent-diagnostics",
            "checkpoint:missing.pt")["status"])


if __name__ == "__main__":
    unittest.main()
