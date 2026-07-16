import json
import os
import tempfile
import unittest

from magic_cabt.research.trust_audit_cli import audit


def decision(sequence=1, selected=0, oracle_in_observation=False):
    options = [
        {"index": 0, "type": "PASS_PRIORITY", "label": "Pass",
         "payload": {"canonicalKey": "pass"}},
        {"index": 1, "type": "CAST_SPELL", "label": "Cast Bear",
         "payload": {"canonicalKey": "cast:bear"}},
    ]
    observation = {
        "current": {
            "matchId": "m", "gameNumber": 1, "gameInstance": "g-1",
            "seq": sequence, "turnNumber": 1, "phase": "MAIN1",
            "localSeat": 1,
            "players": [{"seat": 1, "life": 20},
                        {"seat": 2, "life": 20}],
            "zones": {"hand": {"2": [{
                "name": "Secret", "manaValue": 7,
                "power": 8, "toughness": 8}]}}
        },
        "publicHistory": [],
    }
    if oracle_in_observation:
        observation["beliefLabels"] = {"removal": 1}
    return {
        "schemaVersion": 1,
        "source": "fixture",
        "matchId": "m", "gameNumber": 1,
        "sequenceNumber": sequence,
        "perspectiveSeat": 1,
        "selectedIndices": [selected],
        "observation": observation,
        "select": {"type": "PRIORITY", "option": options},
    }


def write_bundle(path, rows):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "decisions.jsonl"), "w",
              encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def check(result, identifier, scope=None):
    matches = [row for row in result["checks"]
               if row["id"] == identifier and
               (scope is None or row["scope"] == scope)]
    if not matches:
        raise AssertionError("missing check %s" % identifier)
    return matches[0]


class DatasetAuditTest(unittest.TestCase):
    def test_valid_decisions_pass_legality_and_visibility(self):
        with tempfile.TemporaryDirectory() as scratch:
            write_bundle(scratch, [decision(1), decision(2, selected=1)])
            result = audit([scratch], visibility_samples=4)
        self.assertEqual("pass", check(
            result, "dataset.decisions.selected-index")["status"])
        self.assertEqual("pass", check(
            result, "dataset.hidden-information.observation")["status"])
        self.assertEqual("pass", check(
            result, "dataset.hidden-information.perturbation")["status"])
        self.assertTrue(result["summary"]["trusted"])

    def test_invalid_selected_index_is_detected_before_compilation(self):
        with tempfile.TemporaryDirectory() as scratch:
            write_bundle(scratch, [decision(selected=9)])
            result = audit([scratch])
        self.assertEqual("fail", check(
            result, "dataset.decisions.selected-index")["status"])
        self.assertFalse(result["summary"]["trusted"])

    def test_oracle_key_in_observation_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            write_bundle(scratch, [decision(oracle_in_observation=True)])
            result = audit([scratch])
        self.assertEqual("fail", check(
            result, "dataset.hidden-information.observation")["status"])

    def test_strict_mode_promotes_warnings(self):
        value = decision()
        value["observation"]["history"] = [{"unverified": True}]
        with tempfile.TemporaryDirectory() as scratch:
            write_bundle(scratch, [value])
            result = audit([scratch], strict=True)
        row = check(result, "dataset.history.visibility")
        self.assertEqual("fail", row["status"])
        self.assertEqual("warn", row["originalStatus"])


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "requires torch")
class CheckpointAuditTest(unittest.TestCase):
    def metrics(self):
        return {
            "kind": "magic-recurrent-information-state-training-v1",
            "bestEpoch": 1,
            "bestSelectionMetric": 0.5,
            "visibilityPolicy": "public-history-and-perspective-state-v1",
            "collection": {"unit": "complete-game"},
            "split": {
                "unit": "game",
                "trainGameIds": ['["m",1,"g-1"]'],
                "evalGameIds": ['["m",2,"g-2"]'],
            },
            "history": [{"epoch": 1, "eval": {
                "examples": 2, "loss": 0.5,
                "policyTop1": 0.5, "policyTop3": 1.0,
                "policyMRR": 0.75,
            }}],
            "inputs": [],
        }

    def checkpoint(self, path, state=None, kind=None, metrics=None):
        torch.save({
            "kind": kind or "magic-recurrent-information-state-v1",
            "stateDict": state or {"weight": torch.ones(2)},
            "extra": {"metrics": metrics or self.metrics()},
        }, path)

    def test_finite_checkpoint_and_disjoint_split_pass(self):
        with tempfile.TemporaryDirectory() as scratch:
            bundle = os.path.join(scratch, "bundle")
            write_bundle(bundle, [decision()])
            path = os.path.join(scratch, "model.pt")
            self.checkpoint(path)
            result = audit([bundle], checkpoints=[path])
        scope = "checkpoint:model.pt"
        self.assertEqual("pass", check(
            result, "checkpoint.parameters", scope)["status"])
        self.assertEqual("pass", check(
            result, "checkpoint.split", scope)["status"])

    def test_nan_tensor_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            bundle = os.path.join(scratch, "bundle")
            write_bundle(bundle, [decision()])
            path = os.path.join(scratch, "nan.pt")
            self.checkpoint(path, state={"weight": torch.tensor([float("nan")])})
            result = audit([bundle], checkpoints=[path])
        self.assertEqual("fail", check(
            result, "checkpoint.parameters", "checkpoint:nan.pt")["status"])

    def test_overlapping_game_split_fails(self):
        metrics = self.metrics()
        metrics["split"]["evalGameIds"] = list(
            metrics["split"]["trainGameIds"])
        with tempfile.TemporaryDirectory() as scratch:
            bundle = os.path.join(scratch, "bundle")
            write_bundle(bundle, [decision()])
            path = os.path.join(scratch, "overlap.pt")
            self.checkpoint(path, metrics=metrics)
            result = audit([bundle], checkpoints=[path])
        self.assertEqual("fail", check(
            result, "checkpoint.split", "checkpoint:overlap.pt")["status"])

    def test_rssm_missing_rollout_diagnostics_fails(self):
        metrics = self.metrics()
        metrics["kind"] = "magic-structured-rssm-training-v1"
        metrics["history"] = [{"epoch": 1, "eval": {
            "transitionExamples": 2, "loss": 0.5}}]
        with tempfile.TemporaryDirectory() as scratch:
            bundle = os.path.join(scratch, "bundle")
            write_bundle(bundle, [decision()])
            path = os.path.join(scratch, "rssm.pt")
            self.checkpoint(
                path, kind="magic-structured-rssm-v1", metrics=metrics)
            result = audit([bundle], checkpoints=[path])
        self.assertEqual("fail", check(
            result, "checkpoint.rssm-diagnostics",
            "checkpoint:rssm.pt")["status"])


if __name__ == "__main__":
    unittest.main()
