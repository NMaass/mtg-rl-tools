import os
import tempfile
import unittest

from magic_cabt.models.belief import TORCH_AVAILABLE, BeliefInformationStateModel
from magic_cabt.models.structured_jepa import StructuredJEPAConfig
from magic_cabt.models.visibility import VisibilitySafeTensorizer
from magic_cabt.training.train_belief_state import (
    calibration_metrics, compile_belief_record, compile_belief_records, train)


def record(game=1, sequence=1, hidden_name="Secret", mana_value=1,
           labeled=True, chosen=0):
    options = [
        {"index": 0, "type": "PASS_PRIORITY", "label": "Pass",
         "payload": {"canonicalKey": "pass"}},
        {"index": 1, "type": "CAST_SPELL", "label": "Cast Bear",
         "payload": {"canonicalKey": "cast:bear"}},
    ]
    value = {
        "matchId": "m", "gameNumber": game,
        "sequenceNumber": sequence, "perspectiveSeat": 1,
        "selectedIndices": [chosen],
        "select": {"type": "PRIORITY", "option": options},
        "observation": {"current": {
            "matchId": "m", "gameNumber": game,
            "gameInstance": "g-%s" % game, "seq": sequence,
            "localSeat": 1, "turnNumber": 1, "phase": "MAIN1",
            "players": [{"seat": 1, "life": 20},
                        {"seat": 2, "life": 20}],
            "zones": {"hand": {"2": [{
                "name": hidden_name, "manaValue": mana_value,
                "power": 9, "toughness": 9}]}}
        }},
        "_groups": [[0], [1]], "_chosenGroup": chosen,
    }
    if labeled:
        value["trainingLabels"] = {
            "visibility": "oracle-label-only",
            "source": "xmage-engine",
            "belief": {"removal": game % 2,
                       "counterspell": (game + 1) % 2},
        }
    return value


def tiny_config():
    return StructuredJEPAConfig(
        text_dim=32, numeric_dim=40, d_model=16, nhead=4,
        encoder_layers=1, predictor_layers=1, ff_dim=32,
        dropout=0.0, max_objects=16, causal_dim=18,
        horizon_buckets=8, embedding_backend="hash")


class VisibilityContractTest(unittest.TestCase):
    def test_hidden_card_identity_and_stats_do_not_change_rows(self):
        tensorizer = VisibilitySafeTensorizer(tiny_config())
        first = tensorizer.state_rows(record(hidden_name="Lightning Bolt",
                                             mana_value=1))
        second = tensorizer.state_rows(record(hidden_name="Emrakul",
                                              mana_value=15))
        self.assertEqual(first, second)

    def test_generic_private_history_is_not_consumed(self):
        tensorizer = VisibilitySafeTensorizer(tiny_config())
        first = record()
        second = record()
        first["observation"]["history"] = [{"trueOpponentDraw": "Bolt"}]
        second["observation"]["history"] = [{"trueOpponentDraw": "Island"}]
        self.assertEqual(tensorizer.state_rows(first),
                         tensorizer.state_rows(second))


class BeliefLabelContractTest(unittest.TestCase):
    def test_oracle_labels_compile_to_masked_targets(self):
        compiled = compile_belief_record(
            record(), ["removal", "counterspell", "sweeper"])
        self.assertEqual([1.0, 0.0, 0.0], compiled["_beliefTarget"])
        self.assertEqual([True, True, False], compiled["_beliefMask"])

    def test_unlabeled_records_remain_in_sequence(self):
        rows, summary = compile_belief_records(
            [record(labeled=True), record(sequence=2, labeled=False)],
            ["removal"])
        self.assertEqual(2, len(rows))
        self.assertEqual([False], rows[1]["_beliefMask"])
        self.assertEqual(1, summary["labeledRecords"])
        self.assertEqual(1, summary["unlabeledRecords"])

    def test_labels_in_observation_are_rejected(self):
        value = record()
        value["observation"]["beliefLabels"] = {"removal": 1}
        with self.assertRaises(ValueError):
            compile_belief_record(value, ["removal"])

    def test_non_oracle_visibility_is_rejected(self):
        value = record()
        value["trainingLabels"]["visibility"] = "public"
        with self.assertRaises(ValueError):
            compile_belief_record(value, ["removal"])

    def test_calibration_metrics_are_reported(self):
        metrics = calibration_metrics([0.9, 0.1], [1.0, 0.0], bins=2)
        self.assertEqual(2, metrics["examples"])
        self.assertLess(metrics["brier"], 0.02)
        self.assertIsNotNone(metrics["expectedCalibrationError"])


@unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
class BeliefTrainingTest(unittest.TestCase):
    def test_cpu_smoke_reports_held_out_calibration_and_checkpoint(self):
        rows = []
        for game in range(1, 5):
            rows.extend([
                record(game, 1, labeled=True, chosen=game % 2),
                record(game, 2, labeled=False, chosen=(game + 1) % 2),
            ])
        compiled, summary = compile_belief_records(
            rows, ["removal", "counterspell"])
        model, metrics = train(
            compiled, ["removal", "counterspell"],
            config=tiny_config(), epochs=1, batch_size=2,
            sequence_length=2, eval_fraction=0.25, seed=3,
            device="cpu", amp="off")
        self.assertGreater(summary["labeledCells"], 0)
        evaluation = metrics["history"][0]["eval"]
        self.assertGreater(evaluation["beliefCells"], 0)
        self.assertIn("aggregate", evaluation["calibration"])
        self.assertIn("removal", evaluation["calibration"]["perLabel"])
        self.assertFalse(set(metrics["split"]["trainGameIds"]).intersection(
            metrics["split"]["evalGameIds"]))
        with tempfile.TemporaryDirectory() as scratch:
            path = os.path.join(scratch, "belief.pt")
            model.save_checkpoint(path, extra={"metrics": metrics})
            loaded, extra = BeliefInformationStateModel.load_checkpoint(path)
            self.assertEqual(("removal", "counterspell"),
                             loaded.belief_labels)
            self.assertEqual(metrics["kind"], extra["metrics"]["kind"])


if __name__ == "__main__":
    unittest.main()
