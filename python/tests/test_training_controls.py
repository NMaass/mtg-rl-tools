import unittest

from magic_cabt.training.train_reliable import merge_best
from magic_cabt.training.train_structured_bc import build_parser


class FakeModel:
    def __init__(self, best):
        self._best_state_dict = best
        self._best_epoch = 4
        self._training_state = {"completedEpochs": 4}


class ResumeSelectionTest(unittest.TestCase):
    def test_previous_lower_loss_best_is_preserved(self):
        model = FakeModel({"weight": "current"})
        metrics = {"bestEpoch": 4, "bestSelectionMetric": 0.4}
        previous = {"trainingState": {
            "bestStateDict": {"weight": "previous"},
            "bestEpoch": 2,
            "bestSelectionMetric": 0.2,
        }}
        merge_best(model, metrics, previous)
        self.assertEqual({"weight": "previous"}, model._best_state_dict)
        self.assertEqual(2, metrics["bestEpoch"])
        self.assertEqual(0.2, metrics["bestSelectionMetric"])
        self.assertEqual({"weight": "previous"},
                         model._training_state["bestStateDict"])

    def test_new_lower_loss_best_replaces_previous(self):
        model = FakeModel({"weight": "current"})
        metrics = {"bestEpoch": 4, "bestSelectionMetric": 0.1}
        previous = {"trainingState": {
            "bestStateDict": {"weight": "previous"},
            "bestEpoch": 2,
            "bestSelectionMetric": 0.2,
        }}
        merge_best(model, metrics, previous)
        self.assertEqual({"weight": "current"}, model._best_state_dict)
        self.assertEqual(0.1, model._training_state["bestSelectionMetric"])


class StructuredBCCommandTest(unittest.TestCase):
    def test_parser_exposes_capacity_matched_presets(self):
        args = build_parser().parse_args([
            "--input", "decisions.jsonl", "--out", "runs/bc",
            "--preset", "tiny", "--epochs", "2"])
        self.assertEqual("tiny", args.preset)
        self.assertEqual(2, args.epochs)
        self.assertEqual(["decisions.jsonl"], args.input)


if __name__ == "__main__":
    unittest.main()
