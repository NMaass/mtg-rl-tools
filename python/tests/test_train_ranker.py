import json
import os
import tempfile
import unittest

from magic_cabt.models.torch_ranker import TORCH_AVAILABLE, hash_features
from magic_cabt.training.train_ranker import build_examples


def record(selected, options=None, game_id="game-1"):
    if options is None:
        options = [
            {"index": 0, "type": "TARGET", "label": "target instance=101",
             "payload": {"targetInstanceId": 101, "canonicalKey": "tok"}},
            {"index": 1, "type": "TARGET", "label": "target instance=102",
             "payload": {"targetInstanceId": 102, "canonicalKey": "tok"}},
            {"index": 2, "type": "TARGET", "label": "target instance=103",
             "payload": {"targetInstanceId": 103, "canonicalKey": "other"}},
        ]
    return {
        "gameId": game_id,
        "select": {"type": "TARGET_SELECT", "minCount": 1, "maxCount": 1,
                   "option": options},
        "selectedIndices": selected,
        "observation": {"current": {"turnNumber": 2, "players": []}},
    }


class HashFeaturesTest(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        first = hash_features("cast grizzly bears", 64)
        second = hash_features("cast grizzly bears", 64)
        self.assertEqual(first, second)
        self.assertAlmostEqual(1.0, sum(v * v for v in first), places=6)

    def test_empty_text_is_zero_vector(self):
        self.assertEqual([0.0] * 16, hash_features("", 16))


class BuildExamplesTest(unittest.TestCase):
    def test_fungible_choices_fold_to_same_group(self):
        stats = {}
        examples = build_examples([record([0]), record([1])], stats)
        self.assertEqual(2, stats["compiled"])
        # clicking either identical token yields the same training target
        self.assertEqual(examples[0]["chosenGroup"], examples[1]["chosenGroup"])
        self.assertEqual([[0, 1], [2]], examples[0]["groups"])

    def test_bad_records_are_counted_not_patched(self):
        stats = {}
        examples = build_examples([
            record([9]),                       # out of range
            record([0, 1]),                    # multi-select
            record([], options=[]),            # no options
        ], stats)
        self.assertEqual([], examples)
        self.assertEqual(1, stats["skipped_bad_index"])
        self.assertEqual(1, stats["skipped_multi_or_no_select"])
        self.assertEqual(1, stats["skipped_no_options"])

    def test_option_texts_are_identity_free(self):
        examples = build_examples([record([0])])
        texts = examples[0]["optionTexts"]
        self.assertNotIn("101", texts[0])
        self.assertEqual(texts[0], texts[1])


@unittest.skipUnless(TORCH_AVAILABLE, "requires PyTorch")
class TrainRankerTorchTest(unittest.TestCase):
    def _examples(self):
        # the preferred target must differ in canonical text (a different
        # creature), not just instance id -- id noise is stripped by design
        options = [
            {"index": 0, "type": "TARGET", "label": "target Goblin instance=101",
             "payload": {"targetInstanceId": 101, "canonicalKey": "goblin"}},
            {"index": 1, "type": "TARGET", "label": "target Goblin instance=102",
             "payload": {"targetInstanceId": 102, "canonicalKey": "goblin"}},
            {"index": 2, "type": "TARGET", "label": "target Angel instance=103",
             "payload": {"targetInstanceId": 103, "canonicalKey": "angel"}},
        ]
        records = []
        for game in range(6):
            for _ in range(4):
                records.append(record([2], options=list(options),
                                      game_id="game-%d" % game))
        return build_examples(records)

    def test_train_learns_and_roundtrips_checkpoint(self):
        from magic_cabt.models.configs import get_model_config
        from magic_cabt.models.torch_ranker import OptionRanker
        from magic_cabt.training.train_ranker import train

        config = get_model_config("small")
        model, metrics = train(self._examples(), config, epochs=8,
                               batch_size=8, seed=0, device="cpu")
        self.assertGreaterEqual(metrics["train"]["groupTop1"], 0.9)
        self.assertIn("holdout", metrics)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "checkpoint.pt")
            model.save_checkpoint(path)
            restored = OptionRanker.load_checkpoint(path)
            self.assertEqual(model.config, restored.config)

    def test_parameter_count_matches_estimate(self):
        from magic_cabt.models.configs import (
            estimate_parameter_count,
            get_model_config,
        )
        from magic_cabt.models.torch_ranker import OptionRanker

        config = get_model_config("small")
        model = OptionRanker(config)
        actual = sum(p.numel() for p in model.parameters())
        self.assertEqual(estimate_parameter_count(config), actual)


if __name__ == "__main__":
    unittest.main()
