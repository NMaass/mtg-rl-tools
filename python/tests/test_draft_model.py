import unittest

from magic_cabt.models.draft_model import (
    MODES,
    TORCH_AVAILABLE,
    DraftCardResolver,
    DraftModelConfig,
    DraftTensorizer,
    mana_value,
)
from magic_cabt.training.train_draft import (
    build_selection_examples,
    _align_positives,
    _split_by_group,
)


def _config():
    return DraftModelConfig(text_dim=32, numeric_dim=24, d_model=32,
                            nhead=4, encoder_layers=1, ff_dim=64)


def _pick_record(**overrides):
    record = {
        "kind": "draftPick",
        "source": "a.log",
        "draftId": "d1",
        "packNumber": 1,
        "pickNumber": 2,
        "pack": [11, 12, 13],
        "picked": [12, 13],
        "pool": [11],
    }
    record.update(overrides)
    return record


class ManaValueTest(unittest.TestCase):

    def test_parses_arena_mana_strings(self):
        self.assertEqual(mana_value("3UU"), 5)
        self.assertEqual(mana_value("G"), 1)
        self.assertEqual(mana_value("XR"), 1)
        self.assertIsNone(mana_value(None))


class BuildSelectionExamplesTest(unittest.TestCase):

    def test_pick_record_becomes_multi_positive_example(self):
        stats = {}
        examples = build_selection_examples([_pick_record()], stats)
        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example["mode"], "draftPick")
        self.assertEqual(example["contextIds"], [11])
        self.assertEqual(example["candidateIds"], [11, 12, 13])
        self.assertEqual(example["positives"], [1, 2])
        self.assertEqual(stats["examples_draftPick"], 1)

    def test_unaligned_pick_is_skipped_not_patched(self):
        stats = {}
        examples = build_selection_examples(
            [_pick_record(picked=[99])], stats)
        self.assertEqual(examples, [])
        self.assertEqual(stats["skipped_unaligned_pick"], 1)

    def test_deck_build_unrolls_only_pool_cards(self):
        record = {
            "kind": "deckBuild",
            "source": "a.log",
            "draftId": "d1",
            "pool": [1, 2, 3],
            "mainDeck": [2, 900, 3],  # 900: basic land outside the pool
            "sideboard": [1],
        }
        stats = {}
        examples = build_selection_examples([record], stats)
        self.assertEqual(stats["examples_deckBuild"], 2)
        first, second = examples
        self.assertEqual(first["contextIds"], [])
        self.assertEqual(first["candidateIds"], [1, 2, 3])
        # Remaining deck cards from the pool are all positives (2 and 3).
        self.assertEqual(first["positives"], [1, 2])
        # The land joins the context even though it was never a candidate.
        self.assertEqual(second["contextIds"], [2, 900])
        self.assertEqual(second["candidateIds"], [1, 3])
        self.assertEqual(second["positives"], [1])

    def test_sideboard_candidates_span_deck_and_sideboard(self):
        record = {
            "kind": "sideboard",
            "matchId": "m1",
            "offeredDeck": [1, 2],
            "offeredSideboard": [3],
            "chosenDeck": [1, 3],
            "chosenSideboard": [2],
        }
        examples = build_selection_examples([record])
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["candidateIds"], [1, 2, 3])
        self.assertEqual(examples[0]["positives"], [0, 2])

    def test_build_without_pool_is_skipped(self):
        stats = {}
        examples = build_selection_examples(
            [{"kind": "deckBuild", "pool": None, "mainDeck": [1]}], stats)
        self.assertEqual(examples, [])
        self.assertEqual(stats["skipped_build_without_pool"], 1)

    def test_align_positives_respects_multiplicity(self):
        self.assertEqual(_align_positives([5, 5, 6], [5, 5]), [0, 1])
        self.assertIsNone(_align_positives([5, 6], [5, 5]))

    def test_split_by_group_keeps_drafts_whole(self):
        examples = [{"groupKey": "g%d" % (i % 4)} for i in range(20)]
        train, holdout = _split_by_group(examples, 0.25, seed=1)
        train_groups = set(e["groupKey"] for e in train)
        holdout_groups = set(e["groupKey"] for e in holdout)
        self.assertFalse(train_groups & holdout_groups)
        self.assertEqual(len(train) + len(holdout), 20)


class DraftTensorizerTest(unittest.TestCase):

    def test_example_rows_shapes_and_modes(self):
        config = _config()
        tensorizer = DraftTensorizer(config, DraftCardResolver())
        example = {
            "mode": "draftPick",
            "contextIds": [11, 11, 12],
            "candidateIds": [13, 14],
            "packNumber": 2,
            "pickNumber": 3,
        }
        context_rows, candidate_rows = tensorizer.example_rows(example)
        self.assertEqual(len(context_rows), 4)  # status row + 3 cards
        self.assertEqual(len(candidate_rows), 2)
        self.assertEqual(len(context_rows[0]), tensorizer.row_dim)
        mode_offset = config.text_dim + 4 + MODES.index("draftPick")
        self.assertEqual(context_rows[0][mode_offset], 1.0)

    def test_unknown_mode_raises(self):
        tensorizer = DraftTensorizer(_config(), DraftCardResolver())
        with self.assertRaises(ValueError):
            tensorizer.example_rows(
                {"mode": "bogus", "contextIds": [], "candidateIds": [1]})


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class CardSelectionModelTest(unittest.TestCase):

    def test_scores_mask_and_checkpoint_roundtrip(self):
        import os
        import tempfile

        import torch

        from magic_cabt.models.draft_model import CardSelectionModel

        config = _config()
        model = CardSelectionModel(config)
        model.eval()
        tensorizer = DraftTensorizer(config, DraftCardResolver())
        examples = [
            {"mode": "draftPick", "contextIds": [1], "candidateIds": [2, 3],
             "packNumber": 1, "pickNumber": 1},
            {"mode": "deckBuild", "contextIds": [1, 2],
             "candidateIds": [3, 4, 5]},
        ]
        tensors = tensorizer.batch_examples(examples)
        logits = model.score_candidates(*tensors)
        self.assertEqual(tuple(logits.shape), (2, 3))
        # First example has two candidates; the padded slot must be -inf.
        self.assertEqual(float(logits[0, 2]), float("-inf"))
        self.assertTrue(torch.isfinite(logits[1]).all())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            model.save_checkpoint(path, extra={"note": "test"})
            restored, extra = CardSelectionModel.load_checkpoint(path)
            restored.eval()
            self.assertEqual(extra["note"], "test")
            restored_logits = restored.score_candidates(*tensors)
            self.assertTrue(torch.allclose(logits, restored_logits))

    def test_training_step_reduces_loss(self):
        import torch

        from magic_cabt.models.draft_model import CardSelectionModel
        from magic_cabt.training.train_draft import _multi_positive_loss

        torch.manual_seed(0)
        config = _config()
        model = CardSelectionModel(config)
        tensorizer = DraftTensorizer(config, DraftCardResolver())
        batch = [
            {"mode": "draftPick", "contextIds": [1], "candidateIds": [2, 3],
             "positives": [1], "packNumber": 1, "pickNumber": 1},
            {"mode": "sideboard", "contextIds": [2, 3],
             "candidateIds": [2, 3, 4], "positives": [0, 2]},
        ]
        tensors = tensorizer.batch_examples(batch)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        first_loss = None
        for _step in range(12):
            loss = _multi_positive_loss(
                model.score_candidates(*tensors), batch)
            if first_loss is None:
                first_loss = float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss), first_loss)


if __name__ == "__main__":
    unittest.main()
