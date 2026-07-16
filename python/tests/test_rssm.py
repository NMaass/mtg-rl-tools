import os
import tempfile
import unittest
from unittest.mock import patch

from magic_cabt.models.rssm import TORCH_AVAILABLE, StructuredRSSM
from magic_cabt.models.structured_jepa import StructuredJEPAConfig
from magic_cabt.training import train_rssm
from magic_cabt.training.train_rssm import (
    build_transition_sequences, collect_transition_data,
    split_sequences, train, window_sequences)


def state(game, seq, life=20):
    return {
        "matchId": "m", "gameNumber": game,
        "gameInstance": "g-%s" % game, "seq": seq,
        "turnNumber": 1, "phase": "MAIN1", "localSeat": 1,
        "players": [{"seat": 1, "life": life, "handCount": 5,
                     "libraryCount": 50},
                    {"seat": 2, "life": 20, "handCount": 5,
                     "libraryCount": 50}],
        "zones": {"battlefield": []},
    }


def transition(game, seq):
    return {
        "matchId": "m", "gameNumber": game,
        "gameInstance": "g-%s" % game,
        "horizon": 1,
        "prev": state(game, seq),
        "next": state(game, seq + 1, life=19),
        "action": {"promptType": "PRIORITY", "selectedOption": {
            "index": 0, "type": "PASS_PRIORITY", "label": "Pass",
            "payload": {"canonicalKey": "pass"}}},
        "outcome": 1.0 if game % 2 else -1.0,
    }


def tiny_config():
    return StructuredJEPAConfig(
        text_dim=32, numeric_dim=40, d_model=16, nhead=4,
        encoder_layers=1, predictor_layers=1, ff_dim=32,
        dropout=0.0, max_objects=12, causal_dim=18,
        horizon_buckets=8, embedding_backend="hash")


class RSSMSequenceTest(unittest.TestCase):
    def test_sequences_sort_and_split_by_whole_game(self):
        rows = [transition(1, 3), transition(2, 2),
                transition(1, 1), transition(2, 1)]
        sequences = build_transition_sequences(rows)
        first = min(sequences, key=lambda item: item["gameKey"])
        self.assertEqual([1, 3], [row["prev"]["seq"]
                                  for row in first["transitions"]])
        training, evaluation = split_sequences(
            sequences, eval_fraction=0.5, seed=5)
        self.assertFalse({row["gameKey"] for row in training}.intersection(
            row["gameKey"] for row in evaluation))

    def test_windows_do_not_cross_game_boundaries(self):
        sequences = build_transition_sequences([
            transition(1, 1), transition(1, 2), transition(2, 1)])
        windows = window_sequences(sequences, 1)
        self.assertEqual(3, len(windows))
        self.assertTrue(all(len(window["transitions"]) == 1
                            for window in windows))

    def test_collector_never_accepts_partial_later_game(self):
        rows = [transition(1, 1), transition(1, 2),
                transition(2, 1), transition(2, 2)]
        with patch.object(train_rssm.core, "_iter_all_transitions",
                          return_value=iter(rows)):
            accepted, _cards, metadata = collect_transition_data(
                ["unused"], max_transitions=3)
        self.assertEqual([1, 1], [row["gameNumber"] for row in accepted])
        self.assertTrue(metadata["truncatedAtGameBoundary"])


@unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
class StructuredRSSMTest(unittest.TestCase):
    def test_prior_posterior_shapes_kl_and_checkpoint(self):
        import torch
        model = StructuredRSSM(tiny_config(), latent_dim=8)
        deterministic, stochastic = model.initial(3)
        observation = torch.randn(3, 16)
        action = torch.randn(3, 72)
        prior, posterior = model.posterior_step(
            observation, action, deterministic, stochastic, sample=False)
        kl = model.diagonal_kl(
            posterior["mean"], posterior["logScale"],
            prior["mean"], prior["logScale"])
        self.assertEqual((3, 8), tuple(posterior["mean"].shape))
        self.assertEqual((3, 24), tuple(posterior["feature"].shape))
        self.assertTrue(bool(torch.isfinite(kl).all()))
        with tempfile.TemporaryDirectory() as scratch:
            path = os.path.join(scratch, "rssm.pt")
            model.save_checkpoint(path, extra={"test": True})
            loaded, extra = StructuredRSSM.load_checkpoint(path)
            self.assertEqual(8, loaded.latent_dim)
            self.assertTrue(extra["test"])

    def test_cpu_smoke_reports_rollout_and_collapse_diagnostics(self):
        rows = []
        for game in range(1, 5):
            rows.extend([transition(game, 1), transition(game, 2)])
        _model, metrics = train(
            rows, config=tiny_config(), epochs=1, batch_size=2,
            sequence_length=2, eval_fraction=0.25, seed=3,
            device="cpu", amp="off", open_loop_horizon=2)
        self.assertEqual("structured-rssm-v1", metrics["modelFamily"])
        self.assertFalse(set(metrics["split"]["trainGameIds"]).intersection(
            metrics["split"]["evalGameIds"]))
        evaluation = metrics["history"][0]["eval"]
        self.assertGreater(evaluation["transitionExamples"], 0)
        self.assertIn("1", evaluation["openLoopMseByHorizon"])
        self.assertIn("priorNll", evaluation)
        self.assertIn("standardizedResidualRms", evaluation)
        self.assertIn("effectiveRank", evaluation["collapse"])


if __name__ == "__main__":
    unittest.main()
