import os
import tempfile
import unittest

from magic_cabt.models.structured_jepa import (
    TORCH_AVAILABLE,
    MagicJEPA,
    StructuredJEPAConfig,
)
from magic_cabt.training.train_jepa import split_training_data, train


def state(match, game, seq, life=20):
    return {
        "matchId": match,
        "gameNumber": game,
        "gameInstance": "%s-%s" % (match, game),
        "seq": seq,
        "turnNumber": 1,
        "phase": "MAIN1",
        "localSeat": 1,
        "players": [
            {"seat": 1, "life": life, "handCount": 5, "libraryCount": 50},
            {"seat": 2, "life": 20, "handCount": 5, "libraryCount": 50},
        ],
        "zones": {"battlefield": []},
    }


def transition(match, game, seq):
    return {
        "matchId": match,
        "gameNumber": game,
        "gameInstance": "%s-%s" % (match, game),
        "horizon": 1,
        "prev": state(match, game, seq),
        "next": state(match, game, seq + 1, life=19),
        "action": None,
        "outcome": 1.0 if game == 1 else -1.0,
    }


def decision(match, game, chosen=0):
    current = state(match, game, 0)
    options = [
        {"index": 0, "type": "PASS_PRIORITY", "label": "Pass", "payload": {}},
        {"index": 1, "type": "PLAY_LAND", "label": "Play Forest",
         "payload": {"canonicalKey": "play-land:forest"}},
    ]
    return {
        "matchId": match,
        "gameNumber": game,
        "gameInstance": "%s-%s" % (match, game),
        "observation": {"current": current, "select": {
            "type": "PRIORITY", "option": options}},
        "select": {"type": "PRIORITY", "option": options},
        "selectedIndices": [chosen],
        "_groups": [[0], [1]],
        "_chosenGroup": chosen,
    }


class SplitTest(unittest.TestCase):
    def test_split_keeps_transition_and_decision_from_game_together(self):
        transitions = [transition("m", game, 0) for game in range(1, 7)]
        decisions = [decision("m", game, game % 2) for game in range(1, 7)]
        train_t, eval_t, train_d, eval_d, metadata = split_training_data(
            transitions, decisions, eval_fraction=0.34, seed=7)
        train_groups = {(row["matchId"], row["gameNumber"]) for row in train_t}
        train_groups.update((row["matchId"], row["gameNumber"]) for row in train_d)
        eval_groups = {(row["matchId"], row["gameNumber"]) for row in eval_t}
        eval_groups.update((row["matchId"], row["gameNumber"]) for row in eval_d)
        self.assertFalse(train_groups.intersection(eval_groups))
        self.assertEqual(6, len(train_groups.union(eval_groups)))
        self.assertEqual(len(eval_groups), metadata["evalGroups"])


@unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
class ReliableTrainingTest(unittest.TestCase):
    def test_cpu_smoke_run_writes_eval_and_resumable_state(self):
        config = StructuredJEPAConfig(
            text_dim=32,
            numeric_dim=40,
            d_model=32,
            nhead=4,
            encoder_layers=1,
            predictor_layers=1,
            ff_dim=64,
            dropout=0.0,
            max_objects=16,
            causal_dim=18,
            horizon_buckets=8,
            embedding_backend="hash",
        )
        transitions = []
        decisions = []
        for game in range(1, 5):
            transitions.extend([
                transition("smoke", game, 0),
                transition("smoke", game, 1),
            ])
            decisions.append(decision("smoke", game, game % 2))

        model, metrics = train(
            transitions,
            decisions,
            config=config,
            epochs=1,
            batch_size=2,
            eval_fraction=0.25,
            seed=3,
            device="cpu",
            amp="off",
            max_steps_per_epoch=2,
        )
        self.assertEqual("magic-structured-jepa-training-v2", metrics["kind"])
        self.assertGreater(metrics["evalTransitionExamples"], 0)
        self.assertGreater(metrics["evalDecisionExamples"], 0)
        self.assertIn("collapse", metrics["history"][0]["eval"])
        self.assertGreater(metrics["examplesPerSecond"], 0.0)
        self.assertIsNotNone(model._best_state_dict)
        self.assertIn("optimizer", model._training_state)

        with tempfile.TemporaryDirectory() as scratch:
            checkpoint = os.path.join(scratch, "checkpoint.pt")
            model.save_checkpoint(checkpoint, extra={
                "metrics": metrics,
                "trainingState": model._training_state,
            })
            loaded, extra = MagicJEPA.load_checkpoint(checkpoint)
            self.assertEqual(config.d_model, loaded.config.d_model)
            self.assertEqual(1, extra["trainingState"]["completedEpochs"])


if __name__ == "__main__":
    unittest.main()
