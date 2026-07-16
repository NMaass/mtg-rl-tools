import os
import tempfile
import unittest

from magic_cabt.models.information_state import (
    TORCH_AVAILABLE, RecurrentInformationStateModel)
from magic_cabt.models.structured_jepa import StructuredJEPAConfig
from magic_cabt.training.train_information_state import (
    build_game_sequences, selected_action, split_sequences, window_sequences)


def decision(game, sequence, chosen=0):
    options = [
        {"index": 0, "type": "PASS_PRIORITY", "label": "Pass", "payload": {}},
        {"index": 1, "type": "CAST_SPELL", "label": "Cast Bear",
         "payload": {"canonicalKey": "cast:bear"}},
    ]
    return {
        "matchId": "m", "gameNumber": game, "sequenceNumber": sequence,
        "selectedIndices": [chosen],
        "select": {"type": "PRIORITY", "option": options},
        "observation": {"current": {
            "matchId": "m", "gameNumber": game,
            "gameInstance": "g-%s" % game, "seq": sequence,
            "turnNumber": 1, "phase": "MAIN1", "localSeat": 1,
            "players": [{"seat": 1, "life": 20},
                        {"seat": 2, "life": 20}],
            "zones": {"battlefield": []}}},
        "_groups": [[0], [1]], "_chosenGroup": chosen,
    }


class SequenceContractTest(unittest.TestCase):
    def test_sequences_are_sorted_and_split_as_whole_games(self):
        rows = [decision(1, 3), decision(2, 2),
                decision(1, 1), decision(2, 1)]
        sequences = build_game_sequences(rows)
        first = min(sequences, key=lambda item: item["gameKey"])
        self.assertEqual([1, 3], [
            item["sequenceNumber"] for item in first["records"]])
        train, evaluate = split_sequences(sequences, eval_fraction=0.5, seed=4)
        self.assertFalse({row["gameKey"] for row in train}.intersection(
            row["gameKey"] for row in evaluate))

    def test_previous_action_is_concrete_recorded_choice(self):
        action = selected_action(decision(1, 1, chosen=1))
        self.assertEqual("CAST_SPELL", action["selectedOption"]["type"])

    def test_windows_never_cross_games_and_retain_boundary_action(self):
        sequences = build_game_sequences([
            decision(1, 1, chosen=1), decision(1, 2), decision(2, 1)])
        windows = window_sequences(sequences, 1)
        self.assertEqual(3, len(windows))
        second = [row for row in windows if row["start"] == 1][0]
        self.assertEqual("CAST_SPELL",
                         second["previousAction"]["selectedOption"]["type"])


@unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
class InformationStateModelTest(unittest.TestCase):
    def test_forward_and_checkpoint_round_trip(self):
        import torch
        config = StructuredJEPAConfig(
            text_dim=16, numeric_dim=24, d_model=16, nhead=4,
            encoder_layers=1, predictor_layers=1, ff_dim=32,
            dropout=0.0, max_objects=8, causal_dim=18,
            horizon_buckets=8, embedding_backend="hash")
        model = RecurrentInformationStateModel(config)
        rows = torch.randn(2, 3, 4, 40)
        masks = torch.ones(2, 3, 4, dtype=torch.bool)
        previous = torch.randn(2, 3, 40)
        options = torch.randn(2, 3, 5, 40)
        option_mask = torch.ones(2, 3, 5, dtype=torch.bool)
        memory, hidden = model.information_states(rows, masks, previous)
        logits = model.score_from_memory(memory, options, option_mask)
        self.assertEqual((2, 3, 16), tuple(memory.shape))
        self.assertEqual((1, 2, 16), tuple(hidden.shape))
        self.assertEqual((2, 3, 5), tuple(logits.shape))
        with tempfile.TemporaryDirectory() as scratch:
            path = os.path.join(scratch, "model.pt")
            model.save_checkpoint(path, extra={"x": 1})
            loaded, extra = RecurrentInformationStateModel.load_checkpoint(path)
            self.assertEqual(1, extra["x"])
            self.assertEqual(config.d_model, loaded.config.d_model)


if __name__ == "__main__":
    unittest.main()
