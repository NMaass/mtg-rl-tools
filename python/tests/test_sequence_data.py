import unittest
from unittest.mock import patch

from magic_cabt.training import sequence_data
from magic_cabt.training.train_information_state import game_key


def decision(game, sequence):
    options = [{"index": 0, "type": "PASS_PRIORITY", "label": "Pass",
                "payload": {"canonicalKey": "pass"}}]
    return {
        "matchId": "m", "gameNumber": game,
        "sequenceNumber": sequence, "selectedIndices": [0],
        "select": {"type": "PRIORITY", "option": options},
        "observation": {"current": {
            "matchId": "m", "gameNumber": game,
            "gameInstance": "g-%s" % game,
            "seq": sequence,
        }},
        "_groups": [[0]], "_chosenGroup": 0,
    }


class CompleteGameCollectorTest(unittest.TestCase):
    def test_soft_limit_never_keeps_a_partial_later_game(self):
        rows = [decision(1, 1), decision(1, 2),
                decision(2, 1), decision(2, 2)]
        with patch.object(sequence_data.core, "_iter_all_decisions",
                          return_value=iter(rows)):
            accepted, _cards, metadata = \
                sequence_data.collect_complete_decision_games(
                    ["unused"], game_key, max_decisions=3)
        self.assertEqual([1, 1], [row["gameNumber"] for row in accepted])
        self.assertTrue(metadata["truncatedAtGameBoundary"])
        self.assertEqual("complete-game", metadata["unit"])

    def test_first_game_is_retained_even_when_larger_than_soft_limit(self):
        rows = [decision(1, index) for index in range(1, 5)]
        with patch.object(sequence_data.core, "_iter_all_decisions",
                          return_value=iter(rows)):
            accepted, _cards, metadata = \
                sequence_data.collect_complete_decision_games(
                    ["unused"], game_key, max_decisions=2)
        self.assertEqual(4, len(accepted))
        self.assertFalse(metadata["truncatedAtGameBoundary"])


if __name__ == "__main__":
    unittest.main()
