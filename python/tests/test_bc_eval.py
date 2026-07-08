import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.agents.baseline_policy import rank_options
from magic_cabt.training.eval_bc import evaluate


def example(chosen, option_types=None, prompt="PRIORITY"):
    if option_types is None:
        option_types = ["PASS_PRIORITY", "PLAY_LAND", "CAST_SPELL"]
    return {
        "schemaVersion": 1,
        "gameId": "game-1",
        "sequenceNumber": chosen,
        "playerIndex": 0,
        "promptType": prompt,
        "optionTypes": option_types,
        "stateText": "turn=1 | prompt=%s" % prompt,
        "optionTexts": ["Pass priority", "Play Mountain", "Cast Grizzly Bears"][:len(option_types)],
        "chosenIndex": chosen,
        "metadata": {},
    }


class BcEvalTest(unittest.TestCase):

    def test_evaluator_computes_top1_top3_and_mrr(self):
        metrics = evaluate([example(0), example(2)], policy="first")
        self.assertEqual(2, metrics["examples"])
        self.assertEqual(0.5, metrics["top1Accuracy"])
        self.assertEqual(1.0, metrics["top3Accuracy"])
        self.assertAlmostEqual((1.0 + (1.0 / 3.0)) / 2.0,
                               metrics["meanReciprocalRank"])
        self.assertEqual(0.5, metrics["passChoiceRate"])
        self.assertEqual(3.0, metrics["averageOptionCount"])
        self.assertEqual(0.5, metrics["accuracyByPromptType"]["PRIORITY"]["accuracy"])
        self.assertEqual(1.0, metrics["accuracyByChosenOptionType"]["PASS_PRIORITY"]["accuracy"])

    def test_random_policy_returns_in_range_options(self):
        row = example(1)
        ranking = rank_options(row, policy="random", rng=random.Random(0))
        self.assertEqual(len(row["optionTexts"]), len(ranking))
        self.assertEqual(set(range(len(row["optionTexts"]))), set(ranking))
        for index in ranking:
            self.assertGreaterEqual(index, 0)
            self.assertLess(index, len(row["optionTexts"]))


if __name__ == "__main__":
    unittest.main()

