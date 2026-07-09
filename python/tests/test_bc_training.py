import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.agents.bc_agent import BCAgent
from magic_cabt.models import BagOfWordsBCPolicy
from magic_cabt.training.train_bc import evaluate_policy, main as train_bc_main


def example(chosen_index=1):
    return {
        "schemaVersion": 1,
        "gameId": "g1",
        "sequenceNumber": 0,
        "playerIndex": 0,
        "promptType": "PRIORITY",
        "optionTypes": ["PASS_PRIORITY", "CAST_SPELL"],
        "stateText": "turn=1 prompt=PRIORITY",
        "optionTexts": ["type=PASS_PRIORITY label=Pass priority",
                        "type=CAST_SPELL label=Cast Grizzly Bears"],
        "chosenIndex": chosen_index,
        "metadata": {},
    }


def observation():
    return {
        "select": {
            "type": "PRIORITY",
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"index": 0, "type": "PASS_PRIORITY", "label": "Pass priority"},
                {"index": 1, "type": "CAST_SPELL", "label": "Cast Grizzly Bears"},
            ],
        },
    }


class BcTrainingTest(unittest.TestCase):

    def test_policy_trains_scores_and_round_trips(self):
        policy = BagOfWordsBCPolicy.train([example(), example(), example(0)])
        scores = policy.score_example(example())
        self.assertEqual(2, len(scores))
        self.assertIn(policy.rank_example(example())[0], (0, 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "checkpoint.json")
            policy.save(path)
            loaded = BagOfWordsBCPolicy.load(path)
            self.assertEqual(policy.score_example(example()),
                             loaded.score_example(example()))

    def test_evaluate_policy_reports_top1(self):
        rows = [example(), example(), example(0)]
        policy = BagOfWordsBCPolicy.train(rows)
        metrics = evaluate_policy(policy, rows)
        self.assertEqual(3, metrics["examples"])
        self.assertIn("top1Accuracy", metrics)
        self.assertIn("PRIORITY", metrics["accuracyByPromptType"])

    def test_train_cli_writes_checkpoint_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "single_choice_il.jsonl")
            out = os.path.join(tmp, "bc")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(example()) + "\n")
                handle.write(json.dumps(example()) + "\n")
                handle.write(json.dumps(example(0)) + "\n")
            self.assertEqual(0, train_bc_main(["--input", source, "--out", out]))
            self.assertTrue(os.path.exists(os.path.join(out, "checkpoint.json")))
            self.assertTrue(os.path.exists(os.path.join(out, "metrics.json")))
            loaded = BagOfWordsBCPolicy.load(os.path.join(out, "checkpoint.json"))
            self.assertEqual(2, len(loaded.score_example(example())))

    def test_bc_agent_selects_legal_index(self):
        policy = BagOfWordsBCPolicy.train([example(), example(), example(0)])
        agent = BCAgent(policy=policy)
        selection = agent.select(observation())
        self.assertEqual(1, len(selection))
        self.assertIn(selection[0], (0, 1))
        self.assertEqual(2, len(agent.score(observation())))


if __name__ == "__main__":
    unittest.main()
