import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.server.agent_service import AgentService


OBSERVATION = {
    "select": {
        "type": "PRIORITY",
        "minCount": 1,
        "maxCount": 1,
        "option": [
            {"index": 0, "type": "PASS_PRIORITY", "label": "Pass priority"},
            {"index": 1, "type": "PLAY_LAND", "label": "Play Island"},
        ],
    },
}


class AgentServiceTest(unittest.TestCase):

    def test_lists_agents(self):
        service = AgentService()
        agents = service.list_agents()["agents"]
        self.assertIn("random", agents)
        self.assertIn("first", agents)

    def test_decide_returns_selection_scores_and_logs_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = AgentService(log_dir=tmp)
            response = service.decide({
                "agent": "first",
                "observation": OBSERVATION,
                "sequenceNumber": 7,
            })
            self.assertTrue(response["ok"])
            self.assertEqual([0], response["selectedIndices"])
            self.assertEqual(2, len(response["scores"]))
            log_path = os.path.join(tmp, "decisions.jsonl")
            self.assertTrue(os.path.exists(log_path))
            with open(log_path, encoding="utf-8") as handle:
                row = json.loads(handle.readline())
            self.assertEqual("first", row["request"]["agent"])
            self.assertEqual("PRIORITY", row["request"]["promptType"])
            self.assertEqual(2, row["request"]["legalActionCount"])

    def test_rejects_missing_observation(self):
        service = AgentService()
        with self.assertRaises(ValueError):
            service.decide({"agent": "first"})


if __name__ == "__main__":
    unittest.main()
