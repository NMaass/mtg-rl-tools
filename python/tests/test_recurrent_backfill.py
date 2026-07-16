import json
import os
import tempfile
import unittest
from unittest.mock import patch

from magic_cabt.analysis import backfill


def decision(sequence):
    options = [{"index": 0, "type": "PASS_PRIORITY", "label": "Pass",
                "payload": {"canonicalKey": "pass"}}]
    return {
        "matchId": "m", "gameNumber": 1, "sequenceNumber": sequence,
        "selectedIndices": [0],
        "observation": {"current": {"turnNumber": 1},
                        "select": {"type": "PRIORITY", "option": options}},
    }


class FakeStatefulScorer:
    model_info = {"modelId": "fake", "checkpointId": "fake:1"}

    def __init__(self):
        self.scored = []
        self.observed = []

    def reset(self):
        pass

    def score(self, record):
        self.scored.append(record["sequenceNumber"])
        return [1.0]

    def observe(self, record):
        self.observed.append(record["sequenceNumber"])


class StatefulBackfillTest(unittest.TestCase):
    def test_cached_rows_still_advance_recurrent_memory(self):
        with tempfile.TemporaryDirectory() as scratch:
            with open(os.path.join(scratch, "decisions.jsonl"), "w",
                      encoding="utf-8") as handle:
                for row in (decision(1), decision(2)):
                    handle.write(json.dumps(row) + "\n")
            first = FakeStatefulScorer()
            with patch.object(backfill, "load_checkpoint_scorer",
                              return_value=first):
                result = backfill.backfill_bundle(scratch, "unused.pt")
            self.assertEqual(2, result["scored"])
            self.assertEqual([1, 2], first.scored)

            second = FakeStatefulScorer()
            with patch.object(backfill, "load_checkpoint_scorer",
                              return_value=second):
                result = backfill.backfill_bundle(scratch, "unused.pt")
            self.assertEqual(2, result["alreadyCached"])
            self.assertEqual([], second.scored)
            self.assertEqual([1, 2], second.observed)


if __name__ == "__main__":
    unittest.main()
