import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.training import iter_decision_records
from magic_cabt.training.compile_il import main as compile_il_main


OPTIONS = [
    {"index": 0, "type": "PASS_PRIORITY", "label": "Pass priority", "payload": {}},
    {"index": 1, "type": "CAST_SPELL", "label": "Cast spell", "payload": {}},
]


def _self_play_frame(sequence, player, selected):
    return {
        "sequence": sequence,
        "player": player,
        "observation": {
            "select": {
                "type": "PRIORITY",
                "minCount": 1,
                "maxCount": 1,
                "option": OPTIONS,
            },
        },
        "selected": selected,
    }


def _arena_decision(sequence=0, selected=None):
    if selected is None:
        selected = [1]
    return {
        "sequence": sequence,
        "matchId": "match-1",
        "gameNumber": 1,
        "seat": 0,
        "player": "Alice",
        "promptTimestamp": "t1",
        "responseTimestamp": "t2",
        "selectionMatched": True,
        "promptMessageType": "GREMessageType_ActionsAvailableReq",
        "responseMessageType": "ClientMessageType_PerformActionResp",
        "observation": {
            "current": {
                "turnNumber": 1,
                "players": [{"playerIndex": 0, "life": 20, "handCount": 7}],
            },
            "select": {
                "type": "ACTIONS",
                "minCount": 1,
                "maxCount": 1,
                "option": OPTIONS,
            },
        },
        # Raw Arena recorder shape: top-level select is the chosen indices,
        # not the canonical prompt spec. The training reader must normalize it.
        "select": selected,
    }


def _jsonl(records):
    return io.StringIO("".join(json.dumps(record) + "\n" for record in records))


class TrainingIoRegressionTest(unittest.TestCase):

    def test_self_play_terminal_result_is_attached_before_yield(self):
        records = [
            _self_play_frame(0, 0, [0]),
            _self_play_frame(1, 0, [1]),
            {"result": {"winner": 0}},
        ]
        iterator = iter_decision_records(_jsonl(records))
        first = next(iterator)
        self.assertFalse(first["terminal"])

        second = next(iterator)
        # Regression: the old iterator yielded the second frame before reading
        # the trailing result line and mutated it only later, which streaming
        # consumers could not observe at decision-processing time.
        self.assertTrue(second["terminal"])
        self.assertEqual({"winner": 0}, second["result"])
        self.assertEqual(1.0, second["reward"])
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_arena_decision_reader_lifts_prompt_spec_and_selected_indices(self):
        record = next(iter_decision_records(_jsonl([_arena_decision()])))
        self.assertEqual("arena_human", record["source"])
        self.assertEqual("ACTIONS", record["select"]["type"])
        self.assertEqual([1], record["selectedIndices"])
        self.assertNotIn("select", record["observation"])

    def test_compile_il_cli_accepts_raw_arena_decisions_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "decisions.jsonl")
            out = os.path.join(tmp, "single_choice.jsonl")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(_arena_decision()) + "\n")

            stderr = sys.stderr
            try:
                sys.stderr = io.StringIO()
                self.assertEqual(0, compile_il_main(["--input", source, "--out", out]))
                self.assertIn("compiled=1", sys.stderr.getvalue())
            finally:
                sys.stderr = stderr

            with open(out, encoding="utf-8") as handle:
                row = json.loads(handle.readline())
            self.assertEqual("ACTIONS", row["promptType"])
            self.assertEqual(1, row["chosenIndex"])
            self.assertEqual("mirror", row["metadata"]["captureConfidence"])


if __name__ == "__main__":
    unittest.main()
