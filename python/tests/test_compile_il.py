import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.training.compile_il import compile_records, main
from magic_cabt.training.features import state_text


def decision(selected=None, min_count=1, max_count=1, options=None, current=None):
    if selected is None:
        selected = [1]
    if options is None:
        options = [
            {"index": 0, "type": "PASS_PRIORITY", "label": "Pass priority"},
            {"index": 1, "type": "CAST_SPELL", "label": "Cast Grizzly Bears",
             "payload": {"card": {"name": "Grizzly Bears"}}},
        ]
    return {
        "schemaVersion": 1,
        "source": "engine_selfplay",
        "gameId": "game-1",
        "sequenceNumber": 7,
        "playerIndex": 0,
        "observation": {
            "current": current or {
                "turnNumber": 3,
                "activePlayerId": "p0",
                "priorityPlayerId": "p0",
                "phase": "PRECOMBAT_MAIN",
                "players": [
                    {"playerIndex": 0, "playerId": "p0", "name": "Alice",
                     "life": 20, "handCount": 4, "libraryCount": 33}
                ],
                "stack": [{"name": "Shock"}],
                "battlefield": [{"name": "Mountain"}],
            },
            "select": {
                "type": "PRIORITY",
                "minCount": min_count,
                "maxCount": max_count,
                "option": options,
            },
        },
        "select": {
            "type": "PRIORITY",
            "playerIndex": 0,
            "minCount": min_count,
            "maxCount": max_count,
            "option": options,
        },
        "selectedIndices": selected,
        "terminal": False,
        "metadata": {"fixture": True},
    }


class CompileIlTest(unittest.TestCase):

    def test_single_choice_filter_includes_valid_records(self):
        examples, stats = compile_records([decision()])
        self.assertEqual(1, len(examples))
        self.assertEqual(1, stats["compiled"])
        self.assertEqual("PRIORITY", examples[0]["promptType"])
        self.assertEqual(["PASS_PRIORITY", "CAST_SPELL"], examples[0]["optionTypes"])

    def test_multi_select_records_are_discarded_with_statistics(self):
        examples, stats = compile_records([
            decision(selected=[0, 1], min_count=1, max_count=2),
            decision(),
        ])
        self.assertEqual(1, len(examples))
        self.assertEqual(1, stats["discarded_multi_select"])

    def test_missing_optional_state_fields_do_not_crash_feature_extraction(self):
        text = state_text(decision(current={}))
        self.assertIn("prompt=PRIORITY", text)

    def test_compiler_emits_chosen_index_correctly(self):
        examples, _ = compile_records([decision(selected=[1])])
        self.assertEqual(1, examples[0]["chosenIndex"])
        self.assertIn("Cast Grizzly Bears", examples[0]["optionTexts"][1])

    def test_cli_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "records.jsonl")
            out = os.path.join(tmp, "single_choice.jsonl")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(decision()) + "\n")
            stderr = sys.stderr
            try:
                sys.stderr = io.StringIO()
                self.assertEqual(0, main(["--input", source, "--out", out,
                                          "--single-choice-only"]))
                self.assertIn("compiled=1", sys.stderr.getvalue())
            finally:
                sys.stderr = stderr
            with open(out, "r", encoding="utf-8") as handle:
                row = json.loads(handle.readline())
            self.assertEqual(1, row["chosenIndex"])


if __name__ == "__main__":
    unittest.main()

