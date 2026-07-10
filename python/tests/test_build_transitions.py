import json
import os
import tempfile
import unittest

from magic_cabt.training.build_transitions import (
    main,
    transitions_from_decisions,
    transitions_from_states,
)


def state(seq, match_id="m1", game=1):
    return {"matchId": match_id, "gameNumber": game, "gameInstance": 1,
            "seq": seq, "zones": {"battlefield": []}}


def decision(seq, match_id="m1", game=1, selected=None):
    return {
        "matchId": match_id,
        "gameNumber": game,
        "selectedIndices": selected if selected is not None else [0],
        "select": {"type": "PRIORITY",
                   "option": [{"index": 0, "type": "PASS_PRIORITY"}]},
        "observation": {"current": {"gameInstance": 1, "seq": seq}},
    }


class TransitionsFromStatesTest(unittest.TestCase):
    def test_pairs_consecutive_states_within_a_game(self):
        transitions = list(transitions_from_states(
            [state(1), state(2), state(3)]))
        self.assertEqual(2, len(transitions))
        self.assertEqual(1, transitions[0]["prev"]["seq"])
        self.assertEqual(2, transitions[0]["next"]["seq"])
        self.assertIsNone(transitions[0]["action"])

    def test_never_pairs_across_games_or_matches(self):
        transitions = list(transitions_from_states([
            state(1, game=1), state(2, game=2),
            state(3, match_id="m2", game=2),
        ]))
        self.assertEqual([], transitions)


class TransitionsFromDecisionsTest(unittest.TestCase):
    def test_keeps_the_action_between_observations(self):
        transitions = list(transitions_from_decisions(
            [decision(1), decision(2)]))
        self.assertEqual(1, len(transitions))
        self.assertEqual([0], transitions[0]["action"]["selectedIndices"])
        self.assertEqual("PRIORITY", transitions[0]["action"]["promptType"])
        self.assertEqual(1, transitions[0]["action"]["optionCount"])


class CliTest(unittest.TestCase):
    def test_writes_transitions_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            states_path = os.path.join(tmp, "mirror_states.jsonl")
            with open(states_path, "w", encoding="utf-8") as handle:
                for entry in (state(1), state(2), state(3)):
                    handle.write(json.dumps(entry) + "\n")
            out = os.path.join(tmp, "transitions.jsonl")
            self.assertEqual(0, main(["--input", states_path, "--out", out]))
            with open(out, "r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(2, len(rows))
            self.assertEqual("mirror_states", rows[0]["source"])


if __name__ == "__main__":
    unittest.main()
