import json
import os
import tempfile
import unittest

from magic_cabt.training.train_jepa import (_bundle_outcome, _compile_decision,
                                            _decision_transitions,
                                            _horizon_transitions, _reservoir,
                                            collect_training_data)


def options():
    return [
        {"index": 0, "type": "PASS_PRIORITY", "label": "Pass",
         "payload": {"canonicalKey": "pass"}},
        {"index": 1, "type": "CAST_SPELL", "label": "Cast Shock A",
         "payload": {"canonicalKey": "cast-shock"}},
        {"index": 2, "type": "CAST_SPELL", "label": "Cast Shock B",
         "payload": {"canonicalKey": "cast-shock"}},
    ]


def decision(sequence, chosen=0, game_id="game-1", match_id="m-1",
             game_number=1):
    return {
        "schemaVersion": 1,
        "gameId": game_id,
        "matchId": match_id,
        "gameNumber": game_number,
        "sequenceNumber": sequence,
        "selectedIndices": [chosen],
        "select": {"type": "PRIORITY", "minCount": 1, "maxCount": 1,
                   "option": options()},
        "observation": {
            "current": {"turnNumber": 1 + sequence // 3, "phase": "MAIN1",
                        "matchId": match_id, "gameNumber": game_number,
                        "players": [
                            {"seat": 1, "life": 20 - sequence,
                             "handCount": 5},
                            {"seat": 2, "life": 20, "handCount": 6}]},
        },
    }


def mirror_state(sequence, match_id="m-1", game_number=1):
    return {"matchId": match_id, "gameNumber": game_number, "seq": sequence,
            "turnNumber": 1 + sequence // 3,
            "players": [{"seat": 1, "life": 20 - sequence},
                        {"seat": 2, "life": 20}]}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class CompileDecisionTest(unittest.TestCase):
    def test_fungible_choices_share_a_group(self):
        first = _compile_decision(decision(0, chosen=1))
        second = _compile_decision(decision(0, chosen=2))
        self.assertEqual(first["_chosenGroup"], second["_chosenGroup"])
        self.assertNotEqual(first["_chosenGroup"],
                            _compile_decision(decision(0, chosen=0))
                            ["_chosenGroup"])

    def test_multi_select_and_invalid_index_skipped(self):
        multi = decision(0)
        multi["selectedIndices"] = [0, 1]
        self.assertIsNone(_compile_decision(multi))
        bad = decision(0)
        bad["selectedIndices"] = [9]
        self.assertIsNone(_compile_decision(bad))


class TransitionTest(unittest.TestCase):
    def test_horizons_stay_inside_one_game(self):
        states = [mirror_state(index) for index in range(6)] + \
            [mirror_state(index, match_id="m-2") for index in range(3)]
        transitions = list(_horizon_transitions(states))
        self.assertTrue(transitions)
        for item in transitions:
            self.assertEqual(item["prev"]["matchId"], item["next"]["matchId"])
        horizons = {item["horizon"] for item in transitions}
        self.assertIn(1, horizons)
        self.assertIn(4, horizons)
        self.assertNotIn(16, horizons)

    def test_decision_transitions_carry_actions(self):
        records = [decision(index, chosen=1) for index in range(3)]
        transitions = list(_decision_transitions(records, outcome=1.0))
        self.assertEqual(2, len(transitions))
        for item in transitions:
            self.assertEqual(1.0, item["outcome"])
            self.assertEqual("CAST_SPELL",
                             item["action"]["selectedOption"]["type"])


class BundleOutcomeTest(unittest.TestCase):
    def run_outcome(self, summary):
        with tempfile.TemporaryDirectory() as scratch:
            with open(os.path.join(scratch, "summary.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(summary, handle)
            return _bundle_outcome(scratch)

    def test_single_match_win_and_loss(self):
        self.assertEqual(1.0, self.run_outcome(
            {"matchIds": ["m-1"], "result": "win"}))
        self.assertEqual(-1.0, self.run_outcome(
            {"matchIds": ["m-1"], "result": "loss"}))

    def test_ambiguous_bundles_get_no_label(self):
        self.assertIsNone(self.run_outcome(
            {"matchIds": ["m-1", "m-2"], "result": "win"}))
        self.assertIsNone(self.run_outcome({"matchIds": ["m-1"]}))


class CollectTrainingDataTest(unittest.TestCase):
    def test_collects_from_bundle_directory(self):
        with tempfile.TemporaryDirectory() as bundle:
            write_jsonl(os.path.join(bundle, "decisions.jsonl"),
                        [decision(index, chosen=index % 3)
                         for index in range(6)])
            write_jsonl(os.path.join(bundle, "mirror_states.jsonl"),
                        [mirror_state(index) for index in range(6)])
            with open(os.path.join(bundle, "summary.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"matchIds": ["m-1"], "result": "win"}, handle)
            with open(os.path.join(bundle, "card_cache.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"123": {"grpId": 123, "name": "Shock"}}, handle)
            transitions, decisions, cards = collect_training_data([bundle])
        self.assertTrue(transitions)
        self.assertTrue(decisions)
        self.assertTrue(all(item["outcome"] == 1.0 for item in transitions))
        self.assertTrue(all("_groups" in item for item in decisions))
        self.assertEqual("Shock", cards["123"]["name"])


class ReservoirTest(unittest.TestCase):
    def test_bounded_and_unbounded(self):
        self.assertEqual(5, len(_reservoir(iter(range(100)), 5, seed=0)))
        self.assertEqual(list(range(3)), _reservoir(iter(range(3)), 10, seed=0))
        self.assertEqual(list(range(3)), _reservoir(iter(range(3)), 0, seed=0))


if __name__ == "__main__":
    unittest.main()
