import unittest

from magic_cabt.training.macro_actions import (
    classify_decision,
    iter_macro_actions,
    macro_transition,
)


def option(index, option_type, label, key=None):
    payload = {}
    if key is not None:
        payload["canonicalKey"] = key
    return {
        "index": index,
        "type": option_type,
        "label": label,
        "payload": payload,
    }


def record(sequence, prompt, selected, options, player=0, game="g1",
           life=20, terminal=False, next_observation=None):
    return {
        "schemaVersion": 1,
        "source": "engine_selfplay",
        "gameId": game,
        "sequenceNumber": sequence,
        "playerIndex": player,
        "observation": {
            "current": {
                "gameInstance": 1,
                "turnNumber": 2,
                "players": [
                    {"playerIndex": 0, "life": life},
                    {"playerIndex": 1, "life": 20},
                ],
            },
        },
        "select": {
            "type": prompt,
            "playerIndex": player,
            "minCount": 1,
            "maxCount": 1,
            "option": options,
        },
        "selectedIndices": selected,
        "nextObservation": next_observation,
        "terminal": terminal,
        "reward": 1.0 if terminal else None,
        "result": {"winner": player} if terminal else None,
        "metadata": {"captureConfidence": "exact"},
    }


class ClassificationTest(unittest.TestCase):
    def test_known_prompt_families(self):
        root = record(1, "PRIORITY", [0], [option(0, "ACTION", "cast")])
        target = record(2, "CHOOSE_TARGET", [0], [option(0, "TARGET", "opponent")])
        payment = record(3, "PLAY_MANA", [0], [option(0, "PAY", "R")])
        self.assertEqual("root", classify_decision(root)["role"])
        self.assertEqual("parameter", classify_decision(target)["role"])
        self.assertEqual("payment", classify_decision(payment)["role"])

    def test_unknown_prompt_starts_new_root(self):
        value = classify_decision(record(
            1, "SOMETHING_NEW", [0], [option(0, "X", "new")]))
        self.assertEqual("root", value["role"])
        self.assertFalse(value["attachable"])
        self.assertEqual("heuristic", value["confidence"])

    def test_single_option_is_marked_deterministic(self):
        value = classify_decision(record(
            1, "PRIORITY", [0], [option(0, "PASS", "pass")]))
        self.assertTrue(value["deterministic"])
        self.assertTrue(value["passLike"])


class GroupingTest(unittest.TestCase):
    def test_groups_root_target_and_payment(self):
        records = [
            record(1, "PRIORITY", [0], [
                option(0, "ACTION", "cast Lightning Bolt", "cast:bolt"),
                option(1, "PASS", "pass", "pass"),
            ]),
            record(2, "CHOOSE_TARGET", [0], [
                option(0, "TARGET", "opponent", "player:1"),
                option(1, "TARGET", "creature", "card:bear"),
            ]),
            record(3, "PLAY_MANA", [0], [
                option(0, "PAY", "Mountain", "mana:red"),
            ]),
            record(4, "PRIORITY", [1], [
                option(0, "ACTION", "cast response", "cast:response"),
                option(1, "PASS", "pass", "pass"),
            ], player=1),
        ]
        groups = list(iter_macro_actions(records))
        self.assertEqual(2, len(groups))
        first = groups[0]
        self.assertEqual(3, first["decisionCount"])
        self.assertTrue(first["complete"])
        self.assertEqual("player-boundary", first["completionReason"])
        self.assertEqual("cast Lightning Bolt", first["macroAction"]["label"])
        self.assertEqual(2, len(first["macroAction"]["parameters"]))
        self.assertEqual("player:1", first["macroAction"]["parameters"][0]
                         ["selectedOptions"][0]["payload"]["canonicalKey"])
        self.assertEqual(records[3]["observation"], first["nextObservation"])

    def test_never_groups_across_games(self):
        groups = list(iter_macro_actions([
            record(1, "PRIORITY", [0], [option(0, "ACTION", "cast")], game="g1"),
            record(2, "CHOOSE_TARGET", [0], [option(0, "TARGET", "target")], game="g2"),
        ]))
        self.assertEqual(2, len(groups))
        self.assertFalse(groups[0]["complete"])
        self.assertEqual("game-boundary", groups[0]["completionReason"])
        self.assertTrue(groups[1]["orphanedParameterGroup"])

    def test_terminal_group_is_complete_without_boundary_observation(self):
        groups = list(iter_macro_actions([
            record(1, "PRIORITY", [0], [option(0, "ACTION", "win")],
                   terminal=True),
        ]))
        self.assertEqual(1, len(groups))
        self.assertTrue(groups[0]["complete"])
        self.assertEqual("terminal", groups[0]["completionReason"])
        self.assertEqual(1.0, groups[0]["reward"])

    def test_end_of_stream_group_is_explicitly_incomplete(self):
        groups = list(iter_macro_actions([
            record(1, "PRIORITY", [0], [option(0, "ACTION", "cast")]),
        ]))
        self.assertFalse(groups[0]["complete"])
        self.assertEqual("end-of-stream", groups[0]["completionReason"])

    def test_explicit_next_observation_completes_group(self):
        following = {"current": {"players": [
            {"playerIndex": 0, "life": 17},
            {"playerIndex": 1, "life": 20},
        ]}}
        groups = list(iter_macro_actions([
            record(1, "PRIORITY", [0], [option(0, "ACTION", "shock")],
                   next_observation=following),
        ]))
        self.assertTrue(groups[0]["complete"])
        transition = macro_transition(groups[0])
        self.assertIsNotNone(transition)
        self.assertEqual(-3, transition["deltas"]["lifeDelta"]["0"])


if __name__ == "__main__":
    unittest.main()
