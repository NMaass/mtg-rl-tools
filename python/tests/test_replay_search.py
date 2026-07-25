import unittest

from magic_cabt.search.replay_search import (
    ReplayDivergenceError,
    branch_replay,
    branch_to_transition,
    candidate_selections,
    observation_signature,
    replay_to_root,
)


def select(prompt, player, labels):
    return {
        "type": prompt,
        "playerIndex": player,
        "minCount": 1,
        "maxCount": 1,
        "option": [
            {
                "index": index,
                "type": "ACTION",
                "label": label,
                "payload": {"canonicalKey": label.lower()},
            }
            for index, label in enumerate(labels)
        ],
    }


def response(turn, prompt, player, labels, life0=20, life1=20):
    return {
        "ok": True,
        "finished": False,
        "observation": {
            "current": {
                "gameInstance": 999,
                "turnNumber": turn,
                "phase": "MAIN1",
                "players": [
                    {"playerIndex": 0, "life": life0},
                    {"playerIndex": 1, "life": life1},
                ],
            },
            "select": select(prompt, player, labels),
        },
    }


class MockBridge(object):
    def __init__(self):
        self.finished = False
        self.result = None
        self.path = []
        self.closed = False

    def game_start(self, deck0, deck1, player_names=None, seed=None,
                   max_turns=None):
        self.path = []
        self.finished = False
        self.result = None
        return response(1, "PRIORITY", 0, ["Land", "Pass"])

    def game_select(self, selection):
        self.path.append(list(selection))
        if len(self.path) == 1:
            return response(1, "PRIORITY", 1, ["Pass"])
        if len(self.path) == 2:
            # This is the search root after prefix [[0], [0]].
            return response(2, "PRIORITY", 0, ["Attack", "Hold"])
        if len(self.path) == 3:
            if selection == [0]:
                return response(2, "PRIORITY", 1, ["Block"], life1=17)
            return response(2, "PRIORITY", 1, ["Pass"], life1=20)
        self.finished = True
        self.result = {"winner": "Player P0 is the winner"}
        return {"ok": True, "finished": True, "result": self.result}

    def close(self):
        self.closed = True


class CandidateSelectionTest(unittest.TestCase):
    def test_enumerates_single_choice_options(self):
        self.assertEqual([[0], [1]], candidate_selections(
            select("PRIORITY", 0, ["A", "B"])))

    def test_rejects_automatic_multiselect(self):
        value = select("ATTACKERS", 0, ["A", "B"])
        value["maxCount"] = 2
        with self.assertRaises(ValueError):
            candidate_selections(value)


class SignatureTest(unittest.TestCase):
    def test_ignores_volatile_ids_and_unordered_battlefield(self):
        a = response(2, "PRIORITY", 0, ["Attack", "Hold"])["observation"]
        b = response(2, "PRIORITY", 0, ["Attack", "Hold"])["observation"]
        a["current"]["objectId"] = "11111111-1111-1111-1111-111111111111"
        b["current"]["objectId"] = "22222222-2222-2222-2222-222222222222"
        self.assertEqual(
            observation_signature(a)["sha256"],
            observation_signature(b)["sha256"],
        )

    def test_preserves_active_and_priority_player_semantics_across_uuid_changes(self):
        a = response(2, "PRIORITY", 0, ["Attack", "Hold"])["observation"]
        b = response(2, "PRIORITY", 0, ["Attack", "Hold"])["observation"]
        a["current"]["players"][0]["playerId"] = "11111111-1111-1111-1111-111111111111"
        a["current"]["players"][1]["playerId"] = "22222222-2222-2222-2222-222222222222"
        b["current"]["players"][0]["playerId"] = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        b["current"]["players"][1]["playerId"] = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        a["current"]["activePlayerId"] = a["current"]["players"][0]["playerId"]
        a["current"]["priorityPlayerId"] = a["current"]["players"][1]["playerId"]
        b["current"]["activePlayerId"] = b["current"]["players"][0]["playerId"]
        b["current"]["priorityPlayerId"] = b["current"]["players"][1]["playerId"]
        self.assertEqual(
            observation_signature(a)["sha256"],
            observation_signature(b)["sha256"],
        )
        b["current"]["priorityPlayerId"] = b["current"]["players"][0]["playerId"]
        self.assertNotEqual(
            observation_signature(a)["sha256"],
            observation_signature(b)["sha256"],
        )


class ReplayTest(unittest.TestCase):
    def test_reconstructs_and_verifies_root(self):
        expected = response(2, "PRIORITY", 0, ["Attack", "Hold"])
        bridge = MockBridge()
        actual = replay_to_root(
            bridge, [], [], [[0], [0]], seed=3,
            expected_observation=expected["observation"],
            expected_select=expected["observation"]["select"],
        )
        self.assertEqual("PRIORITY", actual["observation"]["select"]["type"])

    def test_divergence_fails_closed(self):
        expected = response(2, "PRIORITY", 0, ["Different"])
        with self.assertRaises(ReplayDivergenceError):
            replay_to_root(
                MockBridge(), [], [], [[0], [0]], seed=3,
                expected_observation=expected["observation"],
                expected_select=expected["observation"]["select"],
            )

    def test_branches_every_option_in_fresh_bridge(self):
        expected = response(2, "PRIORITY", 0, ["Attack", "Hold"])
        created = []

        def factory():
            bridge = MockBridge()
            created.append(bridge)
            return bridge

        result = branch_replay(
            factory, [], [], [[0], [0]],
            expected["observation"], expected["observation"]["select"],
            seed=3, root_player_index=0,
        )
        self.assertEqual(2, len(result["branches"]))
        self.assertEqual([0], result["branches"][0]["selectedIndices"])
        self.assertEqual(17, result["branches"][0]["nextObservation"]
                         ["current"]["players"][1]["life"])
        self.assertEqual(20, result["branches"][1]["nextObservation"]
                         ["current"]["players"][1]["life"])
        self.assertTrue(all(bridge.closed for bridge in created))

        transition = branch_to_transition(result, result["branches"][0])
        self.assertEqual("search", transition["source"])
        self.assertEqual("attack", transition["action"]["selectedOptions"][0]
                         ["payload"]["canonicalKey"])

    def test_optional_rollout_produces_terminal_reward(self):
        expected = response(2, "PRIORITY", 0, ["Attack", "Hold"])
        result = branch_replay(
            MockBridge, [], [], [[0], [0]],
            expected["observation"], expected["observation"]["select"],
            seed=3, root_player_index=0,
            candidates=[[0]], max_rollout_decisions=1,
        )
        branch = result["branches"][0]
        self.assertTrue(branch["finished"])
        self.assertEqual(1.0, branch["reward"])


if __name__ == "__main__":
    unittest.main()
