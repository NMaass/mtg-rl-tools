import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.rl import (
    InvalidActionError,
    MagicCabtEnv,
    action_mask,
    terminal_reward,
)


OPTIONS = [
    {"index": 0, "type": "PASS_PRIORITY", "label": "Pass priority"},
    {"index": 1, "type": "CAST_SPELL", "label": "Cast spell"},
]


def observation(min_count=1, max_count=1, player=0):
    return {
        "current": {"turnNumber": 1},
        "select": {
            "type": "PRIORITY",
            "playerIndex": player,
            "minCount": min_count,
            "maxCount": max_count,
            "option": OPTIONS,
        },
    }


class FakeBridge(object):

    def __init__(self, finish_on_select=True, winner="Player P0 is the winner"):
        self.finish_on_select = finish_on_select
        self.finished = False
        self.result = None
        self.started = False
        self.selections = []
        self.winner = winner

    def game_start(self, deck0, deck1, player_names=None, seed=None, max_turns=None):
        self.started = True
        self.finished = False
        self.result = None
        return {"observation": observation(player=0), "sequence": 0}

    def game_select(self, selection):
        self.selections.append(selection)
        if self.finish_on_select:
            self.finished = True
            self.result = {"winner": self.winner, "finalState": {"turnNumber": 1}}
            return {"finished": True, "result": self.result}
        return {"observation": observation(player=1), "sequence": 1}

    def close(self):
        self.closed = True


class MaskedRlEnvTest(unittest.TestCase):

    def test_action_mask_dynamic_and_padded(self):
        self.assertEqual([True, True], action_mask(observation()))
        self.assertEqual([True, True, False, False],
                         action_mask(observation(), max_actions=4))
        with self.assertRaises(ValueError):
            action_mask(observation(), max_actions=1)

    def test_reset_returns_observation_and_mask_info(self):
        bridge = FakeBridge(finish_on_select=False)
        env = MagicCabtEnv([], [], bridge=bridge)
        obs, info = env.reset(seed=7)
        self.assertTrue(bridge.started)
        self.assertEqual("PRIORITY", obs["select"]["type"])
        self.assertEqual([True, True], info["action_mask"])
        self.assertEqual(2, info["legalActionCount"])

    def test_strict_illegal_action_fails_before_bridge_select(self):
        bridge = FakeBridge()
        env = MagicCabtEnv([], [], bridge=bridge)
        env.reset()
        with self.assertRaises(InvalidActionError):
            env.step(99)
        self.assertEqual([], bridge.selections)

    def test_repair_mode_clamps_and_marks_selection(self):
        bridge = FakeBridge()
        env = MagicCabtEnv([], [], bridge=bridge,
                           illegal_action_mode="repair-and-mark")
        env.reset()
        obs, reward, terminated, truncated, info = env.step(99)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual([[0]], bridge.selections)
        self.assertTrue(info["selectionRepaired"])
        self.assertFalse(info["selectionWasLegal"])
        self.assertEqual(1.0, reward)
        self.assertIn("current", obs)

    def test_terminal_reward_handles_winner_string_and_draw(self):
        self.assertEqual(1.0, terminal_reward(
            {"winner": "Player P1 is the winner"}, 1, ("P0", "P1")))
        self.assertEqual(-1.0, terminal_reward(
            {"winner": "Player P1 is the winner"}, 0, ("P0", "P1")))
        self.assertEqual(0.0, terminal_reward({"winner": None}, 0, ("P0", "P1")))


if __name__ == "__main__":
    unittest.main()
