"""Live protocol smoke test: Python drives a real XMage game end to end.

Needs a built Java bridge: set ``MAGIC_CABT_CLASSPATH`` to a classpath
containing the bridge classes and XMage (scripts/run-cabt-adapter-tests.sh
builds and exports it, then runs this file). Without the variable the tests
skip, so plain ``unittest discover`` stays green on a Python-only checkout.
"""

import os
import unittest

from magic_cabt import CabtBridge, CabtProtocolError
from magic_cabt.protocol import CLASSPATH_ENV_VAR

DECK = [{"name": "Forest", "count": 24}, {"name": "Grizzly Bears", "count": 36}]

PREFERRED = {
    "PRIORITY": ["PLAY_LAND", "CAST_SPELL", "PASS_PRIORITY"],
    "MULLIGAN": ["PROMPT_KEEP"],
    "PAY_MANA": ["PROMPT_MANA_SOURCE", "PROMPT_MANA_POOL", "PROMPT_CANCEL_PAYMENT"],
}


def greedy_choice(select):
    """Same greedy policy as the Java smoke games, from the JSON select."""
    if select["type"] in ("DECLARE_ATTACKERS", "DECLARE_BLOCKERS"):
        return []
    options = select["option"]
    for want in PREFERRED.get(select["type"], []):
        for option in options:
            if option["type"] == want:
                return [option["index"]]
    return list(range(select["minCount"]))


@unittest.skipUnless(
    os.environ.get(CLASSPATH_ENV_VAR),
    "set $%s to run live protocol tests" % CLASSPATH_ENV_VAR,
)
class ProtocolLiveTest(unittest.TestCase):
    def setUp(self):
        self.bridge = CabtBridge()
        self.addCleanup(self.bridge.close)

    def test_ping_and_capabilities(self):
        self.assertTrue(self.bridge.ping()["pong"])
        capabilities = self.bridge.capabilities()
        self.assertIn("game_start", capabilities["commands"])

    def test_full_game_with_greedy_agent(self):
        response = self.bridge.game_start(
            DECK, DECK, player_names=["Alice", "Bob"], seed=20260704, max_turns=4
        )
        select_types = []
        steps = 0
        while not self.bridge.finished:
            self.assertLess(steps, 2000, "runaway game loop")
            steps += 1
            observation = response["observation"]
            select = observation["select"]
            select_types.append(select["type"])

            # hidden information: only the selecting player's hand is visible
            for player in observation["current"]["players"]:
                if player["playerIndex"] != select["playerIndex"]:
                    self.assertEqual(
                        player["hand"], [], "opponent hand leaked: %r" % player
                    )

            response = self.bridge.game_select(greedy_choice(select))

        self.assertIn("PRIORITY", select_types)
        self.assertIn("MULLIGAN", select_types)
        self.assertIsNotNone(self.bridge.result)
        self.assertIn("finalState", self.bridge.result)

        # the greedy policy really played lands/creatures onto the battlefield
        battlefield = self.bridge.result["finalState"]["battlefield"]
        self.assertGreater(len(battlefield), 0)

    def test_card_data_visualization_and_errors(self):
        with self.assertRaises(CabtProtocolError) as ctx:
            self.bridge.all_card_data()
        self.assertEqual(ctx.exception.code, "NO_ACTIVE_GAME")

        self.bridge.game_start(DECK, DECK, seed=1, max_turns=3)

        cards = self.bridge.all_card_data()
        names = sorted(card["name"] for card in cards)
        self.assertEqual(names, ["Forest", "Grizzly Bears"])

        text = self.bridge.visualize_data()
        self.assertIn("turn", text)

        with self.assertRaises(CabtProtocolError) as ctx:
            self.bridge.game_select([99999])
        self.assertEqual(ctx.exception.code, "OPTION_INDEX_OUT_OF_RANGE")

        # the game survived the invalid selection
        select = self.bridge.request({"command": "visualize_data"})
        self.assertTrue(select["ok"])
        self.bridge.game_finish()


if __name__ == "__main__":
    unittest.main()
