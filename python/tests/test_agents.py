"""Agent baselines return legal selections and rank options sanely.

No XMage needed: the agents operate purely on observation dicts, so these
tests build representative prompts by hand and check the contract every agent
must honour -- never an illegal index, count inside ``[minCount, maxCount]``,
distinct entries -- plus the legal-selection envelope.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.agents import (  # noqa: E402
    RandomAgent,
    FirstAgent,
    available_agents,
    clamp_selection,
    is_legal_selection,
    make_agent,
)


def option(index, option_type, label=None):
    return {"index": index, "type": option_type, "label": label or option_type,
            "payload": {}}


def select(select_type, options, min_count=1, max_count=1, player_index=0):
    return {"select": {
        "playerIndex": player_index,
        "type": select_type,
        "minCount": min_count,
        "maxCount": max_count,
        "option": options,
    }}


# Representative prompts covering the select types the agents special-case.
PRIORITY = select("PRIORITY", [
    option(0, "PASS_PRIORITY", "Pass priority"),
    option(1, "PLAY_LAND", "Play Mountain"),
    option(2, "CAST_SPELL", "Cast Grizzly Bears"),
])
MULLIGAN = select("MULLIGAN", [
    option(0, "PROMPT_KEEP", "Keep"),
    option(1, "PROMPT_MULLIGAN", "Mulligan"),
])
PAY_MANA = select("PAY_MANA", [
    option(0, "PROMPT_CANCEL_PAYMENT", "Cancel"),
    option(1, "PROMPT_MANA_SOURCE", "Tap Forest"),
])
ATTACKERS = select("DECLARE_ATTACKERS", [
    option(0, "PROMPT_ATTACKER", "Attack with Bears"),
    option(1, "PROMPT_ATTACKER", "Attack with Elephant"),
], min_count=0, max_count=0)
BLOCKERS = select("DECLARE_BLOCKERS", [
    option(0, "PROMPT_BLOCKER", "Block with Wall"),
], min_count=0, max_count=0)
YES_NO = select("YES_NO", [
    option(0, "PROMPT_YES", "Yes"),
    option(1, "PROMPT_NO", "No"),
])
MULTI = select("MODE", [
    option(0, "PROMPT_MODE", "Mode A"),
    option(1, "PROMPT_MODE", "Mode B"),
    option(2, "PROMPT_MODE", "Mode C"),
], min_count=2, max_count=2)
EMPTY = select("PRIORITY", [], min_count=0, max_count=0)

ALL_PROMPTS = [PRIORITY, MULLIGAN, PAY_MANA, ATTACKERS, BLOCKERS, YES_NO,
               MULTI, EMPTY]


class LegalityTest(unittest.TestCase):
    def test_every_agent_returns_legal_selections(self):
        for spec in ("random", "first"):
            for trial in range(20):  # random agent gets many draws
                agent = make_agent(spec, seed=trial)
                for prompt in ALL_PROMPTS:
                    selection = agent.select(prompt)
                    self.assertTrue(
                        is_legal_selection(selection, prompt["select"]),
                        "%s produced illegal selection %r for %s"
                        % (spec, selection, prompt["select"]["type"]))

    def test_agents_never_touch_empty_option_prompt(self):
        for spec in ("random", "first"):
            agent = make_agent(spec, seed=0)
            self.assertEqual(agent.select(EMPTY), [])


class FirstAgentTest(unittest.TestCase):
    def test_takes_lowest_indexed_option(self):
        self.assertEqual(FirstAgent().select(PRIORITY), [0])

    def test_score_prefers_earlier_options(self):
        scores = FirstAgent().score(PRIORITY)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])


class RandomAgentTest(unittest.TestCase):
    def test_is_reproducible_with_seed(self):
        first = [RandomAgent(seed=7).select(PRIORITY) for _ in range(5)]
        second = [RandomAgent(seed=7).select(PRIORITY) for _ in range(5)]
        self.assertEqual(first, second)

    def test_respects_count_bounds(self):
        prompt = select("MODE", [option(i, "PROMPT_MODE") for i in range(5)],
                        min_count=2, max_count=3)
        for trial in range(50):
            selection = RandomAgent(seed=trial).select(prompt)
            self.assertTrue(2 <= len(selection) <= 3)
            self.assertTrue(is_legal_selection(selection, prompt["select"]))


class HelpersTest(unittest.TestCase):
    def test_clamp_repairs_out_of_range_and_duplicates(self):
        prompt = PRIORITY["select"]
        self.assertEqual(clamp_selection([99, 1, 1, -3], prompt), [1])

    def test_clamp_pads_up_to_min_count(self):
        self.assertEqual(clamp_selection([], MULTI["select"]), [0, 1])

    def test_clamp_trims_to_max_count(self):
        prompt = select("MODE", [option(i, "PROMPT_MODE") for i in range(4)],
                        min_count=0, max_count=2)["select"]
        self.assertEqual(clamp_selection([0, 1, 2, 3], prompt), [0, 1])

    def test_is_legal_rejects_bool_and_non_list(self):
        self.assertFalse(is_legal_selection([True], PRIORITY["select"]))
        self.assertFalse(is_legal_selection("0", PRIORITY["select"]))

    def test_make_agent_unknown_raises(self):
        with self.assertRaises(ValueError):
            make_agent("nonesuch")

    def test_available_agents_lists_builtins(self):
        self.assertEqual(available_agents(), ["first", "random"])


if __name__ == "__main__":
    unittest.main()