import unittest

from mtg_state_contract import CanonicalPlayer, CanonicalState
from xmage_state_follower.follower import (
    FollowConfig, ReplayBackend, ReplayManifest, ReplayResult, XmageFollower)
from xmage_state_follower.matcher import rank_options


class FakeBackend(ReplayBackend):
    def __init__(self):
        self.options = [
            {"index": 0, "type": "PASS_PRIORITY", "label": "Pass"},
            {"index": 1, "type": "CAST_SPELL", "label": "Cast Lightning Bolt",
             "payload": {"cardName": "Lightning Bolt"}},
        ]

    def run(self, selections):
        life = 20
        if selections and selections[-1] == [1]:
            life = 17
        response = {"ok": True, "finished": False,
                    "observation": {"select": {"option": self.options},
                                    "current": {}}}
        state = CanonicalState(
            source="xmage", timestamp_ms=1000,
            players=[CanonicalPlayer("1", life=20),
                     CanonicalPlayer("2", life=life)])
        return ReplayResult(response, state, [list(row) for row in selections])


class ImplicitPassBackend(ReplayBackend):
    def run(self, selections):
        if not selections:
            options = [{"index": 0, "type": "PASS_PRIORITY", "label": "Pass"}]
            life = 20
        elif selections == [[0]]:
            options = [
                {"index": 0, "type": "PASS_PRIORITY", "label": "Pass"},
                {"index": 1, "type": "CAST_SPELL",
                 "label": "Cast Lightning Bolt",
                 "payload": {"cardName": "Lightning Bolt"}},
            ]
            life = 20
        else:
            options = [{"index": 0, "type": "PASS_PRIORITY", "label": "Pass"}]
            life = 17 if selections[-1] == [1] else 20
        response = {"ok": True, "finished": False,
                    "observation": {"select": {"option": options},
                                    "current": {}}}
        state = CanonicalState(
            source="xmage", timestamp_ms=1000,
            players=[CanonicalPlayer("1", life=20),
                     CanonicalPlayer("2", life=life)])
        return ReplayResult(response, state, [list(row) for row in selections])


class FollowerTest(unittest.TestCase):
    def test_option_matcher_prefers_card_and_type(self):
        action = {"action_type": "CAST_SPELL",
                  "card_name": "Lightning Bolt"}
        options = FakeBackend().options
        ranking = rank_options(action, options)
        self.assertEqual(1, ranking[0].option_index)

    def test_follower_replays_and_validates(self):
        manifest = ReplayManifest([], [], seed=7)
        observed = CanonicalState(
            source="mtgo-video", timestamp_ms=1000,
            players=[CanonicalPlayer("1", life=20),
                     CanonicalPlayer("2", life=17)],
            confidence={"/players/1/life": 1.0,
                        "/players/2/life": 1.0})
        report = XmageFollower(
            FakeBackend(), manifest,
            FollowConfig(beam_width=2, minimum_action_score=30,
                         max_hard_mismatches=2)).follow([
                {"timestamp_ms": 1000, "action_type": "CAST_SPELL",
                 "card_name": "Lightning Bolt"}
            ], [observed])
        self.assertTrue(report.passed)
        self.assertTrue(report.verified)
        self.assertEqual([[1]], report.final_hypotheses[0]["selections"])

    def test_follower_can_insert_unobserved_priority_pass(self):
        manifest = ReplayManifest([], [], seed=7)
        observed = CanonicalState(
            source="mtgo-video", timestamp_ms=1000,
            players=[CanonicalPlayer("1", life=20),
                     CanonicalPlayer("2", life=17)],
            confidence={"/players/1/life": 1.0,
                        "/players/2/life": 1.0})
        report = XmageFollower(
            ImplicitPassBackend(), manifest,
            FollowConfig(beam_width=2, minimum_action_score=30,
                         maximum_implicit_steps=2,
                         max_hard_mismatches=0)).follow([
                {"timestamp_ms": 1000, "action_type": "CAST_SPELL",
                 "card_name": "Lightning Bolt"}
            ], [observed])
        self.assertTrue(report.verified)
        self.assertEqual([[0], [1]],
                         report.final_hypotheses[0]["selections"])

    def test_uncompared_replay_is_not_marked_verified(self):
        manifest = ReplayManifest([], [], seed=7)
        report = XmageFollower(
            FakeBackend(), manifest,
            FollowConfig(beam_width=2, minimum_action_score=30)).follow([
                {"timestamp_ms": 1000, "action_type": "CAST_SPELL",
                 "card_name": "Lightning Bolt"}
            ], [])
        self.assertTrue(report.passed)
        self.assertFalse(report.verified)

    def test_manifest_accepts_missing_or_short_deck_array(self):
        self.assertEqual([], ReplayManifest.from_dict({"decks": []}).deck0)
        self.assertEqual([], ReplayManifest.from_dict({"decks": []}).deck1)
        one = ReplayManifest.from_dict({"decks": [[{"name": "Forest", "count": 1}]]})
        self.assertEqual("Forest", one.deck0[0]["name"])
        self.assertEqual([], one.deck1)


if __name__ == "__main__":
    unittest.main()
