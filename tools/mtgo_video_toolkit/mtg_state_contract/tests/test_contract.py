import unittest

from mtg_state_contract import (
    CanonicalObject, CanonicalStateFormatter, ComparisonPolicy, compare_states,
    canonical_to_model_observation,
)


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.formatter = CanonicalStateFormatter()
        self.xmage = {
            "turnNumber": 3, "phase": "MAIN1", "step": "PRECOMBAT_MAIN",
            "activeSeat": "1", "prioritySeat": "1", "localSeat": "1",
            "players": [
                {"seat": 1, "life": 20, "handCount": 4, "libraryCount": 49},
                {"seat": 2, "life": 17, "handCount": 3, "libraryCount": 50},
            ],
            "battlefield": [
                {"name": "Llanowar Elves", "controllerSeat": 1,
                 "tapped": True, "power": 1, "toughness": 1},
            ],
        }

    def test_formatter_and_model_projection_share_one_contract(self):
        state = self.formatter.format(self.xmage, source="xmage")
        model = canonical_to_model_observation(state, perspective_seat=1)
        self.assertEqual(3, model["observation"]["current"]["turnNumber"])
        self.assertEqual("Llanowar Elves",
                         model["observation"]["current"]["zones"]
                              ["battlefield"][0]["name"])

    def test_matching_states_pass(self):
        left = self.formatter.format(self.xmage, source="xmage")
        right = self.formatter.format(self.xmage, source="mtgo-video")
        self.assertTrue(compare_states(left, right).passed)

    def test_unknown_video_field_is_not_false_mismatch(self):
        left = self.formatter.format(self.xmage, source="xmage")
        video = dict(self.xmage)
        video["players"] = [dict(row) for row in self.xmage["players"]]
        video["players"][1]["life"] = None
        right = self.formatter.format(video, source="mtgo-video")
        right.mark_unknown("/players/2/life")
        report = compare_states(left, right)
        row = next(item for item in report.items if item.path == "/players/2/life")
        self.assertEqual("UNKNOWN", row.status)
        self.assertTrue(report.passed)

    def test_real_difference_fails(self):
        left = self.formatter.format(self.xmage, source="xmage")
        video = dict(self.xmage)
        video["players"] = [dict(row) for row in self.xmage["players"]]
        video["players"][1]["life"] = 16
        right = self.formatter.format(video, source="mtgo-video")
        report = compare_states(left, right,
                                ComparisonPolicy(min_confidence=0.5))
        self.assertFalse(report.passed)

    def test_missing_player_is_serializable_and_fails(self):
        left = self.formatter.format(self.xmage, source="xmage")
        right_payload = dict(self.xmage)
        right_payload["players"] = [dict(self.xmage["players"][0])]
        right = self.formatter.format(right_payload, source="video")
        report = compare_states(left, right)
        row = next(item for item in report.items if item.path == "/players/2")
        self.assertEqual("UNKNOWN", row.status)
        self.assertIsInstance(row.expected, dict)

    def test_low_confidence_empty_zone_does_not_false_fail(self):
        left = self.formatter.format(self.xmage, source="xmage")
        right_payload = dict(self.xmage)
        right_payload["battlefield"] = []
        right = self.formatter.format(right_payload, source="video")
        right.confidence["/zones/battlefield/count"] = 0.2
        right.mark_unknown("/zones/battlefield/count")
        report = compare_states(left, right)
        self.assertTrue(report.passed)
        self.assertTrue(any(item.status == "UNKNOWN" and
                            item.path.startswith("/zones/battlefield")
                            for item in report.items))


if __name__ == "__main__":
    unittest.main()
