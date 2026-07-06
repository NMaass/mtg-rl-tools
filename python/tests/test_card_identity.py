"""Card-identity client helpers and cross-language response parsing.

Two layers, both runnable on a Python-only checkout:

- ``FakeBridge`` unit tests exercise ``resolve_card`` / ``validate_deck`` /
  ``repository_card_data`` against a canned responder — the exact request each
  helper sends and how it surfaces success and fail-closed errors — without
  launching the Java subprocess.
- fixture tests parse the ``validate_deck`` / ``repository_card_data`` responses
  the Java ``CardIdentityRepositoryTest`` emits (refreshed by
  scripts/run-cabt-adapter-tests.sh; checked-in copies keep this green
  standalone).
"""

import json
import os
import unittest

from magic_cabt import CabtBridge, CabtProtocolError

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class FakeBridge(CabtBridge):
    """A CabtBridge whose transport is a canned responder — no subprocess."""

    def __init__(self, responder):
        self._responder = responder
        self.sent = []
        self.finished = False
        self.result = None

    def request(self, request_dict):
        self.sent.append(request_dict)
        response = self._responder(request_dict)
        # mirror the real request()'s fail-closed contract
        if response.get("ok") is not True:
            raise CabtProtocolError(
                response.get("error", "UNKNOWN"), response.get("message", "")
            )
        return response


class ResolveCardHelperTest(unittest.TestCase):
    def test_sends_name_and_returns_resolution(self):
        resolution = {
            "requestedName": "Boseiju, Who Endures",
            "normalizedName": "Boseiju, Who Endures",
            "resolved": True,
            "strategy": "EXACT",
            "canonicalName": "Boseiju, Who Endures",
            "setCode": "NEO",
            "cardNumber": "266",
            "error": None,
            "reason": None,
        }
        bridge = FakeBridge(lambda req: {"ok": True, "resolution": resolution})

        result = bridge.resolve_card("Boseiju, Who Endures")

        self.assertEqual(result, resolution)
        self.assertEqual(
            bridge.sent,
            [{"command": "resolve_card", "name": "Boseiju, Who Endures"}],
        )

    def test_unresolved_name_comes_back_without_substitution(self):
        resolution = {
            "requestedName": "No Such Card",
            "normalizedName": "No Such Card",
            "resolved": False,
            "strategy": None,
            "canonicalName": None,
            "error": "UNKNOWN_CARD",
            "reason": 'no XMage card matches name "No Such Card"',
        }
        bridge = FakeBridge(lambda req: {"ok": True, "resolution": resolution})

        result = bridge.resolve_card("No Such Card")

        self.assertFalse(result["resolved"])
        self.assertIsNone(result["canonicalName"])
        self.assertEqual(result["error"], "UNKNOWN_CARD")


class ValidateDeckHelperTest(unittest.TestCase):
    def test_normalizes_decklist_text_before_sending(self):
        captured = {}

        def responder(req):
            captured["deck"] = req["deck"]
            return {"ok": True, "valid": True, "resolutions": [], "failures": []}

        bridge = FakeBridge(responder)
        bridge.validate_deck("24 Forest\n36 Grizzly Bears")

        self.assertEqual(
            captured["deck"],
            [
                {"name": "Forest", "count": 24},
                {"name": "Grizzly Bears", "count": 36},
            ],
        )

    def test_passes_entry_lists_through_and_returns_result(self):
        deck = [{"name": "Forest", "count": 4}]
        payload = {
            "ok": True,
            "valid": False,
            "resolutions": [{"requestedName": "Forest", "resolved": True}],
            "failures": [{"requestedName": "Bogus", "resolved": False,
                          "error": "UNKNOWN_CARD"}],
        }
        bridge = FakeBridge(lambda req: payload)

        result = bridge.validate_deck(deck)

        self.assertEqual(bridge.sent[0]["deck"], deck)
        self.assertFalse(result["valid"])
        self.assertEqual(result["failures"][0]["error"], "UNKNOWN_CARD")


class RepositoryCardDataHelperTest(unittest.TestCase):
    def test_returns_card_list(self):
        cards = [{"name": "Forest"}, {"name": "Lightning Bolt"}]
        bridge = FakeBridge(lambda req: {"ok": True, "cards": cards, "resolutions": []})

        result = bridge.repository_card_data(["Forest", "Lightning Bolt"])

        self.assertEqual(result, cards)
        self.assertEqual(
            bridge.sent[0],
            {"command": "repository_card_data", "names": ["Forest", "Lightning Bolt"]},
        )

    def test_fails_closed_on_unknown_name(self):
        bridge = FakeBridge(
            lambda req: {
                "ok": False,
                "error": "UNKNOWN_CARD",
                "message": "1 requested card name(s) did not resolve",
                "failures": [{"requestedName": "Bogus", "resolved": False}],
            }
        )

        with self.assertRaises(CabtProtocolError) as ctx:
            bridge.repository_card_data(["Forest", "Bogus"])
        self.assertEqual(ctx.exception.code, "UNKNOWN_CARD")


class FixtureParsingTest(unittest.TestCase):
    """Parse the Java-emitted card-identity responses."""

    def _load(self, name):
        with open(os.path.join(FIXTURES, name), "r") as handle:
            return json.load(handle)

    def test_validate_deck_response_shape(self):
        response = self._load("validate_deck_response.json")

        self.assertTrue(response["ok"])
        self.assertTrue(response["valid"])
        self.assertEqual(response["failures"], [])
        names = [entry["requestedName"] for entry in response["resolutions"]]
        for expected in ["Forest", "Lightning Bolt", "Llanowar Elves",
                         "Boseiju, Who Endures", "Fire // Ice"]:
            self.assertIn(expected, names)
        for entry in response["resolutions"]:
            self.assertTrue(entry["resolved"])
            self.assertIsNotNone(entry["canonicalName"])
            self.assertIn(entry["strategy"], ("EXACT", "NORMALIZED", "CLASS_HEURISTIC"))
            self.assertIn("count", entry)

    def test_repository_card_data_response_shape(self):
        response = self._load("repository_card_data_response.json")

        self.assertTrue(response["ok"])
        self.assertGreater(len(response["cards"]), 0)
        for card in response["cards"]:
            self.assertIn("name", card)
            self.assertIn("types", card)
        canonical = {r["canonicalName"] for r in response["resolutions"]}
        self.assertIn("Forest", canonical)
        self.assertIn("Boseiju, Who Endures", canonical)


if __name__ == "__main__":
    unittest.main()
