"""Task 21: all_card_data() parses the Java exporter's protocol response.

The fixture is real Java output: MagicCardDataExporterTest regenerates
target/cabt-fixtures/card_data_response.json on every run, and the checked-in
copy under fixtures/ is taken from there.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt import all_card_data, cards_by_id, cards_by_name

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "card_data_response.json")


class AllCardDataTest(unittest.TestCase):

    def load_cards(self):
        with open(FIXTURE, "r", encoding="utf-8") as handle:
            return all_card_data(handle)

    def test_parses_basic_creature_metadata(self):
        cards = self.load_cards()
        bears = cards_by_name(cards)["Grizzly Bears"]
        self.assertEqual(bears["manaCost"], "{1}{G}")
        self.assertEqual(bears["manaValue"], 2)
        self.assertEqual(bears["types"], ["Creature"])
        self.assertEqual(bears["subtypes"], ["Bear"])
        self.assertEqual(bears["power"], "2")
        self.assertEqual(bears["toughness"], "2")
        self.assertEqual(bears["colors"], ["G"])

    def test_parses_ability_text(self):
        cards = self.load_cards()
        elves = cards_by_name(cards)["Llanowar Elves"]
        rules = [ability["rule"] for ability in elves["abilities"]]
        self.assertTrue(any("Add {G}" in rule for rule in rules))

    def test_cards_join_by_id_and_name(self):
        cards = self.load_cards()
        by_id = cards_by_id(cards)
        by_name = cards_by_name(cards)
        for card in cards:
            self.assertIs(by_id[card["cardId"]], card)
            self.assertIn(card["name"], by_name)

    def test_not_ok_response_raises(self):
        with self.assertRaises(ValueError):
            all_card_data('{"ok": false, "error": "boom"}')

    def test_missing_cards_list_raises(self):
        with self.assertRaises(ValueError):
            all_card_data('{"ok": true}')


if __name__ == "__main__":
    unittest.main()
