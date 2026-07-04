import unittest

from magic_cabt import parse_decklist


class ParseDecklistTest(unittest.TestCase):
    def test_counts_names_comments_and_blanks(self):
        entries = parse_decklist(
            """
            # a comment line
            24 Forest

            36 Grizzly Bears  # trailing comment
            Llanowar Elves
            """
        )
        self.assertEqual(
            entries,
            [
                {"name": "Forest", "count": 24},
                {"name": "Grizzly Bears", "count": 36},
                {"name": "Llanowar Elves", "count": 1},
            ],
        )

    def test_empty_decklist_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_decklist("# only comments\n\n")

    def test_leading_count_wins_over_numeric_name_prefix(self):
        # format limitation: "<digits> <rest>" always parses as count+name,
        # so a card name starting with digits needs an explicit count
        entries = parse_decklist("1 1996 World Champion")
        self.assertEqual(entries, [{"name": "1996 World Champion", "count": 1}])


if __name__ == "__main__":
    unittest.main()
