import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.arena_mirror.gui import _ReplayNamer


def _bundle_with_cache(tmpdir):
    with open(os.path.join(tmpdir, "card_cache.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"70307": {"grpId": 70307, "name": "Gilded Goose"},
                   "67364": {"grpId": 67364, "name": "Goblin Warchief"}},
                  handle)


STATES = [
    {"gameInstance": 1, "seq": 5, "zones": {"battlefield": [
        {"instanceId": 396, "grpId": 67364, "name": "Goblin Warchief"},
        {"instanceId": 181, "grpId": 70307, "name": "Gilded Goose"},
    ]}},
    {"gameInstance": 2, "seq": 3, "zones": {"battlefield": [
        {"instanceId": 396, "grpId": 70307, "name": "Gilded Goose"},
    ]}},
]


class ReplayNamerTest(unittest.TestCase):

    def test_resolves_pairs_singles_and_select_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _bundle_with_cache(tmpdir)
            namer = _ReplayNamer(tmpdir, STATES)

        self.assertEqual(
            namer.resolve("Cast grpId=70307 instance=181", 1),
            "Cast Gilded Goose")
        self.assertEqual(
            namer.resolve("attack with instance=396 -> player seat 2", 1),
            "attack with Goblin Warchief -> player seat 2")
        self.assertEqual(namer.resolve("select id=181", 1),
                         "select Gilded Goose")
        # The same instanceId belongs to a different card in game 2.
        self.assertEqual(namer.resolve("attack with instance=396", 2),
                         "attack with Gilded Goose")

    def test_unknown_ids_are_left_untouched(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _bundle_with_cache(tmpdir)
            namer = _ReplayNamer(tmpdir, STATES)
        self.assertEqual(namer.resolve("Activate grpId=99999", 1),
                         "Activate grpId=99999")
        self.assertEqual(namer.resolve("select id=555", 1), "select id=555")
        self.assertEqual(namer.resolve(None, 1), None)

    def test_missing_cache_still_resolves_from_states(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            namer = _ReplayNamer(tmpdir, STATES)
        self.assertEqual(namer.resolve("Play grpId=67364", 1),
                         "Play Goblin Warchief")


if __name__ == "__main__":
    unittest.main()
