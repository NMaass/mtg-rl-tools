import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.training.build_draft_dataset import (
    collect_limited_records,
    dedupe_records,
)


def _pack(draft_id, pack, pick, cards):
    return {"type": "ARENA_DRAFT_PACK", "draftId": draft_id,
            "packNumber": pack, "pickNumber": pick, "packCards": cards}


def _pick(draft_id, pack, pick, cards):
    return {"type": "ARENA_DRAFT_PICK", "draftId": draft_id,
            "packNumber": pack, "pickNumber": pick, "pickedCardIds": cards}


class CollectLimitedRecordsTest(unittest.TestCase):

    def test_joins_packs_with_picks_and_accumulates_pool(self):
        events = [
            _pack("d1", 1, 1, [1, 2, 3, 4]),
            _pick("d1", 1, 1, [2, 4]),
            _pack("d1", 1, 2, [5, 6]),
            _pick("d1", 1, 2, [5, 6]),
        ]
        stats = {}
        records = collect_limited_records(events, "test.log", stats)
        picks = records["draftPick"]
        self.assertEqual(len(picks), 2)
        self.assertEqual(picks[0]["pool"], [])
        self.assertEqual(picks[0]["picked"], [2, 4])
        self.assertEqual(picks[1]["pool"], [2, 4])
        self.assertEqual(picks[1]["pack"], [5, 6])
        self.assertEqual(records["pools"]["d1"], [2, 4, 5, 6])
        self.assertEqual(stats["picks_compiled"], 2)

    def test_pick_without_pack_is_skipped_but_kept_in_pool(self):
        events = [
            _pick("d1", 1, 1, [9]),
            _pack("d1", 1, 2, [5, 6]),
            _pick("d1", 1, 2, [5]),
        ]
        stats = {}
        records = collect_limited_records(events, "test.log", stats)
        self.assertEqual(stats["picks_missing_pack"], 1)
        self.assertEqual(len(records["draftPick"]), 1)
        # The unmatched pick still contributes to later pools.
        self.assertEqual(records["draftPick"][0]["pool"], [9])

    def test_pick_not_in_pack_raises(self):
        events = [
            _pack("d1", 1, 1, [1, 2]),
            _pick("d1", 1, 1, [3]),
        ]
        with self.assertRaises(ValueError):
            collect_limited_records(events, "test.log")

    def test_deck_build_matches_pool_and_sideboard_pairs(self):
        events = [
            _pack("d1", 1, 1, [1, 2, 3]),
            _pick("d1", 1, 1, [1, 2]),
            {"type": "ARENA_DECK_SUBMIT", "eventName": "TradDraft_TST",
             "mainDeckArenaIds": [1, 2, 900], "sideboardArenaIds": []},
            {"type": "ARENA_SIDEBOARD_PROMPT", "matchId": "m1", "msgId": 9,
             "gameNumber": 1, "deckCards": [1, 2], "sideboardCards": [3]},
            {"type": "ARENA_SIDEBOARD_SUBMIT", "matchId": "m1", "respId": 9,
             "deckCards": [1, 3], "sideboardCards": [2]},
        ]
        records = collect_limited_records(events, "test.log")
        build = records["deckBuild"][0]
        self.assertEqual(build["draftId"], "d1")
        self.assertEqual(build["pool"], [1, 2])
        sideboard = records["sideboard"][0]
        self.assertEqual(sideboard["offeredDeck"], [1, 2])
        self.assertEqual(sideboard["chosenDeck"], [1, 3])
        self.assertTrue(sideboard["changed"])

    def test_dedupe_ignores_source_only(self):
        record = {"kind": "draftPick", "draftId": "d1", "pack": [1],
                  "picked": [1], "pool": []}
        first = dict(record, source="a.log")
        second = dict(record, source="b.log")
        third = dict(record, picked=[2], source="c.log")
        stats = {}
        unique = dedupe_records([first, second, third], stats)
        self.assertEqual(len(unique), 2)
        self.assertEqual(stats["duplicates_dropped"], 1)


if __name__ == "__main__":
    unittest.main()
