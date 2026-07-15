import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.arena_log import iter_log_entries
from magic_cabt.arena_mirror.recorder import MirrorRecorder
from magic_cabt.arena_mirror.session import MirrorSession
from magic_cabt.arena_mirror.tracker import ArenaMatchTracker
from magic_cabt.models.draft_model import TORCH_AVAILABLE

from tests.test_arena_log_draft import DRAFT_LOG


def _pack_event(draft_id="d1", pack=1, pick=1, cards=(1, 2, 3)):
    return {"type": "ARENA_DRAFT_PACK", "draftId": draft_id,
            "packNumber": pack, "pickNumber": pick,
            "packCards": list(cards)}


def _pick_event(draft_id="d1", pack=1, pick=1, cards=(2,)):
    return {"type": "ARENA_DRAFT_PICK", "draftId": draft_id,
            "packNumber": pack, "pickNumber": pick,
            "pickedCardIds": list(cards)}


class TrackerDraftTest(unittest.TestCase):

    def test_tracker_accumulates_pool_and_fires_callback(self):
        seen = []
        tracker = ArenaMatchTracker(
            on_draft=lambda kind, draft, event: seen.append((kind, draft)))
        tracker.handle_event(_pack_event(cards=(1, 2, 3)))
        tracker.handle_event(_pick_event(cards=(2,)))
        tracker.handle_event(_pack_event(pick=2, cards=(4, 5)))
        tracker.handle_event(_pick_event(pick=2, cards=(4, 5)))

        kinds = [kind for kind, _draft in seen]
        self.assertEqual(kinds, ["pack", "pick", "pack", "pick"])
        first_pack = seen[0][1]
        self.assertEqual(first_pack["packCards"], [1, 2, 3])
        self.assertEqual(first_pack["pool"], [])
        last = seen[-1][1]
        self.assertEqual(last["pool"], [2, 4, 5])
        # The tracker's own state must not be aliased by callback copies.
        seen[0][1]["pool"].append(999)
        self.assertEqual(tracker.draft["pool"], [2, 4, 5])

    def test_new_draft_id_resets_pool(self):
        tracker = ArenaMatchTracker()
        tracker.handle_event(_pack_event(draft_id="d1"))
        tracker.handle_event(_pick_event(draft_id="d1", cards=(2,)))
        tracker.handle_event(_pack_event(draft_id="d2", cards=(7, 8)))
        self.assertEqual(tracker.draft["draftId"], "d2")
        self.assertEqual(tracker.draft["pool"], [])
        self.assertEqual(tracker.draft["packCards"], [7, 8])

    def test_deck_submit_fires_callback(self):
        seen = []
        tracker = ArenaMatchTracker(
            on_draft=lambda kind, draft, event: seen.append(kind))
        tracker.handle_event({"type": "ARENA_DECK_SUBMIT",
                              "eventName": "TradDraft_TST",
                              "mainDeckArenaIds": [1], "sideboardArenaIds": []})
        self.assertEqual(seen, ["deck_submit"])


class SessionDraftTest(unittest.TestCase):

    def test_session_forwards_draft_events_and_records_them(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = MirrorRecorder(tmpdir)
            session = MirrorSession(
                recorder=recorder, verbose=False,
                on_draft=lambda kind, draft, event: seen.append(
                    (kind, len(draft["pool"]))))
            session.feed_entries(iter_log_entries(
                DRAFT_LOG.splitlines(True)))
            recorder.close()

            history_path = os.path.join(tmpdir, "game_history.jsonl")
            with open(history_path, encoding="utf-8") as handle:
                types = [json.loads(line)["type"] for line in handle
                         if line.strip()]

        kinds = [kind for kind, _pool in seen]
        self.assertEqual(kinds, ["pack", "pick", "pack", "pick",
                                 "deck_submit"])
        # Pool after the second pick holds both two-card picks.
        self.assertEqual(seen[3], ("pick", 4))
        for event_type in ("ARENA_DRAFT_PACK", "ARENA_DRAFT_PICK",
                           "ARENA_DECK_SUBMIT", "ARENA_SIDEBOARD_PROMPT",
                           "ARENA_SIDEBOARD_SUBMIT"):
            self.assertIn(event_type, types)


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class DraftAdvisorTest(unittest.TestCase):

    def test_maybe_load_scores_packs_and_builds_outlook(self):
        from magic_cabt.arena_mirror.draft_advisor import DraftAdvisor
        from magic_cabt.models.draft_model import (CardSelectionModel,
                                                   DraftModelConfig)

        config = DraftModelConfig(text_dim=32, numeric_dim=24, d_model=32,
                                  nhead=4, encoder_layers=1, ff_dim=64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            CardSelectionModel(config).save_checkpoint(path)
            advisor = DraftAdvisor.maybe_load(checkpoint_path=path)
            self.assertIsNotNone(advisor)

            scored = advisor.score_pack({
                "packCards": [11, 12, 13], "pool": [10],
                "packNumber": 1, "pickNumber": 2})
            self.assertEqual(len(scored), 3)
            self.assertEqual({entry["grpId"] for entry in scored},
                             {11, 12, 13})
            scores = [entry["score"] for entry in scored]
            self.assertEqual(scores, sorted(scores, reverse=True))

            outlook = advisor.outlook([10, 11, 12])
            self.assertEqual(len(outlook["deck"]), 3)
            self.assertIsNone(advisor.outlook([]))

    def test_maybe_load_returns_none_without_checkpoint(self):
        from magic_cabt.arena_mirror.draft_advisor import DraftAdvisor

        missing = os.path.join(tempfile.gettempdir(),
                               "no-such-draft-checkpoint.pt")
        self.assertIsNone(DraftAdvisor.maybe_load(checkpoint_path=missing))


if __name__ == "__main__":
    unittest.main()
