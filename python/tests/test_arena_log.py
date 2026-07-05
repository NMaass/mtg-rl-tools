import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt import iter_log_entries, normalize_arena_log


SAMPLE_LOG = """\
[UnityCrossThreadLogger]2026-07-04 18:22:09 {"payload":"{\\"matchGameRoomStateChangedEvent\\":{\\"gameRoomInfo\\":{\\"gameRoomConfig\\":{\\"matchId\\":\\"match-abc\\",\\"eventId\\":\\"event-1\\"}}}}"}
[Client GRE]2026-07-04 18:22:10 {"greToClientEvent":{"greToClientMessages":[{"type":"GREMessageType_ConnectResp","connectResp":{"deckMessage":{"mainDeck":[{"grpId":1},{"grpId":2}],"sideboard":[{"grpId":3}]}}},{"type":"GREMessageType_GameStateMessage","gameStateMessage":{"gameInfo":{"matchID":"match-abc"},"turnInfo":{"turnNumber":1,"phase":"Phase_Beginning"},"players":[{"systemSeatId":1}],"zones":[{"zoneId":10}],"gameObjects":[{"instanceId":99}]}}]}}
[Client GRE]2026-07-04 18:22:11 {"clientToMatchServiceMessageType":"ClientToMatchServiceMessageType_ClientToGREMessage","payload":"{\\"type\\":\\"ClientMessageType_SelectNResp\\",\\"selectNResp\\":{\\"ids\\":[99]}}"}
[Client GRE]2026-07-04 18:22:12 {"greToClientEvent":{"greToClientMessages":[{"type":"GREMessageType_QueuedGameStateMessage","queuedGameStateMessage":{"gameInfo":{"matchID":"match-abc","stage":"GameStage_GameOver"},"turnInfo":{"turnNumber":2},"players":[],"zones":[],"gameObjects":[]}}]}}
"""


class ArenaLogNormalizerTest(unittest.TestCase):

    def test_iter_log_entries_groups_multiline_json(self):
        text = (
            '[Client GRE]2026-07-04 18:22:10 {"greToClientEvent":{\n'
            '"greToClientMessages":[]}}\n'
            '[Client GRE]2026-07-04 18:22:11 {"ok":true}\n'
        )
        entries = list(iter_log_entries(text.splitlines(True)))
        self.assertEqual(len(entries), 2)
        self.assertIn("greToClientMessages", entries[0]["text"])
        self.assertEqual(entries[0]["timestamp"], "2026-07-04T18:22:10")

    def test_normalizes_core_arena_replay_events(self):
        normalizer = normalize_arena_log(SAMPLE_LOG)
        events = normalizer.records["normalized_events"]
        event_types = [event["type"] for event in events]

        self.assertIn("ARENA_MATCH_STATE_CHANGED", event_types)
        self.assertIn("ARENA_CONNECT_RESP", event_types)
        self.assertIn("ARENA_GAME_STATE", event_types)
        self.assertIn("ARENA_QUEUED_GAME_STATE", event_types)
        self.assertIn("ARENA_SELECT_N_RESP", event_types)
        self.assertIn("ARENA_GAME_OVER", event_types)

        decision = next(event for event in events if event["type"] == "ARENA_SELECT_N_RESP")
        self.assertEqual(decision["precedingGameStateEventId"], "arena-000003")
        self.assertEqual(decision["payload"]["selectNResp"]["ids"], [99])
        self.assertEqual(normalizer.summary["matchId"], "match-abc")
        self.assertEqual(normalizer.summary["gameStateMessages"], 2)
        self.assertEqual(normalizer.summary["clientSelectNRespDecisions"], 1)
        self.assertEqual(normalizer.summary["gameOverEvents"], 1)
        self.assertEqual(normalizer.summary["deckCardIdsFound"], [1, 2, 3])
        self.assertEqual(normalizer.deck_info["mainDeckArenaIds"], [1, 2])

    def test_writes_replay_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            normalize_arena_log(io.StringIO(SAMPLE_LOG), tmpdir)

            expected = {
                "raw_events.jsonl",
                "normalized_events.jsonl",
                "game_history.jsonl",
                "deck_info.json",
                "summary.json",
            }
            self.assertEqual(set(os.listdir(tmpdir)), expected)
            with open(os.path.join(tmpdir, "summary.json"), encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["matchId"], "match-abc")
            with open(os.path.join(tmpdir, "normalized_events.jsonl"), encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertGreaterEqual(len(rows), 6)

    def test_parse_errors_are_reported_not_raised(self):
        normalizer = normalize_arena_log('[Client GRE]2026-07-04 18:22:10 {"bad"\n')
        self.assertEqual(len(normalizer.summary["parseErrors"]), 1)
        self.assertEqual(normalizer.records["normalized_events"], [])


if __name__ == "__main__":
    unittest.main()
