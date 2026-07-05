import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt import iter_log_entries, normalize_arena_log


# Mirrors the real client's layout (validated against 2025 Player.log
# captures): a UnityCrossThreadLogger header line with no JSON, the payload
# as one complete line for GRE events, and pretty-printed multi-line JSON
# for outgoing client messages. Decks arrive as flat grpId lists under
# deckCards/sideboardCards, and the local seat only appears in the GRE
# message envelope (systemSeatIds).
SAMPLE_LOG = """\
[UnityCrossThreadLogger]7/4/2026 6:22:09 PM: Match to CLIENTID: GreToClientEvent
{ "transactionId": "t1", "greToClientEvent": { "greToClientMessages": [ { "type": "GREMessageType_ConnectResp", "systemSeatIds": [ 2 ], "msgId": 1, "connectResp": { "deckMessage": { "deckCards": [ 101, 101, 102 ], "sideboardCards": [ 103 ] } } } ] } }
[UnityCrossThreadLogger]7/4/2026 6:22:10 PM: Match to CLIENTID: MatchGameRoomStateChangedEvent
{ "transactionId": "t2", "matchGameRoomStateChangedEvent": { "gameRoomInfo": { "gameRoomConfig": { "matchId": "match-abc", "eventId": "event-1" }, "stateType": "MatchGameRoomStateType_Playing" } } }
[UnityCrossThreadLogger]7/4/2026 6:22:11 PM: Match to CLIENTID: GreToClientEvent
{ "transactionId": "t3", "greToClientEvent": { "greToClientMessages": [ { "type": "GREMessageType_GameStateMessage", "systemSeatIds": [ 2 ], "msgId": 2, "gameStateId": 1, "gameStateMessage": { "type": "GameStateType_Full", "gameInfo": { "matchID": "match-abc", "gameNumber": 1, "stage": "GameStage_Start" }, "turnInfo": { "turnNumber": 1, "phase": "Phase_Beginning" }, "players": [ { "systemSeatNumber": 2 } ], "zones": [ { "zoneId": 10 } ], "gameObjects": [ { "instanceId": 99 } ] } }, { "type": "GREMessageType_ActionsAvailableReq", "systemSeatIds": [ 2 ], "msgId": 3, "gameStateId": 1, "actionsAvailableReq": { "actions": [ { "actionType": "ActionType_Pass" }, { "actionType": "ActionType_Play", "grpId": 101, "instanceId": 99 } ] } } ] } }
[UnityCrossThreadLogger]7/4/2026 6:22:12 PM: CLIENTID to Match: ClientToGremessage
{
  "requestId": 3,
  "clientToMatchServiceMessageType": "ClientToMatchServiceMessageType_ClientToGREMessage",
  "transactionId": "t4",
  "payload": {
    "type": "ClientMessageType_PerformActionResp",
    "gameStateId": 1,
    "performActionResp": {
      "actions": [
        {
          "actionType": "ActionType_Play",
          "grpId": 101,
          "instanceId": 99
        }
      ]
    }
  }
}
[UnityCrossThreadLogger]7/4/2026 6:22:13 PM: CLIENTID to Match: ClientToGremessage
{
  "requestId": 4,
  "clientToMatchServiceMessageType": "ClientToMatchServiceMessageType_ClientToGREMessage",
  "transactionId": "t5",
  "payload": {
    "type": "ClientMessageType_SelectNResp",
    "gameStateId": 1,
    "selectNResp": {
      "ids": [ 99 ]
    }
  }
}
[UnityCrossThreadLogger]7/4/2026 6:22:14 PM: Match to CLIENTID: GreToClientEvent
{ "transactionId": "t6", "greToClientEvent": { "greToClientMessages": [ { "type": "GREMessageType_GameStateMessage", "systemSeatIds": [ 2 ], "msgId": 4, "gameStateId": 2, "gameStateMessage": { "type": "GameStateType_Diff", "gameInfo": { "matchID": "match-abc", "gameNumber": 1, "stage": "GameStage_GameOver" }, "turnInfo": { "turnNumber": 2 }, "players": [], "zones": [], "gameObjects": [] } } ] } }
[UnityCrossThreadLogger]7/4/2026 6:22:15 PM: Match to CLIENTID: GreToClientEvent
{ "transactionId": "t7", "greToClientEvent": { "greToClientMessages": [ { "type": "GREMessageType_QueuedGameStateMessage", "systemSeatIds": [ 2 ], "msgId": 5, "queuedGameStateMessage": { "type": "GameStateType_Diff", "gameInfo": { "matchID": "match-abc", "gameNumber": 1, "stage": "GameStage_GameOver" }, "turnInfo": { "turnNumber": 2 }, "players": [], "zones": [], "gameObjects": [] } } ] } }
[UnityCrossThreadLogger]7/4/2026 6:22:16 PM: Match to CLIENTID: MatchGameRoomStateChangedEvent
{ "transactionId": "t8", "matchGameRoomStateChangedEvent": { "gameRoomInfo": { "gameRoomConfig": { "matchId": "match-abc", "eventId": "event-1" }, "stateType": "MatchGameRoomStateType_MatchCompleted", "finalMatchResult": { "matchId": "match-abc", "resultList": [ { "scope": "MatchScope_Match", "winningTeamId": 2 } ] } } } }
"""

# Older inline single-line format with string-encoded payloads and
# card-object deck lists; kept parseable for backward compatibility.
LEGACY_LOG = """\
[UnityCrossThreadLogger]2026-07-04 18:22:09 {"payload":"{\\"matchGameRoomStateChangedEvent\\":{\\"gameRoomInfo\\":{\\"gameRoomConfig\\":{\\"matchId\\":\\"match-old\\",\\"eventId\\":\\"event-1\\"}}}}"}
[Client GRE]2026-07-04 18:22:10 {"greToClientEvent":{"greToClientMessages":[{"type":"GREMessageType_ConnectResp","connectResp":{"deckMessage":{"mainDeck":[{"grpId":1},{"grpId":2}],"sideboard":[{"grpId":3}]}}}]}}
[Client GRE]2026-07-04 18:22:11 {"clientToMatchServiceMessageType":"ClientToMatchServiceMessageType_ClientToGREMessage","payload":"{\\"type\\":\\"ClientMessageType_SelectNResp\\",\\"selectNResp\\":{\\"ids\\":[99]}}"}
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
        self.assertIn("ARENA_DECISION_PROMPT", event_types)
        self.assertIn("ARENA_CLIENT_DECISION", event_types)
        self.assertIn("ARENA_GAME_OVER", event_types)

        self.assertEqual(normalizer.summary["matchId"], "match-abc")
        self.assertEqual(normalizer.summary["matchIds"], ["match-abc"])
        self.assertEqual(
            normalizer.summary["games"],
            [{"matchId": "match-abc", "gameNumber": 1}],
        )
        self.assertEqual(normalizer.summary["seatId"], 2)
        self.assertEqual(normalizer.summary["gameStateMessages"], 3)
        self.assertEqual(normalizer.summary["parseErrors"], [])
        self.assertEqual(
            normalizer.summary["decisionPrompts"],
            {"GREMessageType_ActionsAvailableReq": 1},
        )
        self.assertEqual(
            normalizer.summary["clientDecisions"],
            {
                "ClientMessageType_PerformActionResp": 1,
                "ClientMessageType_SelectNResp": 1,
            },
        )
        self.assertEqual(normalizer.summary["clientSelectNRespDecisions"], 1)

    def test_deck_extracted_from_deck_cards_lists(self):
        normalizer = normalize_arena_log(SAMPLE_LOG)
        self.assertEqual(normalizer.summary["deckCardIdsFound"], [101, 102, 103])
        self.assertEqual(len(normalizer.decks), 1)
        deck = normalizer.decks[0]
        self.assertEqual(deck["mainDeckArenaIds"], [101, 101, 102])
        self.assertEqual(deck["sideboardArenaIds"], [103])
        self.assertEqual(deck["seatId"], 2)
        # ConnectResp precedes the first message naming the match; the entry
        # must still end up attributed to it.
        self.assertEqual(deck["matchId"], "match-abc")
        self.assertIs(normalizer.deck_info, deck)

    def test_decision_prompt_and_response_reach_game_history(self):
        normalizer = normalize_arena_log(SAMPLE_LOG)
        history_types = [event["type"] for event in normalizer.records["game_history"]]
        self.assertIn("ARENA_DECISION_PROMPT", history_types)
        self.assertIn("ARENA_CLIENT_DECISION", history_types)

        prompt = next(
            event
            for event in normalizer.records["game_history"]
            if event["type"] == "ARENA_DECISION_PROMPT"
        )
        self.assertEqual(prompt["messageType"], "GREMessageType_ActionsAvailableReq")
        self.assertEqual(prompt["matchId"], "match-abc")
        self.assertEqual(
            len(prompt["payload"]["actionsAvailableReq"]["actions"]), 2
        )

        decision = next(
            event
            for event in normalizer.records["game_history"]
            if event["type"] == "ARENA_CLIENT_DECISION"
        )
        self.assertEqual(decision["messageType"], "ClientMessageType_PerformActionResp")
        self.assertEqual(
            decision["payload"]["performActionResp"]["actions"][0]["instanceId"], 99
        )
        self.assertEqual(
            decision["precedingGameStateEventId"], prompt["precedingGameStateEventId"]
        )

    def test_game_over_emitted_once_per_game_plus_final_result(self):
        normalizer = normalize_arena_log(SAMPLE_LOG)
        game_overs = [
            event
            for event in normalizer.records["normalized_events"]
            if event["type"] == "ARENA_GAME_OVER"
        ]
        # Two diffs carry GameStage_GameOver for the same game -> one event,
        # plus one for the match's finalMatchResult.
        self.assertEqual(len(game_overs), 2)
        self.assertEqual(normalizer.summary["gameOverEvents"], 2)

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
            with open(os.path.join(tmpdir, "deck_info.json"), encoding="utf-8") as handle:
                deck_info = json.load(handle)
            self.assertEqual(len(deck_info["decks"]), 1)
            self.assertEqual(deck_info["decks"][0]["mainDeckArenaIds"], [101, 101, 102])
            with open(os.path.join(tmpdir, "normalized_events.jsonl"), encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertGreaterEqual(len(rows), 8)

    def test_legacy_inline_format_still_parses(self):
        normalizer = normalize_arena_log(LEGACY_LOG)
        event_types = [event["type"] for event in normalizer.records["normalized_events"]]
        self.assertIn("ARENA_MATCH_STATE_CHANGED", event_types)
        self.assertIn("ARENA_CONNECT_RESP", event_types)
        self.assertIn("ARENA_CLIENT_DECISION", event_types)
        self.assertEqual(normalizer.summary["matchId"], "match-old")
        self.assertEqual(normalizer.summary["deckCardIdsFound"], [1, 2, 3])
        self.assertEqual(normalizer.deck_info["mainDeckArenaIds"], [1, 2])
        self.assertEqual(normalizer.summary["clientSelectNRespDecisions"], 1)

    def test_parse_errors_are_reported_not_raised(self):
        normalizer = normalize_arena_log('[Client GRE]2026-07-04 18:22:10 {"bad"\n')
        self.assertEqual(len(normalizer.summary["parseErrors"]), 1)
        self.assertEqual(normalizer.records["normalized_events"], [])

    def test_bracketed_plain_text_is_not_a_parse_error(self):
        normalizer = normalize_arena_log(
            "[UnityCrossThreadLogger][IAPPurchaseController] Filling catalog\n"
        )
        self.assertEqual(normalizer.summary["parseErrors"], [])
        self.assertEqual(normalizer.records["normalized_events"], [])


if __name__ == "__main__":
    unittest.main()
