import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt import normalize_arena_log

# Mirrors real 2025 Player.log captures of a Traditional Draft: Draft.Notify
# carries the pack as a comma string (and is logged twice per pick),
# EventPlayerDraftMakePick wraps its payload in a "request" JSON string with a
# GrpIds list (pick-two formats pick more than one card), EventSetDeckV2
# submits the built deck as {cardId, quantity} entries, and Bo3 sideboarding
# is a GRE SubmitDeckReq answered by a client SubmitDeckResp.
DRAFT_LOG = """\
[UnityCrossThreadLogger]Draft.Notify {"draftId":"draft-1","SelfPick":1,"SelfPack":1,"PackCards":"201,202,203,204"}
[UnityCrossThreadLogger]Draft.Notify {"draftId":"draft-1","SelfPick":1,"SelfPack":1,"PackCards":"201,202,203,204"}
[UnityCrossThreadLogger]==> EventPlayerDraftMakePick {"id":"r1","request":"{\\"DraftId\\":\\"draft-1\\",\\"GrpIds\\":[202,204],\\"Pack\\":1,\\"Pick\\":1}"}
[UnityCrossThreadLogger]Draft.Notify {"draftId":"draft-1","SelfPick":2,"SelfPack":1,"PackCards":"205,206"}
[UnityCrossThreadLogger]==> EventPlayerDraftMakePick {"id":"r2","request":"{\\"DraftId\\":\\"draft-1\\",\\"GrpIds\\":[205,206],\\"Pack\\":1,\\"Pick\\":2}"}
[UnityCrossThreadLogger]==> EventSetDeckV2 {"id":"r3","request":"{\\"EventName\\":\\"TradDraft_TST\\",\\"Summary\\":{\\"DeckId\\":\\"deck-1\\"},\\"Deck\\":{\\"MainDeck\\":[{\\"cardId\\":202,\\"quantity\\":2},{\\"cardId\\":204,\\"quantity\\":1}],\\"Sideboard\\":[{\\"cardId\\":205,\\"quantity\\":1}],\\"CommandZone\\":[],\\"Companions\\":[]}}"}
[UnityCrossThreadLogger]7/4/2026 6:22:10 PM: Match to CLIENTID: MatchGameRoomStateChangedEvent
{ "transactionId": "t1", "matchGameRoomStateChangedEvent": { "gameRoomInfo": { "gameRoomConfig": { "matchId": "match-1", "eventId": "TradDraft_TST" } } } }
[UnityCrossThreadLogger]7/4/2026 6:22:11 PM: Match to CLIENTID: GreToClientEvent
{ "transactionId": "t2", "greToClientEvent": { "greToClientMessages": [ { "type": "GREMessageType_GameStateMessage", "systemSeatIds": [ 1 ], "msgId": 2, "gameStateId": 1, "gameStateMessage": { "type": "GameStateType_Full", "gameInfo": { "matchID": "match-1", "gameNumber": 1, "stage": "GameStage_Play" }, "players": [], "zones": [], "gameObjects": [] } } ] } }
[UnityCrossThreadLogger]7/4/2026 6:22:12 PM: Match to CLIENTID: GreToClientEvent
{ "transactionId": "t3", "greToClientEvent": { "greToClientMessages": [ { "type": "GREMessageType_SubmitDeckReq", "systemSeatIds": [ 1 ], "msgId": 9, "gameStateId": 2, "submitDeckReq": { "deck": { "deckCards": [ 202, 202, 204 ], "sideboardCards": [ 205 ] } } } ] } }
[UnityCrossThreadLogger]7/4/2026 6:22:12 PM: Match to CLIENTID: GreToClientEvent
{ "transactionId": "t3b", "greToClientEvent": { "greToClientMessages": [ { "type": "GREMessageType_SubmitDeckReq", "systemSeatIds": [ 1 ], "msgId": 9, "gameStateId": 2, "submitDeckReq": { "deck": { "deckCards": [ 202, 202, 204 ], "sideboardCards": [ 205 ] } } } ] } }
[UnityCrossThreadLogger]7/4/2026 6:22:13 PM: CLIENTID to Match: ClientToGremessage
{
  "requestId": 5,
  "clientToMatchServiceMessageType": "ClientToMatchServiceMessageType_ClientToGREMessage",
  "transactionId": "t4",
  "payload": {
    "type": "ClientMessageType_SubmitDeckResp",
    "gameStateId": 2,
    "respId": 9,
    "submitDeckResp": {
      "deck": {
        "deckCards": [ 202, 202, 205 ],
        "sideboardCards": [ 204 ]
      }
    }
  }
}
"""


class ArenaLogDraftTest(unittest.TestCase):

    def test_draft_packs_deduplicated_and_parsed(self):
        normalizer = normalize_arena_log(DRAFT_LOG)
        packs = [event for event in normalizer.records["normalized_events"]
                 if event["type"] == "ARENA_DRAFT_PACK"]
        self.assertEqual(len(packs), 2)
        self.assertEqual(packs[0]["draftId"], "draft-1")
        self.assertEqual(packs[0]["packNumber"], 1)
        self.assertEqual(packs[0]["pickNumber"], 1)
        self.assertEqual(packs[0]["packCards"], [201, 202, 203, 204])
        self.assertEqual(packs[1]["packCards"], [205, 206])
        self.assertEqual(normalizer.summary["draftIds"], ["draft-1"])
        self.assertEqual(normalizer.summary["draftEvents"]["packs"], 2)

    def test_draft_picks_parsed_from_request_payload(self):
        normalizer = normalize_arena_log(DRAFT_LOG)
        picks = [event for event in normalizer.records["normalized_events"]
                 if event["type"] == "ARENA_DRAFT_PICK"]
        self.assertEqual(len(picks), 2)
        self.assertEqual(picks[0]["draftId"], "draft-1")
        self.assertEqual(picks[0]["packNumber"], 1)
        self.assertEqual(picks[0]["pickNumber"], 1)
        self.assertEqual(picks[0]["pickedCardIds"], [202, 204])
        self.assertEqual(normalizer.summary["draftEvents"]["picks"], 2)

    def test_deck_submit_expands_quantities(self):
        normalizer = normalize_arena_log(DRAFT_LOG)
        submits = [event for event in normalizer.records["normalized_events"]
                   if event["type"] == "ARENA_DECK_SUBMIT"]
        self.assertEqual(len(submits), 1)
        self.assertEqual(submits[0]["eventName"], "TradDraft_TST")
        self.assertEqual(submits[0]["mainDeckArenaIds"], [202, 202, 204])
        self.assertEqual(submits[0]["sideboardArenaIds"], [205])

    def test_sideboard_prompt_deduplicated_and_paired_with_submit(self):
        normalizer = normalize_arena_log(DRAFT_LOG)
        prompts = [event for event in normalizer.records["normalized_events"]
                   if event["type"] == "ARENA_SIDEBOARD_PROMPT"]
        submits = [event for event in normalizer.records["normalized_events"]
                   if event["type"] == "ARENA_SIDEBOARD_SUBMIT"]
        self.assertEqual(len(prompts), 1)
        self.assertEqual(len(submits), 1)
        self.assertEqual(prompts[0]["matchId"], "match-1")
        self.assertEqual(prompts[0]["gameNumber"], 1)
        self.assertEqual(prompts[0]["msgId"], 9)
        self.assertEqual(prompts[0]["deckCards"], [202, 202, 204])
        self.assertEqual(prompts[0]["sideboardCards"], [205])
        self.assertEqual(submits[0]["respId"], 9)
        self.assertEqual(submits[0]["deckCards"], [202, 202, 205])
        self.assertEqual(submits[0]["sideboardCards"], [204])

    def test_draft_events_reach_game_history(self):
        normalizer = normalize_arena_log(DRAFT_LOG)
        history_types = [event["type"]
                         for event in normalizer.records["game_history"]]
        for event_type in ("ARENA_DRAFT_PACK", "ARENA_DRAFT_PICK",
                           "ARENA_DECK_SUBMIT", "ARENA_SIDEBOARD_PROMPT",
                           "ARENA_SIDEBOARD_SUBMIT"):
            self.assertIn(event_type, history_types)


if __name__ == "__main__":
    unittest.main()
