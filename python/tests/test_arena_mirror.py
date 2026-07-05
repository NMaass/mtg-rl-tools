"""Tests for magic_cabt.arena_mirror: follower, tracker, options, recorder."""

import json
import os
import shutil
import tempfile
import unittest

from magic_cabt.arena_mirror.follower import EntryAssembler, LogFollower
from magic_cabt.arena_mirror.tracker import (
    ArenaMatchTracker, GameStateTracker, StreamingNormalizer)
from magic_cabt.arena_mirror import options
from magic_cabt.arena_mirror.recorder import MirrorRecorder
from magic_cabt.arena_mirror.replay import load_bundle


def _gre_line(message, seat=1):
    """One UnityCrossThreadLogger entry wrapping a GRE message."""
    blob = {
        "greToClientEvent": {"greToClientMessages": [message]},
        "transactionId": "t",
    }
    return ("[UnityCrossThreadLogger]2025-10-10 1:00:00 PM: Match to X: "
            "GreToClientEvent\n" + json.dumps(blob) + "\n")


def _client_line(payload):
    blob = {
        "clientToMatchServiceMessageType":
            "ClientToMatchServiceMessageType_ClientToGREMessage",
        "requestId": 5,
        "payload": payload,
    }
    return ("[UnityCrossThreadLogger]2025-10-10 1:00:01 PM: X to Match: "
            "ClientToMatchServiceMessageType_ClientToGREMessage\n"
            + json.dumps(blob) + "\n")


GAME_STATE_FULL = {
    "type": "GREMessageType_GameStateMessage",
    "systemSeatIds": [1],
    "msgId": 2,
    "gameStateId": 1,
    "gameStateMessage": {
        "type": "GameStateType_Full",
        "gameStateId": 1,
        "gameInfo": {"matchID": "m1", "gameNumber": 1,
                     "stage": "GameStage_Play"},
        "turnInfo": {"turnNumber": 1, "phase": "Phase_Main1",
                     "activePlayer": 1, "priorityPlayer": 1},
        "players": [
            {"systemSeatNumber": 1, "lifeTotal": 20},
            {"systemSeatNumber": 2, "lifeTotal": 20},
        ],
        "zones": [
            {"zoneId": 28, "type": "ZoneType_Battlefield",
             "visibility": "Visibility_Public", "objectInstanceIds": [101]},
            {"zoneId": 31, "type": "ZoneType_Hand", "ownerSeatId": 1,
             "visibility": "Visibility_Private", "objectInstanceIds": [102]},
            {"zoneId": 35, "type": "ZoneType_Hand", "ownerSeatId": 2,
             "visibility": "Visibility_Private", "objectInstanceIds": [103]},
            {"zoneId": 32, "type": "ZoneType_Library", "ownerSeatId": 1,
             "visibility": "Visibility_Hidden", "objectCount": 52},
        ],
        "gameObjects": [
            {"instanceId": 101, "grpId": 90000, "type": "GameObjectType_Card",
             "zoneId": 28, "ownerSeatId": 1, "controllerSeatId": 1,
             "isTapped": True, "cardTypes": ["CardType_Land"]},
            {"instanceId": 102, "grpId": 90001, "type": "GameObjectType_Card",
             "zoneId": 31, "ownerSeatId": 1, "controllerSeatId": 1},
            # seat 2's hand card is never described: hidden info
        ],
    },
}


class EntryAssemblerTest(unittest.TestCase):

    def test_multiline_entry_completes_on_next_header(self):
        assembler = EntryAssembler()
        self.assertIsNone(assembler.feed_line(
            "[UnityCrossThreadLogger]2025-10-10 1:00:00 PM: header\n"))
        self.assertIsNone(assembler.feed_line("{\n"))
        self.assertIsNone(assembler.feed_line("  \"a\": 1\n"))
        self.assertIsNone(assembler.feed_line("}\n"))
        done = assembler.feed_line("[UnityCrossThreadLogger]next\n")
        self.assertIsNotNone(done)
        self.assertIn('"a": 1', done["text"])
        self.assertEqual("UnityCrossThreadLogger", done["channel"])

    def test_flush_pending_returns_partial(self):
        assembler = EntryAssembler()
        assembler.feed_line("[UnityCrossThreadLogger]h\n")
        assembler.feed_line("{\"b\": 2}\n")
        entry = assembler.flush_pending()
        self.assertIn('"b": 2', entry["text"])
        self.assertIsNone(assembler.flush_pending())


class LogFollowerTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "Player.log")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _drain(self, follower):
        entries = []
        for entry in follower.follow():
            entries.append(entry)
        return entries

    def test_reads_appended_content_and_truncation(self):
        with open(self.path, "w") as handle:
            handle.write("[UnityCrossThreadLogger]one\n{\"x\": 1}\n")
        follower = LogFollower(self.path, poll_seconds=0.01, from_start=True)
        follower.IDLE_FLUSH_SECONDS = 0.05
        collected = []
        generator = follower.follow()
        # first entry completes once a second one starts
        with open(self.path, "a") as handle:
            handle.write("[UnityCrossThreadLogger]two\n{\"y\": 2}\n")
        collected.append(next(generator))
        # truncation (client restart) resets and keeps following
        with open(self.path, "w") as handle:
            handle.write("[UnityCrossThreadLogger]three\n{\"z\": 3}\n")
            handle.write("[UnityCrossThreadLogger]four\n")
        collected.append(next(generator))
        follower.stop()
        collected.extend(generator)
        texts = "".join(entry["text"] for entry in collected)
        self.assertIn('"x": 1', texts)
        self.assertIn('"z": 3', texts)
        self.assertEqual(1, follower.rotations)


class GameStateTrackerTest(unittest.TestCase):

    def test_full_state_snapshot_and_hidden_hand(self):
        tracker = GameStateTracker()
        tracker.set_seat_name(1, "Alice")
        tracker.set_seat_name(2, "Bob")
        tracker.apply(GAME_STATE_FULL["gameStateMessage"])
        snapshot = tracker.snapshot()

        self.assertEqual(1, snapshot["turnNumber"])
        self.assertEqual([20, 20],
                         [player["life"] for player in snapshot["players"]])
        battlefield = snapshot["zones"]["battlefield"]
        self.assertEqual(1, len(battlefield))
        self.assertTrue(battlefield[0]["tapped"])
        self.assertFalse(battlefield[0]["faceDown"])
        # own hand card has a grpId; opponent's was never described
        own_hand = snapshot["zones"]["hands"]["1"]
        self.assertEqual(90001, own_hand[0]["grpId"])
        self.assertFalse(own_hand[0]["faceDown"])
        opp_hand = snapshot["zones"]["hands"]["2"]
        self.assertIsNone(opp_hand[0]["grpId"])
        self.assertTrue(opp_hand[0]["faceDown"])
        self.assertEqual(52, snapshot["zones"]["libraries"]["1"])

    def test_diff_moves_object_between_zones(self):
        tracker = GameStateTracker()
        tracker.apply(GAME_STATE_FULL["gameStateMessage"])
        tracker.apply({
            "type": "GameStateType_Diff",
            "gameStateId": 2,
            "zones": [
                {"zoneId": 28, "type": "ZoneType_Battlefield",
                 "objectInstanceIds": [101, 102]},
                {"zoneId": 31, "type": "ZoneType_Hand", "ownerSeatId": 1,
                 "objectInstanceIds": []},
            ],
        })
        snapshot = tracker.snapshot()
        self.assertEqual(2, len(snapshot["zones"]["battlefield"]))
        self.assertEqual([], snapshot["zones"]["hands"]["1"])

    def test_object_id_change_annotation_carries_identity(self):
        tracker = GameStateTracker()
        tracker.apply(GAME_STATE_FULL["gameStateMessage"])
        tracker.apply({
            "type": "GameStateType_Diff",
            "gameStateId": 3,
            "annotations": [{
                "id": 1, "type": ["AnnotationType_ObjectIdChanged"],
                "details": [
                    {"key": "orig_id", "valueInt32": [102]},
                    {"key": "new_id", "valueInt32": [205]},
                ],
            }],
            "zones": [
                {"zoneId": 28, "type": "ZoneType_Battlefield",
                 "objectInstanceIds": [101, 205]},
                {"zoneId": 31, "type": "ZoneType_Hand", "ownerSeatId": 1,
                 "objectInstanceIds": []},
            ],
        })
        battlefield = tracker.snapshot()["zones"]["battlefield"]
        moved = [obj for obj in battlefield if obj["instanceId"] == 205][0]
        self.assertEqual(90001, moved["grpId"])
        self.assertEqual([102, 205], moved["lineage"])


class OptionsTest(unittest.TestCase):

    def _prompt_event(self, message):
        return {"messageType": message["type"], "payload": message,
                "timestamp": "2025-10-10T13:00:00"}

    def test_actions_available_and_perform_action(self):
        prompt = options.build_prompt(self._prompt_event({
            "type": "GREMessageType_ActionsAvailableReq",
            "msgId": 19,
            "actionsAvailableReq": {"actions": [
                {"actionType": "ActionType_Cast", "grpId": 5, "instanceId": 11},
                {"actionType": "ActionType_Play", "grpId": 6, "instanceId": 12},
                {"actionType": "ActionType_Pass"},
            ]},
        }))
        self.assertEqual(3, len(prompt["select"]["option"]))
        result = options.match_response([prompt], {
            "messageType": "ClientMessageType_PerformActionResp",
            "payload": {
                "type": "ClientMessageType_PerformActionResp",
                "respId": 19,
                "performActionResp": {"actions": [
                    {"actionType": "ActionType_Play", "grpId": 6,
                     "instanceId": 12},
                ]},
            },
        })
        matched_prompt, selected, matched = result
        self.assertIs(prompt, matched_prompt)
        self.assertEqual([1], selected)
        self.assertTrue(matched)

    def test_mulligan(self):
        prompt = options.build_prompt(self._prompt_event({
            "type": "GREMessageType_MulliganReq", "msgId": 9,
            "mulliganReq": {"mulliganType": "MulliganType_London"},
        }))
        _, selected, matched = options.match_response([prompt], {
            "messageType": "ClientMessageType_MulliganResp",
            "payload": {"type": "ClientMessageType_MulliganResp", "respId": 9,
                        "mulliganResp": {"decision": "MulliganOption_Mulligan"}},
        })
        self.assertTrue(matched)
        self.assertEqual([1], selected)

    def test_order_preserves_response_order(self):
        prompt = options.build_prompt(self._prompt_event({
            "type": "GREMessageType_OrderReq", "msgId": 70,
            "orderReq": {"ids": [178, 177]},
        }))
        _, selected, matched = options.match_response([prompt], {
            "messageType": "ClientMessageType_OrderResp",
            "payload": {"type": "ClientMessageType_OrderResp", "respId": 70,
                        "orderResp": {"ids": [177, 178],
                                      "ordering": "OrderingType_OrderAsIndicated"}},
        })
        self.assertTrue(matched)
        self.assertEqual([1, 0], selected)

    def test_concede_is_recorded_without_prompt(self):
        result = options.match_response([], {
            "messageType": "ClientMessageType_ConcedeReq",
            "payload": {"type": "ClientMessageType_ConcedeReq",
                        "concedeReq": {"scope": "MatchScope_Game"}},
        })
        prompt, selected, matched = result
        self.assertEqual("CONCEDE", prompt["select"]["type"])
        self.assertEqual([0], selected)
        self.assertTrue(matched)

    def test_unpaired_submit_is_skipped(self):
        self.assertIsNone(options.match_response([], {
            "messageType": "ClientMessageType_SubmitAttackersReq",
            "payload": {"type": "ClientMessageType_SubmitAttackersReq",
                        "respId": 129},
        }))


class PipelineTest(unittest.TestCase):
    """Log text -> normalizer -> tracker -> recorder, end to end."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_prompt_response_pairing_through_tracker(self):
        text = (
            _gre_line(GAME_STATE_FULL)
            + _gre_line({
                "type": "GREMessageType_ActionsAvailableReq",
                "systemSeatIds": [1], "msgId": 19, "gameStateId": 1,
                "actionsAvailableReq": {"actions": [
                    {"actionType": "ActionType_Play", "grpId": 90001,
                     "instanceId": 102},
                    {"actionType": "ActionType_Pass"},
                ]},
            })
            + _client_line({
                "type": "ClientMessageType_PerformActionResp", "respId": 19,
                "gameStateId": 1,
                "performActionResp": {"actions": [
                    {"actionType": "ActionType_Play", "grpId": 90001,
                     "instanceId": 102},
                ]},
            })
        )
        from magic_cabt.arena_log import iter_log_entries

        recorder = MirrorRecorder(self.dir)
        decisions = []
        snapshots = []
        normalizer = StreamingNormalizer()
        tracker = ArenaMatchTracker(
            on_snapshot=lambda snap, event: (snapshots.append(snap),
                                             recorder.record_state(snap, event)),
            on_decision=lambda record: (decisions.append(record),
                                        recorder.record_decision(record)),
        )
        for entry in iter_log_entries(text.splitlines(True)):
            raws, events = normalizer.feed(entry)
            for event in events:
                recorder.record_history_event(event)
                tracker.handle_event(event)
        recorder.close()

        self.assertEqual(1, len(decisions))
        record = decisions[0]
        self.assertEqual([0], record["select"])
        self.assertTrue(record["selectionMatched"])
        self.assertEqual("ACTIONSAVAILABLEREQ",
                         record["observation"]["select"]["type"])
        # the observation carries the pre-decision board snapshot
        current = record["observation"]["current"]
        self.assertEqual(1, current["turnNumber"])
        self.assertTrue(snapshots)

        states, decisions_read, summary = load_bundle(self.dir)
        self.assertEqual(1, len(decisions_read))
        self.assertEqual(len(snapshots), len(states))
        self.assertEqual(1, summary["decisions"])
        self.assertEqual(1, summary["decisionsMatched"])


if __name__ == "__main__":
    unittest.main()
