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
from magic_cabt.arena_mirror import mirror as mirror_mod
from magic_cabt.arena_mirror.mirror import enrich_snapshot
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
        tracker.set_local_seat(1)  # perspective: seat 1 sees its own hand
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
            raws, events, _ = normalizer.feed(entry)
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


def _state_with_described_opponent_hand():
    """A full state where the opponent's hand card IS described with a grpId.

    Arena normally omits hidden-zone identities, but the redaction boundary
    must not depend on that: seat 2's hand object 103 carries grpId 55555.
    """
    message = json.loads(json.dumps(GAME_STATE_FULL["gameStateMessage"]))
    for zone in message["zones"]:
        if zone.get("ownerSeatId") == 2 and zone["type"] == "ZoneType_Hand":
            zone["objectInstanceIds"] = [103]
    message["gameObjects"].append({
        "instanceId": 103, "grpId": 55555, "type": "GameObjectType_Card",
        "zoneId": 35, "ownerSeatId": 2, "controllerSeatId": 2,
        "cardTypes": ["CardType_Creature"],
    })
    return message


class HiddenInfoTest(unittest.TestCase):
    """The opponent's hand identity must never leak, even if Arena sends it."""

    OPPONENT_GRP_ID = 55555

    def _snapshot_local_seat_1(self):
        tracker = GameStateTracker()
        tracker.set_local_seat(1)
        tracker.apply(_state_with_described_opponent_hand())
        return tracker.snapshot()

    def test_opponent_hand_redacted_even_when_described(self):
        snapshot = self._snapshot_local_seat_1()
        opp_hand = snapshot["zones"]["hands"]["2"]
        self.assertEqual(1, len(opp_hand))
        self.assertIsNone(opp_hand[0]["grpId"])
        self.assertTrue(opp_hand[0]["faceDown"])
        self.assertNotIn("name", opp_hand[0])
        # our own hand is still fully visible
        own_hand = snapshot["zones"]["hands"]["1"]
        self.assertEqual(90001, own_hand[0]["grpId"])
        self.assertFalse(own_hand[0]["faceDown"])

    def test_all_hands_redacted_when_local_seat_unknown(self):
        tracker = GameStateTracker()  # never told which seat is local
        tracker.apply(_state_with_described_opponent_hand())
        snapshot = tracker.snapshot()
        for hand in snapshot["zones"]["hands"].values():
            for card in hand:
                self.assertIsNone(card["grpId"])
                self.assertTrue(card["faceDown"])

    def test_enrich_does_not_name_redacted_hand(self):
        looked_up = []

        class RecordingDB(object):
            def lookup(self, grp_id):
                looked_up.append(grp_id)
                return None  # unknown to the DB -> would render face-down

        snapshot = self._snapshot_local_seat_1()
        enrich_snapshot(snapshot, RecordingDB())
        # the opponent's hidden card identity must never be resolved
        self.assertNotIn(self.OPPONENT_GRP_ID, looked_up)
        for card in snapshot["zones"]["hands"]["2"]:
            self.assertNotIn("name", card)
            self.assertTrue(card["faceDown"])

    def test_recorder_output_has_no_opponent_identity(self):
        directory = tempfile.mkdtemp()
        try:
            text = (
                _gre_line({
                    "type": "GREMessageType_GameStateMessage",
                    "systemSeatIds": [1], "msgId": 2, "gameStateId": 1,
                    "gameStateMessage": _state_with_described_opponent_hand(),
                })
            )
            from magic_cabt.arena_log import iter_log_entries

            recorder = MirrorRecorder(directory)
            normalizer = StreamingNormalizer()
            tracker = ArenaMatchTracker(
                on_snapshot=lambda snap, event: recorder.record_state(
                    snap, event))
            tracker.state.set_local_seat(1)
            for entry in iter_log_entries(text.splitlines(True)):
                for event in normalizer.feed(entry)[1]:
                    recorder.record_history_event(event)
                    tracker.handle_event(event)
            recorder.close()

            needle = str(self.OPPONENT_GRP_ID)
            for name in ("mirror_states.jsonl", "decisions.jsonl",
                         "game_history.jsonl"):
                path = os.path.join(directory, name)
                if not os.path.exists(path):
                    continue
                with open(path, encoding="utf-8") as handle:
                    body = handle.read()
                self.assertNotIn(needle, body,
                                 "%s leaked the opponent's card grpId" % name)
                # raw game-state payload must not be persisted either
                self.assertNotIn('"gameObjects"', body)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_raw_audit_opt_in_captures_full_events(self):
        directory = tempfile.mkdtemp()
        try:
            recorder = MirrorRecorder(directory, raw_audit=True)
            recorder.record_history_event({
                "type": "ARENA_GAME_STATE", "inHistory": True,
                "gameObjects": [{"grpId": 55555}], "seq": 1})
            recorder.close()
            audit = os.path.join(directory, "raw_audit.jsonl")
            history = os.path.join(directory, "game_history.jsonl")
            with open(audit, encoding="utf-8") as handle:
                self.assertIn("55555", handle.read())
            with open(history, encoding="utf-8") as handle:
                self.assertNotIn("55555", handle.read())
        finally:
            shutil.rmtree(directory, ignore_errors=True)


def _malformed_line():
    """A log entry whose JSON payload is truncated mid-string (>240 chars).

    The raw newline inside the unterminated string guarantees a decode error,
    and the padding makes the chunk longer than the snippet bound so the test
    can prove truncation actually happened.
    """
    return ("[UnityCrossThreadLogger]2025-10-10 1:00:02 PM: Match to X: "
            "GreToClientEvent\n"
            '{"greToClientEvent": "' + "x" * 400 + "\n")


class ParseErrorRetentionTest(unittest.TestCase):
    """Live recording must retain bounded snippets of unparseable chunks.

    Silently dropping them removes exactly the data needed to debug parser
    gaps (docs/decisions/006-arena-log-retention.md). Full raw chunks stay
    behind the raw_audit opt-in.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run_session(self, raw_audit=False):
        from magic_cabt.arena_log import iter_log_entries
        from magic_cabt.arena_mirror.session import MirrorSession

        recorder = MirrorRecorder(self.dir, raw_audit=raw_audit)
        session = MirrorSession(recorder=recorder, verbose=False)
        text = _gre_line(GAME_STATE_FULL) + _malformed_line()
        session.feed_entries(iter_log_entries(text.splitlines(True)))
        recorder.close()

    def test_streaming_recorder_retains_bounded_snippet(self):
        self._run_session()

        path = os.path.join(self.dir, "parse_errors.jsonl")
        self.assertTrue(os.path.exists(path),
                        "parse error was dropped instead of recorded")
        with open(path, encoding="utf-8") as handle:
            errors = [json.loads(line) for line in handle]
        self.assertEqual(1, len(errors))
        error = errors[0]
        self.assertTrue(error["error"])
        # same bound as the batch path: at most 240 chars of the raw chunk
        self.assertEqual(240, len(error["snippet"]))
        self.assertIn('{"greToClientEvent"', error["snippet"])
        self.assertNotIn("x" * 400, error["snippet"])

        with open(os.path.join(self.dir, "summary.json"),
                  encoding="utf-8") as handle:
            summary = json.load(handle)
        self.assertEqual(1, summary["parseErrors"])
        # the well-formed entry still recorded normally
        self.assertGreater(summary["historyEvents"], 0)
        # full raw chunks require the raw_audit opt-in
        self.assertFalse(
            os.path.exists(os.path.join(self.dir, "raw_audit.jsonl")))

    def test_raw_audit_captures_full_unparseable_chunk(self):
        self._run_session(raw_audit=True)

        with open(os.path.join(self.dir, "raw_audit.jsonl"),
                  encoding="utf-8") as handle:
            audit = [json.loads(line) for line in handle]
        chunks = [record for record in audit
                  if record.get("type") == "ARENA_PARSE_ERROR"]
        self.assertEqual(1, len(chunks))
        self.assertIn("x" * 400, chunks[0]["rawChunk"])


class SessionAutoOpenTest(unittest.TestCase):
    """The GUI relies on lazy display open + status/action callbacks."""

    def _log_text(self):
        return (
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

    def test_display_opens_lazily_and_callbacks_fire(self):
        from magic_cabt.arena_log import iter_log_entries
        from magic_cabt.arena_mirror.session import MirrorSession

        sent_states = []
        factory_calls = []

        class FakeDisplay(object):
            def start_game(self, players, **kwargs):
                pass

            def send_state(self, state):
                sent_states.append(state)

            def finish_game(self, result=None):
                pass

            def close(self):
                pass

        def factory():
            factory_calls.append(1)
            return FakeDisplay()

        statuses = []
        actions = []
        games = []
        session = MirrorSession(
            display_factory=factory, verbose=False,
            on_status=lambda text: statuses.append(text),
            on_action=lambda line, record: actions.append(line),
            on_game=lambda kind, match_id, game: games.append(kind))

        # no display until the first live board update arrives
        self.assertEqual(0, len(factory_calls))
        session.feed_entries(iter_log_entries(self._log_text().splitlines(True)))

        self.assertEqual(1, len(factory_calls), "display opened exactly once")
        self.assertTrue(sent_states, "board states were streamed to XMage")
        self.assertTrue(any("decision #1" in line for line in actions))
        self.assertIn("game_start", games)
        self.assertTrue(statuses)


class GuiDisplayLifecycleTest(unittest.TestCase):
    """The XMage window persists across Start/Stop until the GUI closes it."""

    class FakeDisplay(object):
        def __init__(self, **kwargs):
            self.alive = True
            self.pinged = False
            self.closed = False

        def ping(self):
            self.pinged = True

        def close(self):
            self.closed = True
            self.alive = False

    def _make_app(self):
        # a stand-in for the Tk app that only needs the display-lifecycle
        # attributes/methods under test — no real Tk root required
        from magic_cabt.arena_mirror import gui as gui_mod

        app = gui_mod.ArenaMirrorApp.__new__(gui_mod.ArenaMirrorApp)
        import threading
        app.classpath = "cp.jar"
        app.java = "java"
        app._display = None
        app._display_lock = threading.Lock()
        app._session = None
        app._logs = []
        app._post = lambda item: app._logs.append(item)
        return app, gui_mod

    def test_display_reused_then_closed_only_on_gui_close(self):
        app, gui_mod = self._make_app()
        created = []
        original = gui_mod.MirrorDisplay
        gui_mod.MirrorDisplay = lambda **kwargs: created.append(
            self.FakeDisplay(**kwargs)) or created[-1]
        try:
            # first live game opens the window
            first = app._get_display()
            self.assertTrue(first.pinged)
            self.assertEqual(1, len(created))
            # a later Start (same GUI) reuses the SAME live window
            self.assertIs(first, app._get_display())
            self.assertEqual(1, len(created))
            # if the user closed the XMage window, a later Start relaunches it
            first.alive = False
            second = app._get_display()
            self.assertIsNot(first, second)
            self.assertEqual(2, len(created))
            # closing the GUI closes the XMage window
            app.root = _StubRoot()
            gui_mod.ArenaMirrorApp._on_close(app)
            self.assertTrue(second.closed)
            self.assertIsNone(app._display)
        finally:
            gui_mod.MirrorDisplay = original


class _StubRoot(object):
    def after(self, *args, **kwargs):
        pass

    def destroy(self):
        pass


class ReplayDiscoveryTest(unittest.TestCase):
    """The Replays tab lists any directory holding a mirror_states.jsonl."""

    def test_discovers_bundles_at_runs_dir_and_one_level_down(self):
        from magic_cabt.arena_mirror import gui as gui_mod

        root = tempfile.mkdtemp()
        try:
            # a nested bundle: runs/session/mirror_states.jsonl
            session = os.path.join(root, "session")
            os.makedirs(session)
            open(os.path.join(session, "mirror_states.jsonl"), "w").close()
            # a non-bundle sibling directory is ignored
            os.makedirs(os.path.join(root, "notes"))
            found = gui_mod.ArenaMirrorApp._discover_bundles(root)
            self.assertEqual([session], found)

            # the runs dir may itself be a bundle
            open(os.path.join(root, "mirror_states.jsonl"), "w").close()
            found = gui_mod.ArenaMirrorApp._discover_bundles(root)
            self.assertIn(root, found)
            self.assertIn(session, found)

            # a missing directory yields nothing (no crash)
            self.assertEqual([], gui_mod.ArenaMirrorApp._discover_bundles(
                os.path.join(root, "does-not-exist")))
        finally:
            shutil.rmtree(root, ignore_errors=True)


class MetadataTest(unittest.TestCase):
    """Match metadata: players, colors/archetype, event, result, title."""

    def _fake_card_db(self, colors_by_id, expansion=None):
        class FakeInfo(object):
            def __init__(self, ci):
                self.color_identity = ci
                self.expansion = expansion
                self.supertypes = []

        class FakeDB(object):
            def lookup(self, grp_id):
                ci = colors_by_id.get(grp_id)
                return FakeInfo(ci) if ci is not None else None

        return FakeDB()

    def test_guild_and_event_names(self):
        from magic_cabt.arena_mirror.metadata import (
            guild_name, pretty_event_name)
        self.assertEqual("Dimir", guild_name("UB"))
        self.assertEqual("Dimir", guild_name("BU"))  # order-independent
        self.assertEqual("Mono-Red", guild_name("R"))
        self.assertEqual("Jeskai", guild_name("URW"))
        self.assertEqual("4-Color", guild_name("WUBR"))
        # Limited names the set; Constructed names the format
        self.assertEqual("Marvel's Spider-Man Traditional Draft",
                         pretty_event_name("TradDraft_OM1_20250923"))
        self.assertEqual("Bloomburrow Premier Draft",
                         pretty_event_name("PremierDraft_BLB_20240730"))
        self.assertEqual("Standard", pretty_event_name("Ladder"))
        self.assertEqual("Standard", pretty_event_name("Play"))
        self.assertEqual("Alchemy", pretty_event_name("Alchemy_Ladder"))
        self.assertEqual("Historic (Bo3)",
                         pretty_event_name("Traditional_Historic_Ladder"))
        self.assertEqual("Standard (Bo3)",
                         pretty_event_name("Traditional_Ladder"))
        self.assertEqual("Explorer", pretty_event_name("Explorer_Play"))
        # unknown set code falls back to the raw code, never lies
        self.assertEqual("ZZZ Premier Draft",
                         pretty_event_name("PremierDraft_ZZZ_20990101"))

    def test_collector_builds_match_title(self):
        from magic_cabt.arena_mirror.metadata import MatchMetadataCollector

        match_id = "m-123"
        collector = MatchMetadataCollector()
        collector.observe({
            "type": "ARENA_CONNECT_RESP", "timestamp": "2025-10-10T13:00:00",
            "deckInfo": {"seatId": 1, "mainDeckArenaIds": [70228, 70228, 999]},
        })
        collector.observe({
            "type": "ARENA_MATCH_STATE_CHANGED", "timestamp": "2025-10-10T13:00:01",
            "matchId": match_id,
            "payload": {"gameRoomInfo": {"gameRoomConfig": {
                "matchId": match_id,
                "reservedPlayers": [
                    {"systemSeatId": 1, "teamId": 1, "playerName": "Nick#123",
                     "eventId": "Ladder"},
                    {"systemSeatId": 2, "teamId": 2, "playerName": "Opp"},
                ]}}},
        })
        collector.observe({
            "type": "ARENA_GAME_OVER", "timestamp": "2025-10-10T13:20:00",
            "matchId": match_id,
            "payload": {"resultList": [
                {"scope": "MatchScope_Game", "winningTeamId": 2},
                {"scope": "MatchScope_Game", "winningTeamId": 1},
                {"scope": "MatchScope_Game", "winningTeamId": 1},
                {"scope": "MatchScope_Match", "winningTeamId": 1},
            ]},
        })
        # Cauldron Familiar (70228) is black; 999 unknown -> ignored.
        out = collector.finalize(self._fake_card_db({70228: "B"}))
        self.assertEqual(1, len(out["matches"]))
        match = out["matches"][0]
        self.assertEqual("Nick", match["you"]["name"])
        self.assertEqual("Opp", match["opponent"]["name"])
        self.assertEqual("Mono-Black", match["you"]["archetype"])
        self.assertEqual("Standard", match["eventName"])  # bare Ladder = Standard
        self.assertEqual("win", match["result"])
        self.assertEqual("2-1", match["gameRecord"])
        self.assertIn("Mono-Black vs Opp", match["title"])
        self.assertIn("Win (2-1)", match["title"])
        self.assertIn("Standard", match["title"])
        # headline fields promoted to the top level for the replay list
        self.assertEqual(match["title"], out["title"])
        self.assertEqual("win", out["result"])

    def test_opponent_colors_from_public_board(self):
        from magic_cabt.arena_mirror.metadata import MatchMetadataCollector

        collector = MatchMetadataCollector()
        collector.observe({
            "type": "ARENA_MATCH_STATE_CHANGED", "matchId": "m",
            "timestamp": "2025-10-10T13:00:00",
            "payload": {"gameRoomInfo": {"gameRoomConfig": {
                "matchId": "m",
                "reservedPlayers": [
                    {"systemSeatId": 1, "teamId": 1, "playerName": "Me"},
                    {"systemSeatId": 2, "teamId": 2, "playerName": "You"},
                ]}}},
        })
        collector.note_snapshot({
            "matchId": "m", "localSeat": 1,
            "timestamp": "2025-10-10T13:05:00",
            "players": [{"seat": 1, "name": "Me"}, {"seat": 2, "name": "You"}],
            "zones": {"battlefield": [
                {"controllerSeat": 2, "colors": "U"},
                {"controllerSeat": 2, "colors": "R"},
                {"controllerSeat": 1, "colors": "G"},
            ]},
        })
        out = collector.finalize(card_db=None)
        match = out["matches"][0]
        self.assertEqual("Izzet", match["opponent"]["archetype"])


class ReplayControlTest(unittest.TestCase):
    """Decision classification + seek/jump logic for the interactive viewer."""

    def _decision(self, seq, opt_type, chosen):
        return {
            "sequence": seq, "matchId": "m",
            "observation": {
                "current": {"gameInstance": 1, "seq": seq},
                "select": {"type": "ACTIONSAVAILABLEREQ", "option": [
                    {"index": 0, "type": "PLAY", "label": "Play a land"},
                    {"index": 1, "type": "PASS", "label": "PASS"},
                ]},
            },
            "select": chosen,
        }

    def test_decision_is_pass(self):
        from magic_cabt.arena_mirror.replay import decision_is_pass
        self.assertTrue(decision_is_pass(self._decision(2, "PASS", [1])))
        self.assertFalse(decision_is_pass(self._decision(3, "PLAY", [0])))
        # empty answer to "actions available" is an auto-pass
        self.assertTrue(decision_is_pass(self._decision(4, None, [])))

    def _write_bundle(self, directory):
        def state(seq, turn):
            return {"gameInstance": 1, "seq": seq, "matchId": "m",
                    "turnNumber": turn, "phase": "Phase_Main1",
                    "timestamp": "2025-10-10T13:00:%02d" % seq,
                    "players": [{"seat": 1, "name": "Me"},
                                {"seat": 2, "name": "You"}],
                    "localSeat": 1}
        with open(os.path.join(directory, "mirror_states.jsonl"), "w") as fh:
            for seq in (1, 2, 3):
                fh.write(json.dumps(state(seq, 1)) + "\n")
        with open(os.path.join(directory, "decisions.jsonl"), "w") as fh:
            fh.write(json.dumps(self._decision(2, "PASS", [1])) + "\n")
            fh.write(json.dumps(self._decision(3, "PLAY", [0])) + "\n")

    def test_jump_to_next_and_next_meaningful(self):
        from magic_cabt.arena_mirror.replay import ReplayController
        directory = tempfile.mkdtemp()
        try:
            self._write_bundle(directory)
            reports = []
            controller = ReplayController(
                directory, display=None,
                on_progress=lambda info: reports.append(info))
            self.assertEqual(3, controller.total)
            controller._render(0)
            # next decision (any) is the pass at state index 1
            controller._jump(meaningful=False)
            self.assertEqual(1, controller._index)
            # from the start, the next *non-pass* action is at index 2
            controller._render(0)
            controller._jump(meaningful=True)
            self.assertEqual(2, controller._index)
            # progress reports carry turn + phase for the readout
            self.assertTrue(any(r.get("phase") == "Main1" for r in reports))
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class LauncherTest(unittest.TestCase):
    """The Python launcher must target the real Java class + package."""

    def test_default_command_uses_real_main_class(self):
        captured = {}

        class FakePopen(object):
            def __init__(self, command, **kwargs):
                captured["command"] = command
                self.stdin = self.stdout = None

            def poll(self):
                return None

        original = mirror_mod.subprocess.Popen
        mirror_mod.subprocess.Popen = FakePopen
        try:
            mirror_mod.MirrorDisplay(classpath="cp.jar")
        finally:
            mirror_mod.subprocess.Popen = original

        command = captured["command"]
        self.assertEqual("mage.client.cabtmirror.ArenaMirrorApp", command[-1])
        self.assertIn("cp.jar", command)
        self.assertEqual("mage.client.cabtmirror.ArenaMirrorApp",
                         mirror_mod.APP_MAIN_CLASS)


if __name__ == "__main__":
    unittest.main()
