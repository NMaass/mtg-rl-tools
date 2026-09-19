import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from magic_cabt.replay_review import client
from magic_cabt.replay_review.data import canonical, load_points, make_point
from magic_cabt.replay_review.session import ReviewSession


def record(turn=3):
    return {
        "sequence": turn, "gameNumber": 1, "seat": 1, "select": [1], "selectionMatched": True,
        "responsePayload": {"DO_NOT_SEND": "actual action"}, "reward": 1,
        "observation": {
            "current": {"localSeat": 1, "prioritySeat": 1, "activeSeat": 1,
                        "turnNumber": turn, "phase": "Main", "step": "Precombat",
                        "winner": "DO_NOT_SEND", "players": [
                            {"seat": 1, "name": "DO_NOT_SEND", "life": 16, "handCount": 3},
                            {"seat": 2, "life": 12, "handCount": 4}],
                        "zones": {
                            "hands": {"1": [{"grpId": 7, "name": "Lightning Bolt"}],
                                      "2": [{"name": "SECRET_OPPONENT_HAND"}]},
                            "libraries": {"1": [{"name": "SECRET_DRAW"}]},
                            "battlefield": [{"name": "Mountain", "controllerSeat": 1},
                                            {"name": "Grizzly Bears", "controllerSeat": 2,
                                             "power": 2, "toughness": 2}],
                            "stack": [], "graveyards": {}, "exile": [], "command": []}},
            "select": {"type": "ACTIONSAVAILABLEREQ", "minCount": 1, "maxCount": 1,
                       "option": [{"index": 0, "type": "PASS", "label": "Pass priority", "payload": {}},
                                  {"index": 1, "type": "CAST", "label": "Cast Lightning Bolt", "payload": {"grpId": 7}}]}}}


def response():
    return {"answers": {"action": {"type": "choice", "choice": "option_1", "confidence": .8,
                                   "probabilities": {"option_0": .2, "option_1": .8}}},
            "model": "typesafe/jev-1.13", "id": "fixture", "provider": "TypeSafe",
            "usage": {"cost": .00002, "input_tokens": 500, "output_tokens": 35}}


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out")
        time.sleep(.005)


class ProjectionTests(unittest.TestCase):
    def test_no_outcome_or_hidden_input(self):
        point = make_point(record(), 0)
        self.assertIsNone(point.issue)
        text = canonical(point.request)
        for forbidden in ("DO_NOT_SEND", "SECRET", "responsePayload", "reward", "recordedChoice"):
            self.assertNotIn(forbidden, text)
        changed = record()
        changed["select"] = [0]
        changed["responsePayload"] = {"different": True}
        changed["reward"] = -1
        self.assertEqual(point.request, make_point(changed, 0).request)

    def test_static_card_cache_only_enriches_visible_ids(self):
        rec = record()
        rec["observation"]["current"]["zones"]["exile"] = [{"grpId": 999, "faceDownFlag": True}]
        point = make_point(rec, 0, {"7": {"name": "Lightning Bolt"}, "999": {"name": "SECRET"}})
        self.assertNotIn("SECRET", canonical(point.request))
        self.assertIn("Face-down object", canonical(point.request))

    def test_transport_ids_are_replaced_with_readable_card_names(self):
        rec = record()
        rec["observation"]["current"]["zones"]["battlefield"] = [
            {"grpId": 42, "instanceId": 9001, "controllerSeat": 1}
        ]
        rec["observation"]["select"]["option"][1] = {
            "index": 1,
            "type": "CAST_SPELL",
            "label": "CAST_SPELL grpId=7 instance=9002",
            "payload": {"grpId": 7, "instanceId": 9002},
        }
        point = make_point(
            rec, 0,
            {"7": {"name": "Lightning Bolt"},
             "42": {"name": "Mountain"}})
        text = canonical(point.request)
        self.assertIn("Lightning Bolt", text)
        self.assertIn("Mountain", text)
        self.assertNotIn("grpId=7", text)
        self.assertNotIn("instance=9002", text)
        self.assertNotIn("9001", text)

    def test_fail_closed_bad_options_and_snapshots(self):
        changes = [
            lambda r: r["observation"]["select"].update(option=[]),
            lambda r: r["observation"]["select"].update(maxCount=2),
            lambda r: r["observation"]["current"].update(localSeat=None),
            lambda r: r["observation"]["current"].update(prioritySeat=2),
            lambda r: r["observation"]["current"].update(gameOver=True),
            lambda r: r.update(selectionMatched=False),
            lambda r: r.update(select=[999]),
            lambda r: r["observation"]["select"]["option"][1].update(index=0),
            lambda r: r.update(observation="bad"),
        ]
        for change in changes:
            with self.subTest(change=change):
                rec = record()
                change(rec)
                self.assertIsNone(make_point(rec, 0).request)

    def test_missing_response_is_not_an_invented_pass(self):
        rec = record()
        rec["select"] = []
        rec["observation"]["select"]["option"] = [rec["observation"]["select"]["option"][1]]
        point = make_point(rec, 0)
        self.assertIsNone(point.recorded_choice)
        self.assertFalse(point.is_pass)
        self.assertEqual(len(point.options), 1)

    def test_xmage_captured_priority_and_selected_key(self):
        rec = record()
        rec.pop("select")
        rec["selected"] = [1]
        rec["observation"]["select"]["type"] = "PRIORITY"
        rec["observation"]["current"] = {
            "priorityPlayerId": "hero", "turnNumber": 3,
            "players": [{"playerId": "hero", "life": 20, "hand": [{"ref": {"name": "Island"}}]},
                        {"playerId": "opponent", "life": 20, "hand": [{"ref": {"name": "SECRET"}}]}],
            "battlefield": [], "stack": []}
        point = make_point(rec, 0)
        self.assertIsNone(point.issue)
        self.assertEqual(point.recorded_choice, "option_1")
        self.assertNotIn("SECRET", canonical(point.request))
        self.assertIn("Island", canonical(point.request))

    def test_load_directory_file_unsupported_and_other_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            unsupported = record(4)
            unsupported["selectionMatched"] = False
            path.write_text("\n".join(map(json.dumps, (record(), unsupported, {"result": "SECRET"}))))
            points, ignored = load_points(tmp)
            self.assertEqual((len(points), ignored), (2, 1))
            self.assertIsNotNone(points[1].issue)
            self.assertEqual(load_points(path)[0][0], points[0])

    def test_payload_limit_not_silent_truncation(self):
        rec = record()
        rec["observation"]["select"]["option"][0]["label"] = "x" * 100_000
        self.assertIsNone(make_point(rec, 0).request)


class ClientTests(unittest.TestCase):
    def test_contract_and_actual_cost(self):
        seen = []
        result = client.analyze(make_point(record(), 0).request, "test-key", lambda payload, key: seen.append(payload) or response())
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["costUsd"], .00002)
        self.assertIn("gameState", seen[0]["state"])
        self.assertIn("possibleActions", seen[0]["state"])
        self.assertNotIn("messages", seen[0])

    def test_fixed_https_endpoint_and_no_redirects(self):
        with patch.object(client.http.client, "HTTPSConnection") as cls:
            connection = cls.return_value
            connection.getresponse.return_value.status = 200
            connection.getresponse.return_value.read.return_value = json.dumps(response()).encode()
            payload = make_point(record(), 0).request
            client.post_decision(payload, "test-key")
            cls.assert_called_once_with("openrouter.ai", timeout=20)
            self.assertEqual(connection.request.call_args.args, ("POST", "/api/alpha/decisions"))
            self.assertEqual(connection.request.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")
            connection.close.assert_called_once()

    def test_invalid_distribution_retains_known_charge(self):
        for probabilities in ({"evil": 1}, {"option_0": float("nan"), "option_1": 0},
                              {"option_0": .2, "option_1": .2}, {"option_0": -1, "option_1": 2}):
            body = response()
            body["answers"]["action"]["probabilities"] = probabilities
            result = client.analyze(make_point(record(), 0).request, "test-key", lambda p, k: body)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["costUsd"], .00002)
            self.assertNotIn("probabilities", result)

    def test_auth_timeout_and_missing_usage_are_not_free(self):
        for exception in (client.RequestFailure(401), TimeoutError("test-key")):
            with self.subTest(exception=exception):
                transport = MagicMock(side_effect=exception)
                result = client.analyze(make_point(record(), 0).request, "test-key", transport)
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["costUsd"])
                self.assertNotIn("test-key", canonical(result))
                transport.assert_called_once()
        body = response()
        body.pop("usage")
        result = client.analyze(make_point(record(), 0).request, "test-key", lambda p, k: body)
        self.assertIsNone(result["costUsd"])


class SessionTests(unittest.TestCase):
    def test_coalesces_and_caches_without_leaking_key(self):
        entered, release = threading.Event(), threading.Event()
        def analyzer(payload, key):
            entered.set()
            release.wait(2)
            return client.analyze(payload, key, lambda p, k: response())
        session = ReviewSession(analyzer)
        self.addCleanup(session.close)
        session.connect("test-key")
        a, b, c = [make_point(record(t), t) for t in (3, 4, 5)]
        session.request(a)
        self.assertTrue(entered.wait(2))
        session.request(b)
        session.request(c)
        release.set()
        wait_for(lambda: session.summary()["calls"] == 2)
        self.assertEqual(session.status(b)[0], "idle")
        session.request(a)
        time.sleep(.02)
        self.assertEqual(session.summary()["calls"], 2)
        report = session.report([a, b, c], {a.key: "Useful"})
        self.assertNotIn("test-key", canonical(report))
        self.assertEqual(report["agreement"], {"matched": 2, "scored": 2})

    def test_disconnect_cancels_unsent_and_errors_need_explicit_retry(self):
        analyzer = MagicMock(return_value={"status": "error", "costUsd": None, "error": "failure"})
        session = ReviewSession(analyzer)
        self.addCleanup(session.close)
        point = make_point(record(), 0)
        session.request(point)
        self.assertEqual(session.summary()["calls"], 0)
        session.connect("test-key")
        session.request(point)
        wait_for(lambda: session.summary()["calls"] == 1)
        session.request(point)
        time.sleep(.02)
        self.assertEqual(session.summary()["calls"], 1)
        session.request(point, retry=True)
        wait_for(lambda: session.summary()["calls"] == 2)
        session.disconnect()
        session.request(point, retry=True)
        self.assertFalse(session.connected)
        self.assertEqual(session.summary()["unknownCostCalls"], 2)


if __name__ == "__main__":
    unittest.main()
