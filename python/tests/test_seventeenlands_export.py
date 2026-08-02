import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.seventeenlands_export import (
    DEFAULT_DELAY_SECONDS,
    DRAFT_PARTS,
    ExportError,
    SeventeenLandsClient,
    build_parser,
    estimate,
    export_all,
    format_duration,
    main,
)


class FakeTransport(object):
    """Stands in for the network: maps URL substrings to canned responses.

    A response is ``(status, body)`` or ``(status, body, headers)``; a callable
    receives the URL so a route can change with each call.
    """

    def __init__(self, routes, default=(404, b"nope")):
        self.routes = routes
        self.default = default
        self.calls = []
        self.headers = []

    def __call__(self, url, headers):
        self.calls.append(url)
        self.headers.append(headers)
        response = self.default
        for needle, candidate in self.routes.items():
            if needle in url:
                response = candidate
                break
        if callable(response):
            response = response(url)
        status, body = response[0], response[1]
        out_headers = response[2] if len(response) > 2 else {}
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        return status, body, out_headers


class FakeClock(object):
    """Records every sleep so pacing can be asserted without waiting."""

    def __init__(self):
        self.sleeps = []

    def __call__(self, seconds):
        self.sleeps.append(seconds)

    @property
    def total(self):
        return sum(self.sleeps)


def client_for(routes, **kwargs):
    transport = FakeTransport(routes)
    kwargs.setdefault("cookie", "session=abc")
    kwargs.setdefault("delay", 0)
    kwargs.setdefault("jitter", 0)
    clock = kwargs.pop("clock", None) or FakeClock()
    client = SeventeenLandsClient(opener=transport, sleep=clock, **kwargs)
    client.clock = clock
    return client, transport


DRAFT_ROW = {"draft_id": "d1", "expansion": "TST", "wins": 5, "losses": 2}


def full_routes(draft_ids=("d1",)):
    return {
        "/data/user_game_list": (200, [{"game_id": "g1"}]),
        "/data/user": (200, {"drafts": [
            {"draft_id": did, "expansion": "TST"} for did in draft_ids]}),
        "/data/draft?": (200, {"picks": [{"pick": "Bolt"}]}),
        "/data/event_metadata": (200, {"format": "PremierDraft"}),
        "/data/deck": (200, {"maindeck": []}),
        "/data/pool": (200, {"pool": []}),
    }


class ClientTest(unittest.TestCase):

    def test_cookie_is_sent_and_never_in_the_url(self):
        client, transport = client_for(full_routes())
        client.user_drafts()
        self.assertEqual("session=abc", transport.headers[0]["Cookie"])
        self.assertNotIn("abc", transport.calls[0])

    def test_auth_failure_is_fatal_and_actionable(self):
        client, transport = client_for({"/data/user": (403, b"denied")})
        with self.assertRaises(ExportError) as caught:
            client.user_drafts()
        self.assertIn("cookie", str(caught.exception).lower())
        self.assertEqual(1, len(transport.calls), "must not retry a 403")

    def test_rate_limit_is_retried_then_succeeds(self):
        seen = {"n": 0}

        def flaky(url):
            seen["n"] += 1
            if seen["n"] < 3:
                return 429, b"slow down"
            return 200, {"drafts": []}

        client, transport = client_for({"/data/user": flaky})
        self.assertEqual({"drafts": []}, client.user_drafts())
        self.assertEqual(3, len(transport.calls))

    def test_retries_are_bounded_and_report_the_endpoint(self):
        client, transport = client_for({"/data/user": (503, b"down")},
                                       retries=2)
        with self.assertRaises(ExportError) as caught:
            client.user_drafts()
        self.assertIn("/data/user", str(caught.exception))
        self.assertEqual(3, len(transport.calls))

    def test_html_login_page_is_named_as_such(self):
        client, _ = client_for({"/data/user": (200, b"<html>login</html>")})
        with self.assertRaises(ExportError) as caught:
            client.user_drafts()
        self.assertIn("non-JSON", str(caught.exception))

    def test_sharing_token_selects_the_token_game_list_path(self):
        client, transport = client_for(
            {"/data/user_game_list/tok123": (200, [{"game_id": "g"}])},
            sharing_token="tok123")
        client.game_list()
        self.assertIn("/data/user_game_list/tok123", transport.calls[0])

    def test_delay_is_honored_between_requests(self):
        client, _ = client_for(full_routes(), delay=1.0)
        client.user_drafts()
        client.user_drafts()
        self.assertTrue(client.clock.sleeps, "second request must wait")
        self.assertLessEqual(client.clock.sleeps[-1], 1.0)


class PolitenessTest(unittest.TestCase):
    """The guarantees that keep a multi-thousand-request harvest kind."""

    def test_default_pace_is_conservative(self):
        self.assertGreaterEqual(DEFAULT_DELAY_SECONDS, 2.0)

    def test_retry_after_is_obeyed_over_our_own_backoff(self):
        seen = {"n": 0}

        def busy(url):
            seen["n"] += 1
            if seen["n"] == 1:
                return 429, b"slow down", {"Retry-After": "37"}
            return 200, {"drafts": []}

        client, _ = client_for({"/data/user": busy}, delay=1.0)
        client.user_drafts()
        self.assertIn(37.0, client.clock.sleeps,
                      "the server's own number must win")

    def test_unparsable_retry_after_still_waits(self):
        seen = {"n": 0}

        def busy(url):
            seen["n"] += 1
            if seen["n"] == 1:
                return 503, b"", {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}
            return 200, {"drafts": []}

        client, _ = client_for({"/data/user": busy})
        client.user_drafts()
        self.assertTrue(any(sleep >= 60 for sleep in client.clock.sleeps))

    def test_absurd_retry_after_stops_the_run(self):
        client, _ = client_for(
            {"/data/user": (429, b"", {"Retry-After": "99999"})})
        with self.assertRaises(ExportError) as caught:
            client.user_drafts()
        self.assertIn("resumes", str(caught.exception))

    def test_a_429_slows_the_rest_of_the_run_permanently(self):
        seen = {"n": 0}

        def busy(url):
            seen["n"] += 1
            if seen["n"] == 1:
                return 429, b"slow down"
            return 200, {"drafts": []}

        notes = []
        client, _ = client_for({"/data/user": busy}, delay=2.0,
                               on_slowdown=lambda old, new, why:
                               notes.append((old, new)))
        client.user_drafts()
        self.assertGreater(client.delay, 2.0)
        self.assertEqual(1, client.throttle_events)
        self.assertTrue(notes and notes[0][1] > notes[0][0])

    def test_slowdown_is_capped(self):
        client, _ = client_for({"/data/user": (200, {"drafts": []})},
                               delay=1000.0)
        client._slow_down("test")
        self.assertLessEqual(client.delay, 1000.0)

    def test_breaker_stops_a_struggling_service(self):
        client, transport = client_for({"/data/user": (500, b"boom")},
                                       retries=100, breaker=4)
        with self.assertRaises(ExportError) as caught:
            client.user_drafts()
        self.assertIn("looks unwell", str(caught.exception))
        self.assertEqual(4, len(transport.calls),
                         "must stop at the breaker, not keep retrying")

    def test_success_resets_the_breaker(self):
        seen = {"n": 0}

        def flaky(url):
            seen["n"] += 1
            return (500, b"x") if seen["n"] % 2 else (200, {"drafts": []})

        client, _ = client_for({"/data/user": flaky}, breaker=3)
        for _ in range(4):
            client.user_drafts()
        self.assertEqual(0, client.consecutive_failures)

    def test_jitter_keeps_traffic_off_a_metronome(self):
        class Rng(object):
            def random(self):
                return 0.5

        client, _ = client_for(full_routes(), delay=2.0, jitter=1.0,
                               rng=Rng())
        client.user_drafts()
        self.assertAlmostEqual(2.5, client.clock.sleeps[0], places=6)

    def test_request_budget_stops_cleanly_and_says_how_to_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, transport = client_for(full_routes(("a", "b", "c", "d")),
                                           max_requests=5)
            manifest = export_all(client, tmp)
            self.assertFalse(manifest["complete"])
            self.assertLessEqual(len(transport.calls), 5)
            self.assertTrue(any("re-run" in note
                                for note in manifest["notes"]), manifest)
            # Whatever completed is still on disk and skipped next time.
            client2, transport2 = client_for(full_routes(("a", "b", "c", "d")))
            second = export_all(client2, tmp)
            self.assertTrue(second["complete"])
            self.assertGreater(second["draftsCached"], 0)

    def test_long_pause_happens_on_schedule(self):
        client, _ = client_for(full_routes(("a", "b", "c")),
                               pause_every=2, pause_for=90.0)
        with tempfile.TemporaryDirectory() as tmp:
            export_all(client, tmp, skip_games=True)
        self.assertIn(90.0, client.clock.sleeps)

    def test_manifest_records_what_the_run_cost_the_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, _ = client_for(full_routes(), delay=1.0)
            manifest = export_all(client, tmp, skip_games=True)
            self.assertIn("requests", manifest)
            self.assertIn("sleptSeconds", manifest)
            self.assertIn("throttleEvents", manifest)
            self.assertIn("plan", manifest)


class EstimateTest(unittest.TestCase):

    def test_estimate_multiplies_parts_by_drafts(self):
        plan = estimate(100, ("draft", "deck"), delay=2.0, jitter=0.0)
        self.assertEqual(201, plan["requests"])
        self.assertEqual(402.0, plan["seconds"])

    def test_estimate_discounts_work_already_done(self):
        plan = estimate(100, ("draft",), delay=1.0, done=90)
        self.assertEqual(11, plan["requests"])

    def test_estimate_includes_long_pauses(self):
        plan = estimate(10, ("draft",), delay=1.0, pause_every=5,
                        pause_for=60.0, include_games=False)
        self.assertEqual(10, plan["requests"])
        self.assertEqual(10 + 2 * 60.0, plan["seconds"])

    def test_duration_formatting(self):
        self.assertEqual("45s", format_duration(45))
        self.assertEqual("2m 5s", format_duration(125))
        self.assertEqual("3h 20m", format_duration(12000))


class PlanModeTest(unittest.TestCase):

    def test_plan_spends_exactly_one_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, transport = client_for(
                full_routes(("a", "b", "c", "d", "e")))
            manifest = export_all(client, tmp, plan_only=True)
            self.assertEqual(1, len(transport.calls))
            self.assertEqual(5, manifest["draftsListed"])
            self.assertEqual(0, manifest["draftsWritten"])
            self.assertFalse(os.path.exists(
                os.path.join(tmp, "drafts", "a.json")))

    def test_plan_reports_requests_and_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, _ = client_for(full_routes(("a", "b")), delay=3.0)
            manifest = export_all(client, tmp, plan_only=True,
                                  parts=("draft", "deck"))
            self.assertEqual(5, manifest["plan"]["requests"])
            self.assertIn("human", manifest["plan"])


class PartsTest(unittest.TestCase):

    def test_parts_control_requests_per_draft(self):
        for parts, expected in ((("draft",), 1),
                                (("draft", "deck"), 2),
                                (("draft", "metadata", "deck", "pool"), 4)):
            with tempfile.TemporaryDirectory() as tmp:
                client, transport = client_for(full_routes())
                export_all(client, tmp, parts=parts, skip_games=True)
                # one /data/user plus the per-draft parts
                self.assertEqual(1 + expected, len(transport.calls), parts)

    def test_pool_is_off_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, transport = client_for(full_routes())
            export_all(client, tmp, skip_games=True)
            self.assertFalse(any("/data/pool" in call
                                 for call in transport.calls))

    def test_unknown_part_is_rejected_before_any_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            original, sys.stderr = sys.stderr, stderr
            try:
                code = main(["--out", tmp, "--parts", "draft,nonsense",
                             "--sharing-token", "t"])
            finally:
                sys.stderr = original
            self.assertEqual(1, code)
            self.assertIn("nonsense", stderr.getvalue())


class DraftListShapeTest(unittest.TestCase):

    def _ids(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            routes = dict(full_routes())
            routes["/data/user"] = (200, payload)
            client, _ = client_for(routes)
            manifest = export_all(client, tmp, skip_games=True)
            return manifest["draftsListed"]

    def test_bare_list_shape(self):
        self.assertEqual(2, self._ids([DRAFT_ROW, {"draft_id": "d2"}]))

    def test_wrapped_shapes(self):
        for key in ("drafts", "events", "data", "results"):
            self.assertEqual(1, self._ids({key: [DRAFT_ROW]}), key)

    def test_unknown_wrapper_still_found_by_content(self):
        self.assertEqual(1, self._ids({"somethingNew": [DRAFT_ROW]}))

    def test_unrecognisable_shape_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            routes = dict(full_routes())
            routes["/data/user"] = (200, {"total": 4})
            client, _ = client_for(routes)
            with self.assertRaises(ExportError) as caught:
                export_all(client, tmp, skip_games=True)
            self.assertIn("shape has changed", str(caught.exception))
            # The raw response is still on disk for inspection.
            self.assertTrue(os.path.exists(os.path.join(tmp, "user.json")))


class ExportTest(unittest.TestCase):

    def test_full_export_writes_every_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, _ = client_for(full_routes(("d1", "d2")))
            manifest = export_all(client, tmp, parts=DRAFT_PARTS)
            self.assertEqual(2, manifest["draftsWritten"])
            self.assertEqual([], manifest["failures"])
            self.assertEqual(1, manifest["games"])
            bundle = _read(os.path.join(tmp, "drafts", "d1.json"))
            for part in ("draft", "eventMetadata", "deck", "pool"):
                self.assertIn(part, bundle)
            self.assertTrue(os.path.exists(
                os.path.join(tmp, "export_manifest.json")))

    def test_rerun_resumes_instead_of_refetching(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, first = client_for(full_routes(("d1", "d2")))
            export_all(client, tmp)
            client2, second = client_for(full_routes(("d1", "d2")))
            manifest = export_all(client2, tmp)
            self.assertEqual(0, manifest["draftsWritten"])
            self.assertEqual(2, manifest["draftsCached"])
            self.assertLess(len(second.calls), len(first.calls))

    def test_one_failed_part_does_not_lose_the_draft(self):
        routes = dict(full_routes())
        routes["/data/pool"] = (500, b"boom")
        with tempfile.TemporaryDirectory() as tmp:
            client, _ = client_for(routes, retries=0, breaker=0)
            manifest = export_all(client, tmp, parts=DRAFT_PARTS)
            self.assertEqual(1, manifest["draftsWritten"])
            self.assertEqual(1, len(manifest["failures"]))
            bundle = _read(os.path.join(tmp, "drafts", "d1.json"))
            self.assertIn("draft", bundle)
            self.assertEqual(["pool"], bundle["incomplete"])

    def test_game_ceiling_is_reported_not_hidden(self):
        routes = dict(full_routes())
        routes["/data/user_game_list"] = (
            200, [{"game_id": str(index)} for index in range(100)])
        with tempfile.TemporaryDirectory() as tmp:
            client, _ = client_for(routes)
            manifest = export_all(client, tmp)
            self.assertEqual(100, manifest["games"])
            self.assertTrue(any("ceiling" in note
                                for note in manifest["notes"]), manifest)

    def test_limit_and_no_pool_reduce_the_request_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, transport = client_for(full_routes(("d1", "d2", "d3")))
            manifest = export_all(client, tmp, limit=1,
                                  parts=("draft", "deck"), skip_games=True)
            self.assertEqual(3, manifest["draftsListed"])
            self.assertEqual(1, manifest["draftsWritten"])
            self.assertFalse(any("/data/pool" in call
                                 for call in transport.calls))

    def test_secrets_are_fingerprinted_not_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, _ = client_for(full_routes(), cookie="session=SECRET")
            export_all(client, tmp, skip_games=True)
            manifest = _read(os.path.join(tmp, "export_manifest.json"))
            self.assertTrue(
                manifest["cookieFingerprint"].startswith("sha256:"))
            with open(os.path.join(tmp, "export_manifest.json"),
                      encoding="utf-8") as handle:
                self.assertNotIn("SECRET", handle.read())


class CliTest(unittest.TestCase):

    def _run(self, argv, env=None):
        stderr = io.StringIO()
        original, sys.stderr = sys.stderr, stderr
        saved = dict(os.environ)
        try:
            if env:
                os.environ.update(env)
            code = main(argv)
        finally:
            sys.stderr = original
            os.environ.clear()
            os.environ.update(saved)
        return code, stderr.getvalue()

    def test_missing_credentials_explains_how_to_get_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, stderr = self._run(["--out", os.path.join(tmp, "x")])
            self.assertEqual(1, code)
            self.assertIn("--cookie-file", stderr)

    def test_missing_cookie_file_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, stderr = self._run(
                ["--out", os.path.join(tmp, "x"),
                 "--cookie-file", os.path.join(tmp, "absent")])
            self.assertEqual(1, code)
            self.assertIn("cookie file not found", stderr)

    def test_empty_cookie_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cookie")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("   \n")
            code, stderr = self._run(["--out", os.path.join(tmp, "x"),
                                      "--cookie-file", path])
            self.assertEqual(1, code)
            self.assertIn("empty", stderr)

    def test_argument_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            for argv, needle in (
                    (["--out", tmp, "--limit", "-1"], "--limit"),
                    (["--out", tmp, "--delay", "-1"], "--delay")):
                code, stderr = self._run(argv)
                self.assertEqual(2, code)
                self.assertIn(needle, stderr)

    def test_help_documents_how_to_obtain_the_cookie(self):
        text = build_parser().format_help()
        self.assertIn("developer", text)
        self.assertIn("shell history", text)


def _read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    unittest.main()
