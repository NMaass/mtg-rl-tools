import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.seventeenlands_export import (
    ExportError,
    SeventeenLandsClient,
    build_parser,
    export_all,
    main,
)


class FakeTransport(object):
    """Stands in for the network: maps URL substrings to canned responses."""

    def __init__(self, routes, default=(404, b"nope")):
        self.routes = routes
        self.default = default
        self.calls = []
        self.headers = []

    def __call__(self, url, headers):
        self.calls.append(url)
        self.headers.append(headers)
        for needle, response in self.routes.items():
            if needle in url:
                if callable(response):
                    response = response(url)
                status, body = response
                if isinstance(body, (dict, list)):
                    body = json.dumps(body).encode("utf-8")
                return status, body
        return self.default


def client_for(routes, **kwargs):
    transport = FakeTransport(routes)
    kwargs.setdefault("cookie", "session=abc")
    kwargs.setdefault("delay", 0)
    client = SeventeenLandsClient(opener=transport, sleep=lambda _s: None,
                                 **kwargs)
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
        slept = []
        transport = FakeTransport(full_routes())
        client = SeventeenLandsClient(cookie="c", opener=transport,
                                      sleep=slept.append, delay=1.0)
        client.user_drafts()
        client.user_drafts()
        self.assertTrue(slept, "second request must wait")
        self.assertLessEqual(slept[0], 1.0)


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
            manifest = export_all(client, tmp)
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
            client, _ = client_for(routes, retries=0)
            manifest = export_all(client, tmp)
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
            manifest = export_all(client, tmp, limit=1, include_pool=False,
                                  skip_games=True)
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
