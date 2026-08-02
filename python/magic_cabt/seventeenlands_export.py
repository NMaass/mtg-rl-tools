"""Export your own 17Lands history to local JSON.

17Lands has no "download my data" button. What it has is the private JSON API
its own site calls, and a Mythic patron's account is backfilled with their
whole history. This tool walks that API as *you*, saves every response
verbatim, and reports what it could and could not reach.

    magic-cabt-17lands-export --cookie-file ~/.17lands-cookie --out 17lands-export

The shape of the harvest:

    /data/user                      every draft you have played (id, set, record)
      -> /data/draft?draft_id=       pick-by-pick for that draft
      -> /data/event_metadata?...    format, links, wins/losses
      -> /data/deck?...&deck_index=  the deck you registered
      -> /data/pool?...&deck_index=  the pool you drafted from
    /data/user_game_list            recent games, with replay links

**Credentials.** This tool never asks for, accepts, or stores a password. You
authenticate the way your browser already has: copy the session cookie out of
it into a file (see ``docs/SEVENTEENLANDS_EXPORT.md``), or use a sharing token
for the endpoints that accept one. The cookie is read from a file rather than
an argument so it does not land in your shell history, and it is never written
into the output — the manifest records only a fingerprint of it.

**Being a good guest.** This is someone's free service and an undocumented
API. Requests are serialized with a delay between them (default 1.0s), retry
with backoff on 429/5xx, and stop entirely on 401/403 rather than hammering.
Re-running skips drafts already on disk, so an interrupted export resumes
instead of refetching.

**Known ceiling.** ``/data/user_game_list`` returns roughly the last hundred
games regardless of patron tier, so per-game data does not go back the way
draft data does. The exporter reports that gap explicitly instead of
presenting a partial harvest as complete: getting the full game history needs
17Lands themselves (see the letter in the docs page).

This module is deliberately conservative about what it claims. The API is
undocumented and unversioned, so every endpoint is probed and the result
recorded; an endpoint that has moved shows up as a named failure in the
report, not as silently missing data.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

__all__ = [
    "BASE_URL",
    "ExportError",
    "SeventeenLandsClient",
    "build_parser",
    "export_all",
    "main",
]

BASE_URL = "https://www.17lands.com"

# The site's own XHR paths, as used by its front end. Undocumented and
# unversioned: treat every one as something that may have moved.
USER_DRAFTS_PATH = "/data/user"
DRAFT_PATH = "/data/draft"
EVENT_METADATA_PATH = "/data/event_metadata"
DECK_PATH = "/data/deck"
POOL_PATH = "/data/pool"
GAME_LIST_PATH = "/data/user_game_list"

# The game list is served either from the session (no token) or by sharing
# token as a path segment. Both spellings are tried.
GAME_LIST_TOKEN_PATH = "/data/user_game_list/%s"

DEFAULT_USER_AGENT = ("magic-cabt-17lands-export/1 "
                      "(+https://github.com/NMaass/mtg-rl-tools)")

# Serialized requests with a pause between them. Slower than the service could
# take, deliberately.
DEFAULT_DELAY_SECONDS = 1.0
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 30

_RETRY_STATUS = frozenset((429, 500, 502, 503, 504))
_FATAL_STATUS = frozenset((401, 403))


class ExportError(Exception):
    """Something the user has to resolve; the message is the instruction."""


class SeventeenLandsClient(object):
    """Serialized, retrying JSON client for the 17Lands site API.

    ``opener`` exists so tests can inject a transport: it is called with
    ``(url, headers)`` and returns ``(status, body_bytes)``. The default goes
    to the network through ``urllib``.
    """

    def __init__(self, cookie=None, sharing_token=None, base_url=BASE_URL,
                 delay=DEFAULT_DELAY_SECONDS, retries=DEFAULT_RETRIES,
                 timeout=DEFAULT_TIMEOUT, opener=None, sleep=None,
                 user_agent=DEFAULT_USER_AGENT):
        self.cookie = cookie
        self.sharing_token = sharing_token
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.retries = retries
        self.timeout = timeout
        self.user_agent = user_agent
        self._opener = opener or self._urlopen
        self._sleep = sleep if sleep is not None else time.sleep
        self._last_request = None
        self.requests = 0

    # -- transport ---------------------------------------------------------

    def _urlopen(self, url, headers):
        request = urllib.request.Request(url, headers=headers)
        try:
            handle = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            return error.code, error.read()
        except urllib.error.URLError as error:
            raise ExportError(
                "could not reach %s (%s). Check your network; if you are "
                "behind a proxy that blocks 17lands.com, run this from a "
                "machine that can reach it." % (self.base_url, error.reason))
        try:
            return handle.getcode(), handle.read()
        finally:
            handle.close()

    def _throttle(self):
        if self._last_request is None or not self.delay:
            return
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay:
            self._sleep(self.delay - elapsed)

    def _headers(self):
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        if self.cookie:
            headers["Cookie"] = self.cookie
        return headers

    # -- requests ----------------------------------------------------------

    def get_json(self, path, params=None):
        """Fetch one JSON document, retrying transient failures.

        Returns the decoded body. Raises ExportError with an actionable
        message on auth failure, on a body that is not JSON (which in practice
        means the login page came back instead), and on exhausted retries.
        """
        url = self.base_url + path
        if params:
            url = "%s?%s" % (url, urllib.parse.urlencode(sorted(
                (key, value) for key, value in params.items()
                if value is not None)))

        attempt = 0
        while True:
            self._throttle()
            status, body = self._opener(url, self._headers())
            self._last_request = time.monotonic()
            self.requests += 1

            if status in _FATAL_STATUS:
                raise ExportError(
                    "17Lands rejected the request with HTTP %d (%s). The "
                    "session cookie is missing, expired, or for a different "
                    "account -- copy a fresh one out of your browser."
                    % (status, path))
            if status in _RETRY_STATUS and attempt < self.retries:
                # 429 in particular means back off, not try harder.
                self._sleep(self.delay * (2 ** attempt))
                attempt += 1
                continue
            if status != 200:
                raise ExportError(
                    "17Lands returned HTTP %d for %s after %d attempt(s). "
                    "The endpoint may have moved; nothing was guessed in its "
                    "place." % (status, path, attempt + 1))

            try:
                return json.loads(body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                raise ExportError(
                    "%s returned a non-JSON body -- this usually means the "
                    "request was not recognised as logged in, and the login "
                    "page came back instead. Refresh your cookie." % path)

    # -- typed calls -------------------------------------------------------

    def user_drafts(self):
        """Every draft on the account. This is the unbounded history."""
        return self.get_json(USER_DRAFTS_PATH)

    def draft(self, draft_id):
        return self.get_json(DRAFT_PATH, {"draft_id": draft_id})

    def event_metadata(self, draft_id):
        return self.get_json(EVENT_METADATA_PATH, {"draft_id": draft_id})

    def deck(self, draft_id, deck_index=0):
        return self.get_json(DECK_PATH, {"draft_id": draft_id,
                                         "deck_index": deck_index})

    def pool(self, draft_id, deck_index=0):
        return self.get_json(POOL_PATH, {"draft_id": draft_id,
                                         "deck_index": deck_index})

    def game_list(self):
        """Recent games. Bounded by the service at roughly 100 rows."""
        if self.sharing_token:
            return self.get_json(GAME_LIST_TOKEN_PATH % urllib.parse.quote(
                self.sharing_token))
        return self.get_json(GAME_LIST_PATH)


# ---------------------------------------------------------------------------
# Harvest


def _draft_rows(payload):
    """Pull the draft list out of whatever shape /data/user returns.

    The response has been seen both as a bare list and wrapped under a key.
    Rather than assume, look for the first list of dicts that carries
    something draft-shaped.
    """
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in ("drafts", "events", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
        if rows is None:
            for value in payload.values():
                if isinstance(value, list) and value and \
                        isinstance(value[0], dict):
                    rows = value
                    break
        if rows is None:
            raise ExportError(
                "could not find a draft list in the /data/user response "
                "(keys: %s). The endpoint's shape has changed; the raw "
                "response was saved so it can be inspected."
                % ", ".join(sorted(str(key) for key in payload)))
    else:
        raise ExportError("/data/user returned %s, not a draft list"
                          % type(payload).__name__)
    return [row for row in rows if isinstance(row, dict)]


def _draft_id(row):
    for key in ("draft_id", "id", "aggregate_id", "event_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _atomic_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _fingerprint(secret):
    if not secret:
        return None
    return "sha256:" + hashlib.sha256(
        secret.encode("utf-8")).hexdigest()[:16]


def export_all(client, out_dir, limit=0, include_pool=True, say=None,
               skip_games=False):
    """Walk the account and write everything reachable under ``out_dir``.

    Returns a manifest dict. Per-draft files are written as they arrive and
    skipped when already present, so an interrupted run resumes.
    """
    say = say or (lambda text: None)
    drafts_dir = os.path.join(out_dir, "drafts")
    os.makedirs(drafts_dir, exist_ok=True)

    say("fetching your draft list")
    user_payload = client.user_drafts()
    _atomic_json(os.path.join(out_dir, "user.json"), user_payload)
    rows = _draft_rows(user_payload)
    say("account has %d draft(s)" % len(rows))

    ordered = rows[:limit] if limit else rows
    manifest = {
        "schemaVersion": 1,
        "kind": "seventeenlands-personal-export-v1",
        "source": client.base_url,
        "draftsListed": len(rows),
        "draftsRequested": len(ordered),
        "draftsWritten": 0,
        "draftsCached": 0,
        "failures": [],
        "games": None,
        "notes": [],
    }

    for index, row in enumerate(ordered, start=1):
        draft_id = _draft_id(row)
        if not draft_id:
            manifest["failures"].append(
                {"draft": None, "error": "row carries no draft id",
                 "row": row})
            continue
        path = os.path.join(drafts_dir, "%s.json" % draft_id)
        if os.path.exists(path):
            manifest["draftsCached"] += 1
            continue

        say("draft %d/%d %s" % (index, len(ordered), draft_id[:12]))
        bundle = {"summary": row, "draftId": draft_id}
        failed = []
        for name, call in (
                ("draft", lambda: client.draft(draft_id)),
                ("eventMetadata", lambda: client.event_metadata(draft_id)),
                ("deck", lambda: client.deck(draft_id)),
                ("pool", (lambda: client.pool(draft_id))
                 if include_pool else None)):
            if call is None:
                continue
            try:
                bundle[name] = call()
            except ExportError as error:
                # One missing part must not cost the rest of the draft: record
                # it and keep the pieces that did arrive.
                failed.append({"draft": draft_id, "part": name,
                               "error": str(error)})
        if failed:
            manifest["failures"].extend(failed)
            bundle["incomplete"] = [item["part"] for item in failed]
        _atomic_json(path, bundle)
        manifest["draftsWritten"] += 1

    if not skip_games:
        say("fetching your recent game list")
        try:
            games = client.game_list()
            _atomic_json(os.path.join(out_dir, "game_list.json"), games)
            count = len(games) if isinstance(games, list) else \
                len(games.get("games") or []) if isinstance(games, dict) else 0
            manifest["games"] = count
            if count >= 90:
                manifest["notes"].append(
                    "the game list came back at %d rows, at or near the "
                    "service's ~100-game ceiling: older games are not "
                    "retrievable this way. Ask 17Lands directly for the full "
                    "game history (see docs/SEVENTEENLANDS_EXPORT.md)."
                    % count)
        except ExportError as error:
            manifest["failures"].append({"part": "gameList",
                                         "error": str(error)})

    manifest["requests"] = client.requests
    manifest["cookieFingerprint"] = _fingerprint(client.cookie)
    manifest["sharingTokenFingerprint"] = _fingerprint(client.sharing_token)
    _atomic_json(os.path.join(out_dir, "export_manifest.json"), manifest)
    return manifest


# ---------------------------------------------------------------------------
# CLI


def _read_cookie(args):
    if args.cookie_file:
        path = os.path.abspath(os.path.expanduser(args.cookie_file))
        if not os.path.isfile(path):
            raise ExportError("cookie file not found: %s" % path)
        with open(path, "r", encoding="utf-8") as handle:
            cookie = handle.read().strip()
        if not cookie:
            raise ExportError("cookie file %s is empty" % path)
        return cookie
    return os.environ.get("SEVENTEENLANDS_COOKIE") or None


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-17lands-export",
        description="Export your own 17Lands draft history to local JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="getting a cookie:\n"
               "  Log in to 17lands.com in your browser, open the developer\n"
               "  tools, and copy the whole Cookie request header of any\n"
               "  XHR to /data/. Save it to a file (chmod 600) and pass it\n"
               "  with --cookie-file. Never paste it as an argument: it\n"
               "  would land in your shell history.\n"
               "\n"
               "examples:\n"
               "  magic-cabt-17lands-export --cookie-file ~/.17lands-cookie \\\n"
               "      --out 17lands-export\n"
               "  magic-cabt-17lands-export --cookie-file ~/.17lands-cookie \\\n"
               "      --out 17lands-export --limit 3   # try a few first\n")
    parser.add_argument("--out", required=True,
                        help="directory to write the export into")
    parser.add_argument("--cookie-file",
                        help="file holding your 17Lands Cookie header "
                             "(or set SEVENTEENLANDS_COOKIE)")
    parser.add_argument("--sharing-token",
                        help="sharing token, for the endpoints that take one "
                             "instead of a session")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="only fetch the first N drafts (0 = all)")
    parser.add_argument("--delay", type=float,
                        default=DEFAULT_DELAY_SECONDS,
                        help="seconds between requests (default: %(default)s)")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--no-pool", action="store_true",
                        help="skip the drafted-pool call for each draft")
    parser.add_argument("--skip-games", action="store_true",
                        help="skip the recent-games list")
    parser.add_argument("--base-url", default=BASE_URL,
                        help=argparse.SUPPRESS)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.limit < 0:
        sys.stderr.write("error: --limit must be >= 0\n")
        return 2
    if args.delay < 0:
        sys.stderr.write("error: --delay must be >= 0\n")
        return 2

    def say(text):
        if not args.quiet:
            sys.stderr.write("[17lands-export] %s\n" % text)

    try:
        cookie = _read_cookie(args)
        if not cookie and not args.sharing_token:
            raise ExportError(
                "no credentials: pass --cookie-file (or set "
                "SEVENTEENLANDS_COOKIE), or --sharing-token. See --help for "
                "how to copy the cookie out of your browser.")
        client = SeventeenLandsClient(
            cookie=cookie, sharing_token=args.sharing_token,
            base_url=args.base_url, delay=args.delay, retries=args.retries,
            timeout=args.timeout)
        os.makedirs(os.path.abspath(os.path.expanduser(args.out)),
                    exist_ok=True)
        manifest = export_all(
            client, os.path.abspath(os.path.expanduser(args.out)),
            limit=args.limit, include_pool=not args.no_pool, say=say,
            skip_games=args.skip_games)
    except ExportError as error:
        sys.stderr.write("error: %s\n" % error)
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\n[17lands-export] stopped; re-run to resume "
                         "where it left off\n")
        return 130

    say("wrote %d draft(s) (%d already present), %s game rows, %d request(s)"
        % (manifest["draftsWritten"], manifest["draftsCached"],
           manifest["games"], manifest["requests"]))
    for note in manifest["notes"]:
        say("note: %s" % note)
    if manifest["failures"]:
        say("%d part(s) failed; see export_manifest.json"
            % len(manifest["failures"]))
    say("export: %s" % os.path.join(args.out, "export_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
