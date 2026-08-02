"""Export your own 17Lands history to local JSON, slowly and politely.

17Lands does not offer data exports. What it has is the private JSON API its
own site calls, and a Mythic patron's account is backfilled with their whole
history. This tool walks that API as *you*, saves every response verbatim, and
reports what it could and could not reach.

    magic-cabt-17lands-export --cookie-file ~/.17lands-cookie --out export --plan
    magic-cabt-17lands-export --cookie-file ~/.17lands-cookie --out export

The shape of the harvest:

    /data/user                      every draft you have played (id, set, record)
      -> /data/draft?draft_id=       pick-by-pick for that draft
      -> /data/event_metadata?...    format, links, wins/losses
      -> /data/deck?...&deck_index=  the deck you registered
      -> /data/pool?...&deck_index=  the pool you drafted from (off by default)
    /data/user_game_list            recent games, with replay links

**Credentials.** This tool never asks for, accepts, or stores a password. You
authenticate the way your browser already has: copy the session cookie out of
it into a file (see ``docs/SEVENTEENLANDS_EXPORT.md``). The cookie is read from
a file rather than an argument so it does not land in your shell history, and
it is never written into the output -- the manifest records only a fingerprint.

**This is someone else's free service.** A full history is thousands of
requests, so the whole design is about spending them carefully:

- *Ask for less.* Requests per draft is the multiplier on everything, so only
  the parts you name are fetched (default three of four; the drafted pool is
  the multiset of your own picks, so it is redundant with them). ``--parts
  draft`` alone is one request per draft.
- *Go slowly.* One request at a time, a pause between them, and jitter so the
  traffic never becomes a metronome.
- *Take the hint.* ``Retry-After`` is obeyed when given. Any 429 also slows
  the rest of the run permanently -- if the service says it is busy once, this
  does not go back to its old pace.
- *Give up early.* A run of consecutive failures trips a breaker and stops. A
  struggling server should not also have to carry us.
- *Stop before it hurts.* ``--max-requests`` and ``--pause-every`` let a
  harvest be spread over days. Stopping is free: finished drafts are skipped
  on the next run.
- *Know the cost first.* ``--plan`` spends exactly one request and prints how
  many the real run would need and how long it would take.

**Known ceiling.** ``/data/user_game_list`` returns roughly the last hundred
games regardless of patron tier, so per-game data does not go back the way
draft data does. That gap is reported rather than papered over.

The API is undocumented and unversioned, so every endpoint is probed and the
result recorded; an endpoint that has moved shows up as a named failure in the
report, not as silently missing data.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

__all__ = [
    "BASE_URL",
    "DRAFT_PARTS",
    "ExportError",
    "SeventeenLandsClient",
    "build_parser",
    "estimate",
    "export_all",
    "format_duration",
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
GAME_LIST_TOKEN_PATH = "/data/user_game_list/%s"

# Per-draft parts, in fetch order. "pool" is excluded by default: the pool you
# drafted is the multiset of the picks already in "draft".
DRAFT_PARTS = ("draft", "metadata", "deck", "pool")
DEFAULT_PARTS = ("draft", "metadata", "deck")

DEFAULT_USER_AGENT = ("magic-cabt-17lands-export/1 "
                      "(+https://github.com/NMaass/mtg-rl-tools)")

# Deliberately slower than the service could take.
DEFAULT_DELAY_SECONDS = 2.0
DEFAULT_JITTER_SECONDS = 0.75
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 30
# A 429 multiplies the delay by this for the remainder of the run.
THROTTLE_GROWTH = 1.5
MAX_DELAY_SECONDS = 60.0
# Longest Retry-After we will sit through before giving up on the run.
MAX_RETRY_AFTER_SECONDS = 300
# Consecutive failed requests before concluding the service is unwell.
DEFAULT_BREAKER = 5

_RETRY_STATUS = frozenset((429, 500, 502, 503, 504))
_FATAL_STATUS = frozenset((401, 403))


class ExportError(Exception):
    """Something the user has to resolve; the message is the instruction."""


class BudgetReached(Exception):
    """Internal: the session's self-imposed request budget is spent."""


class SeventeenLandsClient(object):
    """Serialized, self-throttling JSON client for the 17Lands site API.

    ``opener`` exists so tests can inject a transport: it is called with
    ``(url, headers)`` and returns ``(status, body_bytes, headers_dict)``. The
    default goes to the network through ``urllib``.
    """

    def __init__(self, cookie=None, sharing_token=None, base_url=BASE_URL,
                 delay=DEFAULT_DELAY_SECONDS, jitter=DEFAULT_JITTER_SECONDS,
                 retries=DEFAULT_RETRIES, timeout=DEFAULT_TIMEOUT,
                 max_requests=0, pause_every=0, pause_for=0.0,
                 breaker=DEFAULT_BREAKER, opener=None, sleep=None, rng=None,
                 user_agent=DEFAULT_USER_AGENT, on_slowdown=None):
        self.cookie = cookie
        self.sharing_token = sharing_token
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.base_delay = delay
        self.jitter = jitter
        self.retries = retries
        self.timeout = timeout
        self.max_requests = max_requests
        self.pause_every = pause_every
        self.pause_for = pause_for
        self.breaker = breaker
        self.user_agent = user_agent
        self._opener = opener or self._urlopen
        self._sleep = sleep if sleep is not None else time.sleep
        self._rng = rng or random.Random()
        self._on_slowdown = on_slowdown or (lambda old, new, why: None)
        self._last_request = None
        self.requests = 0
        self.throttle_events = 0
        self.consecutive_failures = 0
        self.slept_seconds = 0.0

    # -- transport ---------------------------------------------------------

    def _urlopen(self, url, headers):
        request = urllib.request.Request(url, headers=headers)
        try:
            handle = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            return error.code, error.read(), dict(error.headers or {})
        except urllib.error.URLError as error:
            raise ExportError(
                "could not reach %s (%s). Check your network; if you are "
                "behind a proxy that blocks 17lands.com, run this from a "
                "machine that can reach it." % (self.base_url, error.reason))
        try:
            return handle.getcode(), handle.read(), dict(handle.headers or {})
        finally:
            handle.close()

    # -- pacing ------------------------------------------------------------

    def _wait(self, seconds):
        if seconds <= 0:
            return
        self.slept_seconds += seconds
        self._sleep(seconds)

    def _throttle(self):
        """Space this request from the last, with jitter."""
        target = self.delay + (self._rng.random() * self.jitter
                               if self.jitter else 0.0)
        if self._last_request is not None:
            elapsed = time.monotonic() - self._last_request
            target -= elapsed
        self._wait(target)

    def _slow_down(self, why):
        """A 429 means back off for good, not just for this request."""
        if self.delay >= MAX_DELAY_SECONDS:
            return
        old = self.delay
        self.delay = min(MAX_DELAY_SECONDS, self.delay * THROTTLE_GROWTH)
        self.throttle_events += 1
        self._on_slowdown(old, self.delay, why)

    def _retry_after(self, headers):
        raw = None
        for key, value in (headers or {}).items():
            if key.lower() == "retry-after":
                raw = value
                break
        if raw is None:
            return None
        try:
            seconds = float(str(raw).strip())
        except ValueError:
            # HTTP-date form: we do not need the precision, so treat an
            # unparsable value as "wait a while" rather than ignoring it.
            return min(60.0, MAX_RETRY_AFTER_SECONDS)
        if seconds > MAX_RETRY_AFTER_SECONDS:
            raise ExportError(
                "17Lands asked us to wait %.0f seconds, which is longer than "
                "this run will sit. Stopping; re-run later and it resumes "
                "where it left off." % seconds)
        return max(0.0, seconds)

    def _maybe_long_pause(self, say):
        if not self.pause_every or not self.pause_for:
            return
        if self.requests and self.requests % self.pause_every == 0:
            say("pausing %s after %d requests"
                % (format_duration(self.pause_for), self.requests))
            self._wait(self.pause_for)

    # -- requests ----------------------------------------------------------

    def get_json(self, path, params=None, say=None):
        """Fetch one JSON document, retrying transient failures politely."""
        say = say or (lambda text: None)
        if self.max_requests and self.requests >= self.max_requests:
            raise BudgetReached()

        url = self.base_url + path
        if params:
            url = "%s?%s" % (url, urllib.parse.urlencode(sorted(
                (key, value) for key, value in params.items()
                if value is not None)))

        attempt = 0
        while True:
            self._throttle()
            status, body, headers = self._opener(url, self._headers())
            self._last_request = time.monotonic()
            self.requests += 1

            if status in _FATAL_STATUS:
                raise ExportError(
                    "17Lands rejected the request with HTTP %d (%s). The "
                    "session cookie is missing, expired, or for a different "
                    "account -- copy a fresh one out of your browser."
                    % (status, path))

            if status in _RETRY_STATUS:
                self.consecutive_failures += 1
                if self.breaker and \
                        self.consecutive_failures >= self.breaker:
                    raise ExportError(
                        "%d requests in a row failed (last: HTTP %d on %s). "
                        "The service looks unwell, so this is stopping rather "
                        "than adding to it. Everything fetched so far is "
                        "saved; re-run later to resume."
                        % (self.consecutive_failures, status, path))
                if status == 429:
                    self._slow_down("HTTP 429 on %s" % path)
                wait = self._retry_after(headers)
                if attempt < self.retries:
                    if wait is None:
                        wait = self.delay * (2 ** attempt)
                    say("HTTP %d; waiting %s before retrying"
                        % (status, format_duration(wait)))
                    self._wait(wait)
                    attempt += 1
                    continue
                raise ExportError(
                    "17Lands returned HTTP %d for %s after %d attempt(s). "
                    "Nothing was guessed in its place."
                    % (status, path, attempt + 1))

            if status != 200:
                self.consecutive_failures += 1
                raise ExportError(
                    "17Lands returned HTTP %d for %s. The endpoint may have "
                    "moved; nothing was guessed in its place." % (status, path))

            self.consecutive_failures = 0
            self._maybe_long_pause(say)
            try:
                return json.loads(body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                raise ExportError(
                    "%s returned a non-JSON body -- this usually means the "
                    "request was not recognised as logged in, and the login "
                    "page came back instead. Refresh your cookie." % path)

    def _headers(self):
        headers = {"User-Agent": self.user_agent,
                   "Accept": "application/json"}
        if self.cookie:
            headers["Cookie"] = self.cookie
        return headers

    # -- typed calls -------------------------------------------------------

    def user_drafts(self, say=None):
        """Every draft on the account. This is the unbounded history."""
        return self.get_json(USER_DRAFTS_PATH, say=say)

    def draft(self, draft_id, say=None):
        return self.get_json(DRAFT_PATH, {"draft_id": draft_id}, say=say)

    def event_metadata(self, draft_id, say=None):
        return self.get_json(EVENT_METADATA_PATH, {"draft_id": draft_id},
                             say=say)

    def deck(self, draft_id, deck_index=0, say=None):
        return self.get_json(DECK_PATH, {"draft_id": draft_id,
                                         "deck_index": deck_index}, say=say)

    def pool(self, draft_id, deck_index=0, say=None):
        return self.get_json(POOL_PATH, {"draft_id": draft_id,
                                         "deck_index": deck_index}, say=say)

    def game_list(self, say=None):
        """Recent games. Bounded by the service at roughly 100 rows."""
        if self.sharing_token:
            return self.get_json(GAME_LIST_TOKEN_PATH % urllib.parse.quote(
                self.sharing_token), say=say)
        return self.get_json(GAME_LIST_PATH, say=say)


# ---------------------------------------------------------------------------
# Estimation


def format_duration(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return "%ds" % seconds
    if seconds < 3600:
        return "%dm %ds" % (seconds // 60, seconds % 60)
    return "%dh %dm" % (seconds // 3600, (seconds % 3600) // 60)


def estimate(drafts, parts, delay, jitter=0.0, include_games=True,
             pause_every=0, pause_for=0.0, done=0):
    """What a full run would cost the service, before making it pay."""
    remaining = max(0, drafts - done)
    requests = remaining * len(parts) + (1 if include_games else 0)
    per_request = delay + jitter / 2.0
    seconds = requests * per_request
    if pause_every and pause_for:
        seconds += (requests // pause_every) * pause_for
    return {"drafts": drafts, "draftsRemaining": remaining,
            "requests": requests, "seconds": seconds,
            "human": format_duration(seconds)}


# ---------------------------------------------------------------------------
# Harvest


def _draft_rows(payload):
    """Pull the draft list out of whatever shape /data/user returns."""
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
    return "sha256:" + hashlib.sha256(secret.encode("utf-8")).hexdigest()[:16]


def export_all(client, out_dir, limit=0, parts=DEFAULT_PARTS, say=None,
               skip_games=False, plan_only=False):
    """Walk the account and write everything reachable under ``out_dir``.

    Per-draft files are written as they arrive and skipped when already
    present, so stopping is free and re-running resumes.
    """
    say = say or (lambda text: None)
    drafts_dir = os.path.join(out_dir, "drafts")
    os.makedirs(drafts_dir, exist_ok=True)

    say("fetching your draft list")
    user_payload = client.user_drafts(say=say)
    _atomic_json(os.path.join(out_dir, "user.json"), user_payload)
    rows = _draft_rows(user_payload)

    ordered = rows[:limit] if limit else rows
    already = sum(1 for row in ordered
                  if _draft_id(row) and os.path.exists(os.path.join(
                      drafts_dir, "%s.json" % _draft_id(row))))
    plan = estimate(len(ordered), parts, client.delay, client.jitter,
                    include_games=not skip_games,
                    pause_every=client.pause_every,
                    pause_for=client.pause_for, done=already)
    say("account has %d draft(s); %d already saved" % (len(rows), already))
    say("this run would make %d request(s) over about %s"
        % (plan["requests"], plan["human"]))

    manifest = {
        "schemaVersion": 1,
        "kind": "seventeenlands-personal-export-v1",
        "source": client.base_url,
        "parts": list(parts),
        "draftsListed": len(rows),
        "draftsRequested": len(ordered),
        "draftsWritten": 0,
        "draftsCached": already,
        "failures": [],
        "games": None,
        "notes": [],
        "plan": plan,
        "complete": False,
    }
    if plan_only:
        manifest["notes"].append(
            "plan only: one request was made, to list your drafts")
        _finish(manifest, client, out_dir)
        return manifest

    budget_hit = False
    try:
        for index, row in enumerate(ordered, start=1):
            draft_id = _draft_id(row)
            if not draft_id:
                manifest["failures"].append(
                    {"draft": None, "error": "row carries no draft id",
                     "row": row})
                continue
            path = os.path.join(drafts_dir, "%s.json" % draft_id)
            if os.path.exists(path):
                continue

            say("draft %d/%d %s" % (index, len(ordered), draft_id[:12]))
            bundle = {"summary": row, "draftId": draft_id}
            failed = []
            calls = {
                "draft": lambda: client.draft(draft_id, say=say),
                "metadata": lambda: client.event_metadata(draft_id, say=say),
                "deck": lambda: client.deck(draft_id, say=say),
                "pool": lambda: client.pool(draft_id, say=say),
            }
            keys = {"draft": "draft", "metadata": "eventMetadata",
                    "deck": "deck", "pool": "pool"}
            for part in parts:
                try:
                    bundle[keys[part]] = calls[part]()
                except ExportError as error:
                    # One missing part must not cost the rest of the draft.
                    failed.append({"draft": draft_id, "part": part,
                                   "error": str(error)})
            if failed:
                manifest["failures"].extend(failed)
                bundle["incomplete"] = [item["part"] for item in failed]
            _atomic_json(path, bundle)
            manifest["draftsWritten"] += 1

        if not skip_games:
            say("fetching your recent game list")
            try:
                games = client.game_list(say=say)
                _atomic_json(os.path.join(out_dir, "game_list.json"), games)
                count = len(games) if isinstance(games, list) else \
                    len(games.get("games") or []) \
                    if isinstance(games, dict) else 0
                manifest["games"] = count
                if count >= 90:
                    manifest["notes"].append(
                        "the game list came back at %d rows, at or near the "
                        "service's ~100-game ceiling: older games are not "
                        "retrievable this way. Ask 17Lands directly for the "
                        "full game history (see "
                        "docs/SEVENTEENLANDS_EXPORT.md)." % count)
            except ExportError as error:
                manifest["failures"].append({"part": "gameList",
                                             "error": str(error)})
        manifest["complete"] = True
    except BudgetReached:
        budget_hit = True
        manifest["notes"].append(
            "stopped at the --max-requests budget of %d. Nothing is lost: "
            "re-run the same command to continue from here."
            % client.max_requests)
        say("request budget reached; stopping cleanly")

    if not budget_hit and manifest["draftsWritten"] and \
            manifest["draftsWritten"] + manifest["draftsCached"] < \
            len(ordered):
        manifest["notes"].append(
            "some drafts were listed but not written; see failures")
    _finish(manifest, client, out_dir)
    return manifest


def _finish(manifest, client, out_dir):
    manifest["requests"] = client.requests
    manifest["throttleEvents"] = client.throttle_events
    manifest["finalDelaySeconds"] = round(client.delay, 2)
    manifest["sleptSeconds"] = round(client.slept_seconds, 1)
    manifest["cookieFingerprint"] = _fingerprint(client.cookie)
    manifest["sharingTokenFingerprint"] = _fingerprint(client.sharing_token)
    _atomic_json(os.path.join(out_dir, "export_manifest.json"), manifest)


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


def _parse_parts(value):
    parts = [item.strip() for item in (value or "").split(",")
             if item.strip()]
    unknown = [part for part in parts if part not in DRAFT_PARTS]
    if unknown:
        raise ExportError("unknown --parts value(s): %s (choose from %s)"
                          % (", ".join(unknown), ", ".join(DRAFT_PARTS)))
    if not parts:
        raise ExportError("--parts needs at least one of: %s"
                          % ", ".join(DRAFT_PARTS))
    return tuple(parts)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-17lands-export",
        description="Export your own 17Lands draft history to local JSON, "
                    "slowly enough to be a good guest on a free service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="getting a cookie:\n"
               "  Log in to 17lands.com in your browser, open the developer\n"
               "  tools, and copy the whole Cookie request header of any\n"
               "  XHR to /data/. Save it to a file (chmod 600) and pass it\n"
               "  with --cookie-file. Never paste it as an argument: it\n"
               "  would land in your shell history.\n"
               "\n"
               "examples:\n"
               "  # what would this cost the server? (spends one request)\n"
               "  magic-cabt-17lands-export --cookie-file ~/.17lands-cookie \\\n"
               "      --out 17lands-export --plan\n"
               "\n"
               "  # a cautious first pass over a handful of drafts\n"
               "  magic-cabt-17lands-export --cookie-file ~/.17lands-cookie \\\n"
               "      --out 17lands-export --limit 5\n"
               "\n"
               "  # the whole history, gently, an hour at a time\n"
               "  magic-cabt-17lands-export --cookie-file ~/.17lands-cookie \\\n"
               "      --out 17lands-export --delay 4 --max-requests 900\n"
               "\n"
               "  # the cheapest complete-enough harvest: picks only\n"
               "  magic-cabt-17lands-export --cookie-file ~/.17lands-cookie \\\n"
               "      --out 17lands-export --parts draft\n")

    parser.add_argument("--out", required=True,
                        help="directory to write the export into")
    parser.add_argument("--cookie-file",
                        help="file holding your 17Lands Cookie header "
                             "(or set SEVENTEENLANDS_COOKIE)")
    parser.add_argument("--sharing-token",
                        help="sharing token, for the endpoints that take one "
                             "instead of a session")

    scope = parser.add_argument_group(
        "scope", "fewer requests is the strongest lever you have")
    scope.add_argument("--parts", default=",".join(DEFAULT_PARTS),
                       help="per-draft data to fetch, comma-separated, from "
                            "%s (default: %%(default)s). Each part is one "
                            "request per draft. 'pool' is the multiset of "
                            "your own picks, so it duplicates 'draft'."
                            % ", ".join(DRAFT_PARTS))
    scope.add_argument("--limit", type=int, default=0, metavar="N",
                       help="only fetch the first N drafts (0 = all)")
    scope.add_argument("--skip-games", action="store_true",
                       help="skip the recent-games list")
    scope.add_argument("--plan", action="store_true",
                       help="spend one request to list your drafts, print "
                            "what a full run would cost, and stop")

    pace = parser.add_argument_group("pace")
    pace.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS,
                      help="seconds between requests (default: %(default)s). "
                           "Raise it freely; there is no hurry.")
    pace.add_argument("--jitter", type=float, default=DEFAULT_JITTER_SECONDS,
                      help="random extra delay up to this many seconds, so "
                           "the traffic is not a metronome "
                           "(default: %(default)s)")
    pace.add_argument("--max-requests", type=int, default=0, metavar="N",
                      help="stop cleanly after N requests this run; re-run to "
                           "continue (0 = no limit)")
    pace.add_argument("--pause-every", type=int, default=0, metavar="N",
                      help="take a long breather every N requests")
    pace.add_argument("--pause-for", type=float, default=0.0,
                      metavar="SECONDS",
                      help="how long that breather is")
    pace.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                      help="retries per request (default: %(default)s)")
    pace.add_argument("--breaker", type=int, default=DEFAULT_BREAKER,
                      metavar="N",
                      help="stop after N consecutive failed requests "
                           "(default: %(default)s; 0 disables)")
    pace.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)

    parser.add_argument("--base-url", default=BASE_URL,
                        help=argparse.SUPPRESS)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for name, value, floor in (("--limit", args.limit, 0),
                               ("--delay", args.delay, 0),
                               ("--jitter", args.jitter, 0),
                               ("--max-requests", args.max_requests, 0),
                               ("--pause-every", args.pause_every, 0),
                               ("--pause-for", args.pause_for, 0),
                               ("--retries", args.retries, 0),
                               ("--breaker", args.breaker, 0),
                               ("--timeout", args.timeout, 1)):
        if value < floor:
            sys.stderr.write("error: %s must be >= %s\n" % (name, floor))
            return 2

    def say(text):
        if not args.quiet:
            sys.stderr.write("[17lands-export] %s\n" % text)

    def on_slowdown(old, new, why):
        say("%s -- slowing from %.1fs to %.1fs between requests for the rest "
            "of this run" % (why, old, new))

    try:
        parts = _parse_parts(args.parts)
        cookie = _read_cookie(args)
        if not cookie and not args.sharing_token:
            raise ExportError(
                "no credentials: pass --cookie-file (or set "
                "SEVENTEENLANDS_COOKIE), or --sharing-token. See --help for "
                "how to copy the cookie out of your browser.")
        client = SeventeenLandsClient(
            cookie=cookie, sharing_token=args.sharing_token,
            base_url=args.base_url, delay=args.delay, jitter=args.jitter,
            retries=args.retries, timeout=args.timeout,
            max_requests=args.max_requests, pause_every=args.pause_every,
            pause_for=args.pause_for, breaker=args.breaker,
            on_slowdown=on_slowdown)
        out_dir = os.path.abspath(os.path.expanduser(args.out))
        os.makedirs(out_dir, exist_ok=True)
        manifest = export_all(
            client, out_dir, limit=args.limit, parts=parts, say=say,
            skip_games=args.skip_games, plan_only=args.plan)
    except ExportError as error:
        sys.stderr.write("error: %s\n" % error)
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\n[17lands-export] stopped; re-run to resume "
                         "where it left off\n")
        return 130

    if args.plan:
        say("plan only -- nothing else was fetched")
        return 0

    say("wrote %d draft(s), skipped %d already present, %s game rows, "
        "%d request(s), %s spent waiting"
        % (manifest["draftsWritten"], manifest["draftsCached"],
           manifest["games"], manifest["requests"],
           format_duration(manifest["sleptSeconds"])))
    if manifest["throttleEvents"]:
        say("was asked to slow down %d time(s); ended at %.1fs between "
            "requests" % (manifest["throttleEvents"],
                          manifest["finalDelaySeconds"]))
    for note in manifest["notes"]:
        say("note: %s" % note)
    if manifest["failures"]:
        say("%d part(s) failed; see export_manifest.json"
            % len(manifest["failures"]))
    say("export: %s" % os.path.join(args.out, "export_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
