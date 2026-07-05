"""Live-game client for the CABT-style XMage bridge.

Talks newline-delimited JSON to the Java ``CabtProtocolServer`` subprocess —
the Magic equivalent of the Pokemon CABT ``battle_start`` /
``battle_select`` / ``battle_finish`` / ``visualize_data`` loop:

    with CabtBridge() as bridge:
        obs = bridge.game_start(deck, deck, seed=7, max_turns=30)
        while not bridge.finished:
            options = obs["observation"]["select"]["option"]
            obs = bridge.game_select([0])  # option indices
        print(bridge.result)

Observations are returned as plain dicts, exactly as the Java side
serialized them: ``observation.select.option`` is the indexed legal-option
list, ``observation.current`` the hidden-information-safe state snapshot.

Server discovery: pass ``command=[...]`` (full argv) or ``classpath=...``;
otherwise the ``MAGIC_CABT_CLASSPATH`` environment variable must point at a
Java classpath containing the bridge and XMage (see
scripts/run-cabt-adapter-tests.sh, which writes one).
"""

import json
import os
import subprocess

__all__ = [
    "CabtBridge",
    "CabtProtocolError",
    "CabtGameError",
    "parse_decklist",
    "load_decklist",
]

SERVER_MAIN_CLASS = "mage.player.cabt.CabtProtocolServer"
CLASSPATH_ENV_VAR = "MAGIC_CABT_CLASSPATH"


class CabtProtocolError(RuntimeError):
    """A protocol request failed: carries the stable error code."""

    def __init__(self, code, message):
        super(CabtProtocolError, self).__init__("%s: %s" % (code, message))
        self.code = code
        self.message = message


class CabtGameError(CabtProtocolError):
    """The engine loop itself failed (fail-closed decision, engine error)."""


def parse_decklist(text):
    """Parse ``"4 Forest"``-style decklist text into deck entries.

    One card per line as ``<count> <name>`` (count optional, default 1).
    Blank lines and ``#`` comments are ignored. Returns the
    ``[{"name": ..., "count": ...}, ...]`` list ``game_start`` sends.
    """
    entries = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        first, _, rest = line.partition(" ")
        if first.isdigit() and rest.strip():
            entries.append({"name": rest.strip(), "count": int(first)})
        else:
            entries.append({"name": line, "count": 1})
    if not entries:
        raise ValueError("decklist text contains no cards")
    return entries


def load_decklist(path):
    """Read a decklist file (see ``parse_decklist`` for the format)."""
    with open(path, "r") as file:
        return parse_decklist(file.read())


class CabtBridge(object):
    """One protocol server subprocess and the request/response loop over it."""

    def __init__(self, command=None, classpath=None, java="java", cwd=None):
        if command is None:
            classpath = classpath or os.environ.get(CLASSPATH_ENV_VAR)
            if not classpath:
                raise ValueError(
                    "no way to launch the bridge: pass command=[...] or "
                    "classpath=..., or set $" + CLASSPATH_ENV_VAR
                )
            command = [java, "-cp", classpath, SERVER_MAIN_CLASS]
        self._process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr passes through: server diagnostics stay visible
            universal_newlines=True,
            bufsize=1,
        )
        self.finished = False
        self.result = None

    # --- lifecycle ---

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Finish any active game and stop the server process."""
        if self._process.poll() is None:
            try:
                self.request({"command": "game_finish"})
            except (CabtProtocolError, IOError):
                pass
            try:
                self._process.stdin.close()
            except IOError:
                pass
            try:
                self._process.wait(timeout=10)
            except Exception:
                self._process.kill()
        if self._process.stdout is not None:
            self._process.stdout.close()

    # --- raw protocol ---

    def request(self, request_dict):
        """Send one request object, return the parsed ok-response.

        Raises ``CabtProtocolError`` (with the server's stable error code)
        on an ok=false response, ``CabtGameError`` when the engine loop
        failed, and ``IOError`` when the server process died.
        """
        line = json.dumps(request_dict)
        if "\n" in line:
            raise ValueError("request must serialize to a single line")
        try:
            self._process.stdin.write(line + "\n")
            self._process.stdin.flush()
            response_line = self._process.stdout.readline()
        except OSError as error:
            raise IOError("bridge process is gone: %s" % (error,))
        if not response_line:
            raise IOError(
                "bridge process closed its output (exit code %s)"
                % self._process.poll()
            )
        response = json.loads(response_line)
        if response.get("ok") is not True:
            code = response.get("error", "UNKNOWN")
            message = response.get("message", "")
            if code == "GAME_ERROR":
                self.finished = True
                raise CabtGameError(code, message)
            raise CabtProtocolError(code, message)
        return response

    # --- CABT-parity commands ---

    def ping(self):
        return self.request({"command": "ping"})

    def capabilities(self):
        return self.request({"command": "capabilities"})

    def game_start(self, deck0, deck1, player_names=None, seed=None,
                   max_turns=None, decision_timeout_seconds=None):
        """Start a real engine game; returns the first decision response.

        ``deck0``/``deck1``: entry lists (see ``parse_decklist``) or decklist
        text. The response's ``observation.select`` is the first prompt
        (normally the pregame starting-player or mulligan decision).
        """
        options = {}
        if player_names is not None:
            options["playerNames"] = list(player_names)
        if seed is not None:
            options["seed"] = seed
        if max_turns is not None:
            options["maxTurns"] = max_turns
        if decision_timeout_seconds is not None:
            options["decisionTimeoutSeconds"] = decision_timeout_seconds
        request = {
            "command": "game_start",
            "decks": [self._normalize_deck(deck0), self._normalize_deck(deck1)],
        }
        if options:
            request["options"] = options
        self.finished = False
        self.result = None
        return self._track(self.request(request))

    def game_select(self, select_list):
        """Answer the pending decision with option indices; returns the next
        decision response (or the game result once ``finished``)."""
        if not isinstance(select_list, list) or not all(
            isinstance(i, int) and not isinstance(i, bool) for i in select_list
        ):
            raise ValueError("select_list must be a list of ints")
        return self._track(
            self.request({"command": "game_select", "select": select_list})
        )

    def game_finish(self):
        """End the active game (safe to call when none is active)."""
        response = self.request({"command": "game_finish"})
        self.finished = True
        return response

    def resolve_card(self, name):
        """Resolve a single card name against XMage's card identity.

        Returns the resolution dict — ``requestedName``, ``normalizedName``,
        ``resolved`` (bool), ``strategy`` (``EXACT`` / ``NORMALIZED`` /
        ``CLASS_HEURISTIC`` or ``None``), ``canonicalName``, ``setCode``,
        ``cardNumber``, and on failure ``error`` (``UNKNOWN_CARD``) plus
        ``reason``. Needs no active game. Never substitutes: an unknown name
        comes back ``resolved == False`` rather than as a different card."""
        return self.request({"command": "resolve_card", "name": name})["resolution"]

    def validate_deck(self, deck):
        """Validate a full decklist against XMage's card identity, no game.

        ``deck`` is an entry list (see ``parse_decklist``) or decklist text.
        Returns ``{"valid": bool, "resolutions": [...], "failures": [...]}``:
        one resolution per entry, with the unresolved subset in ``failures``.
        Fail closed by checking ``valid`` (or that ``failures`` is empty)
        before starting a game."""
        return self.request(
            {"command": "validate_deck", "deck": self._normalize_deck(deck)}
        )

    def all_card_data(self):
        """Card metadata for every distinct card in the active game's deduped
        deck pool. This is game-scoped (not a global card database): it
        requires an active game and only covers cards in the decks. Reference
        data only — legal actions always come from ``observation.select``."""
        return self.request({"command": "all_card_data"})["cards"]

    def global_card_data(self, names):
        """Static card metadata for a set of card names, independent of any
        game (distinct from ``all_card_data``, which is the active game's deck
        pool). ``names`` is an iterable of card names; each is resolved through
        the repository and its metadata exported. Fails closed: if any name is
        unknown the server raises ``CabtProtocolError`` (``UNKNOWN_CARD``)
        rather than returning a partial export. Returns the list of card
        dicts."""
        return self.request(
            {"command": "global_card_data", "names": list(names)}
        )["cards"]

    def visualize_data(self):
        """Human-readable board render of the current state."""
        return self.request({"command": "visualize_data"})["text"]

    # --- helpers ---

    @staticmethod
    def _normalize_deck(deck):
        if isinstance(deck, str):
            return parse_decklist(deck)
        return list(deck)

    def _track(self, response):
        if response.get("finished"):
            self.finished = True
            self.result = response.get("result")
        return response
