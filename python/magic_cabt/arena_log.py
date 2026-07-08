"""MTG Arena log normalization for CABT/XMage replay ingestion.

This module is the first, local-only stage of Arena replay ingestion. It reads
saved ``Player.log`` text, extracts Arena JSON payloads, and writes normalized
JSONL files that a later XMage validation layer can consume. It deliberately
does not upload anything and does not attempt to produce legal-option training
frames; those require live XMage state.
"""

import datetime
import json
import os
import re

__all__ = ["ArenaLogNormalizer", "normalize_arena_log", "iter_log_entries"]

SCHEMA_VERSION = 2

# Parse-error snippets are bounded so summaries never embed whole raw chunks;
# the live recorder reuses this bound so batch and streaming retain the same
# shape (docs/decisions/006-arena-log-retention.md).
PARSE_ERROR_SNIPPET_CHARS = 240

LOG_START_RE = re.compile(
    r"^\[(?P<channel>UnityCrossThreadLogger|Client GRE)\](?P<time>\d[\d:/ .T-]*(?:AM|PM)?)?"
)
TIMESTAMP_RE = re.compile(r"^(?P<time>\d[\d/.-]+[ T]\d+:\d+:\d+(?: ?(?:AM|PM))?)")
TIME_FORMATS = (
    "%Y-%m-%d %I:%M:%S %p",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %H:%M:%S",
    "%Y/%m/%d %I:%M:%S %p",
    "%Y/%m/%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %I:%M:%S %p",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %I:%M:%S %p",
)

# GRE messages that ask the local player for a decision. These are paired in
# game_history.jsonl with the ARENA_CLIENT_DECISION the client sent back.
DECISION_PROMPT_GRE_TYPES = frozenset((
    "GREMessageType_ActionsAvailableReq",
    "GREMessageType_AssignDamageReq",
    "GREMessageType_CastingTimeOptionsReq",
    "GREMessageType_ChooseStartingPlayerReq",
    "GREMessageType_DeclareAttackersReq",
    "GREMessageType_DeclareBlockersReq",
    "GREMessageType_GroupReq",
    "GREMessageType_MulliganReq",
    "GREMessageType_NumericInputReq",
    "GREMessageType_OptionalActionMessage",
    "GREMessageType_OrderReq",
    "GREMessageType_PayCostsReq",
    "GREMessageType_SearchReq",
    "GREMessageType_SelectNReq",
    "GREMessageType_SelectTargetsReq",
))

# Client -> GRE messages that carry the player's actual choice.
DECISION_CLIENT_TYPES = frozenset((
    "ClientMessageType_AssignDamageResp",
    "ClientMessageType_CastingTimeOptionsResp",
    "ClientMessageType_ChooseStartingPlayerResp",
    "ClientMessageType_ConcedeReq",
    "ClientMessageType_DeclareAttackersResp",
    "ClientMessageType_DeclareBlockersResp",
    "ClientMessageType_EffectCostResp",
    "ClientMessageType_GroupResp",
    "ClientMessageType_MulliganResp",
    "ClientMessageType_NumericInputResp",
    "ClientMessageType_OptionalActionResp",
    "ClientMessageType_OrderResp",
    "ClientMessageType_PerformActionResp",
    "ClientMessageType_PerformAutoTapActionsResp",
    "ClientMessageType_SearchResp",
    "ClientMessageType_SelectNResp",
    "ClientMessageType_SelectTargetsResp",
    "ClientMessageType_SubmitAttackersReq",
    "ClientMessageType_SubmitBlockersReq",
    "ClientMessageType_SubmitTargetsReq",
))


class ArenaLogNormalizer:
    """Normalize saved Arena log text into local replay artifacts."""

    def __init__(self):
        self._decoder = json.JSONDecoder()
        self._next_raw_id = 1
        self._next_event_id = 1
        self._latest_game_state_event_id = None
        self._current_match_id = None
        self._current_game_number = None
        self._game_over_keys = set()
        self.records = {
            "raw_events": [],
            "normalized_events": [],
            "game_history": [],
        }
        self.decks = []
        self.deck_info = {}
        self.summary = {
            "schemaVersion": SCHEMA_VERSION,
            "matchId": None,
            "matchIds": [],
            "games": [],
            "seatId": None,
            "gameStateMessages": 0,
            "decisionPrompts": {},
            "clientDecisions": {},
            "clientSelectNRespDecisions": 0,
            "gameOverEvents": 0,
            "deckCardIdsFound": [],
            "parseErrors": [],
            "normalizedEvents": 0,
            "rawEvents": 0,
        }

    def normalize_file(self, filename):
        """Parse ``filename`` and return this normalizer."""
        with open(filename, "r", encoding="utf-8", errors="replace") as handle:
            return self.normalize_text(handle.read())

    def normalize_text(self, text):
        """Parse Arena log text and return this normalizer."""
        for entry in iter_log_entries(text.splitlines(True)):
            self._handle_entry(entry)
        self.summary["normalizedEvents"] = len(self.records["normalized_events"])
        self.summary["rawEvents"] = len(self.records["raw_events"])
        return self

    def write_bundle(self, output_dir):
        """Write raw, normalized, history, deck, and summary artifacts."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        _write_jsonl(
            os.path.join(output_dir, "raw_events.jsonl"),
            self.records["raw_events"],
        )
        _write_jsonl(
            os.path.join(output_dir, "normalized_events.jsonl"),
            self.records["normalized_events"],
        )
        _write_jsonl(
            os.path.join(output_dir, "game_history.jsonl"),
            self.records["game_history"],
        )
        _write_json(
            os.path.join(output_dir, "deck_info.json"),
            {"schemaVersion": SCHEMA_VERSION, "decks": self.decks},
        )
        _write_json(os.path.join(output_dir, "summary.json"), self.summary)
        return self

    def _handle_entry(self, entry):
        payload = self._extract_json(entry)
        if payload is None:
            return

        raw_event = {
            "schemaVersion": SCHEMA_VERSION,
            "eventId": self._raw_event_id(),
            "channel": entry["channel"],
            "timestamp": entry["timestamp"],
            "rawTime": entry["rawTime"],
            "payload": payload,
        }
        self.records["raw_events"].append(raw_event)

        unwrapped = _unwrap_payload(payload)
        if not isinstance(unwrapped, dict):
            return

        match_event = unwrapped.get("matchGameRoomStateChangedEvent")
        if isinstance(match_event, dict):
            self._emit_match_state_changed(match_event, raw_event)

        gre_event = unwrapped.get("greToClientEvent")
        if isinstance(gre_event, dict):
            messages = gre_event.get("greToClientMessages")
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict):
                        self._handle_gre_to_client_message(message, raw_event)

        message_type = unwrapped.get("clientToMatchServiceMessageType")
        if message_type in (
            "ClientToMatchServiceMessageType_ClientToGREMessage",
            "ClientToMatchServiceMessageType_ClientToGREUIMessage",
        ):
            client_payload = _decode_json_string(unwrapped.get("payload", {}))
            client_payload = _unwrap_payload(client_payload)
            if isinstance(client_payload, dict):
                self._handle_client_to_gre_message(client_payload, raw_event)

    def _extract_json(self, entry):
        text = entry["text"]
        start = text.find("{")
        if start < 0:
            # No object payload. Plain-text entries routinely contain "["
            # ("[Manifest]...", "[PhysX]..."), so a failed decode from "[" is
            # log noise, not a malformed payload — try it silently.
            bracket = text.find("[")
            if bracket < 0:
                return None
            try:
                payload, _ = self._decoder.raw_decode(text, bracket)
                return payload
            except ValueError:
                return None
        try:
            payload, _ = self._decoder.raw_decode(text, start)
            return payload
        except ValueError as exc:
            self.summary["parseErrors"].append(
                {
                    "timestamp": entry["timestamp"],
                    "rawTime": entry["rawTime"],
                    "error": str(exc),
                    "snippet": text[:PARSE_ERROR_SNIPPET_CHARS],
                }
            )
            return None

    def _handle_gre_to_client_message(self, message, raw_event):
        message_type = message.get("type")
        if message_type == "GREMessageType_ConnectResp":
            self._emit_connect_resp(message, raw_event)
        elif message_type == "GREMessageType_GameStateMessage":
            self._emit_game_state(message, raw_event, queued=False)
        elif message_type == "GREMessageType_QueuedGameStateMessage":
            self._emit_game_state(message, raw_event, queued=True)
        elif message_type in DECISION_PROMPT_GRE_TYPES:
            counts = self.summary["decisionPrompts"]
            counts[message_type] = counts.get(message_type, 0) + 1
            self._emit(
                "ARENA_DECISION_PROMPT",
                raw_event,
                {
                    "messageType": message_type,
                    "matchId": self._current_match_id,
                    "precedingGameStateEventId": self._latest_game_state_event_id,
                    "payload": message,
                },
                history=True,
            )
        else:
            self._emit(
                "ARENA_GRE_MESSAGE",
                raw_event,
                {
                    "messageType": message_type,
                    "payload": message,
                },
                history=False,
            )

    def _handle_client_to_gre_message(self, payload, raw_event):
        message_type = payload.get("type")
        if message_type not in DECISION_CLIENT_TYPES:
            self._emit(
                "ARENA_CLIENT_TO_GRE_MESSAGE",
                raw_event,
                {"messageType": message_type, "payload": payload},
                history=False,
            )
            return

        if message_type == "ClientMessageType_SelectNResp":
            self.summary["clientSelectNRespDecisions"] += 1
        counts = self.summary["clientDecisions"]
        counts[message_type] = counts.get(message_type, 0) + 1
        self._emit(
            "ARENA_CLIENT_DECISION",
            raw_event,
            {
                "messageType": message_type,
                "matchId": self._current_match_id,
                "precedingGameStateEventId": self._latest_game_state_event_id,
                "payload": payload,
            },
            history=True,
        )

    def _register_match(self, match_id):
        if match_id is None:
            return
        if self.summary["matchId"] is None:
            self.summary["matchId"] = match_id
        if match_id not in self.summary["matchIds"]:
            self.summary["matchIds"].append(match_id)
        if match_id != self._current_match_id:
            self._current_match_id = match_id
            self._current_game_number = None
        # A match's ConnectResp (which carries the deck) arrives before the
        # first message that names the match id; adopt pending decks now.
        for deck in self.decks:
            if deck["matchId"] is None:
                deck["matchId"] = match_id

    def _emit_match_state_changed(self, match_event, raw_event):
        game_room_info = match_event.get("gameRoomInfo", {})
        game_room_config = game_room_info.get("gameRoomConfig", {})
        match_id = game_room_config.get("matchId")
        self._register_match(match_id)
        self._emit(
            "ARENA_MATCH_STATE_CHANGED",
            raw_event,
            {
                "matchId": match_id,
                "eventId": game_room_config.get("eventId"),
                "payload": match_event,
            },
            history=True,
        )
        if "finalMatchResult" in game_room_info:
            key = ("final", match_id)
            if key not in self._game_over_keys:
                self._game_over_keys.add(key)
                self.summary["gameOverEvents"] += 1
                self._emit(
                    "ARENA_GAME_OVER",
                    raw_event,
                    {
                        "matchId": match_id,
                        "payload": game_room_info["finalMatchResult"],
                    },
                    history=True,
                )

    def _emit_connect_resp(self, message, raw_event):
        connect_resp = message.get("connectResp", {})
        deck_message = connect_resp.get("deckMessage", {})
        # Current clients send flat grpId lists under deckCards/sideboardCards;
        # mainDeck/sideboard is the older card-object shape.
        main_deck = _card_ids_from_cards(
            _first_present(deck_message, ("deckCards", "mainDeck"))
        )
        sideboard = _card_ids_from_cards(
            _first_present(deck_message, ("sideboardCards", "sideboard"))
        )
        seat_ids = message.get("systemSeatIds") or []
        seat_id = seat_ids[0] if seat_ids else None
        if seat_id is not None and self.summary["seatId"] is None:
            self.summary["seatId"] = seat_id
        deck_entry = {
            "schemaVersion": SCHEMA_VERSION,
            "matchId": None,  # filled in by _register_match
            "seatId": seat_id,
            "mainDeckArenaIds": main_deck,
            "sideboardArenaIds": sideboard,
            "rawDeckMessage": deck_message,
        }
        self.decks.append(deck_entry)
        self.deck_info = deck_entry
        found = set(self.summary["deckCardIdsFound"])
        found.update(main_deck)
        found.update(sideboard)
        self.summary["deckCardIdsFound"] = sorted(found)
        self._emit(
            "ARENA_CONNECT_RESP",
            raw_event,
            {
                "connectResp": connect_resp,
                "deckInfo": deck_entry,
            },
            history=True,
        )

    def _emit_game_state(self, message, raw_event, queued):
        key = "queuedGameStateMessage" if queued else "gameStateMessage"
        game_state = message.get(key) or message.get("gameStateMessage", {})
        game_info = game_state.get("gameInfo", {})
        match_id = game_info.get("matchID") or game_info.get("matchId")
        if match_id is not None:
            self._register_match(match_id)
        else:
            # Diff states omit gameInfo; attribute them to the active match.
            match_id = self._current_match_id
        game_number = game_info.get("gameNumber")
        if game_number is None:
            game_number = self._current_game_number
        else:
            self._current_game_number = game_number
        if match_id is not None and game_number is not None:
            game = {"matchId": match_id, "gameNumber": game_number}
            if game not in self.summary["games"]:
                self.summary["games"].append(game)
        if self.summary["seatId"] is None:
            self.summary["seatId"] = _first_present(
                game_state,
                ("systemSeatId", "seatId", "playerSeatId"),
            )
        event_type = "ARENA_QUEUED_GAME_STATE" if queued else "ARENA_GAME_STATE"
        normalized = self._emit(
            event_type,
            raw_event,
            {
                "matchId": match_id,
                "turnInfo": game_state.get("turnInfo"),
                "players": game_state.get("players", []),
                "zones": game_state.get("zones", []),
                "gameObjects": game_state.get("gameObjects", []),
                "payload": game_state,
            },
            history=True,
        )
        self.summary["gameStateMessages"] += 1
        self._latest_game_state_event_id = normalized["eventId"]
        if _is_game_over(game_state):
            # The GameOver stage persists across several diffs; emit once per game.
            over_key = ("state", match_id, game_number)
            if over_key not in self._game_over_keys:
                self._game_over_keys.add(over_key)
                self.summary["gameOverEvents"] += 1
                self._emit(
                    "ARENA_GAME_OVER",
                    raw_event,
                    {"matchId": match_id, "payload": game_state},
                    history=True,
                )

    def _emit(self, event_type, raw_event, body, history):
        event = {
            "schemaVersion": SCHEMA_VERSION,
            "eventId": self._event_id(),
            "type": event_type,
            "timestamp": raw_event["timestamp"],
            "rawEventId": raw_event["eventId"],
        }
        event.update(body)
        self.records["normalized_events"].append(event)
        if history:
            self.records["game_history"].append(event)
        return event

    def _raw_event_id(self):
        event_id = "arena-raw-%06d" % self._next_raw_id
        self._next_raw_id += 1
        return event_id

    def _event_id(self):
        event_id = "arena-%06d" % self._next_event_id
        self._next_event_id += 1
        return event_id


def normalize_arena_log(source, output_dir=None):
    """Normalize Arena log text or a file path.

    ``source`` may be a path, an open text file, or raw log text. When
    ``output_dir`` is provided, the replay bundle is written there.
    """
    normalizer = ArenaLogNormalizer()
    if hasattr(source, "read"):
        normalizer.normalize_text(source.read())
    elif isinstance(source, str) and os.path.exists(source):
        normalizer.normalize_file(source)
    elif isinstance(source, str):
        normalizer.normalize_text(source)
    else:
        raise TypeError("source must be a path, text, or readable file object")
    if output_dir is not None:
        normalizer.write_bundle(output_dir)
    return normalizer


def iter_log_entries(lines):
    """Yield complete Arena log entries from an iterable of lines."""
    current = None
    for line in lines:
        start = LOG_START_RE.match(line)
        if start:
            if current is not None:
                yield current
            raw_time = (start.group("time") or "").strip()
            timestamp = _format_timestamp(raw_time)
            text = line[start.end() :]
            current = {
                "channel": start.group("channel"),
                "rawTime": raw_time or None,
                "timestamp": timestamp,
                "text": text,
            }
            continue

        if current is None:
            timestamp_match = TIMESTAMP_RE.match(line)
            raw_time = timestamp_match.group("time") if timestamp_match else None
            current = {
                "channel": None,
                "rawTime": raw_time,
                "timestamp": _format_timestamp(raw_time),
                "text": line,
            }
        else:
            current["text"] += line

    if current is not None:
        yield current


def _unwrap_payload(blob):
    if not isinstance(blob, dict):
        return blob
    if "clientToMatchServiceMessageType" in blob:
        return blob
    for key in ("payload", "Payload", "request"):
        if key in blob:
            value = _decode_json_string(blob[key])
            return _unwrap_payload(value)
    return blob


def _decode_json_string(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _format_timestamp(raw_time):
    if raw_time is None:
        return None
    raw_time = raw_time.strip().rstrip(":")
    if not raw_time:
        return None
    if ": " in raw_time:
        raw_time = raw_time.split(": ", 1)[0]
    normalized = raw_time.replace("  ", " ")
    for fmt in TIME_FORMATS:
        try:
            return datetime.datetime.strptime(normalized, fmt).isoformat()
        except ValueError:
            pass
    return raw_time


def _card_ids_from_cards(cards):
    if not isinstance(cards, list):
        return []
    ids = []
    for card in cards:
        if isinstance(card, dict):
            value = _first_present(card, ("grpId", "GrpId", "cardId", "CardId", "id"))
            if value is not None:
                ids.append(value)
        elif isinstance(card, int):
            ids.append(card)
    return ids


def _first_present(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _is_game_over(game_state):
    game_info = game_state.get("gameInfo", {})
    values = [
        game_state.get("gameStage"),
        game_state.get("stage"),
        game_info.get("gameStage"),
        game_info.get("stage"),
    ]
    return "GameStage_GameOver" in values or "GameOver" in values


def _write_jsonl(filename, records):
    with open(filename, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _write_json(filename, document):
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(document, handle, sort_keys=True, indent=2)
        handle.write("\n")
