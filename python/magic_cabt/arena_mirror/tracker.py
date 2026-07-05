"""Streaming Arena log normalization and GRE game-state tracking.

``StreamingNormalizer`` drives the batch-validated ``ArenaLogNormalizer``
entry-by-entry so a live follower can reuse its exact event routing.
``GameStateTracker`` folds full/diff GameStateMessages into an authoritative
board state; ``ArenaMatchTracker`` ties both together with decision
prompt/response pairing and produces JSON-ready mirror snapshots.
"""

from ..arena_log import ArenaLogNormalizer
from . import options as options_mod

__all__ = ["StreamingNormalizer", "GameStateTracker", "ArenaMatchTracker"]

ZONE_TYPE_PREFIX = "ZoneType_"

# Snapshot zones the mirror renders with objects; others are count-only.
VISIBLE_ZONES = ("BATTLEFIELD", "STACK", "HAND", "GRAVEYARD", "EXILE", "COMMAND")


class StreamingNormalizer(object):
    """Feeds entries through ArenaLogNormalizer, draining events as they land.

    The batch normalizer accumulates records in memory; live sessions run for
    hours, so after every entry the new records are drained and handed to the
    caller instead of being retained.
    """

    def __init__(self):
        self._normalizer = ArenaLogNormalizer()

    @property
    def summary(self):
        return self._normalizer.summary

    @property
    def decks(self):
        return self._normalizer.decks

    def feed(self, entry):
        """Process one log entry; returns (raw_events, events) drained lists.

        Each normalized event dict gains an ``"inHistory"`` flag matching the
        batch normalizer's game_history selection.
        """
        self._normalizer._handle_entry(entry)
        records = self._normalizer.records
        raw_events = records["raw_events"][:]
        del records["raw_events"][:]
        history_ids = set(id(event) for event in records["game_history"])
        del records["game_history"][:]
        events = []
        for event in records["normalized_events"]:
            event["inHistory"] = id(event) in history_ids
            events.append(event)
        del records["normalized_events"][:]
        return raw_events, events


class GameStateTracker(object):
    """Authoritative board state folded from GRE GameStateMessages."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.game_id = None
        self.seq = 0
        self.local_seat = None
        self.seat_names = {}
        self.objects = {}          # instanceId -> object dict
        self.zones = {}            # zoneId -> zone dict
        self.players = {}          # seatId -> player dict
        self.object_zone = {}      # instanceId -> zoneId
        self.turn_info = {}
        self.stack_order = []
        self.attacking = set()
        self.blocking = set()
        self.pending_id_changes = {}
        self.game_over = False

    def set_local_seat(self, seat_id):
        if isinstance(seat_id, int):
            self.local_seat = seat_id

    def set_seat_name(self, seat_id, name):
        if isinstance(seat_id, int) and name:
            self.seat_names[seat_id] = name

    def apply(self, message):
        """Apply one gameStateMessage payload (full or diff)."""
        if message.get("type") == "GameStateType_Full":
            preserved_names = dict(self.seat_names)
            preserved_seat = self.local_seat
            self.reset()
            self.seat_names = preserved_names
            self.local_seat = preserved_seat

        self.seq = message.get("gameStateId") or message.get("msgId") or self.seq + 1

        game_info = message.get("gameInfo") or {}
        self.game_id = game_info.get("gameId") or self.game_id
        stage = game_info.get("stage") or message.get("gameStage")
        if stage == "GameStage_GameOver":
            self.game_over = True

        for deleted_id in message.get("diffDeletedInstanceIds") or []:
            if isinstance(deleted_id, int):
                self.objects.pop(deleted_id, None)
                self.object_zone.pop(deleted_id, None)
                self.attacking.discard(deleted_id)
                self.blocking.discard(deleted_id)

        self._apply_annotations(message.get("annotations") or [])

        for entry in message.get("gameObjects") or []:
            if isinstance(entry, dict):
                self._upsert_object(entry)

        for entry in message.get("zones") or []:
            if isinstance(entry, dict):
                self._upsert_zone(entry)

        self._refresh_object_zones()

        for entry in message.get("players") or []:
            if isinstance(entry, dict):
                self._upsert_player(entry)

        turn_info = message.get("turnInfo")
        if isinstance(turn_info, dict):
            self.turn_info.update(turn_info)

        seat_ids = message.get("systemSeatIds")
        if isinstance(seat_ids, list) and seat_ids and isinstance(seat_ids[0], int):
            self.local_seat = seat_ids[0]

    # --- appliers ---

    def _apply_annotations(self, annotations):
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            types = annotation.get("type") or []
            if "AnnotationType_ObjectIdChanged" in types:
                details = _details_map(annotation)
                orig = details.get("orig_id")
                new = details.get("new_id")
                if isinstance(orig, int) and isinstance(new, int) and orig != new:
                    self._remap_object(orig, new)

    def _remap_object(self, orig, new):
        """Arena re-instances objects on zone changes: carry identity over."""
        obj = self.objects.pop(orig, None)
        self.object_zone.pop(orig, None)
        if obj is not None:
            obj = dict(obj)
            obj["instanceId"] = new
            # remember lineage for transition tracking
            lineage = obj.get("lineage") or [orig]
            obj["lineage"] = lineage + [new] if new not in lineage else lineage
            existing = self.objects.get(new)
            if existing is None or not existing.get("grpId"):
                self.objects[new] = obj
        if orig in self.attacking:
            self.attacking.discard(orig)
            self.attacking.add(new)
        if orig in self.blocking:
            self.blocking.discard(orig)
            self.blocking.add(new)

    def _upsert_object(self, entry):
        instance_id = entry.get("instanceId")
        if not isinstance(instance_id, int):
            return
        obj = self.objects.get(instance_id) or {"instanceId": instance_id}
        for source_key, target_key in (
            ("grpId", "grpId"),
            ("overlayGrpId", "overlayGrpId"),
            ("controllerSeatId", "controllerSeat"),
            ("ownerSeatId", "ownerSeat"),
            ("cardTypes", "cardTypes"),
            ("subtypes", "subtypes"),
            ("superTypes", "superTypes"),
            ("color", "colors"),
            ("attackState", "attackState"),
            ("blockState", "blockState"),
            ("attackInfo", "attackInfo"),
            ("blockInfo", "blockInfo"),
            ("objectSourceGrpId", "objectSourceGrpId"),
            ("parentId", "parentId"),
            ("zoneId", "zoneId"),
            ("visibility", "visibility"),
            ("name", "nameId"),
            ("abilities", "abilityIds"),
        ):
            if source_key in entry:
                obj[target_key] = entry[source_key]
        obj["type"] = entry.get("type") or obj.get("type")
        if "isTapped" in entry:
            obj["tapped"] = bool(entry["isTapped"])
        elif "tapped" not in obj:
            obj["tapped"] = False
        for stat_key in ("power", "toughness"):
            if stat_key in entry:
                obj[stat_key] = _stat_value(entry[stat_key])
        if "damage" in entry:
            obj["damage"] = _coerce_int(entry["damage"])
        if "counters" in entry:
            obj["counters"] = _extract_counters(entry.get("counters"))
        if "attachedTo" in entry:
            obj["attachedTo"] = entry.get("attachedTo")
        if "isFaceDown" in entry:
            obj["faceDownFlag"] = bool(entry["isFaceDown"])
        if "turnedFaceUp" in entry:
            obj["turnedFaceUp"] = bool(entry["turnedFaceUp"])

        attack_state = obj.get("attackState")
        if attack_state == "AttackState_Attacking":
            self.attacking.add(instance_id)
        elif attack_state is not None:
            self.attacking.discard(instance_id)
        block_state = obj.get("blockState")
        if block_state == "BlockState_Blocking":
            self.blocking.add(instance_id)
        elif block_state is not None:
            self.blocking.discard(instance_id)

        self.objects[instance_id] = obj

    def _upsert_zone(self, entry):
        zone_id = entry.get("zoneId")
        if not isinstance(zone_id, int):
            return
        existing = self.zones.get(zone_id) or {}
        zone_type = entry.get("type") or existing.get("type")
        name = zone_type or "UNKNOWN"
        if name.startswith(ZONE_TYPE_PREFIX):
            name = name[len(ZONE_TYPE_PREFIX):].upper()
        object_ids = existing.get("objectIds", [])
        if entry.get("objectInstanceIds") is not None:
            object_ids = [i for i in entry["objectInstanceIds"] if isinstance(i, int)]
        self.zones[zone_id] = {
            "zoneId": zone_id,
            "type": zone_type,
            "name": name,
            "ownerSeat": entry.get("ownerSeatId", existing.get("ownerSeat")),
            "visibility": entry.get("visibility") or existing.get("visibility"),
            "objectIds": object_ids,
            "objectCount": entry.get("objectCount", existing.get("objectCount")),
        }

    def _upsert_player(self, entry):
        seat_id = entry.get("systemSeatNumber") or entry.get("systemSeatId")
        if not isinstance(seat_id, int):
            return
        player = self.players.get(seat_id) or {"seat": seat_id}
        if entry.get("lifeTotal") is not None:
            player["life"] = _coerce_int(entry.get("lifeTotal"))
        if entry.get("startingLifeTotal") is not None and "life" not in player:
            player["life"] = _coerce_int(entry.get("startingLifeTotal"))
        for key in ("maxHandSize", "turnNumber", "teamId", "timerIds",
                    "pendingMessageType", "controllerSeatId"):
            if key in entry:
                player[key] = entry[key]
        self.players[seat_id] = player

    def _refresh_object_zones(self):
        self.object_zone = {}
        for zone_id, zone in self.zones.items():
            for object_id in zone["objectIds"]:
                self.object_zone[object_id] = zone_id

    # --- snapshot ---

    def snapshot(self):
        """JSON-ready board state: the mirror-display / recorder payload."""
        players = []
        for seat_id in sorted(self.players):
            player = self.players[seat_id]
            players.append({
                "seat": seat_id,
                "name": self.seat_names.get(seat_id) or "Seat %d" % seat_id,
                "life": player.get("life"),
                "libraryCount": self._zone_count("LIBRARY", seat_id),
                "handCount": self._zone_count("HAND", seat_id),
            })

        zones = {"battlefield": [], "stack": [], "hands": {}, "graveyards": {},
                 "exile": [], "command": [], "libraries": {}}
        for zone in self.zones.values():
            name = zone["name"]
            owner = zone.get("ownerSeat")
            if name not in VISIBLE_ZONES:
                if name == "LIBRARY" and owner is not None:
                    zones["libraries"][str(owner)] = self._count_zone(zone)
                continue
            # A hand is visible only to its owner; the opponent's hand (and
            # every hand when we don't yet know which seat is local) is
            # redacted to face-down placeholders so no hidden identity is
            # ever put in a snapshot. Battlefield/stack/graveyard/exile are
            # public zones both players see.
            hidden = name == "HAND" and (
                self.local_seat is None or owner != self.local_seat)
            objects = [self._object_view(i, hidden=hidden)
                       for i in zone["objectIds"]]
            objects = [o for o in objects if o is not None]
            if name == "BATTLEFIELD":
                zones["battlefield"].extend(objects)
            elif name == "STACK":
                zones["stack"].extend(objects)
            elif name == "HAND" and owner is not None:
                zones["hands"][str(owner)] = objects
            elif name == "GRAVEYARD" and owner is not None:
                zones["graveyards"][str(owner)] = objects
            elif name == "EXILE":
                zones["exile"].extend(objects)
            elif name == "COMMAND":
                zones["command"].extend(objects)

        turn = self.turn_info
        return {
            "seq": self.seq,
            "gameId": self.game_id,
            "gameOver": self.game_over,
            "localSeat": self.local_seat,
            "turnNumber": turn.get("turnNumber"),
            "phase": turn.get("phase"),
            "step": turn.get("step"),
            "activeSeat": turn.get("activePlayer"),
            "prioritySeat": turn.get("priorityPlayer"),
            "decisionSeat": turn.get("decisionPlayer"),
            "players": players,
            "zones": zones,
        }

    def _object_view(self, instance_id, hidden=False):
        obj = self.objects.get(instance_id)
        if obj is None or hidden:
            # Either the GRE never described this instance (hidden card), or
            # it lives in a zone this perspective may not see. Emit a
            # face-down placeholder carrying no card identity — never a grpId,
            # name, or type line — regardless of what the GRE object holds.
            zone_id = self.object_zone.get(instance_id)
            zone = self.zones.get(zone_id) or {}
            owner = (obj.get("ownerSeat") if obj else None) or zone.get("ownerSeat")
            return {
                "instanceId": instance_id,
                "grpId": None,
                "faceDown": True,
                "ownerSeat": owner,
                "controllerSeat": owner,
            }
        grp_id = obj.get("grpId") or obj.get("overlayGrpId")
        face_down = bool(obj.get("faceDownFlag")) or not grp_id
        view = {
            "instanceId": instance_id,
            "grpId": grp_id,
            "faceDown": face_down,
            "ownerSeat": obj.get("ownerSeat"),
            "controllerSeat": obj.get("controllerSeat", obj.get("ownerSeat")),
            "tapped": bool(obj.get("tapped")),
            "objectType": obj.get("type"),
            "cardTypes": obj.get("cardTypes") or [],
            "subtypes": obj.get("subtypes") or [],
            "attacking": instance_id in self.attacking,
            "blocking": instance_id in self.blocking,
        }
        for key in ("power", "toughness", "damage", "counters", "attachedTo",
                    "parentId", "objectSourceGrpId", "lineage"):
            if obj.get(key) is not None:
                view[key] = obj[key]
        return view

    def _zone_count(self, zone_name, seat_id):
        for zone in self.zones.values():
            if zone["name"] == zone_name and zone.get("ownerSeat") == seat_id:
                return self._count_zone(zone)
        return None

    @staticmethod
    def _count_zone(zone):
        if zone.get("objectIds"):
            return len(zone["objectIds"])
        count = zone.get("objectCount")
        return count if isinstance(count, int) else 0


class ArenaMatchTracker(object):
    """Consumes normalized Arena events; emits mirror + decision callbacks.

    Callbacks (all optional constructor kwargs):
        on_snapshot(snapshot, event)      after each game-state application
        on_decision(record)               when a prompt is paired with the
                                          client's response
        on_game_event(kind, event)        lifecycle: connect/match/game_over
    """

    def __init__(self, on_snapshot=None, on_decision=None, on_game_event=None):
        self.state = GameStateTracker()
        self.match_id = None
        self.game_number = None
        self.decks = []
        self._on_snapshot = on_snapshot
        self._on_decision = on_decision
        self._on_game_event = on_game_event
        self._pending_prompts = []
        self._decision_sequence = 0
        # monotonic id per game instance; the GRE gameId is absent on early
        # (pregame/mulligan) states, so stamp our own stable identity that
        # keeps game 2's states from colliding with game 1's restarted seqs
        self._game_instance = 0

    def handle_event(self, event):
        """Route one normalized event (from StreamingNormalizer.feed)."""
        event_type = event.get("type")
        if event_type in ("ARENA_GAME_STATE", "ARENA_QUEUED_GAME_STATE"):
            self._handle_game_state(event)
        elif event_type == "ARENA_CONNECT_RESP":
            self._handle_connect(event)
        elif event_type == "ARENA_MATCH_STATE_CHANGED":
            self._handle_match_changed(event)
        elif event_type == "ARENA_DECISION_PROMPT":
            self._handle_prompt(event)
        elif event_type == "ARENA_CLIENT_DECISION":
            self._handle_client_decision(event)
        elif event_type == "ARENA_GAME_OVER":
            self._fire_game_event("game_over", event)

    # --- event handlers ---

    def _handle_game_state(self, event):
        payload = event.get("payload") or {}
        self.match_id = event.get("matchId") or self.match_id
        game_info = payload.get("gameInfo") or {}
        if game_info.get("gameNumber") is not None:
            game_number = game_info["gameNumber"]
            if game_number != self.game_number:
                self.game_number = game_number
                self._game_instance += 1
                self._fire_game_event("game_start", event)
        if self._game_instance == 0:
            self._game_instance = 1  # states before the first gameInfo
        self.state.apply(payload)
        if self._on_snapshot is not None:
            self._on_snapshot(self._stamp(self.state.snapshot()), event)

    def _stamp(self, snapshot):
        snapshot["matchId"] = self.match_id
        snapshot["gameNumber"] = self.game_number
        snapshot["gameInstance"] = self._game_instance
        return snapshot

    def _handle_connect(self, event):
        deck_info = event.get("deckInfo") or {}
        self.decks.append(deck_info)
        self.state.set_local_seat(deck_info.get("seatId"))
        self._fire_game_event("connect", event)

    def _handle_match_changed(self, event):
        payload = event.get("payload") or {}
        game_room = payload.get("gameRoomInfo") or {}
        config = game_room.get("gameRoomConfig") or {}
        for player in config.get("reservedPlayers") or []:
            if isinstance(player, dict):
                name = player.get("playerName")
                if name:
                    name = name.split("#")[0]
                self.state.set_seat_name(player.get("systemSeatId"), name)
        self._fire_game_event("match_changed", event)

    def _handle_prompt(self, event):
        prompt = options_mod.build_prompt(event)
        if prompt is not None:
            prompt["snapshot"] = self._stamp(self.state.snapshot())
            self._pending_prompts.append(prompt)
            # Arena occasionally abandons prompts (auto-pass); keep the
            # queue short so stale prompts don't mis-pair later responses.
            if len(self._pending_prompts) > 8:
                self._pending_prompts.pop(0)

    def _handle_client_decision(self, event):
        match = options_mod.match_response(self._pending_prompts, event)
        if match is None:
            return
        prompt, selected, matched = match
        if prompt in self._pending_prompts:
            self._pending_prompts.remove(prompt)  # concede prompts are synthetic
        self._decision_sequence += 1
        record = {
            "sequence": self._decision_sequence,
            "matchId": self.match_id,
            "gameNumber": self.game_number,
            "seat": self.state.local_seat,
            "player": self.state.seat_names.get(self.state.local_seat),
            "promptTimestamp": prompt.get("timestamp"),
            "responseTimestamp": event.get("timestamp"),
            "observation": {
                "current": prompt.pop("snapshot", None),
                "select": prompt["select"],
            },
            "select": selected,
            "selectionMatched": matched,
            "promptMessageType": prompt["messageType"],
            "responseMessageType": (event.get("messageType") or ""),
            "responsePayload": event.get("payload"),
        }
        if self._on_decision is not None:
            self._on_decision(record)

    def _fire_game_event(self, kind, event):
        if kind in ("connect", "game_start"):
            # a new game invalidates any prompts left over from the last one
            self._pending_prompts = []
        if self._on_game_event is not None:
            self._on_game_event(kind, event)


# --- small helpers ---

def _details_map(annotation):
    details = {}
    for item in annotation.get("details") or []:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if key is None:
            continue
        values = item.get("valueInt32")
        if isinstance(values, list) and values:
            details[key] = values[0] if len(values) == 1 else values
            continue
        for value_key in ("valueString", "valueInt64", "valueUint32",
                          "valueUint64", "valueBool"):
            values = item.get(value_key)
            if isinstance(values, list) and values:
                details[key] = values[0] if len(values) == 1 else values
                break
    return details


def _stat_value(value):
    if isinstance(value, dict):
        return _coerce_int(value.get("value"))
    return _coerce_int(value)


def _coerce_int(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_counters(counters):
    result = {}
    if isinstance(counters, list):
        for counter in counters:
            if isinstance(counter, dict):
                kind = counter.get("type") or counter.get("counterType")
                count = _coerce_int(counter.get("count")) or 0
                if kind is not None:
                    result[str(kind)] = result.get(str(kind), 0) + count
    return result
