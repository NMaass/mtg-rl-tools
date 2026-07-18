"""Format heterogeneous engine/log/perception states into one symbolic schema."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
import copy

from .schema import CanonicalEvent, CanonicalObject, CanonicalPlayer, CanonicalState


_ZONE_ALIASES = {
    "hands": "hand",
    "hand": "hand",
    "libraries": "library",
    "library": "library",
    "graveyards": "graveyard",
    "graveyard": "graveyard",
    "battlefield": "battlefield",
    "stack": "stack",
    "exile": "exile",
    "command": "command",
    "sideboard": "sideboard",
}


class CanonicalStateFormatter:
    """Shared formatter for XMage, Arena, MTGO logs, and video perception."""

    def format(self, raw: Mapping[str, Any], source: str,
               perspective_seat: Optional[Any] = None,
               timestamp_ms: Optional[int] = None) -> CanonicalState:
        wrapped = copy.deepcopy(dict(raw))
        state = self._unwrap(wrapped)
        perspective = self._first(
            perspective_seat,
            wrapped.get("perspectiveSeat"),
            state.get("localSeat"),
            state.get("perspectiveSeat"),
        )
        result = CanonicalState(
            source=str(source),
            source_id=self._as_text(self._first(
                wrapped.get("sourceId"), wrapped.get("frameId"),
                wrapped.get("decisionFingerprint"))),
            match_id=self._as_text(self._first(
                wrapped.get("matchId"), state.get("matchId"))),
            game_id=self._as_text(self._first(
                wrapped.get("gameId"), state.get("gameId"),
                state.get("gameInstance"))),
            game_number=self._as_int(self._first(
                wrapped.get("gameNumber"), state.get("gameNumber"))),
            timestamp_ms=self._as_int(self._first(
                timestamp_ms, wrapped.get("timestampMs"),
                state.get("timestampMs"))),
            sequence=self._as_int(self._first(
                wrapped.get("sequenceNumber"), wrapped.get("sequence"),
                state.get("seq"), state.get("sequence"))),
            perspective_seat=self._as_text(perspective),
            turn_number=self._as_int(self._first(
                state.get("turnNumber"), state.get("turn"))),
            active_seat=self._as_text(self._first(
                state.get("activeSeat"), state.get("activePlayerId"),
                state.get("activePlayer"))),
            priority_seat=self._as_text(self._first(
                state.get("prioritySeat"), state.get("priorityPlayerId"),
                state.get("priorityPlayer"))),
            phase=self._as_text(state.get("phase")),
            step=self._as_text(state.get("step")),
            confidence=copy.deepcopy(wrapped.get("confidence") or
                                     state.get("confidence") or {}),
            unknown_paths=list(wrapped.get("unknownPaths") or
                               wrapped.get("unknown_paths") or
                               state.get("unknownPaths") or []),
            provenance=copy.deepcopy(wrapped.get("provenance") or {}),
        )
        result.players = [self._player(row, index)
                          for index, row in enumerate(state.get("players") or [])
                          if isinstance(row, Mapping)]
        result.zones = self._zones(state, result.players)
        history = ((wrapped.get("observation") or {}).get("publicHistory")
                   if isinstance(wrapped.get("observation"), Mapping) else None)
        history = history or wrapped.get("publicHistory") or \
            state.get("publicHistory") or []
        result.public_history = [self._event(row, index)
                                 for index, row in enumerate(history)]
        result.provenance.setdefault("rawSource", str(source))
        return result

    @staticmethod
    def _unwrap(wrapped: Mapping[str, Any]) -> Mapping[str, Any]:
        observation = wrapped.get("observation")
        if isinstance(observation, Mapping):
            current = observation.get("current")
            if isinstance(current, Mapping):
                return current
        for key in ("current", "state", "gameState"):
            value = wrapped.get(key)
            if isinstance(value, Mapping):
                return value
        return wrapped

    def _player(self, row: Mapping[str, Any], index: int) -> CanonicalPlayer:
        seat = self._first(row.get("seat"), row.get("playerIndex"),
                           row.get("playerId"), index)
        mana = row.get("manaPool") or row.get("mana_pool") or {}
        if not isinstance(mana, Mapping):
            mana = {}
        return CanonicalPlayer(
            seat=str(seat),
            name=self._as_text(row.get("name")),
            player_id=self._as_text(row.get("playerId")),
            life=self._as_int(row.get("life")),
            poison=self._as_int(row.get("poison")),
            energy=self._as_int(row.get("energy")),
            hand_count=self._as_int(self._first(
                row.get("handCount"), self._safe_len(row.get("hand")))),
            library_count=self._as_int(self._first(
                row.get("libraryCount"), self._safe_len(row.get("library")))),
            graveyard_count=self._as_int(self._first(
                row.get("graveyardCount"), self._safe_len(row.get("graveyard")))),
            mana_pool={str(key): int(value) for key, value in mana.items()
                       if self._as_int(value) is not None},
            in_game=self._as_bool(row.get("inGame")),
            passed=self._as_bool(row.get("passed")),
            metadata={key: copy.deepcopy(value) for key, value in row.items()
                      if key not in {
                          "seat", "playerIndex", "playerId", "name", "life",
                          "poison", "energy", "handCount", "libraryCount",
                          "graveyardCount", "manaPool", "inGame", "passed",
                          "hand", "library", "graveyard", "exile",
                          "revealedHand"}},
        )

    def _zones(self, state: Mapping[str, Any],
               players: Iterable[CanonicalPlayer]) -> Dict[str, list[CanonicalObject]]:
        result: Dict[str, list[CanonicalObject]] = {}
        zones = state.get("zones") or {}
        if isinstance(zones, Mapping):
            for raw_zone, payload in zones.items():
                zone = _ZONE_ALIASES.get(str(raw_zone), str(raw_zone).lower())
                self._append_zone(result, zone, payload)
        for raw_zone in ("battlefield", "stack", "exile", "command",
                         "sideboard"):
            if raw_zone in state:
                self._append_zone(result, raw_zone, state.get(raw_zone))
        player_rows = state.get("players") or []
        for index, row in enumerate(player_rows):
            if not isinstance(row, Mapping):
                continue
            player = players[index] if index < len(players) else None
            seat = player.seat if player else str(index)
            for raw_zone in ("hand", "library", "graveyard", "exile",
                             "revealedHand"):
                payload = row.get(raw_zone)
                if isinstance(payload, list):
                    zone = "hand" if raw_zone == "revealedHand" else raw_zone
                    for obj in payload:
                        result.setdefault(zone, []).append(
                            self._object(obj, zone=zone, default_seat=seat,
                                         revealed=(raw_zone == "revealedHand")))
        return {zone: rows for zone, rows in result.items() if rows}

    def _append_zone(self, result: Dict[str, list[CanonicalObject]], zone: str,
                     payload: Any) -> None:
        if isinstance(payload, Mapping):
            for seat, rows in payload.items():
                if isinstance(rows, list):
                    for obj in rows:
                        result.setdefault(zone, []).append(
                            self._object(obj, zone=zone, default_seat=str(seat)))
                elif isinstance(rows, int):
                    for index in range(max(0, rows)):
                        result.setdefault(zone, []).append(CanonicalObject(
                            object_key=f"count:{zone}:{seat}:{index}",
                            zone=zone, controller=str(seat), owner=str(seat),
                            face_down=True,
                            metadata={"countPlaceholder": True}))
            return
        if isinstance(payload, list):
            for obj in payload:
                result.setdefault(zone, []).append(self._object(obj, zone=zone))

    def _object(self, value: Any, zone: str,
                default_seat: Optional[str] = None,
                revealed: bool = False) -> CanonicalObject:
        row = dict(value) if isinstance(value, Mapping) else {"name": str(value)}
        ref = row.get("ref") if isinstance(row.get("ref"), Mapping) else {}
        name = self._first(row.get("name"), ref.get("name"), row.get("cardName"))
        face_down = bool(row.get("faceDown") or row.get("face_down") or
                         row.get("hidden"))
        if face_down and not row.get("revealed"):
            name = None
        counters = row.get("counters") or {}
        if isinstance(counters, list):
            parsed = {}
            for item in counters:
                if isinstance(item, Mapping):
                    key = item.get("name") or item.get("type")
                    count = self._as_int(item.get("count"))
                    if key is not None and count is not None:
                        parsed[str(key)] = count
            counters = parsed
        if not isinstance(counters, Mapping):
            counters = {}
        types = row.get("types") or row.get("cardTypeNames") or []
        subtypes = row.get("subtypes") or row.get("subtypeNames") or []
        controller = self._first(row.get("controller"), row.get("controllerId"),
                                 row.get("controllerSeat"), default_seat)
        owner = self._first(row.get("owner"), row.get("ownerId"),
                            row.get("ownerSeat"), default_seat)
        return CanonicalObject(
            object_key=self._as_text(self._first(
                row.get("objectKey"), row.get("object_key"), row.get("instanceId"),
                row.get("objectId"), row.get("id"))),
            name=self._as_text(name),
            oracle_id=self._as_text(self._first(row.get("oracleId"),
                                                row.get("oracle_id"))),
            controller=self._as_text(controller),
            owner=self._as_text(owner),
            zone=zone,
            tapped=self._as_bool(row.get("tapped")),
            power=self._as_int(row.get("power")),
            toughness=self._as_int(row.get("toughness")),
            damage=self._as_int(row.get("damage")),
            counters={str(key): int(val) for key, val in counters.items()
                      if self._as_int(val) is not None},
            types=[str(item) for item in types] if isinstance(types, list) else [],
            subtypes=[str(item) for item in subtypes]
                     if isinstance(subtypes, list) else [],
            face_down=face_down,
            revealed=bool(revealed or row.get("revealed") or row.get("known")),
            token=self._as_bool(self._first(row.get("token"), row.get("isToken"))),
            attached_to=self._as_text(self._first(row.get("attachedTo"),
                                                  row.get("attached_to"))),
            attacking=self._as_bool(row.get("attacking")),
            blocking=self._as_bool(row.get("blocking")),
            summoning_sick=self._as_bool(self._first(
                row.get("summoningSick"), row.get("summoning_sick"))),
            metadata={key: copy.deepcopy(val) for key, val in row.items()
                      if key not in {
                          "name", "cardName", "ref", "oracleId", "oracle_id",
                          "controller", "controllerId", "controllerSeat", "owner",
                          "ownerId", "ownerSeat", "tapped", "power", "toughness",
                          "damage", "counters", "types", "cardTypeNames", "subtypes",
                          "subtypeNames", "faceDown", "face_down", "hidden",
                          "revealed", "known", "token", "isToken", "attachedTo",
                          "attached_to", "attacking", "blocking", "summoningSick",
                          "summoning_sick"}},
        )

    def _event(self, value: Any, index: int) -> CanonicalEvent:
        row = dict(value) if isinstance(value, Mapping) else {"text": str(value)}
        return CanonicalEvent(
            sequence=self._as_int(self._first(row.get("sequence"), index)),
            timestamp_ms=self._as_int(row.get("timestampMs")),
            actor=self._as_text(self._first(row.get("actor"), row.get("seat"),
                                            row.get("player"))),
            action_type=self._as_text(self._first(row.get("actionType"),
                                                  row.get("type"))),
            card_name=self._as_text(self._first(row.get("cardName"),
                                                row.get("card"))),
            targets=[str(item) for item in row.get("targets") or []],
            text=self._as_text(row.get("text")),
            confidence=float(row.get("confidence", 1.0) or 0.0),
            metadata={key: copy.deepcopy(val) for key, val in row.items()
                      if key not in {"sequence", "timestampMs", "actor", "seat",
                                     "player", "actionType", "type", "cardName",
                                     "card", "targets", "text", "confidence"}},
        )

    @staticmethod
    def _first(*values: Any) -> Any:
        return next((value for value in values if value is not None), None)

    @staticmethod
    def _safe_len(value: Any) -> Optional[int]:
        return len(value) if isinstance(value, (list, tuple, dict)) else None

    @staticmethod
    def _as_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _as_int(value: Any) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_bool(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "yes", "1", "tapped"}:
            return True
        if text in {"false", "no", "0", "untapped"}:
            return False
        return None


def canonical_to_model_observation(state: CanonicalState,
                                   perspective_seat: Optional[Any] = None
                                   ) -> Dict[str, Any]:
    """Project canonical state to the existing ``magic_cabt`` model input shape."""
    perspective = str(perspective_seat) if perspective_seat is not None \
        else state.perspective_seat
    current: Dict[str, Any] = {
        "matchId": state.match_id,
        "gameNumber": state.game_number,
        "gameInstance": state.game_id,
        "seq": state.sequence,
        "turnNumber": state.turn_number,
        "phase": state.phase,
        "step": state.step,
        "activeSeat": state.active_seat,
        "prioritySeat": state.priority_seat,
        "localSeat": perspective,
        "players": [],
        "zones": {
            "battlefield": [], "stack": [], "exile": [], "command": [],
            "hands": {}, "libraries": {}, "graveyards": {}, "sideboards": {},
        },
    }
    for player in state.players:
        current["players"].append({
            "seat": player.seat,
            "name": player.name,
            "playerId": player.player_id,
            "life": player.life,
            "poison": player.poison,
            "energy": player.energy,
            "handCount": player.hand_count,
            "libraryCount": player.library_count,
            "graveyardCount": player.graveyard_count,
            "manaPool": dict(player.mana_pool),
            "inGame": player.in_game,
            "passed": player.passed,
        })
    plural = {"hand": "hands", "library": "libraries",
              "graveyard": "graveyards", "sideboard": "sideboards"}
    for zone, rows in state.zones.items():
        for obj in rows:
            payload = {
                "objectKey": obj.object_key,
                "name": obj.name,
                "oracleId": obj.oracle_id,
                "controllerSeat": obj.controller,
                "ownerSeat": obj.owner,
                "tapped": obj.tapped,
                "power": obj.power,
                "toughness": obj.toughness,
                "damage": obj.damage,
                "counters": dict(obj.counters),
                "cardTypeNames": list(obj.types),
                "subtypeNames": list(obj.subtypes),
                "faceDown": obj.face_down,
                "revealed": obj.revealed,
                "isToken": obj.token,
                "attachedTo": obj.attached_to,
                "attacking": obj.attacking,
                "blocking": obj.blocking,
                "summoningSick": obj.summoning_sick,
            }
            payload = {key: value for key, value in payload.items()
                       if value is not None and value != []}
            if zone in plural:
                seat = obj.controller or obj.owner or "unknown"
                current["zones"][plural[zone]].setdefault(str(seat), []).append(payload)
            else:
                current["zones"].setdefault(zone, []).append(payload)
    history = [{
        "sequence": row.sequence,
        "timestampMs": row.timestamp_ms,
        "actor": row.actor,
        "actionType": row.action_type,
        "cardName": row.card_name,
        "targets": list(row.targets),
        "text": row.text,
        "confidence": row.confidence,
    } for row in state.public_history]
    return {
        "perspectiveSeat": perspective,
        "observation": {"current": current, "publicHistory": history},
    }
