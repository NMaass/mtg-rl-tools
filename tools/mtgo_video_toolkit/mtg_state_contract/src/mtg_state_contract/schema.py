"""Dependency-free canonical symbolic Magic state.

The schema is deliberately upstream of model tensorization.  Video perception,
MTGO logs, Arena logs, and XMage all format into this object.  The same object is
then projected to the existing ``magic_cabt`` observation shape.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional
import copy
import hashlib
import json


@dataclass
class CanonicalObject:
    object_key: Optional[str] = None
    name: Optional[str] = None
    oracle_id: Optional[str] = None
    controller: Optional[str] = None
    owner: Optional[str] = None
    zone: Optional[str] = None
    tapped: Optional[bool] = None
    power: Optional[int] = None
    toughness: Optional[int] = None
    damage: Optional[int] = None
    counters: Dict[str, int] = field(default_factory=dict)
    types: List[str] = field(default_factory=list)
    subtypes: List[str] = field(default_factory=list)
    face_down: bool = False
    revealed: bool = False
    token: Optional[bool] = None
    attached_to: Optional[str] = None
    attacking: Optional[bool] = None
    blocking: Optional[bool] = None
    summoning_sick: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalObject":
        fields = cls.__dataclass_fields__
        payload = {key: copy.deepcopy(val) for key, val in value.items()
                   if key in fields}
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalPlayer:
    seat: str
    name: Optional[str] = None
    player_id: Optional[str] = None
    life: Optional[int] = None
    poison: Optional[int] = None
    energy: Optional[int] = None
    hand_count: Optional[int] = None
    library_count: Optional[int] = None
    graveyard_count: Optional[int] = None
    mana_pool: Dict[str, int] = field(default_factory=dict)
    in_game: Optional[bool] = None
    passed: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalPlayer":
        fields = cls.__dataclass_fields__
        payload = {key: copy.deepcopy(val) for key, val in value.items()
                   if key in fields}
        payload["seat"] = str(payload.get("seat", "unknown"))
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalEvent:
    sequence: Optional[int] = None
    timestamp_ms: Optional[int] = None
    actor: Optional[str] = None
    action_type: Optional[str] = None
    card_name: Optional[str] = None
    targets: List[str] = field(default_factory=list)
    text: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalEvent":
        fields = cls.__dataclass_fields__
        payload = {key: copy.deepcopy(val) for key, val in value.items()
                   if key in fields}
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalState:
    schema_version: int = 1
    source: str = "unknown"
    source_id: Optional[str] = None
    match_id: Optional[str] = None
    game_id: Optional[str] = None
    game_number: Optional[int] = None
    timestamp_ms: Optional[int] = None
    sequence: Optional[int] = None
    perspective_seat: Optional[str] = None
    turn_number: Optional[int] = None
    active_seat: Optional[str] = None
    priority_seat: Optional[str] = None
    phase: Optional[str] = None
    step: Optional[str] = None
    players: List[CanonicalPlayer] = field(default_factory=list)
    zones: Dict[str, List[CanonicalObject]] = field(default_factory=dict)
    public_history: List[CanonicalEvent] = field(default_factory=list)
    confidence: Dict[str, float] = field(default_factory=dict)
    unknown_paths: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalState":
        payload = copy.deepcopy(dict(value))
        payload["players"] = [CanonicalPlayer.from_dict(row)
                              for row in payload.get("players") or []]
        payload["zones"] = {
            str(zone): [CanonicalObject.from_dict(row) for row in rows or []]
            for zone, rows in (payload.get("zones") or {}).items()
        }
        payload["public_history"] = [CanonicalEvent.from_dict(row)
                                     for row in payload.get("public_history") or []]
        fields = cls.__dataclass_fields__
        payload = {key: val for key, val in payload.items() if key in fields}
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def copy(self) -> "CanonicalState":
        return CanonicalState.from_dict(self.to_dict())

    def confidence_at(self, path: str, default: float = 1.0) -> float:
        if path in self.unknown_paths:
            return 0.0
        value = self.confidence.get(path, default)
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    def mark_unknown(self, *paths: str) -> None:
        existing = set(self.unknown_paths)
        existing.update(str(path) for path in paths)
        self.unknown_paths = sorted(existing)

    def player(self, seat: Any) -> Optional[CanonicalPlayer]:
        target = str(seat)
        return next((row for row in self.players if str(row.seat) == target), None)

    def semantic_fingerprint(self) -> str:
        """Stable content hash excluding volatile provenance and tracking IDs."""
        value = self.to_dict()
        value.pop("timestamp_ms", None)
        value.pop("source_id", None)
        value.pop("provenance", None)
        value.pop("confidence", None)
        value.pop("unknown_paths", None)
        for rows in value.get("zones", {}).values():
            for row in rows:
                row.pop("object_key", None)
                row.pop("metadata", None)
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


def ensure_state(value: Any) -> CanonicalState:
    if isinstance(value, CanonicalState):
        return value
    if isinstance(value, Mapping):
        return CanonicalState.from_dict(value)
    raise TypeError("expected CanonicalState or mapping")


def ensure_objects(values: Iterable[Any]) -> List[CanonicalObject]:
    return [value if isinstance(value, CanonicalObject)
            else CanonicalObject.from_dict(value) for value in values]
