"""Confidence-aware semantic comparison of perceived and engine states."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math

from .schema import CanonicalObject, CanonicalPlayer, CanonicalState, ensure_state


@dataclass
class DiffItem:
    status: str
    path: str
    expected: Any = None
    observed: Any = None
    message: str = ""
    confidence: float = 1.0
    severity: str = "error"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonPolicy:
    min_confidence: float = 0.55
    unknown_is_mismatch: bool = False
    ignore_priority: bool = False
    ignore_object_tracking_ids: bool = True
    ignored_paths: List[str] = field(default_factory=list)
    object_fields: Tuple[str, ...] = (
        "name", "controller", "owner", "tapped", "power", "toughness",
        "damage", "counters", "face_down", "token", "attacking", "blocking",
    )


@dataclass
class ComparisonReport:
    expected_source: str
    observed_source: str
    items: List[DiffItem]

    @property
    def mismatches(self) -> int:
        return sum(item.status == "MISMATCH" for item in self.items)

    @property
    def unknowns(self) -> int:
        return sum(item.status in {"UNKNOWN", "UNOBSERVABLE"}
                   for item in self.items)

    @property
    def matches(self) -> int:
        return sum(item.status == "MATCH" for item in self.items)

    @property
    def passed(self) -> bool:
        return self.mismatches == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expectedSource": self.expected_source,
            "observedSource": self.observed_source,
            "passed": self.passed,
            "counts": {"match": self.matches, "mismatch": self.mismatches,
                       "unknown": self.unknowns},
            "items": [item.to_dict() for item in self.items],
        }


def compare_states(expected: CanonicalState, observed: CanonicalState,
                   policy: Optional[ComparisonPolicy] = None) -> ComparisonReport:
    expected = ensure_state(expected)
    observed = ensure_state(observed)
    policy = policy or ComparisonPolicy()
    items: List[DiffItem] = []

    for path, left, right in (
        ("/turn_number", expected.turn_number, observed.turn_number),
        ("/phase", expected.phase, observed.phase),
        ("/step", expected.step, observed.step),
        ("/active_seat", expected.active_seat, observed.active_seat),
        ("/priority_seat", expected.priority_seat, observed.priority_seat),
    ):
        if path == "/priority_seat" and policy.ignore_priority:
            continue
        _compare_scalar(items, expected, observed, path, left, right, policy)

    expected_players = {str(row.seat): row for row in expected.players}
    observed_players = {str(row.seat): row for row in observed.players}
    for seat in sorted(set(expected_players) | set(observed_players)):
        left = expected_players.get(seat)
        right = observed_players.get(seat)
        path = f"/players/{seat}"
        if left is None or right is None:
            _compare_scalar(items, expected, observed, path,
                            left.to_dict() if left else None,
                            right.to_dict() if right else None, policy)
            continue
        for field_name in ("life", "poison", "energy", "hand_count",
                           "library_count", "graveyard_count"):
            _compare_scalar(items, expected, observed,
                            f"{path}/{field_name}",
                            getattr(left, field_name), getattr(right, field_name),
                            policy)

    for zone in sorted(set(expected.zones) | set(observed.zones)):
        _compare_zone(items, expected, observed, zone,
                      expected.zones.get(zone, []), observed.zones.get(zone, []),
                      policy)
    return ComparisonReport(expected.source, observed.source, items)


def _compare_scalar(items: List[DiffItem], expected_state: CanonicalState,
                    observed_state: CanonicalState, path: str,
                    expected: Any, observed: Any,
                    policy: ComparisonPolicy) -> None:
    if _ignored(path, policy):
        return
    confidence = observed_state.confidence_at(path, 1.0)
    unknown = path in observed_state.unknown_paths or observed is None
    if unknown or confidence < policy.min_confidence:
        status = "MISMATCH" if policy.unknown_is_mismatch else "UNKNOWN"
        items.append(DiffItem(
            status=status, path=path, expected=expected, observed=observed,
            confidence=confidence,
            message="observed value is absent or below confidence threshold",
            severity="error" if status == "MISMATCH" else "info"))
        return
    equal = _semantic_equal(expected, observed)
    items.append(DiffItem(
        status="MATCH" if equal else "MISMATCH",
        path=path, expected=expected, observed=observed,
        confidence=confidence,
        message="values agree" if equal else "values differ",
        severity="info" if equal else "error"))


def _compare_zone(items: List[DiffItem], expected_state: CanonicalState,
                  observed_state: CanonicalState, zone: str,
                  expected: Sequence[CanonicalObject],
                  observed: Sequence[CanonicalObject],
                  policy: ComparisonPolicy) -> None:
    base = f"/zones/{zone}"
    if _ignored(base, policy):
        return
    count_path = base + "/count"
    count_confidence = observed_state.confidence_at(count_path, 1.0)
    count_unknown = count_path in observed_state.unknown_paths
    if not count_unknown and count_confidence >= policy.min_confidence:
        items.append(DiffItem(
            status="MATCH" if len(expected) == len(observed) else "MISMATCH",
            path=count_path, expected=len(expected), observed=len(observed),
            confidence=count_confidence,
            message="zone counts agree" if len(expected) == len(observed)
                    else "zone counts differ",
            severity="info" if len(expected) == len(observed) else "error"))
    else:
        items.append(DiffItem(
            status="MISMATCH" if policy.unknown_is_mismatch else "UNKNOWN",
            path=count_path, expected=len(expected),
            observed=len(observed), confidence=count_confidence,
            message="zone count is unknown or below confidence threshold",
            severity="error" if policy.unknown_is_mismatch else "info"))

    matches, missing_expected, extra_observed = _match_objects(
        expected, observed, policy.object_fields)
    for left_index, right_index, score in matches:
        left, right = expected[left_index], observed[right_index]
        object_path = f"{base}/match-{left_index}-{right_index}"
        for field_name in policy.object_fields:
            left_value = getattr(left, field_name)
            right_value = getattr(right, field_name)
            field_path = f"{object_path}/{field_name}"
            object_confidence = _object_confidence(right, score)
            if right_value is None or (field_name == "name" and right.face_down) \
                    or object_confidence < policy.min_confidence:
                items.append(DiffItem(
                    status="UNKNOWN", path=field_path, expected=left_value,
                    observed=right_value, confidence=object_confidence,
                    message="object field is not observable", severity="info"))
            else:
                equal = _semantic_equal(left_value, right_value)
                items.append(DiffItem(
                    status="MATCH" if equal else "MISMATCH", path=field_path,
                    expected=left_value, observed=right_value,
                    confidence=object_confidence,
                    message="object field agrees" if equal else
                            "object field differs",
                    severity="info" if equal else "error"))
    for index in missing_expected:
        status = "MISMATCH" if not count_unknown and \
            count_confidence >= policy.min_confidence else "UNKNOWN"
        items.append(DiffItem(
            status=status, path=f"{base}/missing/{index}",
            expected=expected[index].name or expected[index].to_dict(),
            observed=None, confidence=count_confidence,
            message="engine object has no perceived counterpart",
            severity="error" if status == "MISMATCH" else "info"))
    for index in extra_observed:
        status = "MISMATCH" if not count_unknown and \
            count_confidence >= policy.min_confidence else "UNKNOWN"
        items.append(DiffItem(
            status=status, path=f"{base}/extra/{index}", expected=None,
            observed=observed[index].name or observed[index].to_dict(),
            confidence=count_confidence,
            message="perceived object has no engine counterpart",
            severity="error" if status == "MISMATCH" else "info"))


def _match_objects(expected: Sequence[CanonicalObject],
                   observed: Sequence[CanonicalObject],
                   fields: Iterable[str]
                   ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    candidates: List[Tuple[float, int, int]] = []
    for left_index, left in enumerate(expected):
        for right_index, right in enumerate(observed):
            candidates.append((_object_similarity(left, right, fields),
                               left_index, right_index))
    candidates.sort(reverse=True)
    used_left, used_right = set(), set()
    matches: List[Tuple[int, int, float]] = []
    for score, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        if score < 0.25:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append((left_index, right_index, score))
    return (matches,
            [index for index in range(len(expected)) if index not in used_left],
            [index for index in range(len(observed)) if index not in used_right])


def _object_similarity(left: CanonicalObject, right: CanonicalObject,
                       fields: Iterable[str]) -> float:
    weight = 0.0
    score = 0.0
    for field_name in fields:
        left_value = getattr(left, field_name)
        right_value = getattr(right, field_name)
        field_weight = 4.0 if field_name == "name" else 1.0
        if right_value is None or (field_name == "name" and right.face_down):
            continue
        weight += field_weight
        if _semantic_equal(left_value, right_value):
            score += field_weight
    if weight == 0:
        return 0.5
    return score / weight


def _object_confidence(value: CanonicalObject, fallback: float) -> float:
    raw = value.metadata.get("confidence", fallback)
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(fallback)))


def _semantic_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, str) or isinstance(right, str):
        return str(left).strip().casefold() == str(right).strip().casefold()
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0,
                                abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    return left == right


def _ignored(path: str, policy: ComparisonPolicy) -> bool:
    return any(path == ignored or path.startswith(ignored.rstrip("/") + "/")
               for ignored in policy.ignored_paths)
