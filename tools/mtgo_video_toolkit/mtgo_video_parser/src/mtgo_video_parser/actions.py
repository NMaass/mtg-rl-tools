"""Parse OCR'd MTGO game-log lines into semantic actions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import re

from .types import OCRSpan


@dataclass
class ObservedAction:
    timestamp_ms: int
    action_type: str
    actor: Optional[str] = None
    card_name: Optional[str] = None
    targets: List[str] = field(default_factory=list)
    text: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class MTGOLogActionParser:
    """Conservative regex parser for common English MTGO log sentences."""

    _RULES = [
        ("CAST_SPELL", re.compile(r"^(?P<actor>.+?) casts (?P<card>.+?)(?: targeting (?P<target>.+))?\.?$", re.I)),
        ("PLAY_LAND", re.compile(r"^(?P<actor>.+?) plays (?P<card>.+?)\.?$", re.I)),
        ("ACTIVATE_ABILITY", re.compile(r"^(?P<actor>.+?) activates (?:an ability of )?(?P<card>.+?)(?: targeting (?P<target>.+))?\.?$", re.I)),
        ("ATTACK", re.compile(r"^(?P<actor>.+?) attacks with (?P<card>.+?)\.?$", re.I)),
        ("BLOCK", re.compile(r"^(?P<actor>.+?) blocks with (?P<card>.+?)\.?$", re.I)),
        ("DRAW", re.compile(r"^(?P<actor>.+?) draws? (?:a card|(?P<card>.+?))\.?$", re.I)),
        ("DISCARD", re.compile(r"^(?P<actor>.+?) discards (?P<card>.+?)\.?$", re.I)),
        ("SACRIFICE", re.compile(r"^(?P<actor>.+?) sacrifices (?P<card>.+?)\.?$", re.I)),
        ("DESTROY", re.compile(r"^(?P<card>.+?) is destroyed\.?$", re.I)),
        ("GAIN_LIFE", re.compile(r"^(?P<actor>.+?) gains (?P<amount>\d+) life\.?$", re.I)),
        ("LOSE_LIFE", re.compile(r"^(?P<actor>.+?) loses (?P<amount>\d+) life\.?$", re.I)),
        ("CONCEDE", re.compile(r"^(?P<actor>.+?) concedes\.?$", re.I)),
        ("MULLIGAN", re.compile(r"^(?P<actor>.+?) mulligans?.*$", re.I)),
        ("KEEP", re.compile(r"^(?P<actor>.+?) keeps? (?:their )?hand.*$", re.I)),
    ]

    def parse(self, spans: Iterable[OCRSpan], timestamp_ms: int) -> List[ObservedAction]:
        lines = merge_spans_to_lines(spans)
        result = []
        for text, confidence in lines:
            for action_type, pattern in self._RULES:
                match = pattern.match(text.strip())
                if not match:
                    continue
                groups = match.groupdict()
                targets = [groups["target"]] if groups.get("target") else []
                metadata = {key: value for key, value in groups.items()
                            if key not in {"actor", "card", "target"}
                            and value is not None}
                result.append(ObservedAction(
                    timestamp_ms=timestamp_ms,
                    action_type=action_type,
                    actor=_clean(groups.get("actor")),
                    card_name=_clean(groups.get("card")),
                    targets=[_clean(value) for value in targets if _clean(value)],
                    text=text,
                    confidence=confidence,
                    metadata=metadata,
                ))
                break
        return result


def merge_spans_to_lines(spans: Iterable[OCRSpan]):
    """Return ordered ``(text, confidence)`` lines from OCR word spans."""
    rows = list(spans)
    if not rows:
        return []
    if not any(row.bbox for row in rows):
        return [(row.text, row.confidence) for row in rows]
    rows.sort(key=lambda row: ((row.bbox or [0, 0])[1],
                              (row.bbox or [0, 0])[0]))
    lines = []
    current = []
    baseline = None
    for row in rows:
        y = (row.bbox or [0, 0])[1]
        height = (row.bbox or [0, 0, 0, 12])[3]
        if baseline is None or abs(y - baseline) <= max(8, height * 0.6):
            current.append(row)
            baseline = y if baseline is None else (baseline + y) / 2
        else:
            lines.append(_line(current))
            current = [row]
            baseline = y
    if current:
        lines.append(_line(current))
    return lines


def _line(rows):
    rows = sorted(rows, key=lambda row: (row.bbox or [0])[0])
    text = " ".join(row.text for row in rows).strip()
    confidence = sum(row.confidence for row in rows) / max(1, len(rows))
    return text, confidence


def _clean(value):
    if value is None:
        return None
    return str(value).strip(" .,:;\t\n") or None
