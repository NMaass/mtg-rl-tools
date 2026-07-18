"""Serializable perception records."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OCRSpan:
    text: str
    confidence: float
    bbox: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectedCard:
    bbox: List[int]
    zone: str
    controller: Optional[str] = None
    name: Optional[str] = None
    tapped: Optional[bool] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerceivedFrame:
    frame_index: int
    timestamp_ms: int
    change_score: float
    fields: Dict[str, Any] = field(default_factory=dict)
    field_confidence: Dict[str, float] = field(default_factory=dict)
    zone_confidence: Dict[str, float] = field(default_factory=dict)
    cards: List[DetectedCard] = field(default_factory=list)
    log_lines: List[OCRSpan] = field(default_factory=list)
    raw_regions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        return value
