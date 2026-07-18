"""Optional VLM second pass for uncertain MTGO frames.

This is a verifier/repair layer, not the default high-throughput reader.  It is
called only on configured anchor or low-confidence frames and is constrained by
an explicit JSON schema.  The prompt forbids guessing hidden card identities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .ocr import OCRBackend
from .types import DetectedCard, OCRSpan, PerceivedFrame


@dataclass
class SecondaryVisionPolicy:
    every_n_samples: int = 0
    minimum_primary_confidence: float = 0.48
    max_calls: Optional[int] = None

    def should_run(self, sample_number: int, frame: PerceivedFrame,
                   calls_so_far: int) -> bool:
        if self.max_calls is not None and calls_so_far >= self.max_calls:
            return False
        periodic = self.every_n_samples > 0 and \
            sample_number % self.every_n_samples == 0
        values = list(frame.field_confidence.values())
        primary = sum(values) / len(values) if values else 0.0
        uncertain = primary < self.minimum_primary_confidence
        return periodic or uncertain


class StructuredMTGOScreenReader:
    def __init__(self, backend: OCRBackend):
        self.backend = backend

    def read(self, image: np.ndarray) -> Dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "object",
                    "properties": {
                        "local_life": _nullable_integer(),
                        "opponent_life": _nullable_integer(),
                        "local_hand_count": _nullable_integer(),
                        "opponent_hand_count": _nullable_integer(),
                        "turn_number": _nullable_integer(),
                        "phase": {"type": ["string", "null"]},
                    },
                    "required": [
                        "local_life", "opponent_life", "local_hand_count",
                        "opponent_hand_count", "turn_number", "phase"],
                },
                "field_confidence": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "cards": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "zone": {"type": "string"},
                            "controller": {"type": ["string", "null"]},
                            "name": {"type": ["string", "null"]},
                            "tapped": {"type": ["boolean", "null"]},
                            "bbox": {
                                "type": ["array", "null"],
                                "items": {"type": "integer"},
                            },
                            "confidence": {"type": "number"},
                            "hidden": {"type": "boolean"},
                        },
                        "required": ["zone", "controller", "name", "tapped",
                                     "bbox", "confidence", "hidden"],
                    },
                },
                "log_lines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["text", "confidence"],
                    },
                },
            },
            "required": ["fields", "field_confidence", "cards", "log_lines"],
        }
        prompt = """
Inspect this Magic: The Gathering Online duel screenshot and return only the
requested JSON. The local player is at the bottom and the opponent is at the
top. Read visible life totals, visible hand counts, turn number, phase, public
battlefield/stack cards, and visible game-log lines. Use pixel bounding boxes in
[x,y,width,height] coordinates for cards when possible. Set a card name to null
when its title is unreadable. Never infer or guess face-down cards, hidden hand
cards, library contents, or off-screen information. Confidence is 0 to 1 and
must reflect only visual evidence.
""".strip()
        return self.backend.extract_json(image, prompt, schema=schema)


def merge_structured_read(frame: PerceivedFrame,
                          payload: Dict[str, Any]) -> PerceivedFrame:
    fields = payload.get("fields") or {}
    confidence = payload.get("field_confidence") or {}
    for key, value in fields.items():
        score = _clamp(confidence.get(key, 0.0))
        if value is not None and score > frame.field_confidence.get(key, 0.0):
            frame.fields[key] = value
            frame.field_confidence[key] = score
    for row in payload.get("cards") or []:
        if not isinstance(row, dict) or row.get("hidden"):
            # Preserve count evidence while refusing hidden identity.
            name = None
        else:
            name = row.get("name")
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            bbox = [0, 0, 0, 0]
        frame.cards.append(DetectedCard(
            bbox=[int(value) for value in bbox[:4]],
            zone=str(row.get("zone") or "unknown"),
            controller=str(row["controller"])
                if row.get("controller") is not None else None,
            name=str(name) if name else None,
            tapped=row.get("tapped") if isinstance(row.get("tapped"), bool)
                else None,
            confidence=_clamp(row.get("confidence", 0.0)),
            metadata={"detector": "structured-vlm", "secondaryPass": True},
        ))
    for row in payload.get("log_lines") or []:
        if isinstance(row, dict) and str(row.get("text", "")).strip():
            frame.log_lines.append(OCRSpan(
                text=str(row["text"]), confidence=_clamp(row.get("confidence", 0.0))))
    frame.raw_regions["secondaryVision"] = {
        "used": True,
        "backend": getattr(frame, "backend", None),
    }
    return frame


def _nullable_integer():
    return {"type": ["integer", "null"]}


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
