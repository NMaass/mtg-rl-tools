"""Card identity recognition from detected MTGO card rectangles.

The recognizer is intentionally conservative.  It OCRs the title bar of a card,
normalizes the candidate against a local Scryfall bulk database, and only writes
an identity when both OCR and card-database agreement clear configured gates.
Unknown cards remain unknown rather than becoming plausible hallucinations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import re

import cv2
import numpy as np

from .carddb import CardNameResolver, CardResolution
from .ocr import OCRBackend
from .types import DetectedCard, OCRSpan


@dataclass
class RecognitionAttempt:
    raw_text: str
    ocr_confidence: float
    resolved_name: Optional[str]
    resolver_score: float
    oracle_id: Optional[str]
    combined_score: float
    orientation: str

    def to_dict(self):
        return asdict(self)


class CardIdentityRecognizer:
    """Recognize card names from title bars and validate against known cards."""

    def __init__(self, ocr: OCRBackend,
                 resolver: Optional[CardNameResolver] = None,
                 minimum_resolver_score: float = 76.0,
                 minimum_combined_score: float = 0.60,
                 title_fraction: float = 0.28):
        self.ocr = ocr
        self.resolver = resolver
        self.minimum_resolver_score = float(minimum_resolver_score)
        self.minimum_combined_score = float(minimum_combined_score)
        self.title_fraction = max(0.12, min(0.55, float(title_fraction)))

    def recognize(self, frame: np.ndarray, card: DetectedCard) -> DetectedCard:
        crop = _safe_crop(frame, card.bbox)
        if crop.size == 0:
            return card
        attempts: List[RecognitionAttempt] = []
        orientations = [("upright", crop)]
        # Tapped cards in MTGO may be rotated in either direction depending on
        # client rendering.  Try both; the resolver gate prevents random OCR
        # fragments from being accepted.
        if card.tapped:
            orientations.extend([
                ("clockwise", cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)),
                ("counter-clockwise", cv2.rotate(
                    crop, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ])
        for orientation, oriented in orientations:
            title = oriented[:max(1, int(oriented.shape[0] * self.title_fraction)), :]
            for variant_name, variant in _title_variants(title):
                spans = self.ocr.read(
                    variant, hint=f"card-title:{orientation}:{variant_name}")
                attempts.extend(self._attempts(spans, orientation))
        if not attempts:
            card.metadata.setdefault("recognition", {})["status"] = "no-text"
            return card
        attempts.sort(key=lambda row: row.combined_score, reverse=True)
        best = attempts[0]
        card.metadata["recognition"] = {
            "status": "accepted" if best.resolved_name and
                      best.combined_score >= self.minimum_combined_score else
                      "below-threshold",
            "best": best.to_dict(),
            "attempts": [row.to_dict() for row in attempts[:8]],
        }
        if best.resolved_name and best.combined_score >= self.minimum_combined_score:
            card.name = best.resolved_name
            card.confidence = max(card.confidence, best.combined_score)
            card.metadata["oracleId"] = best.oracle_id
            card.metadata["identitySource"] = "title-ocr+scryfall"
        return card

    def _attempts(self, spans: Sequence[OCRSpan],
                  orientation: str) -> List[RecognitionAttempt]:
        candidates = _text_candidates(spans)
        result: List[RecognitionAttempt] = []
        for raw_text, ocr_confidence in candidates:
            if self.resolver is None:
                # Keep the evidence, but do not assert a card identity without a
                # dictionary.  This is a deliberate fail-closed policy.
                result.append(RecognitionAttempt(
                    raw_text=raw_text, ocr_confidence=ocr_confidence,
                    resolved_name=None, resolver_score=0.0, oracle_id=None,
                    combined_score=0.0, orientation=orientation))
                continue
            resolution = self.resolver.resolve(
                raw_text, minimum_score=self.minimum_resolver_score)
            combined = _combine_scores(ocr_confidence, resolution)
            result.append(RecognitionAttempt(
                raw_text=raw_text,
                ocr_confidence=ocr_confidence,
                resolved_name=resolution.name,
                resolver_score=resolution.score,
                oracle_id=resolution.oracle_id,
                combined_score=combined,
                orientation=orientation,
            ))
        return result


def _safe_crop(frame: np.ndarray, bbox: Sequence[int]) -> np.ndarray:
    if len(bbox) < 4:
        return np.empty((0, 0, 3), dtype=np.uint8)
    x, y, width, height = [int(value) for value in bbox[:4]]
    left, top = max(0, x), max(0, y)
    right = min(frame.shape[1], x + max(0, width))
    bottom = min(frame.shape[0], y + max(0, height))
    if right <= left or bottom <= top:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return frame[top:bottom, left:right]


def _title_variants(image: np.ndarray):
    scale = max(2.0, 54.0 / max(1, image.shape[0]))
    enlarged = cv2.resize(image, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(equalized, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 9)
    for name, value in (
        ("rgb", enlarged),
        ("gray", cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)),
        ("otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)),
        ("adaptive", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)),
    ):
        yield name, value


def _text_candidates(spans: Sequence[OCRSpan]) -> List[Tuple[str, float]]:
    cleaned = []
    for span in spans:
        text = _clean_title(span.text)
        if text:
            cleaned.append((text, max(0.0, min(1.0, span.confidence))))
    candidates = list(cleaned)
    if cleaned:
        joined = " ".join(row[0] for row in cleaned)
        confidence = sum(row[1] for row in cleaned) / len(cleaned)
        candidates.append((_clean_title(joined), confidence))
    # Preserve order while removing duplicate normalized strings.
    result = []
    seen = set()
    for text, confidence in candidates:
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append((text, confidence))
    return result


def _clean_title(value: str) -> str:
    text = str(value or "")
    text = text.replace("|", "I")
    text = re.sub(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ'’\- ,.!/:]+", " ", text)
    text = " ".join(text.split()).strip(" .,:;/")
    # Mana costs and lone OCR artifacts are common on the title bar.
    text = re.sub(r"\s+(?:[0-9WUBRGCXYZ/]{1,8})$", "", text,
                  flags=re.I).strip()
    return text if len(text) >= 2 else ""


def _combine_scores(ocr_confidence: float,
                    resolution: CardResolution) -> float:
    if resolution.name is None:
        return 0.0
    dictionary = max(0.0, min(1.0, resolution.score / 100.0))
    ocr = max(0.0, min(1.0, float(ocr_confidence)))
    # Dictionary agreement is more reliable than OCR's raw confidence on tiny
    # title bars, but neither signal can independently force acceptance.
    return dictionary * (0.35 + 0.65 * ocr)
