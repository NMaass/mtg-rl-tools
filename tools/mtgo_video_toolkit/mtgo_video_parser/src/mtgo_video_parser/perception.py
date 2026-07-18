"""Per-frame MTGO UI perception driven by a layout profile."""
from __future__ import annotations

from typing import Dict, List, Optional
import re

import cv2
import numpy as np

from .carddb import CardNameResolver
from .detector import CardRectangleDetector
from .layout import LayoutProfile
from .ocr import OCRBackend
from .recognition import CardIdentityRecognizer
from .structured_vision import (
    SecondaryVisionPolicy, StructuredMTGOScreenReader, merge_structured_read)
from .types import DetectedCard, OCRSpan, PerceivedFrame


class FramePerceiver:
    def __init__(self, layout: LayoutProfile, ocr: OCRBackend,
                 detector: Optional[CardRectangleDetector] = None,
                 card_resolver: Optional[CardNameResolver] = None,
                 card_ocr: Optional[OCRBackend] = None,
                 secondary_vision: Optional[OCRBackend] = None,
                 secondary_policy: Optional[SecondaryVisionPolicy] = None):
        self.layout = layout
        self.ocr = ocr
        self.detector = detector or CardRectangleDetector()
        self.card_recognizer = CardIdentityRecognizer(
            card_ocr or ocr, card_resolver) if card_resolver is not None else None
        self.secondary_reader = StructuredMTGOScreenReader(secondary_vision) \
            if secondary_vision is not None else None
        self.secondary_policy = secondary_policy or SecondaryVisionPolicy()
        self._sample_number = 0
        self._secondary_calls = 0

    def perceive(self, frame_index: int, timestamp_ms: int,
                 change_score: float, image: np.ndarray) -> PerceivedFrame:
        fields: Dict[str, object] = {}
        confidence: Dict[str, float] = {}
        zone_confidence: Dict[str, float] = {}
        cards: List[DetectedCard] = []
        log_lines: List[OCRSpan] = []
        raw_regions = {}
        self._sample_number += 1
        for region in self.layout.regions:
            crop = region.crop(image)
            region_quality = _image_quality(crop)
            raw_regions[region.name] = {
                "pixelBox": list(region.pixel_box(image.shape)),
                "kind": region.kind,
                "imageQuality": region_quality,
            }
            if region.kind == "phase_bar":
                # The old MTGO client marks the active step with a colour
                # highlight on a fixed step bar, not distinct text, so this is a
                # colour detector rather than OCR. Its confidence reflects the
                # highlight, not text sharpness, so it bypasses region_quality.
                value, score = _read_phase_bar(crop, region.config)
                fields[region.name] = value
                confidence[region.name] = score
            elif region.kind in {"integer", "text", "log", "phase"}:
                prepared = _preprocess(crop, region.config)
                spans = self.ocr.read(prepared, hint=region.name)
                if region.kind == "integer":
                    value, score = _read_integer(spans)
                    fields[region.name] = value
                    confidence[region.name] = _evidence_confidence(score, region_quality)
                elif region.kind == "phase":
                    text, score = _join(spans)
                    fields[region.name] = _normalize_phase(text)
                    confidence[region.name] = _evidence_confidence(score, region_quality)
                elif region.kind == "log":
                    log_lines.extend(spans)
                else:
                    text, score = _join(spans)
                    fields[region.name] = text
                    confidence[region.name] = _evidence_confidence(score, region_quality)
            elif region.kind == "card_zone":
                zone = str(region.config.get("zone") or region.name)
                controller = region.config.get("controller")
                detected = self.detector.detect(crop, zone=zone,
                                                controller=str(controller)
                                                if controller is not None else None)
                left, top, _, _ = region.pixel_box(image.shape)
                for card in detected:
                    card.bbox[0] += left
                    card.bbox[1] += top
                    card.metadata["confidence"] = card.confidence
                    card.metadata["region"] = region.name
                    if self.card_recognizer is not None and \
                            region.config.get("recognize_names", True):
                        self.card_recognizer.recognize(image, card)
                cards.extend(detected)
                if detected:
                    detector_score = sum(row.confidence for row in detected) / len(detected)
                    zone_confidence[zone] = max(
                        zone_confidence.get(zone, 0.0),
                        min(region_quality, 0.35 + 0.65 * detector_score))
                else:
                    # An empty detection is weak evidence, not proof that the
                    # zone is empty.  Layouts can raise this after calibration.
                    empty_confidence = float(
                        region.config.get("empty_confidence", 0.30))
                    zone_confidence[zone] = max(
                        zone_confidence.get(zone, 0.0),
                        min(region_quality, empty_confidence))
        perceived = PerceivedFrame(
            frame_index=frame_index, timestamp_ms=timestamp_ms,
            change_score=change_score, fields=fields,
            field_confidence=confidence, zone_confidence=zone_confidence,
            cards=cards, log_lines=log_lines, raw_regions=raw_regions)
        if self.secondary_reader is not None and self.secondary_policy.should_run(
                self._sample_number, perceived, self._secondary_calls):
            try:
                payload = self.secondary_reader.read(image)
                merge_structured_read(perceived, payload)
                perceived.cards = _deduplicate_cards(perceived.cards)
                self._secondary_calls += 1
                perceived.raw_regions["secondaryVision"] = {
                    "used": True,
                    "backend": getattr(self.secondary_reader.backend, "name", None),
                    "callNumber": self._secondary_calls,
                }
            except Exception as error:
                # Extraction continues with the local reader.  The failure is
                # retained for audit rather than silently swallowed.
                perceived.raw_regions["secondaryVision"] = {
                    "used": False, "error": f"{type(error).__name__}: {error}"}
        return perceived


def _preprocess(image: np.ndarray, config: Dict[str, object]) -> np.ndarray:
    scale = float(config.get("scale", 2.0))
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)
    if config.get("grayscale", True):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if config.get("threshold") == "otsu":
            _, gray = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif config.get("threshold") == "adaptive":
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 9)
        elif config.get("invert"):
            gray = 255 - gray
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return image


def _read_integer(spans: List[OCRSpan]):
    candidates = []
    for span in spans:
        normalized = span.text.replace("O", "0").replace("o", "0")
        for match in re.findall(r"-?\d+", normalized):
            try:
                candidates.append((int(match), span.confidence))
            except ValueError:
                pass
    return max(candidates, key=lambda row: row[1]) if candidates else (None, 0.0)


def _read_phase_bar(image: np.ndarray, config: Dict[str, object]):
    """Locate the colour-highlighted active step on a fixed MTGO phase bar.

    config keys: `steps` (ordered canonical phase names, left to right),
    `centers` (matching x-fractions of each step within the crop), optional
    `hsv_low`/`hsv_high` (highlight colour range) and `min_center` (ignore
    highlights left of this fraction, e.g. the active-player turn label).
    Returns (canonical_phase, confidence) or (None, 0.0) when no step is lit.
    """
    steps = list(config.get("steps") or [])
    centers = [float(value) for value in (config.get("centers") or [])]
    if not steps or len(steps) != len(centers) or image.size == 0:
        return None, 0.0
    low = np.array(config.get("hsv_low", [8, 80, 120]), dtype=np.uint8)
    high = np.array(config.get("hsv_high", [28, 255, 255]), dtype=np.uint8)
    min_center = float(config.get("min_center", 0.07))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low, high)
    columns = mask.sum(axis=0).astype(np.float64)
    if columns.max() <= 0:
        return None, 0.0
    lit = np.where(columns > columns.max() * 0.3)[0]
    if lit.size == 0:
        return None, 0.0
    center = float((lit.min() + lit.max()) / 2 / image.shape[1])
    if center < min_center:
        return None, 0.0
    distances = [abs(center - value) for value in centers]
    index = int(np.argmin(distances))
    proximity = max(0.0, 1.0 - distances[index] / 0.06)
    score = max(0.0, min(1.0, 0.6 + 0.4 * proximity))
    return steps[index], score


def _join(spans: List[OCRSpan]):
    if not spans:
        return None, 0.0
    ordered = sorted(spans, key=lambda row: (
        (row.bbox or [0, 0])[1], (row.bbox or [0, 0])[0]))
    text = " ".join(row.text for row in ordered).strip()
    score = sum(row.confidence for row in ordered) / len(ordered)
    return text or None, score


def _normalize_phase(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")
    aliases = {
        "PRECOMBAT_MAIN": "MAIN1", "FIRST_MAIN": "MAIN1",
        "POSTCOMBAT_MAIN": "MAIN2", "SECOND_MAIN": "MAIN2",
        "BEGIN_COMBAT": "BEGIN_COMBAT", "DECLARE_ATTACKERS": "ATTACKERS",
        "DECLARE_BLOCKERS": "BLOCKERS", "END_COMBAT": "END_COMBAT",
        "UPKEEP": "UPKEEP", "DRAW": "DRAW", "END": "END",
    }
    return aliases.get(text, text)


def _image_quality(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())
    sharp_score = min(1.0, sharpness / 180.0)
    contrast_score = min(1.0, contrast / 45.0)
    exposure_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
    return max(0.0, min(1.0,
                        0.50 * sharp_score + 0.35 * contrast_score +
                        0.15 * exposure_score))


def _evidence_confidence(ocr: float, quality: float) -> float:
    return max(0.0, min(1.0, float(ocr) * (0.55 + 0.45 * quality)))


def _deduplicate_cards(cards: List[DetectedCard]) -> List[DetectedCard]:
    ordered = sorted(cards, key=lambda row: row.confidence, reverse=True)
    result: List[DetectedCard] = []
    for card in ordered:
        duplicate = next((row for row in result
                          if row.zone == card.zone and
                          _bbox_iou(row.bbox, card.bbox) >= 0.55), None)
        if duplicate is None:
            result.append(card)
            continue
        # Merge complementary evidence from local CV and VLM.
        if duplicate.name is None and card.name:
            duplicate.name = card.name
        if duplicate.tapped is None and card.tapped is not None:
            duplicate.tapped = card.tapped
        duplicate.confidence = max(duplicate.confidence, card.confidence)
        duplicate.metadata.setdefault("mergedDetectors", []).append(
            card.metadata.get("detector"))
    return result


def _bbox_iou(left: List[int], right: List[int]) -> float:
    if len(left) < 4 or len(right) < 4:
        return 0.0
    lx, ly, lw, lh = left[:4]
    rx, ry, rw, rh = right[:4]
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(1, lw * lh + rw * rh - intersection)
    return intersection / union
