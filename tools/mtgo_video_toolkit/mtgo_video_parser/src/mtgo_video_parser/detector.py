"""Classical card-rectangle detection for configured MTGO zones."""
from __future__ import annotations

from typing import List
import cv2
import numpy as np

from .types import DetectedCard


class CardRectangleDetector:
    def __init__(self, min_area_ratio: float = 0.002,
                 max_area_ratio: float = 0.15,
                 aspect_min: float = 0.48,
                 aspect_max: float = 1.65):
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.aspect_min = float(aspect_min)
        self.aspect_max = float(aspect_max)

    def detect(self, image: np.ndarray, zone: str,
               controller: str | None = None) -> List[DetectedCard]:
        if image.size == 0:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 45, 130)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(image.shape[0] * image.shape[1])
        boxes = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area_ratio = width * height / max(1.0, frame_area)
            if not self.min_area_ratio <= area_ratio <= self.max_area_ratio:
                continue
            ratio = width / max(1.0, height)
            tapped = ratio > 1.05
            normalized_ratio = 1.0 / ratio if tapped else ratio
            if not self.aspect_min <= normalized_ratio <= self.aspect_max:
                continue
            rectangularity = cv2.contourArea(contour) / max(1.0, width * height)
            if rectangularity < 0.45:
                continue
            boxes.append((x, y, width, height, rectangularity, tapped))
        boxes = _non_maximum_suppression(boxes, 0.5)
        return [DetectedCard(
            bbox=[x, y, width, height], zone=zone, controller=controller,
            tapped=tapped, confidence=min(0.98, max(0.25, score)),
            metadata={"detector": "contour-rectangle"})
            for x, y, width, height, score, tapped in boxes]


def _non_maximum_suppression(boxes, threshold):
    ordered = sorted(boxes, key=lambda row: row[4], reverse=True)
    kept = []
    for candidate in ordered:
        if all(_iou(candidate, row) < threshold for row in kept):
            kept.append(candidate)
    return kept


def _iou(left, right):
    lx, ly, lw, lh = left[:4]
    rx, ry, rw, rh = right[:4]
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union else 0.0
