"""Headless adaptive video sampling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import cv2
import numpy as np


@dataclass
class FrameSample:
    frame_index: int
    timestamp_ms: int
    image: np.ndarray
    change_score: float


class FrameSampler:
    """Sample fixed-FPS frames, retaining scene changes and periodic anchors."""

    def __init__(self, video_path: str, fps: float = 2.0,
                 change_threshold: float = 0.018,
                 max_interval_seconds: float = 3.0,
                 start_seconds: float = 0.0,
                 end_seconds: Optional[float] = None):
        self.video_path = str(video_path)
        self.fps = max(0.05, float(fps))
        self.change_threshold = max(0.0, float(change_threshold))
        self.max_interval_ms = max(100, int(max_interval_seconds * 1000))
        self.start_seconds = max(0.0, float(start_seconds))
        self.end_seconds = None if end_seconds is None else float(end_seconds)

    def __iter__(self) -> Iterator[FrameSample]:
        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            raise IOError(f"unable to open video: {self.video_path}")
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start_frame = int(self.start_seconds * source_fps)
        end_frame = total_frames if self.end_seconds is None \
            else min(total_frames, int(self.end_seconds * source_fps))
        stride = max(1, int(round(source_fps / self.fps)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        previous_signature = None
        last_emitted_ms = -10**9
        frame_index = start_frame
        try:
            while frame_index < end_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                if (frame_index - start_frame) % stride != 0:
                    frame_index += 1
                    continue
                timestamp_ms = int(round(frame_index / source_fps * 1000))
                signature = self._signature(frame)
                score = 1.0 if previous_signature is None else float(
                    np.mean(np.abs(signature.astype(np.float32) -
                                   previous_signature.astype(np.float32))) / 255.0)
                periodic = timestamp_ms - last_emitted_ms >= self.max_interval_ms
                if previous_signature is None or score >= self.change_threshold or periodic:
                    yield FrameSample(frame_index, timestamp_ms, frame, score)
                    previous_signature = signature
                    last_emitted_ms = timestamp_ms
                frame_index += 1
        finally:
            capture.release()

    @staticmethod
    def _signature(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)


def write_frame(path: str, image: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise IOError(f"failed to write frame: {path}")
