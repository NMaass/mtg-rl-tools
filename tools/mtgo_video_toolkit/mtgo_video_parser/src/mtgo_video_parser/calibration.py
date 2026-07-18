"""Create visual layout-calibration artifacts from a representative frame."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import json

import cv2

from .layout import LayoutProfile


def read_frame(video_path: str, timestamp_seconds: float = 0.0):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError(f"unable to open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_seconds) * 1000.0)
        ok, frame = capture.read()
        if not ok:
            raise IOError(
                f"unable to read frame at {timestamp_seconds:.3f}s from {video_path}")
        return frame
    finally:
        capture.release()


def write_calibration(video_path: str, profile: LayoutProfile,
                      output_image: str, timestamp_seconds: float = 0.0,
                      crops_dir: Optional[str] = None) -> Dict[str, object]:
    frame = read_frame(video_path, timestamp_seconds)
    overlay = frame.copy()
    crops = Path(crops_dir) if crops_dir else None
    if crops:
        crops.mkdir(parents=True, exist_ok=True)
    regions = []
    for index, region in enumerate(profile.regions):
        left, top, right, bottom = region.pixel_box(frame.shape)
        # Fixed deterministic colors make overlays easy to compare without
        # requiring a plotting stack.
        color = ((71 * (index + 1)) % 255,
                 (137 * (index + 1)) % 255,
                 (211 * (index + 1)) % 255)
        cv2.rectangle(overlay, (left, top), (right, bottom), color, 2)
        cv2.putText(overlay, f"{region.name} [{region.kind}]",
                    (left + 3, max(15, top + 16)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.47, color, 1, cv2.LINE_AA)
        if crops:
            cv2.imwrite(str(crops / f"{index:02d}-{region.name}.png"),
                        frame[top:bottom, left:right])
        regions.append({"name": region.name, "kind": region.kind,
                        "pixelBox": [left, top, right, bottom]})
    target = Path(output_image)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), overlay):
        raise IOError(f"unable to write calibration overlay: {target}")
    result = {
        "video": str(Path(video_path).resolve()),
        "timestampSeconds": timestamp_seconds,
        "frameShape": list(frame.shape),
        "profile": profile.name,
        "overlay": str(target.resolve()),
        "regions": regions,
    }
    with target.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result
