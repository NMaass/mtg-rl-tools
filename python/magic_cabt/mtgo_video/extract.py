"""Frame extraction from MTGO footage via ffmpeg."""

import os
import subprocess
from typing import List, Optional, Tuple

from .regions import Region, LOG_PANE_1080


def extract_log_frames(
    video_path: str,
    out_dir: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
    fps: float = 1.0,
    region: Region = LOG_PANE_1080,
    scale: int = 3,
    ffmpeg: str = "ffmpeg",
) -> List[Tuple[float, str]]:
    """Extract cropped, upscaled, grayscale log-pane frames.

    Returns a list of (timestamp_seconds, png_path) where timestamp is
    relative to the start of the video.
    """
    # Resolve symlinks up front: frame paths are handed to tesseract, and
    # some sandboxes refuse to traverse a symlinked prefix (e.g. macOS's
    # /tmp -> /private/tmp) in a nested subprocess.
    out_dir = os.path.realpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    vf = "fps=%g,%s,scale=%d:%d:flags=lanczos,format=gray" % (
        fps,
        region.ffmpeg_crop(),
        region.width * scale,
        region.height * scale,
    )
    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += ["-i", video_path, "-vf", vf, os.path.join(out_dir, "f_%06d.png")]
    subprocess.run(cmd, check=True)

    frames = []
    base = start or 0.0
    for name in sorted(os.listdir(out_dir)):
        if not (name.startswith("f_") and name.endswith(".png")):
            continue
        index = int(name[2:-4])
        # ffmpeg's fps filter emits frame k at source time ~ (k-1)/fps.
        timestamp = base + (index - 1) / fps
        frames.append((timestamp, os.path.join(out_dir, name)))
    return frames
