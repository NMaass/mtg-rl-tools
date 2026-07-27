"""Frame extraction from MTGO footage via ffmpeg."""

import os
import subprocess
from typing import List, Optional, Tuple

from .regions import Region, LOG_PANE_1080

# The log pane is upscaled to this width before OCR, whatever the capture
# resolution. Tesseract is sensitive to glyph size, so normalizing here is
# what lets the same game decode identically from a 720p and a 1440p
# recording instead of drifting with the source.
OCR_TARGET_WIDTH = 1035


def probe_frame_size(video_path: str, ffprobe: str = "ffprobe") -> Tuple[int, int]:
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
         video_path],
        stdout=subprocess.PIPE, check=True,
    )
    width, _, height = proc.stdout.decode().strip().split("\n")[0].partition("x")
    return int(width), int(height)


def grab_frame(video_path: str, timestamp: float, out_path: str,
               ffmpeg: str = "ffmpeg") -> str:
    """Extract a single full frame, for layout detection."""
    out_path = os.path.realpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-ss", str(timestamp),
         "-i", video_path, "-frames:v", "1", out_path],
        check=True,
    )
    return out_path


def extract_log_frames(
    video_path: str,
    out_dir: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
    fps: float = 1.0,
    region: Region = LOG_PANE_1080,
    scale: Optional[int] = None,
    target_width: int = OCR_TARGET_WIDTH,
    ffmpeg: str = "ffmpeg",
) -> List[Tuple[float, str]]:
    """Extract cropped, upscaled, grayscale log-pane frames.

    Returns a list of (timestamp_seconds, png_path) where timestamp is
    relative to the start of the video. `scale` forces a fixed multiplier;
    by default the crop is resampled to `target_width` so OCR sees the same
    glyph size regardless of capture resolution.
    """
    # Resolve symlinks up front: frame paths are handed to tesseract, and
    # some sandboxes refuse to traverse a symlinked prefix (e.g. macOS's
    # /tmp -> /private/tmp) in a nested subprocess.
    out_dir = os.path.realpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    factor = scale if scale else max(1.0, target_width / max(1, region.width))
    vf = "fps=%g,%s,scale=%d:%d:flags=lanczos,format=gray" % (
        fps,
        region.ffmpeg_crop(),
        int(round(region.width * factor)),
        int(round(region.height * factor)),
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
