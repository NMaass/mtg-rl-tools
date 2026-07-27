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


def probe_duration(video_path: str, ffprobe: str = "ffprobe") -> Optional[float]:
    """The capture's length in seconds, or None if the container omits it."""
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", video_path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    text = proc.stdout.decode().strip().splitlines()
    try:
        return float(text[0])
    except (IndexError, ValueError):
        return None


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
    stack: int = 1,
) -> List[Tuple[float, str]]:
    """Extract cropped, upscaled, grayscale log-pane frames.

    Returns a list of (timestamp_seconds, png_path) where timestamp is
    relative to the start of the video. `scale` forces a fixed multiplier;
    by default the crop is resampled to `target_width` so OCR sees the same
    glyph size regardless of capture resolution.

    With `stack` above one, each returned frame is the average of that many
    consecutive source frames instead of a single one -- see
    `composite_frames` for why that is worth doing.
    """
    if stack > 1:
        return _extract_stacked(video_path, out_dir, start, end, fps, region,
                                scale, target_width, ffmpeg, stack)
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


# How different two frames of the same still text may be and still be
# averaged together, as mean absolute difference in grey levels. Compression
# noise on identical content sits well under this; a scrolled pane is a
# different picture entirely and sits far above it.
STILL_TOLERANCE = 6.0


def composite_frames(paths: List[str], out_path: str,
                     tolerance: float = STILL_TOLERANCE) -> int:
    """Average frames showing the same thing into one cleaner frame.

    The log pane holds still between scrolls, so several consecutive frames
    of it are the same picture plus independent compression noise -- and
    averaging N of them cuts that noise by about the square root of N. This
    is the one thing that adds information rather than merely reading the
    same information more carefully: it is why a capture whose text is at the
    edge of legibility can be pushed over the line.

    Only frames that agree with the newest one are averaged. A scroll
    partway through the group would otherwise blend two different states of
    the log into a ghosted frame holding lines from both, which reads as
    plausible text that was never on screen -- much worse than a noisy frame.

    Returns how many frames went into the composite.
    """
    from PIL import Image, ImageChops, ImageStat

    if not paths:
        raise ValueError("no frames to composite")
    newest = Image.open(paths[-1]).convert("L")
    agreeing = [newest]
    for path in reversed(paths[:-1]):
        candidate = Image.open(path).convert("L")
        if candidate.size != newest.size:
            break
        difference = ImageStat.Stat(
            ImageChops.difference(candidate, newest)).mean[0]
        if difference > tolerance:
            # The pane scrolled here; everything older belongs to a
            # different picture.
            break
        agreeing.append(candidate)

    if len(agreeing) == 1:
        newest.save(out_path)
        return 1
    accumulated = agreeing[0]
    for index, image in enumerate(agreeing[1:], start=2):
        accumulated = Image.blend(accumulated, image, 1.0 / index)
    accumulated.save(out_path)
    return len(agreeing)


def _extract_stacked(video_path, out_dir, start, end, fps, region, scale,
                     target_width, ffmpeg, stack):
    """Extract at `stack` times the rate, then average each group."""
    import shutil
    import tempfile

    out_dir = os.path.realpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    raw_dir = tempfile.mkdtemp(prefix="mtgo_stack_")
    try:
        raw = extract_log_frames(video_path, raw_dir, start=start, end=end,
                                 fps=fps * stack, region=region, scale=scale,
                                 target_width=target_width, ffmpeg=ffmpeg)
        frames = []
        for index in range(0, len(raw) - stack + 1, stack):
            group = raw[index:index + stack]
            out_path = os.path.join(out_dir, "f_%06d.png"
                                    % (index // stack + 1))
            composite_frames([path for _, path in group], out_path)
            # The composite shows the group's newest state, so it carries the
            # newest frame's timestamp.
            frames.append((group[-1][0], out_path))
        return frames
    finally:
        shutil.rmtree(raw_dir, ignore_errors=True)
