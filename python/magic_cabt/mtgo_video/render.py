"""Render an MTGO video bundle as an XMage replay.

Drives the ArenaMirrorApp XMage window state-by-state, captures a screenshot
per state, and assembles them into a video. Because every state carries the
video timestamp of the log line that produced it, the XMage side can be cut
to the footage's own clock -- which is what makes a side-by-side comparison
meaningful rather than merely suggestive.
"""

import json
import os
import subprocess
from typing import List, Optional, Tuple

from magic_cabt.arena_mirror.mirror import MirrorDisplay


def load_states(bundle_dir: str) -> List[dict]:
    states = []
    with open(os.path.join(bundle_dir, "mirror_states.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                states.append(json.loads(line))
    return states


def capture_states(
    bundle_dir: str,
    frames_dir: Optional[str] = None,
    classpath: Optional[str] = None,
    java: str = "java",
    cwd: Optional[str] = None,
    caption: bool = True,
    reuse: bool = False,
) -> List[Tuple[str, dict]]:
    """Replay every state in XMage, returning (screenshot, state) pairs.

    With reuse=True, an existing complete set of screenshots is used as-is so
    a re-encode does not have to relaunch XMage.
    """
    states = load_states(bundle_dir)
    if not states:
        raise ValueError("bundle has no mirror states")
    frames_dir = os.path.realpath(frames_dir
                                  or os.path.join(bundle_dir, "replay_frames"))
    os.makedirs(frames_dir, exist_ok=True)

    if reuse:
        existing = [(os.path.join(frames_dir, "state_%05d.png" % i), state)
                    for i, state in enumerate(states)]
        if all(os.path.exists(shot) for shot, _ in existing):
            return existing

    first = states[0]
    players = [
        {"seat": p["seat"], "name": p.get("name") or "Seat %d" % p["seat"]}
        for p in first.get("players", [])
    ]

    display = MirrorDisplay(classpath=classpath, java=java, cwd=cwd)
    shots = []
    try:
        display.ping()
        display.start_game(
            players,
            local_seat=first.get("localSeat"),
            match_id=first.get("matchId"),
            game_number=first.get("gameNumber"),
        )
        for i, state in enumerate(states):
            display.send_state(state)
            source = state.get("sourceEvent") or {}
            if caption and source.get("text"):
                display.send_message(source["text"])
            shot = os.path.join(frames_dir, "state_%05d.png" % i)
            display.screenshot(shot)
            shots.append((shot, state))
    finally:
        display.close()
    return shots


def _write_concat(path: str, entries: List[Tuple[str, float]]):
    """ffmpeg concat demuxer script: (file, duration) pairs."""
    with open(path, "w") as f:
        for shot, duration in entries:
            f.write("file '%s'\n" % shot)
            f.write("duration %g\n" % max(duration, 0.04))
        # The concat demuxer ignores the last entry's duration unless the
        # final file is repeated.
        f.write("file '%s'\n" % entries[-1][0])


def render_replay(
    bundle_dir: str,
    out_video: str,
    classpath: Optional[str] = None,
    java: str = "java",
    cwd: Optional[str] = None,
    seconds_per_state: float = 1.0,
    frames_dir: Optional[str] = None,
    caption: bool = True,
    shots: Optional[List[Tuple[str, dict]]] = None,
) -> str:
    """Replay the bundle in XMage, screenshot each state, encode a video."""
    if shots is None:
        shots = capture_states(bundle_dir, frames_dir, classpath, java, cwd, caption)
    out_parent = os.path.dirname(os.path.abspath(out_video))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    concat_path = os.path.join(os.path.dirname(shots[0][0]), "concat.txt")
    _write_concat(concat_path, [(shot, seconds_per_state) for shot, _ in shots])
    _encode(concat_path, out_video)
    return out_video


def render_side_by_side(
    bundle_dir: str,
    source_video: str,
    out_video: str,
    classpath: Optional[str] = None,
    java: str = "java",
    cwd: Optional[str] = None,
    frames_dir: Optional[str] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    speed: float = 1.0,
    shots: Optional[List[Tuple[str, dict]]] = None,
) -> str:
    """Render the MTGO footage beside the XMage board it was decoded into.

    Each XMage screenshot is held until the video timestamp of the next
    state, so the two panes advance together: whenever a play appears in the
    MTGO log pane on the left, the XMage board on the right updates with it.
    """
    if shots is None:
        shots = capture_states(bundle_dir, frames_dir, classpath, java, cwd)

    timed = [(shot, state.get("videoTime")) for shot, state in shots]
    if any(t is None for _, t in timed):
        raise ValueError(
            "bundle has states without videoTime; re-ingest so log entries "
            "carry their frame timestamps"
        )

    clip_start = start if start is not None else timed[0][1]
    clip_end = end if end is not None else timed[-1][1] + 3.0

    durations = []
    for i, (shot, when) in enumerate(timed):
        nxt = timed[i + 1][1] if i + 1 < len(timed) else clip_end
        durations.append((shot, (nxt - when) / speed))
    # The first screenshot also covers the run-up before the first log line.
    durations[0] = (durations[0][0],
                    durations[0][1] + (timed[0][1] - clip_start) / speed)

    frames_root = os.path.dirname(shots[0][0])
    concat_path = os.path.join(frames_root, "concat_timed.txt")
    _write_concat(concat_path, durations)

    xmage_only = os.path.join(frames_root, "xmage_timed.mp4")
    _encode(concat_path, xmage_only)

    out_parent = os.path.dirname(os.path.abspath(out_video))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    pane_w, pane_h = 960, 600
    fit = ("scale=%d:%d:force_original_aspect_ratio=decrease,"
           "pad=%d:%d:(ow-iw)/2:(oh-ih)/2:black" % (pane_w, pane_h, pane_w, pane_h))
    banner = _banner_image(frames_root, pane_w * 2,
                           ["MTGO footage (source)", "XMage mirror (decoded)"])
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", str(clip_start), "-to", str(clip_end), "-i", source_video,
         "-i", xmage_only, "-i", banner,
         "-filter_complex",
         "[0:v]setpts=PTS/%g,%s[l];[1:v]%s[r];"
         "[l][r]hstack=inputs=2[panes];"
         "[2:v][panes]vstack=inputs=2,format=yuv420p[v]" % (speed, fit, fit),
         "-map", "[v]", "-r", "30", "-shortest", out_video],
        check=True,
    )
    return out_video


def _banner_image(out_dir: str, width: int, labels: List[str],
                  height: int = 48) -> str:
    """A two-column caption strip stacked above the panes."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), (16, 16, 20))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    column = width // max(1, len(labels))
    for i, label in enumerate(labels):
        box = draw.textbbox((0, 0), label, font=font)
        x = i * column + (column - (box[2] - box[0])) // 2
        draw.text((x, (height - (box[3] - box[1])) // 2 - box[1]), label,
                  fill=(235, 235, 235), font=font)
        if i:
            draw.line([(i * column, 0), (i * column, height)], fill=(80, 80, 90))
    path = os.path.join(out_dir, "banner.png")
    image.save(path)
    return path


def _encode(concat_path: str, out_video: str):
    subprocess.run(
        # -r resamples the concat demuxer's per-image durations to a constant
        # 30fps; do not add -vsync/-fps_mode, which contradicts it.
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", concat_path,
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
         "-r", "30", out_video],
        check=True,
    )
