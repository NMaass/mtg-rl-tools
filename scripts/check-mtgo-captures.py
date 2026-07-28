#!/usr/bin/env python3
"""Check the layout detector against the real recordings it was built on.

The unit tests draw their own MTGO client, which is what makes them fast and
deterministic -- and also what makes them unable to catch the thing that
actually goes wrong: real footage is compressed, overlaid, differently
skinned, and drawn at whatever size the streamer chose. Every claim in
docs/MTGO_VIDEO.md about real captures was checked by hand once. This makes
that repeatable.

The videos are other people's and are not in the repository. The manifest
(python/tests/captures.json) records a YouTube id and a timestamp for each,
which is enough to fetch the same frame again:

    scripts/check-mtgo-captures.py                # fetch what is missing, check all
    scripts/check-mtgo-captures.py --offline      # only what is already cached
    scripts/check-mtgo-captures.py --cache DIR    # keep downloads somewhere else

Needs yt-dlp, ffmpeg and tesseract on PATH.
"""

import argparse
import difflib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "..", "python", "tests", "captures.json")
DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "magic_cabt",
                             "captures")
WINDOW = 6.0   # seconds either side of the wanted frame to fetch


def fetch(capture, cache, offline):
    """The clip for one capture, downloading it if it is not cached yet."""
    path = os.path.join(cache, "%s.mp4" % capture["id"])
    if os.path.exists(path):
        return path
    if offline:
        return None
    os.makedirs(cache, exist_ok=True)
    url = "https://www.youtube.com/watch?v=" + capture["id"]
    command = ["yt-dlp", "-q", "--no-warnings",
               "-f", capture.get("format", "bv*[height<=1080][ext=mp4]"),
               "-o", path]
    if not capture.get("whole"):
        at = capture["at"]
        command += ["--download-sections",
                    "*%d-%d" % (max(0, at - WINDOW), at + WINDOW),
                    "--force-keyframes-at-cuts"]
    command.append(url)
    if subprocess.run(command).returncode != 0 or not os.path.exists(path):
        return None
    return path


SAMPLES = 5


def frames_of(capture, video, work):
    """Several frames around the moment the manifest points at.

    The pipeline never decides a layout from one frame -- the log pane's text
    column is only as wide as that frame's longest line, and a life numeral
    can be behind a card being dragged over it -- so checking it against one
    frame would be checking something it does not do.
    """
    base = capture["at"] if capture.get("whole") else WINDOW
    paths = []
    for index in range(SAMPLES):
        at = max(0.0, base - WINDOW / 2 + index * (WINDOW / SAMPLES))
        out = os.path.join(work, "%s_%d.png" % (capture["id"], index))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(at),
                        "-i", video, "-frames:v", "1", out], check=True)
        if os.path.exists(out):
            paths.append(out)
    return paths


def close_enough(seen, wanted, ratio=0.8):
    """Names are OCR'd; hold them to a likeness rather than to the letter."""
    return difflib.SequenceMatcher(None, (seen or "").lower(),
                                   wanted.lower()).ratio() >= ratio


def check(capture, frames):
    """Every way these frames disagree with what the manifest expects."""
    from magic_cabt.mtgo_video.layout import (detect_layout_from_frames,
                                              validate_layout)

    layout = detect_layout_from_frames(frames)
    # Same rule the ingest uses: validate against every sample and take the
    # one that read the seats, since a single frame can catch a numeral
    # mid-repaint or behind a card.
    results = [validate_layout(layout, frame) for frame in frames]
    result = next((r for r in results if r["ok"] and r.get("seatsReadable")),
                  next((r for r in results if r["ok"]), results[-1]))
    expect = capture["expect"]
    problems = []

    if "logDetected" in expect and layout.detected != expect["logDetected"]:
        problems.append("log pane %s, expected %s"
                        % ("detected" if layout.detected else "not detected",
                           "detected" if expect["logDetected"] else "not"))
    if ("seatsDetected" in expect
            and layout.seats_detected != expect["seatsDetected"]):
        problems.append("seats %s, expected %s"
                        % ("detected" if layout.seats_detected else "not detected",
                           "detected" if expect["seatsDetected"] else "not"))
    for slot, wanted in (expect.get("life") or {}).items():
        if result["life"].get(slot) != wanted:
            problems.append("%s life read %s, expected %s"
                            % (slot, result["life"].get(slot), wanted))
    if expect.get("lifeReadable"):
        # Mid-game the number itself is a property of the moment, not of the
        # detector. What the detector owes is a box both seats can be read
        # out of at all.
        for slot in ("top", "bottom"):
            value = result["life"].get(slot)
            if value is None or not 0 <= value <= 40:
                problems.append("%s life read %r, expected a life total"
                                % (slot, value))
    for slot, wanted in (expect.get("names") or {}).items():
        if not close_enough(result["names"].get(slot), wanted):
            problems.append("%s name read %r, expected %r"
                            % (slot, result["names"].get(slot), wanted))
    refused = expect.get("refused")
    if refused:
        # The capture is supposed to be turned away, and for the stated
        # reason: refusing it for some other reason is not the same result.
        stated = "; ".join(result["problems"]) + "; ".join(result["warnings"])
        if refused not in stated:
            problems.append("expected to be refused for %r, got %r"
                            % (refused, result["problems"] or "no problems"))
    elif result["problems"]:
        problems.append("unexpectedly refused: %s" % "; ".join(result["problems"]))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--offline", action="store_true",
                        help="check only the captures already downloaded")
    parser.add_argument("--only", nargs="*", default=None,
                        help="check only these capture ids")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(HERE, "..", "python"))
    with open(MANIFEST) as handle:
        manifest = json.load(handle)

    import tempfile

    work = tempfile.mkdtemp(prefix="mtgo_captures_")
    failures = skipped = 0
    try:
        for capture in manifest["captures"]:
            if args.only and capture["id"] not in args.only:
                continue
            video = fetch(capture, args.cache, args.offline)
            if video is None:
                print("SKIP %-14s not downloaded" % capture["id"])
                skipped += 1
                continue
            problems = check(capture, frames_of(capture, video, work))
            if problems:
                failures += 1
                print("FAIL %-14s %s" % (capture["id"], capture["arrangement"]))
                for problem in problems:
                    print("       %s" % problem)
            else:
                print("ok   %-14s %s" % (capture["id"], capture["arrangement"]))
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)

    print("\n%d checked, %d failed, %d skipped"
          % (len(manifest["captures"]) - skipped, failures, skipped))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
