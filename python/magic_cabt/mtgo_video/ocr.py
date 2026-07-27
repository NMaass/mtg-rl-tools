"""OCR of MTGO log-pane frames via tesseract."""

import os
import re
import subprocess
from typing import List, Optional

# MTGO log entries start with a clock timestamp like "7:07 AM:". OCR mangles
# it freely ("/:06 AN:", "7:04 AWM:", "7:01 4M:"), so be tolerant: an
# hour-ish glyph, colon-ish separator, two digits, then a short AM/PM-ish
# token ending in M/N/I (e.g. AM, PM, AN, AWM, AMI, 4M, Ali).
# The clock's punctuation is the least reliable thing on screen: the colon
# has been seen as ".", ",", "-" and even as a digit ("7203 AM" for 7:03),
# and "AM" as "AN", "AWM", "Alvi", "4M". Only the shape -- one or two digits,
# a separator, two digits, an AM/PM-ish token, a colon -- is dependable, and
# that shape is distinctive enough not to match ordinary log text.
_TS = (r"[\dOlI|/t]{1,2}\s*[^\sA-Za-z]?\s*[\dO]{2}\s*"
       r"[AP4/][A-Za-z]{0,3}\s*[^\sA-Za-z0-9]")
TIMESTAMP_RE = re.compile(r"^\s*" + _TS + r"\s*")
INLINE_TIMESTAMP_RE = re.compile(r"\s+(?=" + _TS + r"\s)")

# Scrollbar/edge artifacts OCR'd as short junk tokens at line ends, e.g.
# " A", " VU", " Lv", " [a", " a]", " S|", " ry", " QQ", " z.".
_JUNK_TOKEN_RE = re.compile(r"\s+[\[\(\{]?[A-Za-z|/\\]{1,2}[\]\)\}|]?\.?$")


def ocr_image(
    png_path: str,
    tesseract: str = "tesseract",
    psm: int = 6,
    strict: bool = True,
    whitelist: Optional[str] = None,
) -> List[str]:
    """Run tesseract and return raw non-empty output lines.

    With strict=False a failing frame yields an empty list instead of
    raising: one unreadable frame out of thousands should not abort an
    ingest, and the scrolling-window reconstruction recovers its content
    from neighbouring frames. Failures are silent, so the caller is
    responsible for reporting how many frames came back empty.
    """
    png_path = os.path.realpath(png_path)
    env = dict(os.environ)
    # Tesseract's OpenMP pool fails intermittently when several instances run
    # concurrently, and single-threaded is faster per-page for small images.
    env.setdefault("OMP_THREAD_LIMIT", "1")
    command = [tesseract, png_path, "stdout", "--psm", str(psm)]
    if whitelist:
        command += ["-c", "tessedit_char_whitelist=" + whitelist]
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        # tesseract's diagnostics are not always valid UTF-8; they are noise
        # here and must not be allowed to fail the decode of a good frame.
        stderr=subprocess.PIPE,
        env=env,
    )
    if proc.returncode != 0:
        if strict:
            raise RuntimeError(
                "tesseract failed on %s (exit %d): %s"
                % (png_path, proc.returncode,
                   proc.stderr.decode("utf-8", errors="replace")[-500:])
            )
        return []
    text = proc.stdout.decode("utf-8", errors="replace")
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def strip_junk(text: str) -> str:
    """Strip trailing scrollbar-glyph junk tokens (repeatedly)."""
    prev = None
    while prev != text:
        prev = text
        stripped = _JUNK_TOKEN_RE.sub("", text.rstrip())
        if stripped and " " in stripped:
            text = stripped
    return text.strip()


def clean_line(line: str) -> str:
    line = line.replace("’", "'").replace("‘", "'")
    return line.strip()


def lines_to_entries(lines: List[str]) -> List[str]:
    """Assemble raw OCR lines into full log entries.

    A line starting with a clock timestamp begins a new entry; other lines
    continue the previous entry (the pane word-wraps long entries). OCR
    sometimes glues two physical lines together, so lines are also split on
    inline timestamps. Leading continuation lines with no parent entry are
    dropped: the pane scrolls upward, so their parent was fully visible in
    an earlier frame.
    """
    pieces: List[str] = []
    for raw in lines:
        line = clean_line(raw)
        if not line:
            continue
        pieces.extend(p for p in INLINE_TIMESTAMP_RE.split(line) if p.strip())

    entries: List[str] = []
    current: List[str] = []
    for line in pieces:
        if TIMESTAMP_RE.match(line):
            if current:
                entries.append(strip_junk(" ".join(current)))
            current = [line]
        elif current:
            current.append(line)
        # else: orphan continuation at the top of the pane -> drop.
    if current:
        entries.append(strip_junk(" ".join(current)))
    return [e for e in entries if e]


def strip_timestamp(entry: str) -> str:
    return TIMESTAMP_RE.sub("", entry).strip()
