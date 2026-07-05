"""Live tailing of MTG Arena's Player.log.

Arena keeps ``Player.log`` open for writing and truncates it on client
restart (the previous session moves to ``Player-prev.log``). The follower
polls for appended bytes, detects truncation, and assembles the byte stream
into complete log entries — the same entry shape ``magic_cabt.arena_log``
parses in batch mode.
"""

import io
import os
import time

from ..arena_log import LOG_START_RE, TIMESTAMP_RE, _format_timestamp

__all__ = ["LogFollower", "EntryAssembler"]


class EntryAssembler(object):
    """Turns a stream of log lines into complete Arena log entries.

    An entry starts at a ``[UnityCrossThreadLogger]``/``[Client GRE]`` header
    and includes every following line until the next header. Because a live
    tail never sees "the next header" for the final entry, ``flush_pending``
    closes out the buffered entry when the stream goes idle.
    """

    def __init__(self):
        self._current = None

    def feed_line(self, line):
        """Feed one line (with newline); returns a completed entry or None."""
        completed = None
        start = LOG_START_RE.match(line)
        if start:
            completed = self._current
            raw_time = (start.group("time") or "").strip()
            self._current = {
                "channel": start.group("channel"),
                "rawTime": raw_time or None,
                "timestamp": _format_timestamp(raw_time),
                "text": line[start.end():],
            }
        elif self._current is None:
            timestamp_match = TIMESTAMP_RE.match(line)
            raw_time = timestamp_match.group("time") if timestamp_match else None
            self._current = {
                "channel": None,
                "rawTime": raw_time,
                "timestamp": _format_timestamp(raw_time),
                "text": line,
            }
        else:
            self._current["text"] += line
        return completed

    def flush_pending(self):
        """Return and clear the buffered partial entry (idle stream)."""
        entry = self._current
        self._current = None
        return entry

    def reset(self):
        self._current = None


class LogFollower(object):
    """Polls a Player.log file and yields complete entries as they land.

    Usage::

        follower = LogFollower(path)
        for entry in follower.follow():   # blocks; stop via stop() or
            handle(entry)                 # KeyboardInterrupt

    ``from_start=True`` replays existing content before going live (useful
    when Arena is already mid-session); the default starts at the file's
    current end so only new gameplay is mirrored.
    """

    # Seconds of stream silence after which a buffered partial entry is
    # considered complete. Arena writes each entry in one burst, so anything
    # still buffered after a quiet poll is a finished entry.
    IDLE_FLUSH_SECONDS = 0.5

    def __init__(self, path, poll_seconds=0.25, from_start=False):
        self.path = path
        self.poll_seconds = poll_seconds
        self.from_start = from_start
        self._stop = False
        self.rotations = 0

    def stop(self):
        self._stop = True

    def follow(self):
        """Generator of complete log entries; blocks between polls."""
        assembler = EntryAssembler()
        position = 0
        buffer = b""
        last_data_time = time.monotonic()
        waiting_for_file_logged = False

        if not self.from_start:
            try:
                position = os.path.getsize(self.path)
            except OSError:
                position = 0

        while not self._stop:
            try:
                size = os.path.getsize(self.path)
            except OSError:
                # File missing (client not started yet, or mid-rotation).
                if not waiting_for_file_logged:
                    waiting_for_file_logged = True
                position = 0
                buffer = b""
                assembler.reset()
                time.sleep(self.poll_seconds)
                continue
            waiting_for_file_logged = False

            if size < position:
                # Truncated: Arena restarted and started a fresh log.
                self.rotations += 1
                position = 0
                buffer = b""
                assembler.reset()

            if size > position:
                chunk = self._read_chunk(position, size - position)
                if chunk is None:
                    time.sleep(self.poll_seconds)
                    continue
                position += len(chunk)
                buffer += chunk
                lines, buffer = _split_lines(buffer)
                for line in lines:
                    entry = assembler.feed_line(line)
                    if entry is not None:
                        yield entry
                last_data_time = time.monotonic()
            else:
                idle = time.monotonic() - last_data_time
                if idle >= self.IDLE_FLUSH_SECONDS:
                    if buffer:
                        # Idle stream: treat the unterminated tail as a line.
                        entry = assembler.feed_line(_decode(buffer))
                        buffer = b""
                        if entry is not None:
                            yield entry
                    entry = assembler.flush_pending()
                    if entry is not None:
                        yield entry
                time.sleep(self.poll_seconds)

        # Drain on stop.
        if buffer:
            entry = assembler.feed_line(_decode(buffer))
            if entry is not None:
                yield entry
        entry = assembler.flush_pending()
        if entry is not None:
            yield entry

    def _read_chunk(self, position, length):
        try:
            with io.open(self.path, "rb") as handle:
                handle.seek(position)
                return handle.read(length)
        except OSError:
            return None


def _split_lines(buffer):
    """Split a byte buffer into complete decoded lines + leftover bytes."""
    if b"\n" not in buffer:
        return [], buffer
    head, _, rest = buffer.rpartition(b"\n")
    lines = [_decode(line) + "\n" for line in head.split(b"\n")]
    return lines, rest


def _decode(raw):
    return raw.decode("utf-8", errors="replace").replace("\r", "")
