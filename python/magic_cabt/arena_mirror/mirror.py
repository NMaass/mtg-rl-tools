"""Client for the Java XMage mirror display.

Spawns ``mage.player.cabt.mirror.ArenaMirrorApp`` (an XMage client window
plus a puppet game) and speaks newline-delimited JSON over stdin/stdout,
protocol-server style: one command line in, one ``{"ok": ...}`` line out.

``enrich_snapshot`` resolves Arena grpIds to card names so the Java side can
look real cards up in XMage's card repository; anything unresolved renders
face-down.
"""

import json
import os
import subprocess

__all__ = ["MirrorDisplay", "MirrorDisplayError", "enrich_snapshot"]

APP_MAIN_CLASS = "mage.client.cabtmirror.ArenaMirrorApp"
CLASSPATH_ENV_VAR = "MAGIC_CABT_CLASSPATH"


class MirrorDisplayError(RuntimeError):
    pass


def enrich_snapshot(snapshot, card_db):
    """Resolve card names/types into a snapshot (in place; returns it)."""
    if card_db is None:
        return snapshot

    def enrich_object(obj):
        # never resolve a name for a redacted/face-down object: the tracker
        # strips grpId from hidden cards, and enrichment must not reintroduce
        # any identity for them
        if obj.get("faceDown") or not obj.get("grpId"):
            obj["faceDown"] = True
            return
        info = card_db.lookup(obj.get("grpId"))
        if info is not None:
            obj["name"] = info.name
            obj["cardTypeNames"] = info.types
            obj["subtypeNames"] = info.subtypes
            obj["isToken"] = info.is_token
            if info.power is not None and obj.get("power") is None:
                obj["power"] = _to_int(info.power)
            if info.toughness is not None and obj.get("toughness") is None:
                obj["toughness"] = _to_int(info.toughness)
        else:
            obj["faceDown"] = True
        source_grp = obj.get("objectSourceGrpId")
        if source_grp:
            source_info = card_db.lookup(source_grp)
            if source_info is not None:
                obj["sourceName"] = source_info.name

    zones = snapshot.get("zones") or {}
    for objects in (zones.get("battlefield"), zones.get("stack"),
                    zones.get("exile"), zones.get("command")):
        for obj in objects or []:
            enrich_object(obj)
    for seat_objects in (zones.get("hands") or {}).values():
        for obj in seat_objects:
            enrich_object(obj)
    for seat_objects in (zones.get("graveyards") or {}).values():
        for obj in seat_objects:
            enrich_object(obj)
    return snapshot


class MirrorDisplay(object):
    """One ArenaMirrorApp subprocess and the command loop over it."""

    def __init__(self, command=None, classpath=None, java="java", cwd=None):
        if command is None:
            classpath = classpath or os.environ.get(CLASSPATH_ENV_VAR)
            if not classpath:
                raise ValueError(
                    "no way to launch the display: pass command=[...] or "
                    "classpath=..., or set $" + CLASSPATH_ENV_VAR)
            command = [java, "-Xmx1g", "-cp", classpath, APP_MAIN_CLASS]
        self._process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr passes through so XMage client logging stays visible
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def alive(self):
        return self._process.poll() is None

    def close(self):
        if self._process.poll() is None:
            try:
                self.request({"command": "mirror_finish"})
            except (MirrorDisplayError, IOError):
                pass
            try:
                self._process.stdin.close()
            except IOError:
                pass
            try:
                self._process.wait(timeout=15)
            except Exception:
                self._process.kill()
        if self._process.stdout is not None:
            self._process.stdout.close()

    def request(self, request_dict):
        line = json.dumps(request_dict, separators=(",", ":"))
        if "\n" in line:
            raise ValueError("request must serialize to a single line")
        try:
            self._process.stdin.write(line + "\n")
            self._process.stdin.flush()
            response_line = self._process.stdout.readline()
        except OSError as error:
            raise IOError("display process is gone: %s" % (error,))
        if not response_line:
            raise IOError("display process closed its output (exit code %s)"
                          % self._process.poll())
        response = json.loads(response_line)
        if response.get("ok") is not True:
            raise MirrorDisplayError("%s: %s" % (
                response.get("error", "UNKNOWN"), response.get("message", "")))
        return response

    # --- commands ---

    def ping(self):
        return self.request({"command": "ping"})

    def start_game(self, players, local_seat=None, match_id=None,
                   game_number=None):
        """Open (or reset) the board for a new Arena game."""
        return self.request({
            "command": "mirror_start",
            "players": players,
            "localSeat": local_seat,
            "matchId": match_id,
            "gameNumber": game_number,
        })

    def send_state(self, snapshot):
        return self.request({"command": "mirror_state", "state": snapshot})

    def send_message(self, text):
        """Append a line to the display's game log panel."""
        return self.request({"command": "mirror_message", "text": text})

    def finish_game(self, result_text=None):
        return self.request({"command": "mirror_game_over",
                             "result": result_text})

    def screenshot(self, path):
        """Render the mirror window to a PNG file; returns its path."""
        return self.request({"command": "mirror_screenshot",
                             "path": path})["path"]


def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
