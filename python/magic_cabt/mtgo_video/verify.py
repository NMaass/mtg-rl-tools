"""Verify a video-derived MTGO game against XMage's mirrored behavior.

The MTGO log gives us what *should* be true after each event. Feeding those
snapshots to XMage's real MirrorStateApplier + GameView tells us what XMage
actually renders. This module runs the headless MirrorVerify harness over a
bundle and diffs the two, state by state.

A game verifies when every state agrees on turn, both players' life,
library and hand counts, and the exact multiset of battlefield card names
(plus tapped state) and graveyard contents.
"""

import json
import os
import subprocess
from collections import Counter
from typing import Dict, List, Optional

from .catalog import fold as _fold

VERIFY_MAIN_CLASS = "mage.client.cabtmirror.MirrorVerify"


def fold(name: Optional[str]) -> str:
    """Compare card identities, not spellings.

    XMage's repository indexes accented cards without their diacritics
    ("Lorien Revealed") while Scryfall carries them ("Lorien Revealed" with
    an accent), so both sides are folded before comparison. A difference in
    orthography between the two databases is not a mirroring error.
    """
    return _fold(name or "")


CARD_DB = os.path.join("db", "cards.h2.mv.db")


def find_card_database(classpath: str) -> Optional[str]:
    """The directory XMage's card database sits in, if it is on the classpath.

    XMage opens its card database relative to the working directory, so a JVM
    started anywhere else builds every card as null -- which surfaces as an
    empty library rather than as an error. Finding it here means callers do
    not have to know that.
    """
    for entry in classpath.split(os.pathsep):
        directory = os.path.abspath(entry)
        if os.path.isfile(directory):
            directory = os.path.dirname(directory)
        for _ in range(4):
            if os.path.exists(os.path.join(directory, CARD_DB)):
                return directory
            parent = os.path.dirname(directory)
            if parent == directory:
                break
            directory = parent
    return None


def resolve_cwd(classpath: str, cwd: Optional[str]) -> Optional[str]:
    """Where to start the JVM: what the caller asked for, or where the DB is."""
    if cwd:
        return cwd
    return find_card_database(classpath)


def assert_card_database(states: List[dict], summaries: List[dict],
                         classpath: str, cwd: Optional[str]) -> None:
    """Fail loudly when XMage ran without its card database.

    Without it every card resolves to null, so libraries and hands come back
    empty and *every* comparison disagrees -- which reads like a broken
    decode rather than a missing file. Say which it is.
    """
    claimed = any((player.get("libraryCount") or 0) > 0
                  for state in states for player in state.get("players", []))
    reported = any((player.get("libraryCount") or 0) > 0
                   for summary in summaries
                   for player in summary.get("players", []))
    if claimed and not reported:
        raise RuntimeError(
            "XMage reported an empty library for every state, which means it "
            "could not open its card database (%s). Start the JVM in the "
            "directory that holds it: pass --cwd <Mage.Client dir>. Looked "
            "from classpath, ran in %r."
            % (CARD_DB, cwd or os.getcwd()))


def run_mirror_verify(
    states_path: str,
    classpath: str,
    java: str = "java",
    cwd: Optional[str] = None,
    heap: str = "-Xmx2g",
) -> List[dict]:
    """Apply every state in XMage and return one GameView summary per state."""
    cwd = resolve_cwd(classpath, cwd)
    proc = subprocess.run(
        [java, heap, "-Dfile.encoding=UTF-8", "-Djava.awt.headless=true",
         "-cp", classpath, VERIFY_MAIN_CLASS, states_path, "--all"],
        capture_output=True, text=True, cwd=cwd,
    )
    if proc.returncode != 0:
        raise RuntimeError("MirrorVerify failed (%d):\n%s"
                           % (proc.returncode, proc.stderr[-4000:]))
    summaries = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            summaries.append(json.loads(line))
    return summaries


def _expected(state: dict) -> dict:
    """What the MTGO log says the board is, in MirrorVerify's vocabulary."""
    zones = state.get("zones") or {}
    players = []
    for player in sorted(state.get("players", []), key=lambda p: p["seat"]):
        seat = str(player["seat"])
        battlefield = [
            (fold(o.get("name")), bool(o.get("tapped")))
            for o in zones.get("battlefield", [])
            if str(o.get("controllerSeat", o.get("ownerSeat"))) == seat
        ]
        graveyard = [fold(o.get("name"))
                     for o in (zones.get("graveyards") or {}).get(seat, [])]
        players.append({
            "name": player.get("name"),
            "life": player.get("life"),
            "libraryCount": player.get("libraryCount"),
            "handCount": player.get("handCount"),
            "battlefield": battlefield,
            "graveyard": graveyard,
        })
    return {"turn": state.get("turnNumber"), "players": players}


def _actual(summary: dict) -> dict:
    players = []
    for player in summary.get("players", []):
        players.append({
            "name": player.get("name"),
            "life": player.get("life"),
            "libraryCount": player.get("libraryCount"),
            "handCount": player.get("handCount"),
            "battlefield": [(fold(p.get("name")), bool(p.get("tapped")))
                            for p in player.get("battlefield", [])],
            "graveyard": [fold(n) for n in player.get("graveyard", [])],
        })
    return {"turn": summary.get("turn"), "players": players}


def compare_state(state: dict, summary: dict) -> List[str]:
    """Return a list of mismatch descriptions (empty means the state agrees)."""
    expected, actual = _expected(state), _actual(summary)
    problems = []
    if expected["turn"] != actual["turn"]:
        problems.append("turn: log=%s xmage=%s" % (expected["turn"], actual["turn"]))

    by_name = {p["name"]: p for p in actual["players"]}
    for want in expected["players"]:
        got = by_name.get(want["name"])
        if got is None:
            problems.append("player %r missing in XMage" % want["name"])
            continue
        for field in ("life", "libraryCount", "handCount"):
            if want[field] is not None and want[field] != got[field]:
                problems.append("%s %s: log=%s xmage=%s"
                                % (want["name"], field, want[field], got[field]))
        # XMage renders an unresolvable card name as a same-named token, so
        # comparing names (not identities) is the right granularity here.
        if Counter(want["battlefield"]) != Counter(got["battlefield"]):
            problems.append(
                "%s battlefield: log=%s xmage=%s"
                % (want["name"], sorted(want["battlefield"]), sorted(got["battlefield"]))
            )
        if Counter(want["graveyard"]) != Counter(got["graveyard"]):
            problems.append(
                "%s graveyard: log=%s xmage=%s"
                % (want["name"], sorted(want["graveyard"]), sorted(got["graveyard"]))
            )
    return problems


def verify_bundle(
    bundle_dir: str,
    classpath: str,
    java: str = "java",
    cwd: Optional[str] = None,
) -> Dict:
    """Verify every state of a bundle; returns a structured report."""
    states_path = os.path.join(bundle_dir, "mirror_states.jsonl")
    states = []
    with open(states_path) as f:
        for line in f:
            line = line.strip()
            if line:
                states.append(json.loads(line))

    summaries = run_mirror_verify(states_path, classpath, java=java, cwd=cwd)
    if len(summaries) != len(states):
        raise RuntimeError("XMage returned %d summaries for %d states"
                           % (len(summaries), len(states)))
    assert_card_database(states, summaries, classpath, cwd)

    mismatches = []
    for i, (state, summary) in enumerate(zip(states, summaries)):
        problems = compare_state(state, summary)
        if problems:
            source = (state.get("sourceEvent") or {}).get("text", "")
            mismatches.append({"stateIndex": i, "seq": state.get("seq"),
                               "event": source, "problems": problems})

    return {
        "bundle": os.path.abspath(bundle_dir),
        "states": len(states),
        "statesVerified": len(states) - len(mismatches),
        "mismatches": mismatches,
        "ok": not mismatches,
    }
