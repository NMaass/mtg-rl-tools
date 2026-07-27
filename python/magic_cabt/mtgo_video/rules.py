"""Check each decoded event against the board XMage holds when it happens.

The other verifications ask whether the decoded game *renders* consistently.
This one asks whether it *makes sense*: MTGO's log is a stream of assertions
about a game -- "MarshFlats sacrifices Chromatic Star", "Mafuhsa is being
attacked by Faerie Seer" -- and every one of them presupposes something about
the board. A creature cannot be destroyed unless it is on the battlefield. A
player cannot discard from an empty hand. A card cannot attack from a
graveyard.

Those presuppositions are what makes this check independent of the others.
The pipeline derives the rules of the game by watching a log describe it, so
the decoded stream can be internally consistent and still be wrong: a missed
"casts X" line leaves the later "X is destroyed" with nothing to destroy, and
neither the XMage render check (XMage renders whatever the state claimed) nor
the HUD life check (a destroyed creature does not change anyone's life) can
see that. Feeding the states through XMage and asking, one step ahead of each
event, whether the board it is about to describe can support it, does.

What it is not: XMage is not being asked to *play* the game here, so this
does not prove the original match was legal. It proves the decoded event
stream is coherent against the board the mirror actually built -- which is
where OCR and parse damage shows up.
"""

import json
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence

from .verify import assert_card_database, fold, run_mirror_verify

# Events whose presuppositions this module knows how to state. Anything not
# listed is reported as unchecked rather than silently passed, so the check's
# coverage is visible instead of assumed.
CHECKED_TYPES = {
    "DESTROYED", "DIES", "SACRIFICE", "EXILE_CARD", "RETURN_HAND",
    "TRANSFORM", "BLOCK", "ATTACKED_BY", "ACTIVATE", "DISCARD", "DRAW",
    "DRAW_WITH", "MILL", "TURN", "CAST", "PLAY_LAND", "GAIN_LIFE",
    "LOSE_LIFE", "LIFE_TOTAL",
}

# How alike two names have to be to be the same person. Player names are
# OCR'd from the log, so an exact match is too much to ask; but a name that
# is only *nearly* a seat's is itself evidence of a misread, and is reported
# as one rather than quietly accepted.
_SAME_PLAYER = 0.8


def _similar(a: str, b: str) -> float:
    import difflib

    return difflib.SequenceMatcher(None, (a or "").lower(),
                                   (b or "").lower()).ratio()


# Abilities activated from a zone the board view cannot see -- almost always
# the hand, whose contents MTGO never logs. An ability with one of these
# names says nothing about where its card is, so "the source is nowhere" is
# not evidence of a decode error.
_HAND_ABILITIES = ("ninjutsu", "cycling", "channel", "forecast", "transmute",
                   "evoke", "foretell")


def _activated_from_hand(event: Dict) -> bool:
    text = (event.get("text") or "").lower()
    return any(keyword in text for keyword in _HAND_ABILITIES)


class _Board:
    """XMage's view of one state, in the terms the log talks about."""

    def __init__(self, summary: dict):
        self.summary = summary or {}
        self.turn = self.summary.get("turn")
        self.players = {}
        for player in self.summary.get("players", []):
            name = player.get("name") or ""
            self.players[name.lower()] = player

    def player(self, name: Optional[str]) -> Optional[dict]:
        if not name:
            return None
        return self.players.get(name.lower())

    def closest_player(self, name: Optional[str]):
        """The seat a name most resembles, and how much: (name, ratio)."""
        best, score = None, 0.0
        for seat in self.players:
            ratio = _similar(name or "", seat)
            if ratio > score:
                best, score = seat, ratio
        return best, score

    def battlefield(self, controller: Optional[str] = None) -> Counter:
        cards: Counter = Counter()
        for name, player in self.players.items():
            if controller is not None and name != controller.lower():
                continue
            for card in player.get("battlefield", []):
                cards[fold(card.get("name"))] += 1
        return cards

    def graveyard(self, controller: Optional[str] = None) -> Counter:
        cards: Counter = Counter()
        for name, player in self.players.items():
            if controller is not None and name != controller.lower():
                continue
            for card in player.get("graveyard", []):
                cards[fold(card)] += 1
        return cards

    def opponents_of(self, name: Optional[str]) -> List[str]:
        return [other for other in self.players if other != (name or "").lower()]


def _violation(kind: str, detail: str) -> Dict:
    return {"rule": kind, "detail": detail}


# Rules that assume the board holds everything the game has ever put there.
# In a clip that starts mid-game it does not: the permanents cast before the
# recording began are not in it, and neither are the cards drawn from the
# library or the turns already taken. Findings from these rules are reported
# separately when the game's opening was never captured -- they are missing
# history, not a contradiction.
_NEEDS_GAME_START = {
    "missing-permanent", "empty-battlefield", "wrong-controller",
    "missing-source", "empty-hand", "empty-library", "turn-skipped",
    "turn-went-backwards",
}

# Events that only occur at a game's start. If none was captured, the clip
# joined the game in progress.
_OPENING_EVENTS = {"JOIN", "ROLL", "OPENING_HAND", "MULLIGAN", "BOTTOM_BEGIN",
                   "PLAY_FIRST"}


def _needs_battlefield(board: _Board, card: str, controller: Optional[str],
                       verb: str) -> Optional[Dict]:
    present = board.battlefield(controller)
    if present.get(fold(card)):
        return None
    where = "%s's battlefield" % controller if controller else "the battlefield"
    also = board.battlefield()
    if not controller and not also:
        return _violation("empty-battlefield",
                          "%s %s, but nothing is on the battlefield" % (card, verb))
    if controller and also.get(fold(card)):
        return _violation("wrong-controller",
                          "%s %s under %s, but XMage has it under the other seat"
                          % (card, verb, controller))
    return _violation("missing-permanent",
                      "%s %s, but it is not on %s" % (card, verb, where))


def check_event(event: dict, board: _Board) -> List[Dict]:
    """Every way this event contradicts the board XMage holds before it."""
    kind = event.get("type")
    problems: List[Dict] = []
    player = event.get("player")

    # A card is never a player. MTGO does print a player's name where a
    # card's belongs -- so this is not always a misread -- but either way the
    # decoded event names no card, and saying so beats letting a permanent
    # named after a player onto the board.
    card = event.get("card")
    if card and board.player(card) is not None:
        problems.append(_violation(
            "card-is-player",
            "%s names the player %r where a card should be" % (kind, card)))

    # Whoever the log says did this has to be one of the two people playing.
    # An event attributed to nobody in the game is a misread line, and it
    # would otherwise be applied to a player invented on the spot.
    if player and board.players and board.player(player) is None:
        closest, score = board.closest_player(player)
        problems.append(_violation(
            "garbled-player" if score >= 0.6 else "unknown-player",
            "%r is not a seat in this game%s"
            % (player, " (closest: %r)" % closest if closest else "")))

    if kind in ("DESTROYED", "DIES", "EXILE_CARD", "RETURN_HAND", "TRANSFORM"):
        card = event.get("card")
        if card:
            found = _needs_battlefield(board, card, None, _VERBS[kind])
            if found:
                problems.append(found)

    elif kind == "SACRIFICE":
        card = event.get("card")
        if card:
            found = _needs_battlefield(board, card, player, "was sacrificed")
            if found:
                problems.append(found)

    elif kind == "BLOCK":
        for role, card in (("blocker", event.get("blocker")),
                           ("attacker", event.get("attacker"))):
            if not card:
                continue
            found = _needs_battlefield(board, card, None, "was declared as a %s" % role)
            if found:
                problems.append(found)

    elif kind == "ATTACKED_BY":
        # "<defender> is being attacked by A and B": the attackers belong to
        # whoever is not the defender.
        for card in event.get("cards", []):
            found = _needs_battlefield(board, card, None, "attacked")
            if found:
                problems.append(found)

    elif kind == "ACTIVATE":
        card = event.get("card")
        if (card and not _activated_from_hand(event)
                and not (board.battlefield().get(fold(card))
                         or board.graveyard().get(fold(card)))):
            problems.append(_violation(
                "missing-source",
                "%s activated an ability of %s, which is neither on the "
                "battlefield nor in a graveyard" % (player, card)))

    elif kind == "DISCARD":
        seat = board.player(player)
        if seat is not None and (seat.get("handCount") or 0) < 1:
            problems.append(_violation(
                "empty-hand",
                "%s discarded %s from a hand XMage shows as empty"
                % (player, event.get("card"))))

    elif kind in ("DRAW", "DRAW_WITH"):
        seat = board.player(player)
        count = int(event.get("count") or 1)
        if seat is not None and (seat.get("libraryCount") or 0) < count:
            problems.append(_violation(
                "empty-library",
                "%s drew %d with only %s cards left in library"
                % (player, count, seat.get("libraryCount"))))

    elif kind == "MILL":
        seat = board.player(player)
        count = len(event.get("cards") or [])
        if seat is not None and (seat.get("libraryCount") or 0) < count:
            problems.append(_violation(
                "empty-library",
                "%s milled %d with only %s cards left in library"
                % (player, count, seat.get("libraryCount"))))

    elif kind == "TURN":
        turn = event.get("turn")
        if board.turn is not None and turn is not None:
            if turn < board.turn:
                problems.append(_violation(
                    "turn-went-backwards",
                    "log starts turn %s while XMage is on turn %s"
                    % (turn, board.turn)))
            elif turn > board.turn + 1:
                problems.append(_violation(
                    "turn-skipped",
                    "log jumps from turn %s to turn %s" % (board.turn, turn)))

    elif kind == "CAST":
        # A spell may target a player, another spell on the stack, or a
        # permanent. Only the last is checkable from a board view, so a
        # target that names neither a seat nor a permanent is reported and
        # anything else is left alone.
        for target in event.get("targets", []) or []:
            if board.player(target) is not None:
                continue
            if board.battlefield().get(fold(target)):
                continue
            closest, score = board.closest_player(target)
            if closest and (score >= _SAME_PLAYER
                            or closest in (target or "").lower()):
                # Nearly a seat's name: the target is a player and the log
                # line picked up something extra on the way through OCR.
                problems.append(_violation(
                    "garbled-target",
                    "%s cast %s targeting %r, which is %r with something "
                    "stuck to it" % (player, event.get("card"), target, closest)))
            # A target that is neither a player nor a permanent is *not*
            # reported. Countering targets a spell on the stack, and the
            # stack is not modelled here, so "not on the battlefield" does
            # not distinguish a decode error from an ordinary counterspell.
            # A check that fires on legal play is worse than no check.

    return problems


_VERBS = {
    "DESTROYED": "was destroyed",
    "DIES": "was put into a graveyard",
    "EXILE_CARD": "was exiled",
    "RETURN_HAND": "was returned to hand",
    "TRANSFORM": "transformed",
}


def check_bundle(bundle_dir: str, classpath: str, java: str = "java",
                 cwd: Optional[str] = None) -> Dict:
    """Walk a game's states, checking each event against the board before it.

    The board comes from XMage rather than from the simulator that produced
    the states: what is being audited is the decoded event stream, and
    checking it against the bookkeeping that produced it would agree with
    itself by construction.
    """
    states_path = os.path.join(bundle_dir, "mirror_states.jsonl")
    states = []
    with open(states_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                states.append(json.loads(line))

    summaries = run_mirror_verify(states_path, classpath, java=java, cwd=cwd)
    if len(summaries) != len(states):
        raise RuntimeError("XMage returned %d summaries for %d states"
                           % (len(summaries), len(states)))
    # Without the card database every library reads as empty, which would
    # turn every draw in the game into a violation of this check.
    assert_card_database(states, summaries, classpath, cwd)

    from_the_start = any(
        (state.get("sourceEvent") or {}).get("type") in _OPENING_EVENTS
        for state in states)

    violations = []
    unestablished = []
    checked = unchecked = 0
    unchecked_types: Counter = Counter()
    for index, state in enumerate(states):
        event = state.get("sourceEvent") or {}
        kind = event.get("type")
        if not kind:
            continue
        if kind not in CHECKED_TYPES:
            unchecked += 1
            unchecked_types[kind] += 1
            continue
        if index == 0:
            # Nothing has happened yet; there is no prior board to contradict.
            continue
        checked += 1
        board = _Board(summaries[index - 1])
        for problem in check_event(event, board):
            finding = dict(problem, stateIndex=index, seq=state.get("seq"),
                           videoTime=state.get("videoTime"),
                           event=event.get("text"), type=kind)
            if not from_the_start and problem["rule"] in _NEEDS_GAME_START:
                unestablished.append(finding)
            else:
                violations.append(finding)

    return {
        "bundle": os.path.abspath(bundle_dir),
        "states": len(states),
        "eventsChecked": checked,
        "eventsUnchecked": unchecked,
        "uncheckedTypes": dict(unchecked_types),
        "capturedFromGameStart": from_the_start,
        "violations": violations,
        # Findings that only mean the recording joined the game in progress.
        "unestablished": unestablished,
        "ok": not violations,
    }
