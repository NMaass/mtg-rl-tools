"""Board-state simulation over parsed MTGO log events.

Applies events in order and emits mirror-format snapshots (the same schema
`arena_mirror.tracker.GameStateTracker.snapshot()` produces) so the game can
be replayed through the XMage mirror and verified headlessly.
"""

import difflib
from typing import Callable, Dict, List, Optional

from . import catalog as cardinfo


class _Player:
    def __init__(self, seat: int, name: str, deck_size: int):
        self.seat = seat
        self.name = name
        self.life = 20
        self.hand = 0
        self.library = deck_size


class GameSimulator:
    def __init__(
        self,
        hero: Optional[str] = None,
        match_id: str = "mtgo",
        game_number: int = 1,
        deck_size: int = 60,
        card_info: Optional[Callable[[str], Optional[dict]]] = None,
        max_players: int = 2,
    ):
        self.hero = hero
        self.match_id = match_id
        self.game_number = game_number
        self.deck_size = deck_size
        self.card_info = card_info or (lambda name: None)
        self.max_players = max_players

        self.players: Dict[str, _Player] = {}
        self.battlefield: List[dict] = []
        self.graveyards: Dict[int, List[dict]] = {}
        self.exile: List[dict] = []
        self.turn_number = 1
        self.pending_combat: Optional[dict] = None
        self.phase = "Phase_Beginning"
        self.step: Optional[str] = None
        self.active_seat: Optional[int] = None
        self.game_over = False
        self.winner: Optional[str] = None
        self.seq = 0
        self._next_instance = 100
        self.warnings: List[str] = []

    # -- player helpers ----------------------------------------------------

    def seed_players(self, names: List[str]):
        """Register the game's players up front, in seat order.

        The first snapshot has to name every player: the XMage mirror creates
        its seats once, from the state it opens with. Seeding also stops an
        OCR-mangled name later in the game from being read as a new player.
        """
        for name in names:
            self._player(name)

    def _player(self, name: str) -> _Player:
        resolved = self._resolve_player_name(name)
        if resolved not in self.players:
            seat = len(self.players) + 1
            self.players[resolved] = _Player(seat, resolved, self.deck_size)
        return self.players[resolved]

    def _resolve_player_name(self, name: str) -> str:
        name = name.strip().rstrip(".,:;")
        if name in self.players:
            return name
        best, score = None, 0.0
        for known in self.players:
            r = difflib.SequenceMatcher(None, name.lower(), known.lower()).ratio()
            if r > score:
                best, score = known, r
        if best and score >= 0.7:
            return best
        # A duel has exactly two players. Once both seats are known, an
        # unrecognized name is OCR noise, so snap it to the nearest seat
        # rather than inventing a third player.
        if best and len(self.players) >= self.max_players:
            return best
        return name

    @property
    def local_seat(self) -> Optional[int]:
        if not self.hero:
            return None
        resolved = self._resolve_player_name(self.hero)
        player = self.players.get(resolved)
        return player.seat if player else None

    # -- object helpers ----------------------------------------------------

    def _new_object(self, name: str, seat: int) -> dict:
        self._next_instance += 1
        info = self.card_info(name)
        obj = {
            "instanceId": self._next_instance,
            "name": name,
            "faceDown": False,
            "ownerSeat": seat,
            "controllerSeat": seat,
            "tapped": False,
        }
        if info and cardinfo.is_creature(info):
            face = info
            if face.get("power") is None and info.get("faces"):
                face = info["faces"][0]
            if face.get("power") is not None:
                obj["power"] = _num(face.get("power"))
                obj["toughness"] = _num(face.get("toughness"))
        return obj

    def _find_battlefield(self, name: str, seat: Optional[int] = None) -> Optional[dict]:
        candidates = [
            o
            for o in self.battlefield
            if _loose(o["name"]) == _loose(name)
            and (seat is None or o["controllerSeat"] == seat)
        ]
        return candidates[-1] if candidates else None

    def _to_graveyard(self, obj: dict):
        seat = obj["ownerSeat"]
        self.graveyards.setdefault(seat, []).append(
            {
                "instanceId": obj["instanceId"],
                "name": obj["name"],
                "faceDown": False,
                "ownerSeat": seat,
            }
        )

    def _add_graveyard_card(self, name: str, seat: int):
        self._next_instance += 1
        self.graveyards.setdefault(seat, []).append(
            {
                "instanceId": self._next_instance,
                "name": name,
                "faceDown": False,
                "ownerSeat": seat,
            }
        )

    # -- event application -------------------------------------------------

    def apply(self, event: dict) -> List[dict]:
        """Apply one event; returns the snapshots it produced, in order.

        Usually one, but an event that also resolves a pending combat emits
        the combat damage as its own snapshot first: the damage happens when
        combat resolves, not when the next spell is cast, and MTGO's life
        display moves at that earlier moment.
        """
        kind = event["type"]
        snapshots = []
        if kind != "BLOCK":
            damage = self._flush_combat()
            if damage is not None:
                snapshots.append(damage)

        handler = getattr(self, "_on_" + kind.lower(), None)
        changed = True
        if handler:
            changed = handler(event)
        elif kind in ("ROLL", "JOIN", "CHAT", "UNPARSED", "REVEAL", "CYCLE",
                      "TRIGGER", "SHUFFLE", "SCRY_BOTTOM", "PUT_TOP", "ACTIVATE",
                      "PLAY_FIRST", "SKIP_DRAW", "LEADS_MATCH", "PUTS_TOP_N"):
            changed = False
        if changed:
            self.seq += 1
            snapshots.append(self.snapshot())
        return snapshots

    def _on_opening_hand(self, e):
        p = self._player(e["player"])
        p.hand = int(e["count"])
        p.library = self.deck_size - p.hand
        return True

    def _on_mulligan(self, e):
        p = self._player(e["player"])
        p.hand = int(e["count"])
        p.library = self.deck_size - p.hand
        return True

    def _on_bottom_begin(self, e):
        return self._on_opening_hand(e)

    def _on_turn(self, e):
        self.turn_number = e["turn"]
        p = self._player(e["player"])
        self.active_seat = p.seat
        self.phase = "Phase_Beginning"
        self.step = "Step_Upkeep"
        for obj in self.battlefield:
            if obj["controllerSeat"] == p.seat:
                obj["tapped"] = False
        return True

    def _on_play_land(self, e):
        p = self._player(e["player"])
        p.hand = max(0, p.hand - 1)
        if self.phase == "Phase_Beginning":
            self.phase, self.step = "Phase_Main1", None
        self.battlefield.append(self._new_object(e["card"], p.seat))
        return True

    def _on_cast(self, e):
        p = self._player(e["player"])
        p.hand = max(0, p.hand - 1)
        if self.phase == "Phase_Beginning":
            self.phase, self.step = "Phase_Main1", None
        info = self.card_info(e["card"])
        if cardinfo.is_permanent(info):
            self.battlefield.append(self._new_object(e["card"], p.seat))
        else:
            self._add_graveyard_card(e["card"], p.seat)
        return True

    def _on_draw(self, e):
        p = self._player(e["player"])
        count = int(e.get("count", 1))
        p.hand += count
        if p.library is not None:
            p.library = max(0, p.library - count)
        return True

    def _on_draw_with(self, e):
        return self._on_draw(e)

    def _on_counters_with(self, e):
        # "<player> counters <card> with <counter>": pull the countered spell
        # off the battlefield if the cast optimistically resolved it there
        # (instants/sorceries are already in their owner's graveyard).
        self._remove_to_graveyard(e["card"])
        return True

    def _on_mill(self, e):
        p = self._player(e["player"])
        for card in e["cards"]:
            self._add_graveyard_card(card, p.seat)
            if p.library is not None:
                p.library = max(0, p.library - 1)
        return True

    def _on_discard(self, e):
        p = self._player(e["player"])
        p.hand = max(0, p.hand - 1)
        self._add_graveyard_card(e["card"], p.seat)
        return True

    def _on_transform(self, e):
        obj = self._find_battlefield(e["card"])
        if not obj:
            self.warnings.append("transform: %s not on battlefield" % e["card"])
            return False
        obj["name"] = e["into"]
        # The back face is a card in its own right, so look it up by name;
        # only fall back to walking the front face's face list. Getting this
        # wrong leaves the permanent with its pre-transform power, which then
        # silently under-counts combat damage.
        face = self.card_info(e["into"])
        if not face or face.get("power") is None:
            for candidate in (self.card_info(e["card"]) or {}).get("faces", []):
                if _loose(candidate.get("name", "")) == _loose(e["into"]):
                    face = candidate
                    break
        if face and face.get("power") is not None:
            obj["power"] = _num(face["power"])
            obj["toughness"] = _num(face["toughness"])
        else:
            self.warnings.append("transform: unknown power for %s" % e["into"])
        return True

    def _on_attacked_by(self, e):
        """Declare attackers and hold their damage until combat resolves.

        MTGO's game log never prints combat damage dealt to a player -- it
        only shows the result in the on-screen life total -- so it has to be
        derived from the attack itself. The damage is held pending here so
        that a "blocks" line can remove an attacker before it lands; see
        _flush_combat.
        """
        self.phase, self.step = "Phase_Combat", "Step_DeclareAttack"
        defender = self._player(e["player"])
        self.pending_combat = {"defenderSeat": defender.seat, "attackers": {},
                               # Damage lands within a second or two of this
                               # line, so the attack's timestamp is the right
                               # one to carry onto the damage snapshot.
                               "videoTime": e.get("videoTime")}
        for card in e["cards"]:
            obj = self._find_battlefield(card, self.active_seat)
            if not obj:
                self.warnings.append("attacker not on battlefield: %s" % card)
                continue
            obj["tapped"] = True
            power = obj.get("power")
            if power is None:
                self.warnings.append("attacker has unknown power: %s" % card)
                continue
            self.pending_combat["attackers"].setdefault(_loose(card), []).append(power)
        return True

    def _on_block(self, e):
        """A blocked attacker deals no damage to the defending player."""
        pending = self.pending_combat
        if not pending:
            return False
        pending_powers = pending["attackers"].get(_loose(e["attacker"]))
        if not pending_powers:
            return False
        pending_powers.pop()
        self.step = "Step_DeclareBlock"
        return True

    def _flush_combat(self) -> Optional[dict]:
        """Deal the pending unblocked attackers' damage; snapshot the result.

        The snapshot inherits the *attack's* video timestamp, not the next
        log line's: combat damage lands moments after attackers are declared,
        which can be many seconds before anything else is logged.
        """
        pending = self.pending_combat
        self.pending_combat = None
        if not pending:
            return None
        total = sum(sum(powers) for powers in pending["attackers"].values())
        if not total:
            return None
        for player in self.players.values():
            if player.seat == pending["defenderSeat"]:
                player.life -= total
        self.step = "Step_CombatDamage"
        self.seq += 1
        snapshot = self.snapshot()
        snapshot["sourceEvent"] = {"type": "COMBAT_DAMAGE",
                                   "text": "combat damage: %d" % total}
        if pending.get("videoTime") is not None:
            snapshot["videoTime"] = pending["videoTime"]
        return snapshot

    def _on_damage(self, e):
        target = self._resolve_player_name(e["target"])
        if target in self.players:
            self.players[target].life -= e["amount"]
            if self.phase == "Phase_Combat":
                self.step = "Step_CombatDamage"
            return True
        return False  # damage to creatures tracked implicitly via death events

    def _on_lose_life(self, e):
        self._player(e["player"]).life -= e["amount"]
        return True

    def _on_gain_life(self, e):
        self._player(e["player"]).life += e["amount"]
        return True

    def _on_life_total(self, e):
        self._player(e["player"]).life = e["total"]
        return True

    def _remove_to_graveyard(self, card_name: str) -> bool:
        obj = self._find_battlefield(card_name)
        if obj:
            self.battlefield.remove(obj)
            self._to_graveyard(obj)
            return True
        return False

    def _on_dies(self, e):
        if not self._remove_to_graveyard(e["card"]):
            self.warnings.append("dies: %s not on battlefield" % e["card"])
            return False
        return True

    def _on_destroyed(self, e):
        return self._on_dies(e)

    def _on_sacrifice(self, e):
        if not self._remove_to_graveyard(e["card"]):
            self.warnings.append("sacrifice: %s not found" % e["card"])
            return False
        return True

    def _on_countered(self, e):
        # The spell most recently cast with this name: if we optimistically
        # put it on the battlefield, pull it back to its owner's graveyard.
        return self._remove_to_graveyard(e["card"]) or True

    def _on_return_hand(self, e):
        obj = self._find_battlefield(e["card"])
        if not obj:
            self.warnings.append("return: %s not on battlefield" % e["card"])
            return False
        self.battlefield.remove(obj)
        self.players_by_seat()[obj["ownerSeat"]].hand += 1
        return True

    def _on_exile_card(self, e):
        obj = self._find_battlefield(e["card"])
        if not obj:
            return False
        self.battlefield.remove(obj)
        self.exile.append(obj)
        return True

    def _on_concede(self, e):
        self.game_over = True
        loser = self._resolve_player_name(e["player"])
        for name in self.players:
            if name != loser:
                self.winner = name
        return True

    def _on_win(self, e):
        self.game_over = True
        self.winner = self._resolve_player_name(e["player"])
        return True

    def _on_lose_game(self, e):
        self.game_over = True
        loser = self._resolve_player_name(e["player"])
        for name in self.players:
            if name != loser:
                self.winner = name
        return True

    # -- snapshot ----------------------------------------------------------

    def players_by_seat(self) -> Dict[int, _Player]:
        return {p.seat: p for p in self.players.values()}

    def snapshot(self) -> dict:
        hands = {}
        for p in self.players.values():
            cards = []
            for i in range(p.hand):
                cards.append(
                    {
                        "instanceId": 9000 + p.seat * 100 + i,
                        "faceDown": True,
                        "ownerSeat": p.seat,
                        "controllerSeat": p.seat,
                    }
                )
            hands[str(p.seat)] = cards
        return {
            "seq": self.seq,
            "gameInstance": self.game_number,
            "gameNumber": self.game_number,
            "matchId": self.match_id,
            "localSeat": self.local_seat,
            "gameOver": self.game_over,
            "turnNumber": self.turn_number,
            "phase": self.phase,
            "step": self.step,
            "activeSeat": self.active_seat,
            "prioritySeat": self.active_seat,
            "players": [
                {
                    "seat": p.seat,
                    "name": p.name,
                    "life": p.life,
                    "libraryCount": p.library,
                    "handCount": p.hand,
                }
                for p in sorted(self.players.values(), key=lambda q: q.seat)
            ],
            "zones": {
                "battlefield": [dict(o) for o in self.battlefield],
                "stack": [],
                "hands": hands,
                "graveyards": {
                    str(seat): [dict(o) for o in objs]
                    for seat, objs in self.graveyards.items()
                },
                "exile": [dict(o) for o in self.exile],
                "command": [],
                "libraries": {
                    str(p.seat): p.library
                    for p in self.players.values()
                    if p.library is not None
                },
            },
        }


def _loose(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _num(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
