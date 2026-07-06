"""Human-friendly match metadata for recorded replays.

Turns the raw Arena lifecycle events a session already produces into the kind
of summary a replay browser wants to show: *who* played *whom*, with *what*
colors, in *which* event, and *who won* — the same fields 17Lands / Untapped
lead their match history with, so a recording reads as "Dimir vs Boros — Win
(2-1) — Standard Ladder" instead of a bare timestamp.

``MatchMetadataCollector`` observes normalized events (and enriched board
snapshots) and, on ``finalize``, resolves deck colors through the card
database and emits a ``matches`` list plus a headline ``title``.
"""

__all__ = ["MatchMetadataCollector", "guild_name", "pretty_event_name",
           "set_name", "SET_NAMES"]

# WUBRG-sorted color-letter string -> the name players actually use.
_COLOR_COMBO_NAMES = {
    "": "Colorless",
    "W": "Mono-White", "U": "Mono-Blue", "B": "Mono-Black",
    "R": "Mono-Red", "G": "Mono-Green",
    "WU": "Azorius", "WB": "Orzhov", "WR": "Boros", "WG": "Selesnya",
    "UB": "Dimir", "UR": "Izzet", "UG": "Simic",
    "BR": "Rakdos", "BG": "Golgari", "RG": "Gruul",
    "WUB": "Esper", "WUR": "Jeskai", "WUG": "Bant",
    "WBR": "Mardu", "WBG": "Abzan", "WRG": "Naya",
    "UBR": "Grixis", "UBG": "Sultai", "URG": "Temur", "BRG": "Jund",
    "WUBRG": "Five-Color",
}

_WUBRG_ORDER = "WUBRG"

# MTGA event ids encode the event *type* and, for Limited, the *set* being
# drafted: "TradDraft_OM1_20250923", "PremierDraft_BLB_20240730". Constructed
# events instead encode the *format*: "Alchemy_Ladder", "Historic_Play",
# "Traditional_Ladder", or a bare "Ladder"/"Play" (which is Standard). This is
# the same internal-event-name string 17Lands/the mtga-log-client capture; the
# human names below are the mapping their site applies to it.

# Limited event-type token -> label. Presence of one of these marks a Limited
# event, so the accompanying set code is shown ("Bloomburrow Premier Draft").
_DRAFT_TYPES = {
    "PremierDraft": "Premier Draft",
    "QuickDraft": "Quick Draft",
    "TradDraft": "Traditional Draft",
    "CompDraft": "Competitive Draft",
    "OpenDraft": "Draft",
    "Draft": "Draft",
    "Sealed": "Sealed",
    "TradSealed": "Traditional Sealed",
    "OpenSealed": "Sealed",
    "CompSealed": "Competitive Sealed",
}

# Constructed format token -> label. "Traditional" is a Best-of-Three modifier,
# not a format; a bare Ladder/Play/Constructed event is Standard.
_CONSTRUCTED_FORMATS = {
    "Standard": "Standard",
    "Alchemy": "Alchemy",
    "Historic": "Historic",
    "HistoricBrawl": "Historic Brawl",
    "Explorer": "Explorer",
    "Pioneer": "Pioneer",
    "Timeless": "Timeless",
    "Pauper": "Pauper",
    "Brawl": "Brawl",
    "Gladiator": "Gladiator",
    "Singleton": "Singleton",
    "Pandemonium": "Pandemonium",
}
_BO3_TOKENS = frozenset(("Traditional", "BestOf3", "Bo3"))
_CONSTRUCTED_HINTS = frozenset(("Ladder", "Play", "Constructed", "Event",
                                "BestOf3", "BestOf1", "Bo3", "Bo1"))

# Arena set code -> set name. Covers current/recent Standard and popular older
# sets; anything not listed falls back to the raw code, so it never lies.
SET_NAMES = {
    "OM1": "Marvel's Spider-Man", "SPM": "Marvel's Spider-Man",
    "EOE": "Edge of Eternities", "FIN": "Final Fantasy",
    "TDM": "Tarkir: Dragonstorm", "DFT": "Aetherdrift",
    "FDN": "Foundations", "DSK": "Duskmourn: House of Horror",
    "BLB": "Bloomburrow", "OTJ": "Outlaws of Thunder Junction",
    "MKM": "Murders at Karlov Manor", "LCI": "The Lost Caverns of Ixalan",
    "WOE": "Wilds of Eldraine", "LTR": "The Lord of the Rings",
    "MOM": "March of the Machine", "MAT": "March of the Machine: The Aftermath",
    "ONE": "Phyrexia: All Will Be One", "BRO": "The Brothers' War",
    "DMU": "Dominaria United", "HBG": "Alchemy Horizons: Baldur's Gate",
    "SNC": "Streets of New Capenna", "NEO": "Kamigawa: Neon Dynasty",
    "VOW": "Crimson Vow", "MID": "Midnight Hunt", "AFR": "Adventures in the Forgotten Realms",
    "STX": "Strixhaven", "KHM": "Kaldheim", "ZNR": "Zendikar Rising",
    "M21": "Core Set 2021", "IKO": "Ikoria", "THB": "Theros Beyond Death",
    "ELD": "Throne of Eldraine", "M20": "Core Set 2020", "WAR": "War of the Spark",
    "RNA": "Ravnica Allegiance", "GRN": "Guilds of Ravnica",
    "Cube": "Cube", "Chaos": "Chaos Draft",
}


def set_name(code):
    """Arena set code -> readable set name (falls back to the code)."""
    if not code:
        return None
    return SET_NAMES.get(code, SET_NAMES.get(code.upper(), code))


def guild_name(color_letters):
    """WUBRG letters ('UB') -> a color/guild name ('Dimir')."""
    key = "".join(c for c in _WUBRG_ORDER if c in (color_letters or ""))
    if key in _COLOR_COMBO_NAMES:
        return _COLOR_COMBO_NAMES[key]
    return "%d-Color" % len(key) if key else "Colorless"


def pretty_event_name(event_id, deck_set=None):
    """Readable event label: Limited names the set, Constructed the format.

    'TradDraft_OM1_20250923'      -> "Marvel's Spider-Man Traditional Draft"
    'Alchemy_Ladder'              -> 'Alchemy'
    'Traditional_Historic_Ladder' -> 'Historic (Bo3)'
    'Ladder' / 'Play'             -> 'Standard'

    ``deck_set`` (an Arena set code derived from the recorded deck) is used as
    the Limited set when the event id itself doesn't carry one.
    """
    if not event_id:
        return set_name(deck_set) if deck_set else None
    tokens = str(event_id).split("_")

    draft_label = _limited_label(tokens)
    if draft_label is not None:
        code = _set_token(tokens) or deck_set
        name = set_name(code)
        return "%s %s" % (name, draft_label) if name else draft_label

    # Constructed, or an unrecognized event: name the format.
    fmt = None
    for token in tokens:
        if token in _CONSTRUCTED_FORMATS:
            fmt = _CONSTRUCTED_FORMATS[token]
            break
    if fmt is None and any(t in _CONSTRUCTED_HINTS for t in tokens):
        fmt = "Standard"  # bare Ladder / Play / Constructed is Standard
    if fmt is None:
        return _camel_words(tokens[0]) or str(event_id)
    if any(t in _BO3_TOKENS for t in tokens):
        fmt += " (Bo3)"
    return fmt


def _limited_label(tokens):
    for token in tokens:
        if token in _DRAFT_TYPES:
            return _DRAFT_TYPES[token]
    # "DirectGameTournamentLimited" and similar carry no separate type token
    joined = "".join(tokens)
    if "Sealed" in joined and "Draft" not in joined:
        return "Sealed"
    if "Limited" in joined:
        return "Draft"
    return None


def _set_token(tokens):
    """The set code embedded in a Limited event id ('OM1', 'BLB')."""
    for token in tokens:
        if token in _DRAFT_TYPES:
            continue
        if token in SET_NAMES:
            return token
        # a set code is short, uppercase-ish, and not an 8-digit date
        if 2 <= len(token) <= 5 and token.isalnum() and not token.isdigit() \
                and token.upper() == token:
            return token
    return None


def _camel_words(text):
    words = []
    current = ""
    for char in text or "":
        if char.isupper() and current and not current[-1].isupper():
            words.append(current)
            current = char
        else:
            current += char
    if current:
        words.append(current)
    return " ".join(words).strip()


class MatchMetadataCollector(object):
    """Accumulates per-match metadata from normalized events + snapshots."""

    def __init__(self):
        self._matches = {}          # matchId -> match record
        self._order = []            # matchId order of first appearance
        self._local_seat = None
        self._local_deck_ids = []
        self._pending_deck_ids = None

    # --- ingestion ---

    def observe(self, event):
        etype = event.get("type")
        if etype == "ARENA_CONNECT_RESP":
            self._observe_connect(event)
        elif etype == "ARENA_MATCH_STATE_CHANGED":
            self._observe_match_state(event)
        elif etype == "ARENA_GAME_OVER":
            self._observe_game_over(event)
        self._note_time(event.get("matchId"), event.get("timestamp"))

    def note_snapshot(self, snapshot):
        """Enriched board snapshot: learn seat names, colors seen, timing."""
        match_id = snapshot.get("matchId")
        match = self._match(match_id)
        if match is None:
            return
        local = snapshot.get("localSeat")
        if local is not None:
            self._local_seat = local
            match["localSeat"] = local
        for player in snapshot.get("players") or []:
            seat = player.get("seat")
            name = player.get("name")
            if seat is not None and name and not name.startswith("Seat "):
                match["names"].setdefault(seat, name)
        # opponent colors from their public (non-hidden) cards
        zones = snapshot.get("zones") or {}
        objects = list(zones.get("battlefield") or [])
        for seat_objs in (zones.get("graveyards") or {}).values():
            objects.extend(seat_objs)
        for obj in objects:
            colors = obj.get("colors")
            controller = obj.get("controllerSeat")
            if not colors or controller is None:
                continue
            if local is not None and controller == local:
                match["youColorsSeen"].update(colors)
            else:
                match["oppColorsSeen"].update(colors)
        self._note_time(match_id, snapshot.get("timestamp"))

    # --- event handlers ---

    def _observe_connect(self, event):
        deck = event.get("deckInfo") or {}
        seat = deck.get("seatId")
        if seat is not None:
            self._local_seat = seat
        ids = deck.get("mainDeckArenaIds") or []
        if ids:
            # ConnectResp arrives before the match id is known; hold the deck
            # until a match record exists, then attach it.
            self._local_deck_ids = ids
            self._pending_deck_ids = ids
        self._attach_pending_deck()

    def _observe_match_state(self, event):
        payload = event.get("payload") or {}
        room = payload.get("gameRoomInfo") or {}
        config = room.get("gameRoomConfig") or {}
        match_id = config.get("matchId") or event.get("matchId")
        match = self._match(match_id, create=True)
        if match is None:
            return
        for player in config.get("reservedPlayers") or []:
            if not isinstance(player, dict):
                continue
            seat = player.get("systemSeatId")
            name = player.get("playerName")
            if name:
                match["names"][seat] = name.split("#")[0]
            if player.get("teamId") is not None:
                match["teamBySeat"][seat] = player.get("teamId")
            if player.get("eventId") and not match.get("eventId"):
                match["eventId"] = player.get("eventId")
        if config.get("eventId") and not match.get("eventId"):
            match["eventId"] = config.get("eventId")
        # a MatchCompleted state may carry the result inline
        final = room.get("finalMatchResult")
        if isinstance(final, dict):
            self._apply_result(match, final.get("resultList"))
        self._attach_pending_deck()

    def _observe_game_over(self, event):
        payload = event.get("payload") or {}
        result_list = payload.get("resultList")
        if result_list is None:
            final = payload.get("finalMatchResult")
            if isinstance(final, dict):
                result_list = final.get("resultList")
        if result_list is not None:
            match = self._match(event.get("matchId"), create=True)
            if match is not None:
                self._apply_result(match, result_list)

    def _apply_result(self, match, result_list):
        if not isinstance(result_list, list):
            return
        games = []
        match_winner = None
        for item in result_list:
            if not isinstance(item, dict):
                continue
            scope = item.get("scope")
            winner = item.get("winningTeamId")
            if scope == "MatchScope_Game":
                games.append(winner)
            elif scope == "MatchScope_Match":
                match_winner = winner
        if games:
            match["gameWinners"] = games
        if match_winner is not None:
            match["matchWinner"] = match_winner

    # --- helpers ---

    def _match(self, match_id, create=False):
        if match_id is None:
            return None
        if match_id not in self._matches:
            if not create:
                return None
            self._matches[match_id] = {
                "matchId": match_id,
                "names": {},
                "teamBySeat": {},
                "eventId": None,
                "localSeat": self._local_seat,
                "deckIds": [],
                "gameWinners": [],
                "matchWinner": None,
                "youColorsSeen": set(),
                "oppColorsSeen": set(),
                "startTime": None,
                "endTime": None,
            }
            self._order.append(match_id)
        return self._matches[match_id]

    def _attach_pending_deck(self):
        if not self._pending_deck_ids:
            return
        for match_id in self._order:
            match = self._matches[match_id]
            if not match["deckIds"]:
                match["deckIds"] = list(self._pending_deck_ids)

    def _note_time(self, match_id, timestamp):
        match = self._match(match_id)
        if match is None or not timestamp:
            return
        if match["startTime"] is None or timestamp < match["startTime"]:
            match["startTime"] = timestamp
        if match["endTime"] is None or timestamp > match["endTime"]:
            match["endTime"] = timestamp

    # --- output ---

    def finalize(self, card_db=None):
        """Return {'matches': [...], 'title': ..., ...} for summary.json."""
        matches = [self._render_match(self._matches[m], card_db)
                   for m in self._order]
        result = {"matches": matches}
        if matches:
            primary = matches[0]
            result["title"] = primary.get("title")
            for key in ("you", "opponent", "result", "gameRecord",
                        "eventName", "startTime"):
                if primary.get(key) is not None:
                    result[key] = primary[key]
        return result

    def _render_match(self, match, card_db):
        local_seat = match.get("localSeat")
        if local_seat is None:
            local_seat = self._local_seat
        seats = sorted(match["names"].keys())
        opp_seat = None
        for seat in seats:
            if seat != local_seat:
                opp_seat = seat
                break

        you_colors = self._deck_colors(match["deckIds"], card_db) \
            or _letters(match["youColorsSeen"])
        opp_colors = _letters(match["oppColorsSeen"])

        you = {
            "seat": local_seat,
            "name": match["names"].get(local_seat) if local_seat else None,
            "colors": you_colors,
            "archetype": guild_name(you_colors) if you_colors else None,
            "deckSize": len(match["deckIds"]) or None,
        }
        opponent = {
            "seat": opp_seat,
            "name": match["names"].get(opp_seat) if opp_seat else None,
            "colors": opp_colors,
            "archetype": guild_name(opp_colors) if opp_colors else None,
        }

        result, record = self._result(match, local_seat)
        deck_set = self._deck_set(match["deckIds"], card_db)
        event_name = pretty_event_name(match.get("eventId"), deck_set=deck_set)

        rendered = {
            "matchId": match["matchId"],
            "eventId": match.get("eventId"),
            "eventName": event_name,
            "startTime": match.get("startTime"),
            "endTime": match.get("endTime"),
            "you": you,
            "opponent": opponent,
            "result": result,
            "gameRecord": record,
            "games": len(match["gameWinners"]) or None,
        }
        rendered["title"] = _build_title(rendered)
        return rendered

    def _result(self, match, local_seat):
        team = match["teamBySeat"].get(local_seat, local_seat)
        winners = match["gameWinners"]
        wins = sum(1 for w in winners if w == team)
        losses = sum(1 for w in winners if w is not None and w != team)
        record = "%d-%d" % (wins, losses) if winners else None
        outcome = None
        if match["matchWinner"] is not None and team is not None:
            outcome = "win" if match["matchWinner"] == team else "loss"
        elif winners:
            outcome = "win" if wins > losses else (
                "loss" if losses > wins else "draw")
        return outcome, record

    def _deck_set(self, deck_ids, card_db):
        """The set most of the deck comes from — the drafted set, in Limited."""
        if not deck_ids or card_db is None:
            return None
        counts = {}
        for grp_id in deck_ids:
            try:
                info = card_db.lookup(grp_id)
            except Exception:
                info = None
            if info is None or not info.expansion:
                continue
            if "Basic" in (info.supertypes or []):
                continue  # basics come from any set; ignore them
            counts[info.expansion] = counts.get(info.expansion, 0) + 1
        if not counts:
            return None
        return max(counts, key=counts.get)

    def _deck_colors(self, deck_ids, card_db):
        if not deck_ids or card_db is None:
            return ""
        seen = set()
        for grp_id in deck_ids:
            try:
                info = card_db.lookup(grp_id)
            except Exception:
                info = None
            if info is not None and info.color_identity:
                seen.update(info.color_identity)
        return _letters(seen)


def _letters(color_set):
    return "".join(c for c in _WUBRG_ORDER if c in (color_set or set()))


def _build_title(match):
    you = match["you"]
    opp = match["opponent"]
    left = you.get("archetype") or (you.get("name") or "You")
    right = opp.get("archetype") or (opp.get("name") or "Opponent")
    parts = ["%s vs %s" % (left, right)]
    result = match.get("result")
    if result:
        label = {"win": "Win", "loss": "Loss", "draw": "Draw"}.get(result, result)
        if match.get("gameRecord"):
            label = "%s (%s)" % (label, match["gameRecord"])
        parts.append(label)
    if match.get("eventName"):
        parts.append(match["eventName"])
    date = _short_date(match.get("startTime"))
    if date:
        parts.append(date)
    return " — ".join(parts)


def _short_date(iso):
    if not iso:
        return None
    import datetime
    try:
        dt = datetime.datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return None
    return dt.strftime("%b %d, %Y")
