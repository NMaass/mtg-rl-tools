"""Arena grpId -> card metadata resolution.

The authoritative source is MTG Arena's own card database, a SQLite file the
client keeps under ``MTGA_Data/Downloads/Raw/Raw_CardDatabase_*.mtga`` — it is
always in sync with whatever client wrote the log. A JSON cache export lets
replays resolve cards on machines without MTGA installed.
"""

import glob
import json
import os
import re
import sqlite3

# MTGA localization strings occasionally embed inline sprite markup, e.g.
# ``<sprite="SpriteSheet_MiscIcons" name="arena_a">Cauldron Familiar`` for the
# Alchemy "A" badge. Strip any such tags so a clean card name comes through.
_SPRITE_TAG_RE = re.compile(r"<[^>]*>")

__all__ = ["CardDatabase", "CardInfo", "default_mtga_card_db_path"]

DEFAULT_MTGA_RAW_DIRS = (
    r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw",
    r"C:\Program Files (x86)\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw",
)


def default_mtga_card_db_path():
    """Newest Raw_CardDatabase_*.mtga from a standard MTGA install, or None."""
    candidates = []
    for raw_dir in DEFAULT_MTGA_RAW_DIRS:
        candidates.extend(glob.glob(os.path.join(raw_dir, "Raw_CardDatabase_*.mtga")))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


class CardInfo(object):
    """One Arena card: the fields the mirror and recorder need."""

    __slots__ = ("grp_id", "name", "types", "subtypes", "supertypes",
                 "power", "toughness", "is_token", "linked_face_grp_ids",
                 "colors", "color_identity", "is_rebalanced", "expansion")

    def __init__(self, grp_id, name, types=None, subtypes=None, supertypes=None,
                 power=None, toughness=None, is_token=False,
                 linked_face_grp_ids=None, colors="", color_identity="",
                 is_rebalanced=False, expansion=None):
        self.grp_id = grp_id
        self.name = name
        self.types = types or []
        self.subtypes = subtypes or []
        self.supertypes = supertypes or []
        self.power = power
        self.toughness = toughness
        self.is_token = is_token
        self.linked_face_grp_ids = linked_face_grp_ids or []
        # WUBRG letters, e.g. "UB"; colors = the card's face color, while
        # color_identity also counts mana symbols (so a dual land is colored).
        self.colors = colors or ""
        self.color_identity = color_identity or ""
        self.is_rebalanced = bool(is_rebalanced)
        self.expansion = expansion or None

    def to_dict(self):
        return {
            "grpId": self.grp_id,
            "name": self.name,
            "types": self.types,
            "subtypes": self.subtypes,
            "supertypes": self.supertypes,
            "power": self.power,
            "toughness": self.toughness,
            "isToken": self.is_token,
            "linkedFaceGrpIds": self.linked_face_grp_ids,
            "colors": self.colors,
            "colorIdentity": self.color_identity,
            "isRebalanced": self.is_rebalanced,
            "expansion": self.expansion,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            grp_id=data.get("grpId"),
            name=data.get("name"),
            types=data.get("types") or [],
            subtypes=data.get("subtypes") or [],
            supertypes=data.get("supertypes") or [],
            power=data.get("power"),
            toughness=data.get("toughness"),
            is_token=bool(data.get("isToken")),
            linked_face_grp_ids=data.get("linkedFaceGrpIds") or [],
            colors=data.get("colors") or "",
            color_identity=data.get("colorIdentity") or "",
            is_rebalanced=bool(data.get("isRebalanced")),
            expansion=data.get("expansion"),
        )


class CardDatabase(object):
    """grpId lookups against the MTGA SQLite DB or a JSON cache export."""

    def __init__(self, db_path=None, cache_path=None):
        """``db_path``: Raw_CardDatabase_*.mtga (auto-discovered when None).
        ``cache_path``: JSON cache; used when the SQLite DB is unavailable
        and refreshed from it when it is."""
        self._cards = {}
        self._connection = None
        self._enum_cache = {}
        self.source = None

        if db_path is None:
            db_path = default_mtga_card_db_path()
        if db_path is not None and os.path.exists(db_path):
            # sqlite3 cannot open the file read-only via a plain path when it
            # sits under Program Files; use a read-only URI.
            uri = "file:%s?mode=ro" % db_path.replace("\\", "/")
            self._connection = sqlite3.connect(uri, uri=True)
            self.source = db_path
        elif cache_path is not None and os.path.exists(cache_path):
            self._load_cache(cache_path)
            self.source = cache_path
        else:
            raise IOError(
                "no card database: no MTGA Raw_CardDatabase found and no cache "
                "file at %r; pass db_path= or cache_path=" % (cache_path,)
            )

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def lookup(self, grp_id):
        """CardInfo for an Arena grpId, or None when unknown / hidden (0)."""
        if not grp_id:
            return None
        grp_id = int(grp_id)
        if grp_id in self._cards:
            return self._cards[grp_id]
        info = self._query(grp_id) if self._connection is not None else None
        self._cards[grp_id] = info
        return info

    def name_for(self, grp_id):
        info = self.lookup(grp_id)
        return info.name if info is not None else None

    def export_cache(self, cache_path, grp_ids=None):
        """Write the JSON cache (all cards seen so far, plus ``grp_ids``)."""
        if grp_ids:
            for grp_id in grp_ids:
                self.lookup(grp_id)
        payload = {
            str(grp_id): info.to_dict()
            for grp_id, info in self._cards.items()
            if info is not None
        }
        directory = os.path.dirname(cache_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)

    # --- internals ---

    def _load_cache(self, cache_path):
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for key, value in payload.items():
            try:
                grp_id = int(key)
            except ValueError:
                continue
            self._cards[grp_id] = CardInfo.from_dict(value)

    def _query(self, grp_id):
        row = self._connection.execute(
            "SELECT TitleId, Types, Subtypes, Supertypes, Power, Toughness,"
            "       IsToken, LinkedFaceGrpIds, Colors, ColorIdentity,"
            "       IsRebalanced, ExpansionCode"
            "  FROM Cards WHERE GrpId = ?",
            (grp_id,),
        ).fetchone()
        if row is None:
            return None
        (title_id, types, subtypes, supertypes, power, toughness, is_token,
         faces, colors, color_identity, is_rebalanced, expansion) = row
        name = self._localize(title_id)
        if name is None:
            return None
        return CardInfo(
            grp_id=grp_id,
            name=name,
            types=self._enum_names("CardType", types),
            subtypes=self._enum_names("SubType", subtypes),
            supertypes=self._enum_names("SuperType", supertypes),
            power=power if power not in ("", None) else None,
            toughness=toughness if toughness not in ("", None) else None,
            is_token=bool(is_token),
            linked_face_grp_ids=_int_list(faces),
            colors=_color_letters(colors),
            color_identity=_color_letters(color_identity),
            is_rebalanced=bool(is_rebalanced),
            expansion=expansion or None,
        )

    def _localize(self, loc_id):
        if loc_id is None:
            return None
        # Formatted variants exist per LocId; the lowest is the plain form.
        row = self._connection.execute(
            "SELECT Loc FROM Localizations_enUS WHERE LocId = ?"
            " ORDER BY Formatted LIMIT 1",
            (loc_id,),
        ).fetchone()
        if not row or row[0] is None:
            return None
        cleaned = _SPRITE_TAG_RE.sub("", row[0]).strip()
        return cleaned or None

    def _enum_names(self, enum_type, value):
        names = []
        for raw in _int_list(value):
            key = (enum_type, raw)
            if key not in self._enum_cache:
                row = self._connection.execute(
                    "SELECT l.Loc FROM Enums e"
                    " JOIN Localizations_enUS l ON e.LocId = l.LocId"
                    " WHERE e.Type = ? AND e.Value = ?"
                    " ORDER BY l.Formatted LIMIT 1",
                    (enum_type, raw),
                ).fetchone()
                self._enum_cache[key] = row[0] if row else str(raw)
            names.append(self._enum_cache[key])
        return names


# MTGA color enum -> WUBRG letter (1=W, 2=U, 3=B, 4=R, 5=G).
_COLOR_LETTERS = {1: "W", 2: "U", 3: "B", 4: "R", 5: "G"}


def _color_letters(value):
    """MTGA Colors/ColorIdentity ('3,2') -> WUBRG-ordered letters ('UB')."""
    letters = [_COLOR_LETTERS[i] for i in sorted(set(_int_list(value)))
               if i in _COLOR_LETTERS]
    return "".join(letters)


def _int_list(value):
    """MTGA stores lists as comma-separated text columns."""
    if value in (None, ""):
        return []
    if isinstance(value, int):
        return [value]
    parts = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if chunk:
            try:
                parts.append(int(chunk))
            except ValueError:
                pass
    return parts
