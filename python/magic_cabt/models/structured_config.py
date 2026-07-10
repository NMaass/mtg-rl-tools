"""Configuration and card metadata lookup for the structured Magic JEPA."""
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass
class StructuredJEPAConfig:
    text_dim: int = 384
    numeric_dim: int = 40
    d_model: int = 320
    nhead: int = 8
    encoder_layers: int = 6
    predictor_layers: int = 3
    ff_dim: int = 1280
    dropout: float = 0.1
    max_objects: int = 128
    causal_dim: int = 18
    horizon_buckets: int = 32
    embedding_backend: str = "hash"

    @classmethod
    def preset(cls, name):
        name = (name or "local").lower()
        if name == "tiny":
            return cls(d_model=192, nhead=6, encoder_layers=4,
                       predictor_layers=2, ff_dim=768, max_objects=96)
        if name == "local":
            return cls()
        if name == "large":
            return cls(d_model=448, nhead=8, encoder_layers=8,
                       predictor_layers=4, ff_dim=1792, max_objects=160)
        raise ValueError("unknown model preset: %s" % name)


class CardTextResolver:
    """Card cache plus optional read-only Arena database rules-text lookup."""
    def __init__(self, cards: Optional[Mapping] = None,
                 arena_db_path: Optional[str] = None):
        self.by_id, self.by_name = {}, {}
        self.arena_db_path = arena_db_path if arena_db_path and \
            os.path.isfile(arena_db_path) else None
        self.connection, self.arena_cache = None, {}
        if cards:
            self.update(cards)

    @classmethod
    def from_path(cls, path, arena_db_path=None):
        if not path or not os.path.exists(path):
            return cls(arena_db_path=arena_db_path)
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle), arena_db_path)

    def update(self, cards):
        values = cards.values() if isinstance(cards, dict) else cards
        for card in values or []:
            if not isinstance(card, dict):
                continue
            for key in ("grpId", "arenaId", "id"):
                if card.get(key) is not None:
                    self.by_id[str(card[key])] = card
            if card.get("name"):
                self.by_name[str(card["name"]).lower()] = card

    def resolve(self, obj):
        if not isinstance(obj, dict):
            return None
        grp_id, found = None, None
        for key in ("grpId", "arenaId", "cardId"):
            if obj.get(key) is not None:
                grp_id = str(obj[key])
                found = self.by_id.get(grp_id)
                break
        if found is None:
            name = obj.get("name") or obj.get("cardName") or obj.get("label")
            found = self.by_name.get(str(name).lower()) if name else None
        if not grp_id or not self.arena_db_path:
            return found
        if grp_id not in self.arena_cache:
            self.arena_cache[grp_id] = self._query_arena(grp_id) or {}
        if not self.arena_cache[grp_id]:
            return found
        merged = dict(found or {})
        merged.update(self.arena_cache[grp_id])
        return merged

    def _query_arena(self, grp_id):
        try:
            if self.connection is None:
                uri = "file:%s?mode=ro" % self.arena_db_path.replace("\\", "/")
                self.connection = sqlite3.connect(uri, uri=True)
            row = self.connection.execute(
                "SELECT OldSchoolManaText, AbilityIds FROM Cards WHERE GrpId=?",
                (int(grp_id),)).fetchone()
            if not row:
                return None
            mana_cost, ability_ids = row
            texts = []
            for item in str(ability_ids or "").split(","):
                ability, sep, loc = item.partition(":")
                result = None
                if sep and loc.strip().isdigit():
                    result = self.connection.execute(
                        "SELECT Loc FROM Localizations_enUS WHERE LocId=? "
                        "ORDER BY Formatted DESC LIMIT 1",
                        (int(loc),)).fetchone()
                if result and result[0]:
                    clean = re.sub(r"<[^>]*>", "", str(result[0])).strip()
                    if clean and clean not in texts:
                        texts.append(clean)
            return {"manaCost": mana_cost or None,
                    "oracleText": "\n".join(texts) or None}
        except (sqlite3.Error, OSError, ValueError):
            return None
