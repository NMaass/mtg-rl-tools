"""Local card-name resolver backed by a Scryfall bulk-data JSON file."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import os

import requests
from rapidfuzz import fuzz, process


@dataclass
class CardResolution:
    query: str
    name: Optional[str]
    score: float
    oracle_id: Optional[str] = None


class CardNameResolver:
    def __init__(self, records: Iterable[Dict[str, Any]]):
        self.by_name: Dict[str, Dict[str, Any]] = {}
        for row in records:
            name = row.get("name")
            if not name:
                continue
            self.by_name.setdefault(str(name), row)
            for face in row.get("card_faces") or []:
                if face.get("name"):
                    self.by_name.setdefault(str(face["name"]), row)
        self.names = sorted(self.by_name)
        self.casefold_names = {name.casefold(): name for name in self.names}

    @classmethod
    def from_json(cls, path: str) -> "CardNameResolver":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "data" in payload:
            payload = payload["data"]
        if not isinstance(payload, list):
            raise ValueError("card database JSON must contain a list")
        return cls(payload)

    def resolve(self, text: str, minimum_score: float = 72.0) -> CardResolution:
        query = " ".join(str(text).split())
        exact = self.casefold_names.get(query.casefold())
        if exact:
            row = self.by_name[exact]
            return CardResolution(query, exact, 100.0, row.get("oracle_id"))
        match = process.extractOne(query, self.names, scorer=fuzz.WRatio)
        if not match or match[1] < minimum_score:
            return CardResolution(query, None, float(match[1]) if match else 0.0)
        name, score, _ = match
        row = self.by_name[name]
        return CardResolution(query, name, float(score), row.get("oracle_id"))


def download_scryfall_bulk(destination: str, bulk_type: str = "oracle_cards",
                            user_agent: str = "mtgo-video-parser/0.1") -> str:
    """Download one Scryfall bulk file with explicit headers and no API loop."""
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json;q=0.9,*/*;q=0.8",
    }
    metadata = requests.get("https://api.scryfall.com/bulk-data",
                            headers=headers, timeout=60)
    metadata.raise_for_status()
    entries = metadata.json().get("data") or []
    row = next((item for item in entries if item.get("type") == bulk_type), None)
    if row is None:
        raise ValueError(f"Scryfall bulk type not found: {bulk_type}")
    response = requests.get(row["download_uri"], headers=headers,
                            timeout=300, stream=True)
    response.raise_for_status()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                handle.write(chunk)
    os.replace(temporary, target)
    return str(target)
