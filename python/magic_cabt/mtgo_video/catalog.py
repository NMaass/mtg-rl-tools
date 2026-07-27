"""A local, offline card-name catalog for resolving OCR'd names.

Resolving each OCR string with a live Scryfall fuzzy lookup is the wrong
shape for this pipeline: it is rate limited, it needs the network on every
run, and -- worst -- it is not deterministic across runs, so the same
footage could decode to different card names. Scryfall's fuzzy endpoint also
refuses names it cannot match at all ("Bojuko Bog" is a 404), which loses a
card that the very same log spells correctly elsewhere.

Instead, download Scryfall's `oracle_cards` bulk file once (one row per
unique card, ~200MB, refreshed daily), condense it to the few fields this
pipeline needs, and match locally:

    1. exact match on a folded key (casefold, strip diacritics and
       punctuation) -- catches the overwhelming majority, including
       "Lorien Revealed" vs "Lórien Revealed";
    2. fuzzy match against the *run vocabulary* -- the cards already seen in
       this log -- which is a few dozen names rather than thirty thousand,
       so a garbled reading snaps onto the correct card the log already
       spelled properly;
    3. fuzzy match against the whole catalog, with a length-bucketed
       candidate set to keep it cheap.

Every resolution is memoized per run, so an OCR string always decodes to the
same card no matter how often it appears.
"""

import difflib
import gzip
import json
import os
import re
import time
import unicodedata
import urllib.request
from typing import Dict, Iterable, List, Optional

BULK_INDEX = "https://api.scryfall.com/bulk-data"
DEFAULT_CATALOG = os.path.join(
    os.path.expanduser("~"), ".cache", "magic_cabt", "card_catalog.json.gz")

_HEADERS = {
    # Scryfall requires both and answers 400 without them.
    "User-Agent": "mtg-rl-tools-mtgo-video/0.1",
    "Accept": "application/json",
}

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")

# Scryfall layouts that are not cards a game log can refer to. Tokens are
# kept: they do appear on the battlefield.
SKIP_LAYOUTS = frozenset({"art_series", "emblem", "scheme", "planar",
                          "vanguard", "augment", "host"})


def _informative(entry: dict) -> bool:
    """Whether an entry carries a real type line rather than a placeholder."""
    return (entry.get("type_line") or "").strip() not in ("", "Card")


def fold(name: str) -> str:
    """Normalize a card name for matching: casefold, drop accents/punctuation."""
    text = unicodedata.normalize("NFD", name.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def _condense(card: dict) -> dict:
    faces = card.get("card_faces") or []
    entry = {
        "name": card.get("name"),
        "type_line": card.get("type_line", ""),
        "power": card.get("power"),
        "toughness": card.get("toughness"),
        "colors": "".join(card.get("colors") or []),
    }
    if faces:
        entry["faces"] = [
            {"name": f.get("name"), "type_line": f.get("type_line", ""),
             "power": f.get("power"), "toughness": f.get("toughness")}
            for f in faces
        ]
    return entry


def bulk_download_uri(kind: str = "oracle_cards") -> str:
    request = urllib.request.Request(BULK_INDEX, headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=60) as resp:
        index = json.loads(resp.read().decode("utf-8"))
    for item in index.get("data", []):
        if item.get("type") == kind:
            return item["download_uri"]
    raise LookupError("no Scryfall bulk file of type %r" % kind)


def build_catalog(path: str = DEFAULT_CATALOG, kind: str = "oracle_cards",
                  mtgo_only: bool = True) -> str:
    """Download Scryfall bulk data and write the condensed local catalog.

    Restricting to cards that exist on MTGO is not just a size win: across
    the full card pool a handful of distinct cards normalize to the same key
    ("Bat"/"Bat-", "Goblin // Soldier"/"Goblin Soldier"), whereas within the
    MTGO-legal pool the normalized keys are unique. It also removes every
    Alchemy "A-" card, which cannot appear in MTGO footage.
    """
    uri = bulk_download_uri(kind)
    request = urllib.request.Request(uri, headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=600) as resp:
        cards = json.loads(resp.read().decode("utf-8"))

    entries: Dict[str, dict] = {}

    def offer(name: str, entry: dict):
        """Index an entry, never replacing a typed card with an untyped one.

        Scryfall's bulk data includes art-series cards that carry a real
        card's name with the placeholder type line "Card". Left unchecked
        they overwrite the playable card and it stops being recognized as a
        permanent.
        """
        existing = entries.get(name)
        if existing is not None and not _informative(entry):
            return
        if existing is not None and _informative(existing) and not _informative(entry):
            return
        entries[name] = entry

    for card in cards:
        if card.get("lang") not in (None, "en"):
            continue
        if card.get("layout") in SKIP_LAYOUTS:
            continue
        if mtgo_only and "mtgo" not in (card.get("games") or []):
            continue
        entry = _condense(card)
        if not entry["name"]:
            continue
        offer(entry["name"], entry)
        # Index each face of a split/MDFC card under its own name too: MTGO
        # logs the face that was played, not the combined name.
        for face in entry.get("faces", []):
            if face.get("name"):
                merged = dict(face)
                merged["colors"] = entry["colors"]
                offer(face["name"], merged)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"source": uri, "builtAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "mtgoOnly": mtgo_only, "cards": entries}
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def is_permanent(info: Optional[dict]) -> bool:
    """Whether a resolved card stays on the battlefield once it resolves."""
    if not info:
        return False
    type_line = (info.get("type_line") or "").split("//")[0]
    return any(
        kind in type_line
        for kind in ("Creature", "Artifact", "Enchantment", "Planeswalker",
                     "Land", "Battle")
    )


def is_creature(info: Optional[dict]) -> bool:
    return bool(info) and "Creature" in (info.get("type_line") or "")


class CardCatalog:
    """Offline card lookup with deterministic fuzzy matching."""

    def __init__(self, path: str = DEFAULT_CATALOG, vocabulary: Iterable[str] = (),
                 payload: Optional[dict] = None):
        if payload is None:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                payload = json.load(f)
        self.source = payload.get("source")
        self.built_at = payload.get("builtAt")
        self.cards: Dict[str, dict] = payload["cards"]
        # Sort so that ties break the same way on every run and machine,
        # rather than following dict insertion order.
        self.by_fold: Dict[str, str] = {}
        for name in sorted(self.cards):
            self.by_fold.setdefault(fold(name), name)
        self._by_length: Dict[int, List[str]] = {}
        for folded in sorted(self.by_fold):
            self._by_length.setdefault(len(folded), []).append(folded)

        # Names known to be in play this run. Resolutions prefer these, so a
        # garbled reading lands on a card the log already spelled correctly.
        self.vocabulary: List[str] = []
        for name in vocabulary:
            self.add_to_vocabulary(name)
        self._memo: Dict[str, Optional[str]] = {}
        self.unresolved: Dict[str, int] = {}

    @classmethod
    def from_cards(cls, cards: Dict[str, dict], **kwargs) -> "CardCatalog":
        """Build a catalog from an in-memory card map (for tests)."""
        return cls(payload={"cards": cards, "source": "memory",
                            "builtAt": None}, **kwargs)

    def add_to_vocabulary(self, name: str):
        canonical = self.by_fold.get(fold(name))
        if canonical and canonical not in self.vocabulary:
            self.vocabulary.append(canonical)

    def _best_two(self, key: str, candidates: Iterable[str]):
        """Top two (score, candidate) pairs, best first."""
        matcher = difflib.SequenceMatcher(autojunk=False)
        matcher.set_seq2(key)
        best = (0.0, None)
        second = (0.0, None)
        for candidate in candidates:
            matcher.set_seq1(candidate)
            # Prune against the *runner-up*, not the leader: the margin
            # between first and second is what decides acceptance, so the
            # second score has to stay accurate.
            if (matcher.real_quick_ratio() <= second[0]
                    or matcher.quick_ratio() <= second[0]):
                continue
            score = matcher.ratio()
            if score > best[0]:
                best, second = (score, candidate), best
            elif score > second[0]:
                second = (score, candidate)
        return best, second

    def _gated(self, key: str, candidates: Iterable[str], cutoff: float,
               margin: float) -> Optional[str]:
        """Best match, accepted only if it also beats the runner-up clearly.

        A high absolute score alone is a poor gate: with tens of thousands of
        names, almost any string has a plausible-looking neighbour, and short
        names sit one or two edits apart in their hundreds. Requiring the top
        match to lead the second by a margin is what separates "this is that
        card, misread" from "this is not a card at all" -- it is what makes a
        garbled player name like "Buz2Caldera" get rejected rather than
        resolved to whatever it happens to look like.
        """
        best, second = self._best_two(key, candidates)
        if best[1] is None or best[0] < cutoff:
            return None
        if best[0] - second[0] < margin:
            return None
        return best[1]

    def resolve(self, name: str,
                vocabulary_cutoff: float = 0.55, vocabulary_margin: float = 0.05,
                catalog_cutoff: float = 0.70, catalog_margin: float = 0.08
                ) -> Optional[str]:
        """Canonical card name for an OCR'd string, or None if unresolvable.

        Matching is tried against the run vocabulary first -- the cards this
        log has already named exactly -- where a much lower threshold is
        safe because the candidate set is a few dozen names known to be in
        play rather than the whole game's card pool.
        """
        key = fold(name)
        if not key:
            return None
        if key in self._memo:
            return self._memo[key]

        canonical = self.by_fold.get(key)
        if canonical is None and self.vocabulary:
            hit = self._gated(key, [fold(v) for v in self.vocabulary],
                              vocabulary_cutoff, vocabulary_margin)
            canonical = self.by_fold.get(hit) if hit else None
        if canonical is None:
            # Only names of a similar length can score above the threshold,
            # so the rest of the catalog can be skipped.
            window = range(max(1, len(key) - 5), len(key) + 6)
            bucket = [f for size in window for f in self._by_length.get(size, ())]
            hit = self._gated(key, bucket, catalog_cutoff, catalog_margin)
            canonical = self.by_fold.get(hit) if hit else None

        if canonical is None:
            self.unresolved[name] = self.unresolved.get(name, 0) + 1
        self._memo[key] = canonical
        return canonical

    def lookup(self, name: str) -> Optional[dict]:
        """Card metadata for an OCR'd name (the CardResolver interface)."""
        canonical = self.resolve(name)
        return self.cards.get(canonical) if canonical else None

    def canonical_name(self, name: str) -> str:
        return self.resolve(name) or name

    def seed_from_log(self, names: Iterable[str]):
        """Lock in the run vocabulary from names that match the catalog exactly.

        Done before any fuzzy resolution so that a correctly spelled sighting
        of a card always wins over a garbled one, regardless of which came
        first in the log.
        """
        for name in names:
            if fold(name) in self.by_fold:
                self.add_to_vocabulary(name)
