"""Perceptual signatures of card art, for recognising cards on screen.

A card on MTGO's battlefield is about a hundred pixels wide, and its art
window perhaps a hundred by fifty. That is far too small for the gradient
hashes usually used to match card scans -- tried on this footage, a 16x16
dHash put the right card outside the top two for three cards in four,
because at that size the gradient structure it measures is mostly the
capture's compression noise.

What survives at that size is *colour*. Magic art is strongly and
distinctively coloured, and a coarse grid of average colour still separates
one card from another when nothing finer does. Measured on the same four
cards, a 6x6 colour signature identified all four, each beating the runner-up
by a factor of about two.

Signatures are built from Scryfall's `art_crop` -- the same window MTGO
shows -- and only for the cards a match has actually named, which is a few
dozen rather than the tens of thousands in the catalog. That is what makes
this a closed-set problem: not "which card is this" but "which of the cards
this game is known to contain is this, or none of them".
"""

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .catalog import DEFAULT_CATALOG

DEFAULT_ART_CACHE = os.path.join(os.path.dirname(DEFAULT_CATALOG),
                                 "art_signatures.json")
SCRYFALL_NAMED = "https://api.scryfall.com/cards/named?exact="
USER_AGENT = "mtg-rl-tools/0.1 (MTGO video ingestion)"

# The grid the signature is averaged over. Coarse on purpose: finer grids
# start measuring the capture's compression rather than the art.
GRID = 6
# Signatures are brightness-normalised before comparison, because MTGO draws
# card art dimmer than Scryfall serves it and a whole-image brightness offset
# is not a difference in what the card is.
_TARGET_MEAN = 128.0


def signature(image, grid: int = GRID) -> List[float]:
    """A brightness-normalised grid of average colour."""
    from PIL import Image

    small = image.convert("RGB").resize((grid, grid), Image.LANCZOS)
    pixels = list(small.getdata())
    mean = sum(sum(pixel) for pixel in pixels) / float(3 * len(pixels)) or 1.0
    return [channel * _TARGET_MEAN / mean
            for pixel in pixels for channel in pixel]


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Mean absolute difference between two signatures."""
    if not a or not b or len(a) != len(b):
        return float("inf")
    return sum(abs(x - y) for x, y in zip(a, b)) / float(len(a))


class ArtIndex:
    """Signatures for a run's vocabulary, cached between runs."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or DEFAULT_ART_CACHE
        self.signatures: Dict[str, List[float]] = {}
        self.missing: List[str] = []
        if os.path.exists(self.path):
            try:
                with open(self.path) as handle:
                    stored = json.load(handle)
                self.signatures = {k: v for k, v in stored.get("cards", {}).items()
                                   if isinstance(v, list)}
                self.missing = list(stored.get("missing", []))
            except (ValueError, OSError):
                self.signatures, self.missing = {}, []

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as handle:
            json.dump({"cards": self.signatures, "missing": self.missing},
                      handle)

    def ensure(self, names: Iterable[str], pause: float = 0.12,
               log=None) -> Tuple[int, int]:
        """Fetch art for any of these names not already known.

        Returns (fetched, failed). Names that Scryfall has no art for are
        remembered as missing so a later run does not ask again.
        """
        fetched = failed = 0
        for name in sorted({n for n in names if n}):
            if name in self.signatures or name in self.missing:
                continue
            art = self._fetch(name)
            if art is None:
                self.missing.append(name)
                failed += 1
            else:
                self.signatures[name] = signature(art)
                fetched += 1
            if log:
                log(name, art is not None)
            time.sleep(pause)
        if fetched or failed:
            self.save()
        return fetched, failed

    def _fetch(self, name: str):
        from PIL import Image
        import io

        request = urllib.request.Request(
            SCRYFALL_NAMED + urllib.parse.quote(name),
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            card = json.load(urllib.request.urlopen(request, timeout=30))
        except Exception:
            return None
        uris = card.get("image_uris")
        if not uris:
            faces = card.get("card_faces") or []
            uris = (faces[0].get("image_uris") if faces else None) or {}
        url = uris.get("art_crop") or uris.get("small")
        if not url:
            return None
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
                    timeout=30) as response:
                return Image.open(io.BytesIO(response.read()))
        except Exception:
            return None

    def match(self, art, limit: float = 30.0, margin: float = 1.25,
              min_detail: float = 20.0) -> Tuple[Optional[str], Optional[float]]:
        """The card this art is, or None.

        Three gates, all necessary.

        The first is that the crop has to contain a picture at all. An empty
        stretch of battlefield is flat grey, and flat grey is closest to
        whichever card in the vocabulary has the flattest art -- so without
        this, every empty region confidently reports the same wrong card.
        Measured on one frame, real card art stood at 28-55 standard
        deviations of brightness and bare panel or overlapped slivers at
        1.6-14.

        The absolute distance gate rejects art that is not any card in the
        vocabulary -- the interesting case, since a permanent on screen that
        the log never mentioned is the evidence this exists to find. The
        margin rejects a match no better than the next candidate, for the
        same reason the card catalog demands one.
        """
        if not self.signatures:
            return None, None
        from PIL import ImageStat

        if ImageStat.Stat(art.convert("L")).stddev[0] < min_detail:
            return None, None
        query = signature(art)
        scored = sorted((distance(query, value), name)
                        for name, value in self.signatures.items())
        best, name = scored[0]
        if best > limit:
            return None, best
        if len(scored) > 1 and scored[1][0] < best * margin:
            return None, best
        return name, best
