"""Python side of the CABT-style XMage bridge.

Two layers live here:

- parsers for the Java bridge's stable output formats: static card data
  (offline from exported documents) and the JSONL transition dataset;
- the live-game protocol client (``CabtBridge``): a subprocess loop over
  the Java ``CabtProtocolServer`` with the CABT-parity commands
  ``game_start`` / ``game_select`` / ``game_finish`` / ``all_card_data`` /
  ``visualize_data``, plus the card-identity commands ``resolve_card`` /
  ``validate_deck`` / ``repository_card_data`` (which need no active game).

Note: ``all_card_data`` is exported in two senses —
  ``from magic_cabt import all_card_data``  parses an offline card-data
  document (the static export from earlier tasks), while
  ``bridge.all_card_data()``  requests live, game-scoped deck-pool metadata
  through the subprocess protocol.  The latter requires an active game.
"""

from .arena_log import ArenaLogNormalizer, iter_log_entries, normalize_arena_log
from .card_data import all_card_data, cards_by_id, cards_by_name
from .dataset import read_dataset
from .protocol import (
    CabtBridge,
    CabtGameError,
    CabtProtocolError,
    load_decklist,
    parse_decklist,
)

__all__ = [
    "ArenaLogNormalizer",
    "all_card_data",
    "cards_by_id",
    "cards_by_name",
    "iter_log_entries",
    "normalize_arena_log",
    "read_dataset",
    "CabtBridge",
    "CabtGameError",
    "CabtProtocolError",
    "load_decklist",
    "parse_decklist",
]
