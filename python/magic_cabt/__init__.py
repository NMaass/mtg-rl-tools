"""Python side of the CABT-style XMage bridge.

Two layers live here:

- parsers for the Java bridge's stable output formats: static card data
  and the JSONL transition dataset;
- the live-game protocol client (``CabtBridge``): a subprocess loop over
  the Java ``CabtProtocolServer`` with the CABT-parity commands
  ``game_start`` / ``game_select`` / ``game_finish`` / ``all_card_data`` /
  ``visualize_data``.
"""

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
    "all_card_data",
    "cards_by_id",
    "cards_by_name",
    "read_dataset",
    "CabtBridge",
    "CabtGameError",
    "CabtProtocolError",
    "load_decklist",
    "parse_decklist",
]
