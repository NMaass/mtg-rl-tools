"""Python side of the CABT-style XMage bridge.

Only the pieces that parse the Java bridge's stable output formats live here
so far: static card data (Task 21) and the JSONL transition dataset
(Task 22). The subprocess protocol client (Tasks 18-19) is not built yet;
until it exists, callers hand these functions the documents the Java side
produced.
"""

from .card_data import all_card_data, cards_by_id, cards_by_name
from .dataset import read_dataset

__all__ = [
    "all_card_data",
    "cards_by_id",
    "cards_by_name",
    "read_dataset",
]
