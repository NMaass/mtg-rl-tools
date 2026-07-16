"""Complete-game collectors for recurrent model training."""
from __future__ import annotations

import json
import os

from . import train_jepa as core


def collect_complete_decision_games(inputs, game_key,
                                    max_decisions=100000):
    """Collect compiled decisions without truncating or sampling inside games.

    Input streams are expected in chronological game order, as produced by the
    repository bundle writers. The limit is a soft upper bound: the first game
    and the final accepted game are retained whole.
    """
    accepted = []
    current = []
    current_key = None
    unknown = 0
    limit = int(max_decisions) if max_decisions is not None else -1
    stopped = False

    def flush():
        nonlocal stopped
        if not current:
            return
        if limit > 0 and accepted and len(accepted) + len(current) > limit:
            stopped = True
            return
        accepted.extend(current)

    for record in core._iter_all_decisions(inputs):
        key = game_key(record)
        if key is None:
            key = "unknown:%d" % unknown
            unknown += 1
        if current_key is not None and key != current_key:
            flush()
            if stopped:
                break
            current = []
        current_key = key
        current.append(record)
    if not stopped:
        flush()
    return accepted, _card_metadata(inputs), {
        "unit": "complete-game",
        "softLimit": limit,
        "acceptedDecisions": len(accepted),
        "truncatedAtGameBoundary": stopped,
        "unknownGameRecords": unknown,
    }


def _card_metadata(inputs):
    cards = {}
    for path in inputs:
        if not os.path.isdir(path):
            continue
        cache = os.path.join(path, "card_cache.json")
        if not os.path.isfile(cache):
            continue
        try:
            with open(cache, encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                cards.update(payload)
        except (OSError, ValueError):
            pass
    return cards
