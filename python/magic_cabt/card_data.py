"""Static Magic card metadata: the CABT all_card_data() equivalent.

The Java bridge answers the protocol command ``{"command": "all_card_data"}``
with ``{"ok": true, "cards": [...]}`` (see MagicCardDataProtocolCommand).
The subprocess transport that will carry that command (Task 18) is not built
yet, so ``all_card_data`` parses a response document directly: JSON text, a
parsed dict, or an open file.

Card data is reference data. Legal choices always come from live XMage state
through the option-index prompts, never from this metadata.
"""

import json

__all__ = ["all_card_data", "cards_by_id", "cards_by_name"]


def all_card_data(source):
    """Return the list of card dicts from an all_card_data response.

    ``source`` may be JSON text (str or bytes), an already-parsed response
    dict, or a readable file object. Raises ``ValueError`` when the response
    is malformed or reports ``ok: false`` — a failed export must not be
    silently treated as an empty card pool.
    """
    if isinstance(source, (str, bytes)):
        response = json.loads(source)
    elif isinstance(source, dict):
        response = source
    elif hasattr(source, "read"):
        response = json.load(source)
    else:
        raise TypeError(
            "source must be JSON text, a response dict, or a file object, "
            "not %r" % type(source).__name__
        )
    if not isinstance(response, dict):
        raise ValueError("card data response must be a JSON object")
    if response.get("ok") is not True:
        raise ValueError(
            "card data response is not ok: %s" % response.get("error", "no error message")
        )
    cards = response.get("cards")
    if not isinstance(cards, list):
        raise ValueError("card data response has no cards list")
    return cards


def cards_by_id(cards):
    """Index card dicts by their ``cardId`` for joining observation object IDs."""
    return {card["cardId"]: card for card in cards if "cardId" in card}


def cards_by_name(cards):
    """Index card dicts by name for joining observation card names.

    Multiple printings share a name; the first occurrence wins.
    """
    index = {}
    for card in cards:
        name = card.get("name")
        if name is not None and name not in index:
            index[name] = card
    return index
