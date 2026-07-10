"""Redaction helpers for opt-in training-log uploads."""

SENSITIVE_KEYS = {
    "accountid", "accountidstr", "useremail", "email", "token",
    "accesstoken", "refreshtoken", "auth", "password",
    "rawlogpath", "logpath", "deviceid", "machineid",
    # Titles embed free-form player names ("Alice vs Bob — Win ..."), which
    # key-based pseudonymization cannot catch inside strings, so drop them.
    "title",
    # Deck names are user-authored free text and can identify the player.
    "deckname",
}

# Values under these keys are identifiers: scalars are pseudonymized, list
# elements are pseudonymized one by one, and dict values (player objects such
# as summary.json's `opponent`) are recursed into with only identifying
# fields pseudonymized, so colors/archetype/etc. survive.
PSEUDONYM_KEYS = {
    "player", "playername", "opponent", "opponentname",
    "matchid", "matchids", "gameid", "gameids",
}

# Keys whose dict/list values describe players (recorder summary.json nests
# {"you": {"name": ...}}; observation snapshots carry a `players` list).
# Inside such objects, PLAYER_NAME_KEYS hold real Arena screen names.
PLAYER_OBJECT_KEYS = {"you", "opponent", "player", "players"}
PLAYER_NAME_KEYS = {"name", "screenname", "playername", "opponentname"}

__all__ = ["redact_json_value", "redact_decision_record"]


def redact_decision_record(record):
    """Return a redacted copy of one DecisionRecord-like object."""
    return redact_json_value(record)


def redact_json_value(value, _in_player_object=False):
    """Recursively redact sensitive JSON fields.

    This keeps game-state/action content while dropping obvious identifiers and
    credentials. It is intentionally conservative and deterministic so uploaded
    bundles can be audited locally before sending.
    """
    if isinstance(value, dict):
        redacted = {}
        for key, child in value.items():
            normalized = _normalize_key(key)
            if normalized in SENSITIVE_KEYS:
                redacted[key] = "<redacted>"
            elif normalized in PSEUDONYM_KEYS:
                redacted[key] = _pseudonymize_value(child)
            elif _in_player_object and normalized in PLAYER_NAME_KEYS:
                redacted[key] = _pseudonymize_value(child)
            elif normalized in PLAYER_OBJECT_KEYS:
                redacted[key] = redact_json_value(child, _in_player_object=True)
            else:
                redacted[key] = redact_json_value(child)
        return redacted
    if isinstance(value, list):
        return [redact_json_value(item, _in_player_object=_in_player_object)
                for item in value]
    return value


def _pseudonymize_value(value):
    """Pseudonymize an identifier value without destroying structure.

    Scalars become stable pseudonyms; lists are pseudonymized element-wise
    (e.g. summary.json's `matchIds`); dicts (player objects) are recursed
    into so only identifying fields are replaced and fields like colors or
    archetype are preserved.
    """
    if isinstance(value, dict):
        return redact_json_value(value, _in_player_object=True)
    if isinstance(value, list):
        return [_pseudonymize_value(item) for item in value]
    if value is None:
        return None
    return _stable_pseudonym(value)


def _normalize_key(key):
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _stable_pseudonym(value):
    text = str(value)
    if text == "":
        return text
    # Non-cryptographic, but deterministic and dependency-free. The uploader's
    # goal is removing readable identifiers from default bundles, not providing
    # irreversible anonymization guarantees.
    acc = 2166136261
    for char in text:
        acc ^= ord(char)
        acc = (acc * 16777619) & 0xFFFFFFFF
    return "id_%08x" % acc
