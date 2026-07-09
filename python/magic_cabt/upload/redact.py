"""Redaction helpers for opt-in training-log uploads."""

SENSITIVE_KEYS = {
    "accountid", "accountidstr", "useremail", "email", "token",
    "accesstoken", "refreshToken".lower(), "auth", "password",
    "rawlogpath", "logpath", "deviceid", "machineid",
}

PSEUDONYM_KEYS = {"player", "playername", "opponent", "opponentname", "matchid"}

__all__ = ["redact_json_value", "redact_decision_record"]


def redact_decision_record(record):
    """Return a redacted copy of one DecisionRecord-like object."""
    return redact_json_value(record)


def redact_json_value(value):
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
                redacted[key] = _stable_pseudonym(child)
            else:
                redacted[key] = redact_json_value(child)
        return redacted
    if isinstance(value, list):
        return [redact_json_value(item) for item in value]
    return value


def _normalize_key(key):
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _stable_pseudonym(value):
    text = str(value)
    if text in ("", "None"):
        return text
    # Non-cryptographic, but deterministic and dependency-free. The uploader's
    # goal is removing readable identifiers from default bundles, not providing
    # irreversible anonymization guarantees.
    acc = 2166136261
    for char in text:
        acc ^= ord(char)
        acc = (acc * 16777619) & 0xFFFFFFFF
    return "id_%08x" % acc
