"""Match a perceived semantic action to XMage's current legal options."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional
import json
import re

from rapidfuzz import fuzz


@dataclass
class OptionMatch:
    option_index: int
    score: float
    option: Dict[str, Any]
    reasons: List[str]

    def to_dict(self):
        return asdict(self)


def rank_options(action: Dict[str, Any], options: Iterable[Dict[str, Any]],
                 minimum_score: float = 0.0) -> List[OptionMatch]:
    result = []
    for position, option in enumerate(options):
        index = option.get("index")
        if not isinstance(index, int):
            index = position
        score, reasons = _score(action, option)
        if score >= minimum_score:
            result.append(OptionMatch(index, score, dict(option), reasons))
    result.sort(key=lambda row: (-row.score, row.option_index))
    return result


def _score(action: Dict[str, Any], option: Dict[str, Any]):
    reasons = []
    action_type = _normalize_type(action.get("action_type") or
                                  action.get("actionType") or action.get("type"))
    option_type = _normalize_type(option.get("type") or
                                  (option.get("payload") or {}).get("actionType"))
    score = 0.0
    if action_type and option_type:
        if action_type == option_type:
            score += 45.0
            reasons.append("action type exact")
        elif _type_family(action_type) == _type_family(option_type):
            score += 28.0
            reasons.append("action type family")
    action_card = _clean(action.get("card_name") or action.get("cardName") or
                         action.get("card"))
    option_text = _option_text(option)
    if action_card:
        card_score = fuzz.WRatio(action_card, option_text)
        score += 0.38 * card_score
        reasons.append(f"card/text fuzzy={card_score:.1f}")
    targets = action.get("targets") or []
    if targets:
        target_text = " ".join(map(str, targets))
        target_score = fuzz.token_set_ratio(target_text, option_text)
        score += 0.12 * target_score
        reasons.append(f"target fuzzy={target_score:.1f}")
    raw_text = _clean(action.get("text"))
    if raw_text:
        text_score = fuzz.token_set_ratio(raw_text, option_text)
        score += 0.05 * text_score
        reasons.append(f"line fuzzy={text_score:.1f}")
    if not action_type and not action_card:
        score = fuzz.token_set_ratio(json.dumps(action, sort_keys=True), option_text)
        reasons.append("fallback text")
    return min(100.0, score), reasons


def _option_text(option: Dict[str, Any]) -> str:
    payload = option.get("payload") or {}
    pieces = [
        option.get("type"), option.get("label"),
        payload.get("canonicalKey"), payload.get("cardName"),
        payload.get("sourceName"), payload.get("targetName"),
        payload.get("promptType"),
    ]
    return " ".join(_flatten(value) for value in pieces if value is not None)


def _flatten(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    return " ".join(str(value).strip().split()) or None


def _normalize_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")
    aliases = {
        "CAST": "CAST_SPELL",
        "PLAY": "PLAY_LAND",
        "ACTIVATE": "ACTIVATE_ABILITY",
        "ATTACKS": "ATTACK",
        "BLOCKS": "BLOCK",
    }
    return aliases.get(text, text)


def _type_family(value: str) -> str:
    if "CAST" in value:
        return "CAST"
    if "LAND" in value or value == "PLAY":
        return "LAND"
    if "ACTIVAT" in value:
        return "ACTIVATE"
    if "ATTACK" in value:
        return "ATTACK"
    if "BLOCK" in value:
        return "BLOCK"
    if "PASS" in value:
        return "PASS"
    if "MULL" in value or "KEEP" in value:
        return "MULLIGAN"
    return value
