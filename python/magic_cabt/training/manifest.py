"""Dataset manifest builder for training/evaluation runs.

A manifest is a compact, reproducible summary of a DecisionRecord stream:
schema/source counts, prompt/action distributions, option-count statistics,
reward coverage, terminal coverage, observed causal-factor ranges, and a
training-readiness inventory. It is small enough to commit alongside generated
datasets and rich enough to catch schema or corpus-coverage drift before
training starts.
"""

import json

from magic_cabt.analysis.schema import decision_fingerprint

from .action_dedup import canonical_groups
from .causal import FACTOR_NAMES, causal_variables
from .io import iter_decision_records
from .macro_actions import classify_decision
from .records import validate_record

__all__ = [
    "build_manifest",
    "write_manifest",
]

_CARD_IDENTIFIER_KEYS = (
    "grpId", "arenaId", "mtgaId", "oracleId", "scryfallId",
)
_CARD_CONTEXT_KEYS = frozenset((
    "manaCost", "oracleText", "rulesText", "types", "subtypes", "typeLine",
    "power", "toughness", "loyalty", "zone", "cardNumber", "setCode",
))
_FINGERPRINT_TRACKING_LIMIT = 250000
_DECK_IDENTIFIER_KEYS = (
    "deckId", "deck0Id", "deck1Id", "opponentDeckId", "playerDeckId",
)


def build_manifest(records, name=None):
    """Return a JSON-serializable manifest for an iterable of records."""
    summary = _empty_manifest(name)
    for record in records:
        _add_record(summary, record)
    _finalize(summary)
    return summary


def write_manifest(path, manifest):
    """Write ``manifest`` to ``path`` as deterministic pretty JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _empty_manifest(name):
    return {
        "name": name,
        "schema": "DecisionRecord-v1",
        "records": 0,
        "validRecords": 0,
        "invalidRecords": 0,
        "sources": {},
        "promptTypes": {},
        "optionTypes": {},
        "selectedCount": {},
        "optionCount": {
            "min": None,
            "max": None,
            "sum": 0,
            "mean": None,
        },
        "terminalRecords": 0,
        "rewardRecords": 0,
        "resultRecords": 0,
        "captureConfidence": {},
        "causalFactors": {
            name: {"observed": 0, "min": None, "max": None}
            for name in FACTOR_NAMES
        },
        "inventory": {
            "games": {
                "known": 0,
                "unknownRecords": 0,
            },
            "decisions": {
                "roles": {
                    "root": 0,
                    "parameter": 0,
                    "payment": 0,
                    "ordering": 0,
                },
                "strategicRootRecords": 0,
                "deterministicRecords": 0,
                "noOptionRecords": 0,
                "singleSemanticOptionRecords": 0,
                "nontrivialSemanticOptionRecords": 0,
                "trainableSingleChoiceRecords": 0,
                "semanticOptionCount": {
                    "min": None,
                    "max": None,
                    "sum": 0,
                    "mean": None,
                },
            },
            "transitions": {
                "nonTerminalRecords": 0,
                "nonTerminalWithNextObservation": 0,
                "nonTerminalNextObservationRate": None,
                "terminalOutcomeRecords": 0,
            },
            "duplicates": {
                "uniquePublicFingerprintsTracked": 0,
                "duplicateRecordsObserved": 0,
                "trackingLimit": _FINGERPRINT_TRACKING_LIMIT,
                "trackingTruncated": False,
            },
            "cards": {
                "uniqueObservedIdentifiers": 0,
                "recordsWithObservedIdentifiers": 0,
                "identifierKinds": {},
            },
            "decks": {
                "uniqueIdentifiers": 0,
                "recordsWithIdentifiers": 0,
            },
        },
        "firstErrors": [],
        "_inventorySets": {
            "games": set(),
            "fingerprints": {},
            "cards": set(),
            "decks": set(),
        },
    }


def _add_record(summary, record):
    summary["records"] += 1
    errors = validate_record(record)
    if errors:
        summary["invalidRecords"] += 1
        if len(summary["firstErrors"]) < 5:
            summary["firstErrors"].append({
                "record": summary["records"] - 1,
                "line": record.get("__source_line") if isinstance(record, dict) else None,
                "messages": errors,
            })
    else:
        summary["validRecords"] += 1

    if not isinstance(record, dict):
        return

    _inc(summary["sources"], record.get("source") or "UNKNOWN")
    metadata = record.get("metadata") or {}
    if isinstance(metadata, dict):
        _inc(summary["captureConfidence"],
             metadata.get("captureConfidence") or "UNKNOWN")

    game = _game_identity(record)
    if game is None:
        summary["inventory"]["games"]["unknownRecords"] += 1
    else:
        summary["_inventorySets"]["games"].add(game)

    select = record.get("select") or \
        (record.get("observation") or {}).get("select") or {}
    options = []
    if isinstance(select, dict):
        _inc(summary["promptTypes"], select.get("type") or "UNKNOWN")
        options = select.get("option") or []
        if isinstance(options, list):
            _add_option_count(summary["optionCount"], len(options))
            for option in options:
                if isinstance(option, dict):
                    _inc(summary["optionTypes"], option.get("type") or "UNKNOWN")
        else:
            options = []

    selected = record.get("selectedIndices")
    if isinstance(selected, list):
        _inc(summary["selectedCount"], str(len(selected)))

    _add_decision_inventory(summary, record, select, options, selected)
    _add_transition_inventory(summary, record)
    _add_duplicate_inventory(summary, record)
    _add_card_inventory(summary, record, select)
    _add_deck_inventory(summary, record, metadata)

    if record.get("terminal") is True:
        summary["terminalRecords"] += 1
    if record.get("reward") is not None:
        summary["rewardRecords"] += 1
    if record.get("result") is not None:
        summary["resultRecords"] += 1

    factors = causal_variables(record)
    for name, value in factors.items():
        if isinstance(value, (int, float)):
            _add_factor(summary["causalFactors"][name], value)


def _add_decision_inventory(summary, record, select, options, selected):
    inventory = summary["inventory"]["decisions"]
    if not isinstance(select, dict) or not options:
        inventory["noOptionRecords"] += 1
        return
    classification = classify_decision(record)
    role = classification.get("role") or "root"
    if role not in inventory["roles"]:
        inventory["roles"][role] = 0
    inventory["roles"][role] += 1
    if classification.get("deterministic"):
        inventory["deterministicRecords"] += 1

    groups = canonical_groups(select)
    semantic_count = len(groups)
    if role == "root" and semantic_count > 1 and not classification.get("deterministic"):
        inventory["strategicRootRecords"] += 1
    _add_option_count(inventory["semanticOptionCount"], semantic_count)
    if semantic_count <= 1:
        inventory["singleSemanticOptionRecords"] += 1
    else:
        inventory["nontrivialSemanticOptionRecords"] += 1
    selected_valid = (
        isinstance(selected, list) and len(selected) == 1 and
        isinstance(selected[0], int) and not isinstance(selected[0], bool) and
        0 <= selected[0] < len(options))
    if (select.get("minCount") == 1 and select.get("maxCount") == 1 and
            selected_valid and semantic_count > 1):
        inventory["trainableSingleChoiceRecords"] += 1


def _add_transition_inventory(summary, record):
    inventory = summary["inventory"]["transitions"]
    if record.get("terminal") is True:
        if record.get("reward") is not None or record.get("result") is not None:
            inventory["terminalOutcomeRecords"] += 1
        return
    inventory["nonTerminalRecords"] += 1
    if isinstance(record.get("nextObservation"), dict):
        inventory["nonTerminalWithNextObservation"] += 1


def _add_duplicate_inventory(summary, record):
    fingerprint = decision_fingerprint(record)
    digest = str(fingerprint).split(":")[-1]
    try:
        key = int(digest[:16], 16)
    except ValueError:
        key = str(fingerprint)
    counts = summary["_inventorySets"]["fingerprints"]
    if key in counts:
        counts[key] += 1
    elif len(counts) < _FINGERPRINT_TRACKING_LIMIT:
        counts[key] = 1
    else:
        summary["inventory"]["duplicates"]["trackingTruncated"] = True


def _add_card_inventory(summary, record, select):
    identifiers = set()
    kinds = {}
    _collect_card_identifiers(record.get("observation"), identifiers, kinds)
    _collect_card_identifiers(select, identifiers, kinds)
    if identifiers:
        summary["inventory"]["cards"]["recordsWithObservedIdentifiers"] += 1
        summary["_inventorySets"]["cards"].update(identifiers)
        for kind, values in kinds.items():
            bucket = summary["inventory"]["cards"]["identifierKinds"]
            bucket.setdefault(kind, set()).update(values)


def _add_deck_inventory(summary, record, metadata):
    identifiers = set()
    for source in (record, metadata):
        if not isinstance(source, dict):
            continue
        for key in _DECK_IDENTIFIER_KEYS:
            value = source.get(key)
            if value not in (None, ""):
                identifiers.add("id:%s" % value)
        values = source.get("deckIds")
        if isinstance(values, list):
            identifiers.update("id:%s" % value for value in values
                               if value not in (None, ""))
    if identifiers:
        summary["inventory"]["decks"]["recordsWithIdentifiers"] += 1
        summary["_inventorySets"]["decks"].update(identifiers)


def _collect_card_identifiers(value, identifiers, kinds):
    if isinstance(value, list):
        for child in value:
            _collect_card_identifiers(child, identifiers, kinds)
        return
    if not isinstance(value, dict):
        return

    found = False
    for key in _CARD_IDENTIFIER_KEYS:
        identifier = value.get(key)
        if identifier not in (None, ""):
            token = "%s:%s" % (key, identifier)
            identifiers.add(token)
            kinds.setdefault(key, set()).add(token)
            found = True
    if not found and any(key in value for key in _CARD_CONTEXT_KEYS):
        name = value.get("canonicalName") or value.get("cardName") or value.get("name")
        if name not in (None, ""):
            set_code = value.get("setCode")
            number = value.get("cardNumber")
            suffix = ":%s:%s" % (set_code or "", number or "")
            token = "name:%s%s" % (name, suffix)
            identifiers.add(token)
            kinds.setdefault("name", set()).add(token)
    for child in value.values():
        _collect_card_identifiers(child, identifiers, kinds)


def _game_identity(record):
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    observation = record.get("observation") if isinstance(record.get("observation"), dict) else {}
    current = observation.get("current") if isinstance(observation.get("current"), dict) else {}
    match = (record.get("gameId") or record.get("matchId") or
             metadata.get("matchId") or current.get("matchId"))
    game = record.get("gameNumber")
    if game is None:
        game = metadata.get("gameNumber")
    if game is None:
        game = current.get("gameNumber")
    instance = current.get("gameInstance")
    if match is None and game is None and instance is None:
        return None
    return json.dumps([match, game, instance], separators=(",", ":"))


def _finalize(summary):
    option_count = summary["optionCount"]
    if summary["records"]:
        option_count["mean"] = option_count["sum"] / float(summary["records"])

    inventory = summary["inventory"]
    sets = summary.pop("_inventorySets")
    inventory["games"]["known"] = len(sets["games"])

    semantic = inventory["decisions"]["semanticOptionCount"]
    observed = sum(inventory["decisions"]["roles"].values())
    if observed:
        semantic["mean"] = semantic["sum"] / float(observed)

    transitions = inventory["transitions"]
    if transitions["nonTerminalRecords"]:
        transitions["nonTerminalNextObservationRate"] = (
            transitions["nonTerminalWithNextObservation"] /
            float(transitions["nonTerminalRecords"]))

    fingerprints = sets["fingerprints"]
    inventory["duplicates"]["uniquePublicFingerprintsTracked"] = len(fingerprints)
    inventory["duplicates"]["duplicateRecordsObserved"] = sum(
        count - 1 for count in fingerprints.values() if count > 1)

    inventory["cards"]["uniqueObservedIdentifiers"] = len(sets["cards"])
    inventory["cards"]["identifierKinds"] = {
        key: len(values)
        for key, values in sorted(
            inventory["cards"]["identifierKinds"].items())
    }
    inventory["decks"]["uniqueIdentifiers"] = len(sets["decks"])


def _inc(bucket, key):
    key = str(key)
    bucket[key] = bucket.get(key, 0) + 1


def _add_option_count(stats, count):
    stats["min"] = count if stats["min"] is None else min(stats["min"], count)
    stats["max"] = count if stats["max"] is None else max(stats["max"], count)
    stats["sum"] += count


def _add_factor(stats, value):
    stats["observed"] += 1
    stats["min"] = value if stats["min"] is None else min(stats["min"], value)
    stats["max"] = value if stats["max"] is None else max(stats["max"], value)


def _records_from_path(path, source_hint=None):
    return iter_decision_records(path, source_hint=source_hint)
