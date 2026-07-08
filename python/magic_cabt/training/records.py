"""Canonical DecisionRecord normalizer + dataset validator (issue #10).

Three input formats feed in here:

1. Java ``CabtDatasetWriter`` JSONL transitions, the format
   ``python/magic_cabt/dataset.py:read_dataset`` returns. Each record already
   carries ``schemaVersion``, top-level ``select`` spec, ``selectedIndices``
   and ``terminal``.
2. ``examples/run_selfplay.py`` replay frames: one line per bridge step
   as ``{sequence, player, observation, selected}`` and a trailing
   ``{result}`` line at game end.
3. Arena ``decisions.jsonl`` bundle records from the arena-mirror recorder:
   each line carries the prompt spec under ``observation.select`` and the
   chosen indices under a top-level ``select`` (the recorder reuses that
   field name for indices; the normalizer repairs the naming clash).

All three become identical ``DecisionRecord v1`` dicts via
``normalize_record`` so that downstream training / evaluation code never has
to branch on source.

The module is intentionally dependency-free: the only stdlib pieces used are
``json`` plus plain container iteration, matching the rest of the package.
"""

import json
import os  # noqa: F401  (kept for callers that import from this module)

__all__ = [
    "SCHEMA_VERSION",
    "CANONICAL_SOURCES",
    "DEFAULT_SOURCE",
    "normalize_record",
    "iter_decision_records",
    "validate_record",
    "validate_records",
    "is_decision_record",
]

SCHEMA_VERSION = 1

# Source labels documented in docs/decision-record-schema.md.
CANONICAL_SOURCES = (
    "arena_human",
    "engine_selfplay",
    "engine_human",
    "search",
)

# Default source when a record cannot tell us where it came from. Engine
# captures (Java transition dataset, self-play replay, future XMage-driven
# human play) all sit on the indexed legal-action protocol and produce exact
# observations, so we default to engine_selfplay rather than guessing human.
DEFAULT_SOURCE = "engine_selfplay"

# Arena decisions.jsonl carries these top-level keys alongside the canonical
# fields. They are preserved under ``metadata`` so the top-level surface stays
# canonical while nothing provenance-relevant is dropped.
_ARENA_PRESERVED_KEYS = (
    "matchId",
    "gameNumber",
    "player",
    "promptTimestamp",
    "responseTimestamp",
    "selectionMatched",
    "promptMessageType",
    "responseMessageType",
    "responsePayload",
)

# Default captureConfidence per source when the writer has not set one.
_DEFAULT_CAPTURE_CONFIDENCE = {
    "arena_human": "mirror",
    "engine_selfplay": "exact",
    "engine_human": "exact",
    "search": "partial",
}


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

def normalize_record(raw, source_hint=None):
    """Return one raw record as a canonical ``DecisionRecord v1`` dict.

    Detection is best-effort. Records that already match the canonical shape
    (Java transition dataset, anything previously normalized) are largely
    passed through with only canonicalization of ``playerIndex`` /
    ``metadata``. The self-play replay frame and Arena decision shapes are
    field-detected by their distinctive keys.

    ``source_hint`` overrides the detected ``source`` when it is one of the
    canonical labels; otherwise it is ignored. This lets a caller route the
    same replay file through ``source_hint="engine_human"`` when it knows
    the live game came from a human instead of an agent.

    Raises ``ValueError`` only when the input cannot possibly be a record
    (not a ``dict``). Missing canonical fields are filled with their null /
    empty defaults so the validator can report them rather than the normalizer
    raising.
    """
    if not isinstance(raw, dict):
        raise ValueError("record is not a JSON object: %r" % (type(raw),))

    source = _resolve_source(raw, source_hint)
    metadata = dict(raw.get("metadata") or {})

    # three shape patterns, in detection order from most-specific to least.
    if _looks_like_self_play_replay(raw):
        normalized = _normalize_self_play_replay(raw, source, metadata)
    elif _looks_like_arena_decision(raw):
        normalized = _normalize_arena_decision(raw, source, metadata)
    elif _looks_like_canonical_or_java(raw):
        normalized = _normalize_canonical_or_java(raw, source, metadata)
    else:
        # Unknown shape: still emit a record so the validator can flag it
        # rather than the normalizer guessing. Preserve everything as
        # observation/metadata and let validate_record report the gaps.
        normalized = _normalize_unknown(raw, source, metadata)

    normalized.setdefault("schemaVersion", SCHEMA_VERSION)
    metadata.setdefault(
        "captureConfidence",
        _DEFAULT_CAPTURE_CONFIDENCE.get(source, "partial"))
    normalized["metadata"] = metadata
    return normalized


def is_decision_record(candidate):
    """True if ``candidate`` already has the canonical v1 surface required for
    validation to make sense (used by callers iterating mixed inputs)."""
    return (
        isinstance(candidate, dict)
        and candidate.get("schemaVersion") == SCHEMA_VERSION
        and "select" in candidate
        and "selectedIndices" in candidate
    )


# --- shape detection --------------------------------------------------------

def _looks_like_self_play_replay(raw):
    # examples/run_selfplay.py writes {sequence, player, observation, selected}
    # plus a trailing {result}. We accept both arms here; the trailing result
    # line is filtered out earlier in iter_decision_records.
    return (
        "sequence" in raw
        and "player" in raw
        and "observation" in raw
        and "selected" in raw
    )


def _looks_like_arena_decision(raw):
    # Arena decisions.jsonl: top-level select holds chosen indices (list of
    # ints), the prompt spec lives under observation.select, and the envelope
    # carries promptMessageType / responseMessageType / selectionMatched.
    top_select = raw.get("select")
    observation = raw.get("observation") or {}
    nested_select = observation.get("select")
    return (
        isinstance(top_select, list)
        and isinstance(nested_select, dict)
        and any(k in raw for k in ("promptMessageType", "selectionMatched",
                                   "responseMessageType"))
    )


def _looks_like_canonical_or_java(raw):
    # Java transition dataset / already-canon records carry schemaVersion +
    # top-level select spec (a dict) + selectedIndices (a list).
    top_select = raw.get("select")
    return (
        "schemaVersion" in raw
        and isinstance(top_select, dict)
        and "selectedIndices" in raw
    )


# --- per-source normalizers --------------------------------------------------

def _normalize_self_play_replay(raw, source, metadata):
    observation = dict(raw.get("observation") or {})
    nested_select = observation.pop("select", None)
    if not isinstance(nested_select, dict):
        nested_select = _ensure_select({})
    return {
        "schemaVersion": SCHEMA_VERSION,
        "source": source,
        "gameId": raw.get("gameId"),
        "sequenceNumber": _as_int(raw.get("sequence"), raw.get("sequenceNumber")),
        "playerIndex": _as_int(raw.get("player"), raw.get("playerIndex"),
                               (nested_select or {}).get("playerIndex")),
        "observation": observation,
        "select": _ensure_select(nested_select),
        "selectedIndices": _as_index_list(raw.get("selected")),
        "nextObservation": raw.get("nextObservation"),
        "terminal": bool(raw.get("terminal")),
        "reward": raw.get("reward"),
        "result": raw.get("result"),
        "metadata": metadata,
    }


def _normalize_arena_decision(raw, source, metadata):
    observation = dict(raw.get("observation") or {})
    nested_select = observation.pop("select", None)
    if not isinstance(nested_select, dict):
        nested_select = _ensure_select({})

    # The arena recorder writes the chosen indices in a top-level ``select``
    # field (a list of ints). The canonical ``select`` field is the prompt
    # spec, so the indices move into ``selectedIndices`` and the prompt spec
    # takes the canonical ``select`` slot.
    selected_indices = raw.get("select")
    if not isinstance(selected_indices, list):
        # defensive: never raise; let the validator grumble if needed.
        selected_indices = raw.get("selectedIndices") or []

    seat = raw.get("seat")
    player_index = seat if isinstance(seat, int) and not isinstance(seat, bool) \
        else raw.get("playerIndex")

    for key in _ARENA_PRESERVED_KEYS:
        if key in raw:
            # ``player`` name collides with the self-play replay field; the
            # arena recorder uses it as a seat *name* rather than a seat
            # index, so we keep the original under metadata.playerName.
            if key == "player":
                metadata.setdefault("playerName", raw[key])
                continue
            metadata.setdefault(key, raw[key])

    return {
        "schemaVersion": SCHEMA_VERSION,
        "source": source,
        "gameId": raw.get("gameId") or metadata.get("matchId"),
        "sequenceNumber": _as_int(raw.get("sequence"),
                                  raw.get("sequenceNumber")),
        "playerIndex": player_index,
        "observation": observation,
        "select": _ensure_select(nested_select),
        "selectedIndices": _as_index_list(selected_indices),
        "nextObservation": raw.get("nextObservation"),
        "terminal": bool(raw.get("terminal")) or bool(raw.get("gameOver")),
        "reward": raw.get("reward"),
        "result": raw.get("result"),
        "metadata": metadata,
    }


def _normalize_canonical_or_java(raw, source, metadata):
    # Both Java transition dataset and previously-normalized records share the
    # canonical top-level shape. All we do:
    #   - lift select.playerIndex (when present) to top-level playerIndex,
    #   - copy decisionMethod into metadata (transition dataset provenance),
    #   - accept whatever nextObservation/terminal/reward/result the writer
    #     already produced.
    top_select = raw.get("select") or {}

    player_index = raw.get("playerIndex")
    if player_index is None:
        player_index = (top_select.get("playerIndex")
                        if isinstance(top_select, dict) else None)

    if "decisionMethod" in raw and "decisionMethod" not in metadata:
        metadata["decisionMethod"] = raw["decisionMethod"]
    # The transition dataset packs source-specific deck/pool/xmage metadata
    # already under metadata; preserve any keys not yet copied.
    for key, value in (raw.get("metadata") or {}).items():
        metadata.setdefault(key, value)

    return {
        "schemaVersion": SCHEMA_VERSION,
        "source": source,
        "gameId": raw.get("gameId"),
        "sequenceNumber": _as_int(raw.get("sequenceNumber")),
        "playerIndex": _as_int(player_index),
        "observation": raw.get("observation") or {},
        "select": _ensure_select(top_select),
        "selectedIndices": raw.get("selectedIndices") or [],
        "nextObservation": raw.get("nextObservation"),
        "terminal": raw.get("terminal"),
        "reward": raw.get("reward"),
        "result": raw.get("result"),
        "metadata": metadata,
    }


def _normalize_unknown(raw, source, metadata):
    # Last resort: never raise. Stuff everything we cannot interpret under
    # observation so the validator can flag the missing canonical fields.
    unknown_payload = {
        key: value for key, value in raw.items()
        if key not in ("schemaVersion", "metadata", "select")
    }
    top_select = raw.get("select")
    if isinstance(top_select, dict):
        unknown_payload.pop("select", None)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "source": source,
        "gameId": raw.get("gameId"),
        "sequenceNumber": _as_int(raw.get("sequenceNumber")),
        "playerIndex": _as_int(raw.get("playerIndex")),
        "observation": unknown_payload,
        "select": _ensure_select(top_select if isinstance(top_select, dict)
                                 else {}),
        "selectedIndices": raw.get("selectedIndices") or [],
        "nextObservation": raw.get("nextObservation"),
        "terminal": raw.get("terminal"),
        "reward": raw.get("reward"),
        "result": raw.get("result"),
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# streaming iteration
# ---------------------------------------------------------------------------

def iter_decision_records(path_or_file, source_hint=None):
    """Yield normalized ``DecisionRecord`` dicts from a JSONL file or handle.

    Empty lines are skipped. A line that is not valid JSON raises
    ``ValueError`` naming the line number, matching
    ``magic_cabt.dataset.read_dataset``'s contract so callers can swap them
    one-for-one.

    Self-play replay JSONL is handled specially: the trailing ``{result}``
    line emitted by ``examples/run_selfplay.py`` is folded back into the
    previous decision as its terminal marker (``terminal = True`` and
    ``result`` attached). It is not yielded on its own — the canonical shape
    requires a ``select`` on every record and a bare result line has none.

    The generator is source-agnostic: when the input genuinely is a Java
    transition dataset, the per-record output is the same shape as
    ``normalize_record`` produces for those records directly.
    """
    own_handle = not hasattr(path_or_file, "read")
    if own_handle:
        handle = open(path_or_file, "r", encoding="utf-8")
    else:
        handle = path_or_file

    try:
        pending = []  # buffered self-play decisions awaiting their result
        is_self_play = None  # decided on the first non-blank line

        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except ValueError:
                raise ValueError("dataset line %d is not valid JSON"
                                 % line_number)
            if not isinstance(raw, dict):
                raise ValueError("dataset line %d is not a JSON object"
                                 % line_number)

            # Detect the self-play replay shape on the first record. The
            # bare trailing {result} line is the only deviation; we treat it
            # as a continuation marker rather than a standalone record.
            if is_self_play is None:
                is_self_play = (
                    "sequence" in raw
                    and "player" in raw
                    and "observation" in raw
                    and "selected" in raw
                )

            if is_self_play and _is_self_play_result_line(raw):
                _attach_self_play_terminal(pending, raw.get("result"))
                continue

            record = normalize_record(raw, source_hint=source_hint)
            record["__source_line"] = line_number
            if is_self_play:
                pending.append(record)
            yield record

            if is_self_play and record.get("terminal"):
                # A transition with terminal=True already carries its outcome;
                # don't keep waiting for a trailing result line.
                pending.pop()

        # If the writer forgot the {result} line, don't synthesize one: the
        # decisions are still valid non-terminal records.
    finally:
        if own_handle:
            handle.close()


def _is_self_play_result_line(raw):
    return (
        "result" in raw
        and "selected" not in raw
        and "observation" not in raw
    )


def _attach_self_play_terminal(pending, result):
    if not pending:
        return
    final = pending.pop()
    final["terminal"] = True
    if result is not None:
        final["result"] = result
        if "winner" in result and isinstance(result["winner"], int):
            winner = result["winner"]
            final["reward"] = 1.0 if final.get("playerIndex") == winner \
                else 0.0 if winner in (0, 1) else None


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate_record(record):
    """Return a list of error message strings; empty when the record is valid.

    Every rule documented in ``docs/decision-record-schema.md`` is enforced.
    The validator is intentionally lenient about ``observation`` internals
    (which are source-specific) so it stays useful across all four sources.
    """
    errors = []
    if not isinstance(record, dict):
        return ["record is not a JSON object"]
    if record.get("schemaVersion") != SCHEMA_VERSION:
        errors.append("schemaVersion != %d (got %r)"
                      % (SCHEMA_VERSION, record.get("schemaVersion")))

    select = _select_block(record)
    if select is None:
        errors.append("missing select block (observation.select or top-level "
                      "select)")
    elif not isinstance(select, dict):
        errors.append("select is not an object")
    else:
        errors.extend(_validate_select_block(select))

    selected = record.get("selectedIndices")
    if selected is None:
        errors.append("selectedIndices is missing")
    elif not isinstance(selected, list):
        errors.append("selectedIndices is not a list (got %r)"
                      % (type(selected).__name__,))
    else:
        option_count = _option_count(select)
        errors.extend(_validate_selected_indices(selected, option_count))

    # cross-check: count vs minCount/maxCount, only when both sides exist.
    if isinstance(select, dict) and isinstance(selected, list):
        errors.extend(_validate_count_bounds(selected, select))

    # playerIndex is required when the source had it; we accept ``None`` only
    # when the record explicitly declines (e.g. Arena concede before seat
    # assignment). Both situations normalize to a present key.
    if "playerIndex" not in record:
        errors.append("playerIndex is missing")
    elif record.get("playerIndex") is not None:
        if not isinstance(record["playerIndex"], int) or \
                isinstance(record["playerIndex"], bool):
            errors.append("playerIndex is not an int (%r)"
                          % (record.get("playerIndex"),))

    terminal = record.get("terminal")
    if terminal is True:
        if record.get("result") is None and record.get("reward") is None:
            errors.append("terminal record has no result or reward "
                          "(allowed but flagged)")
    elif terminal is not None and not isinstance(terminal, bool):
        errors.append("terminal is not a boolean (got %r)"
                      % (type(terminal).__name__,))

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata is missing or not an object")
    else:
        confidence = metadata.get("captureConfidence")
        if confidence is not None and confidence not in (
                "exact", "mirror", "partial"):
            errors.append("metadata.captureConfidence is not a known value: "
                         "%r" % (confidence,))
        # optional known-hidden-leak marker: a writer can opt in to declaring
        # that this record leaked private information the acting seat would
        # not have had. The validator fails on it loudly.
        if metadata.get("knownHiddenLeak"):
            errors.append("metadata.knownHiddenLeak is set: record contains "
                          "detectable hidden-information leakage")

    source = record.get("source")
    if source is not None and source not in CANONICAL_SOURCES:
        errors.append("source is not a canonical label: %r" % (source,))

    return errors


def validate_records(records):
    """Aggregate ``validate_record`` across an iterable of records.

    Returns a summary dict::

        {"total": int, "valid": int, "invalid": int,
         "errors": [{"record": int, "line": int, "messages": [...]}],
         "selectTypes": {...}, "optionTypes": {...}, "selectedCount": {...}}

    ``line`` is the ``__source_line`` stamp ``iter_decision_records`` adds when
    reading from a file, falling back to ``1``-based enumeration when records
    are iterated in isolation.
    """
    total = 0
    valid = 0
    invalid = 0
    errors = []
    select_types = {}
    option_types = {}
    selected_count = {}
    for idx, raw in enumerate(records):
        total += 1
        messages = validate_record(raw)
        if messages:
            invalid += 1
            errors.append({
                "record": idx,
                "line": raw.get("__source_line", idx + 1)
                if isinstance(raw, dict) else idx + 1,
                "messages": messages,
            })
        else:
            valid += 1
        _tally(raw, select_types, option_types, selected_count)
    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "errors": errors,
        "selectTypes": select_types,
        "optionTypes": option_types,
        "selectedCount": selected_count,
    }


# --- validation helpers -----------------------------------------------------

def _select_block(record):
    if "select" in record:
        return record["select"]
    observation = record.get("observation")
    if isinstance(observation, dict) and "select" in observation:
        return observation["select"]
    return None


def _validate_select_block(select):
    errors = []
    if "type" not in select:
        errors.append("select.type is missing")
    for key in ("minCount", "maxCount"):
        if key in select and not _is_int(select[key]):
            errors.append("select.%s is not an int (%r)" % (key, select[key]))
    options = select.get("option")
    if not isinstance(options, list):
        errors.append("select.option is not a list")
        return errors
    for option_idx, option in enumerate(options):
        if not isinstance(option, dict):
            errors.append("select.option[%d] is not an object" % option_idx)
            continue
        index = option.get("index")
        if not _is_int(index):
            errors.append("select.option[%d].index is not an int (%r)"
                         % (option_idx, index))
            continue
        if index != option_idx:
            # tolerance: the canonical contract requires index == position,
            # but the canonical example is non-exhaustive on this strictness.
            # We flag rather than fail hard so legacy records remain loadable.
            errors.append("select.option[%d].index != position (%r != %d)"
                         % (option_idx, index, option_idx))
    return errors


def _validate_selected_indices(selected, option_count):
    errors = []
    for offset, value in enumerate(selected):
        if isinstance(value, bool) or not _is_int(value):
            errors.append("selectedIndices[%d] is not an int (%r)"
                         % (offset, value))
            continue
        if value < 0:
            errors.append("selectedIndices[%d] is negative: %r"
                         % (offset, value))
            continue
        if option_count is not None and value >= option_count:
            errors.append("selectedIndices[%d] = %d is out of range "
                          "(option count %d)" % (offset, value, option_count))
    if len(selected) != len(set(selected)):
        duplicates = _duplicates(selected)
        errors.append("selectedIndices has duplicates: %r" % (duplicates,))
    return errors


def _validate_count_bounds(selected, select):
    errors = []
    count = len(selected)
    min_count = select.get("minCount")
    max_count = select.get("maxCount")
    if _is_int(min_count) and count < min_count:
        errors.append("selected count %d is below minCount %d"
                      % (count, min_count))
    if _is_int(max_count) and max_count > 0 and count > max_count:
        errors.append("selected count %d exceeds maxCount %d"
                      % (count, max_count))
    return errors


def _tally(record, select_types, option_types, selected_count):
    if not isinstance(record, dict):
        return
    select = _select_block(record)
    if isinstance(select, dict) and select.get("type") is not None:
        select_types[select["type"]] = select_types.get(select["type"], 0) + 1
    if isinstance(select, dict) and isinstance(select.get("option"), list):
        for option in select["option"]:
            if isinstance(option, dict) and option.get("type") is not None:
                option_types[option["type"]] = \
                    option_types.get(option["type"], 0) + 1
    selected = record.get("selectedIndices")
    if isinstance(selected, list):
        bucket = str(len(selected))
        selected_count[bucket] = selected_count.get(bucket, 0) + 1


def _option_count(select):
    options = select.get("option") if isinstance(select, dict) else None
    if not isinstance(options, list):
        return None
    return len(options)


# ---------------------------------------------------------------------------
# small coercion utilities
# ---------------------------------------------------------------------------

def _resolve_source(raw, source_hint):
    hint = source_hint
    if hint in CANONICAL_SOURCES:
        return hint
    if hint is not None and hint not in CANONICAL_SOURCES:
        # ignore non-canonical hints; fall through to detection.
        pass
    explicit = raw.get("source")
    if explicit in CANONICAL_SOURCES:
        return explicit
    # Arena recorder records are recognizable by their envelope keys.
    if any(k in raw for k in ("promptMessageType", "selectionMatched",
                             "responseMessageType")):
        return "arena_human"
    return DEFAULT_SOURCE


def _ensure_select(candidate):
    if not isinstance(candidate, dict):
        return {"type": None, "minCount": None, "maxCount": None, "option": []}
    select = dict(candidate)
    select.setdefault("option", [])
    return select


def _as_index_list(value):
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    return [value]


def _as_int(*candidates):
    for value in candidates:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _duplicates(values):
    seen = set()
    duplicated = []
    for value in values:
        if value in seen and value not in duplicated:
            duplicated.append(value)
        seen.add(value)
    return duplicated