"""Streaming readers for canonical DecisionRecord datasets.

This module is the public dataset-ingestion surface for training and
evaluation CLIs. It wraps ``records.normalize_record`` but owns the file/JSONL
iteration details, including the self-play replay convention where a trailing
``{"result": ...}`` line annotates the previous decision.

Unlike the original iterator in ``records.py``, this implementation never
mutates a record after yielding it. That property matters for streaming
consumers that process records one at a time rather than materializing the
whole dataset first.
"""

import json

from .records import normalize_record

__all__ = ["iter_decision_records"]


def iter_decision_records(path_or_file, source_hint=None):
    """Yield normalized ``DecisionRecord`` dicts from JSONL.

    ``path_or_file`` may be a filesystem path or an already-open text handle.
    Empty lines are skipped. Invalid JSON and non-object rows raise
    ``ValueError`` with the source line number.

    Self-play replay JSONL is handled specially: the trailing ``{"result": ...}``
    line emitted by ``examples/run_selfplay.py`` is folded into the previous
    decision before that decision is yielded. This avoids post-yield mutation,
    so streaming consumers see terminal/result/reward fields consistently.
    """
    own_handle = not hasattr(path_or_file, "read")
    handle = open(path_or_file, "r", encoding="utf-8") if own_handle else path_or_file

    try:
        pending_self_play = None
        is_self_play = None

        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except ValueError:
                raise ValueError("dataset line %d is not valid JSON" % line_number)
            if not isinstance(raw, dict):
                raise ValueError("dataset line %d is not a JSON object" % line_number)

            if is_self_play is None:
                is_self_play = _looks_like_self_play_replay(raw)

            if is_self_play and _is_self_play_result_line(raw):
                if pending_self_play is not None:
                    _attach_self_play_terminal(pending_self_play, raw.get("result"))
                    yield pending_self_play
                    pending_self_play = None
                continue

            record = normalize_record(raw, source_hint=source_hint)
            record["__source_line"] = line_number

            if not is_self_play:
                yield record
                continue

            if pending_self_play is not None:
                yield pending_self_play
            pending_self_play = record
            if record.get("terminal"):
                yield pending_self_play
                pending_self_play = None

        if pending_self_play is not None:
            yield pending_self_play
    finally:
        if own_handle:
            handle.close()


def _looks_like_self_play_replay(raw):
    return (
        "sequence" in raw
        and "player" in raw
        and "observation" in raw
        and "selected" in raw
    )


def _is_self_play_result_line(raw):
    return (
        "result" in raw
        and "selected" not in raw
        and "observation" not in raw
    )


def _attach_self_play_terminal(record, result):
    record["terminal"] = True
    if result is None:
        return
    record["result"] = result
    winner = result.get("winner") if isinstance(result, dict) else None
    if isinstance(winner, int):
        record["reward"] = 1.0 if record.get("playerIndex") == winner \
            else 0.0 if winner in (0, 1) else None
