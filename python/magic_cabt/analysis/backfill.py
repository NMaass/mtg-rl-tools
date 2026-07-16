"""Offline analysis backfill for recorded bundles.

Stateful scorers are advanced through every decision, including rows whose
analysis is already cached. Recurrent scorers also receive a per-decision
analysis context so identical public states at different history positions do
not collide in the cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

from .cache import AnalysisCache
from .schema import analysis_cache_key, make_analysis_record
from .scorer import load_checkpoint_scorer

__all__ = ["backfill_bundle"]


def backfill_bundle(bundle_dir, checkpoint, device=None, top_k=5,
                    progress=None, source="backfill"):
    bundle_dir = os.path.abspath(os.path.expanduser(bundle_dir))
    decisions_path = os.path.join(bundle_dir, "decisions.jsonl")
    if not os.path.isfile(decisions_path):
        raise IOError("no decisions.jsonl in %s" % bundle_dir)
    card_cache = os.path.join(bundle_dir, "card_cache.json")
    scorer = load_checkpoint_scorer(
        checkpoint, device=device or None,
        card_cache=card_cache if os.path.isfile(card_cache) else None)
    reset = getattr(scorer, "reset", None)
    if reset is not None:
        reset()
    cache = AnalysisCache(os.path.join(bundle_dir, "analysis.jsonl"))

    records = list(_iter_raw_jsonl(decisions_path))
    scored, cached = 0, 0
    for index, record in enumerate(records):
        model_info = dict(scorer.model_info)
        context_method = getattr(scorer, "cache_context", None)
        if context_method is not None:
            context = context_method(record)
        elif getattr(scorer, "observe", None) is not None:
            context = _default_recurrent_context(record)
        else:
            context = None
        if context:
            model_info["analysisContext"] = context
        key = analysis_cache_key(record, model_info)
        existing = cache.get(key)
        if existing is not None:
            cached += 1
            observe = getattr(scorer, "observe", None)
            if observe is not None:
                observe(record)
        else:
            started = time.perf_counter()
            scores = scorer.score(record)
            latency = int((time.perf_counter() - started) * 1000)
            value_method = getattr(scorer, "state_value", None)
            cache.add(make_analysis_record(
                record, scores, model_info, top_k=top_k,
                latency_ms=latency,
                value=value_method(record) if value_method else None,
                source=source), persist=True)
            scored += 1
        if progress is not None and (index + 1) % 25 == 0:
            progress(index + 1, len(records))
    if progress is not None and records:
        progress(len(records), len(records))
    return {"bundle": bundle_dir, "checkpoint": checkpoint,
            "scored": scored, "alreadyCached": cached,
            "model": scorer.model_info}


def _default_recurrent_context(record):
    observation = record.get("observation") or {}
    current = observation.get("current") or record.get("current") or {}
    payload = [
        record.get("matchId") or record.get("gameId") or current.get("matchId"),
        record.get("gameNumber", current.get("gameNumber")),
        record.get("gameInstance") or current.get("gameInstance"),
        record.get("sequenceNumber", record.get("sequence", current.get("seq"))),
    ]
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _iter_raw_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
