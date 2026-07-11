"""Offline analysis backfill for recorded bundles.

Scores every decision in a bundle with a checkpoint and persists the results
to the bundle's ``analysis.jsonl`` cache — the exact records the replay
overlays look up. Idempotent per checkpoint: already-cached decisions are
skipped, so calling this before every replay is cheap once a bundle has been
scored.

Decisions are read raw, byte-for-byte as recorded: the cache key is the
fingerprint of the *recorded* decision, and any normalization here would
orphan the records being written.
"""
from __future__ import annotations

import json
import os
import time

from .cache import AnalysisCache
from .schema import analysis_cache_key, make_analysis_record
from .scorer import load_checkpoint_scorer

__all__ = ["backfill_bundle"]


def backfill_bundle(bundle_dir, checkpoint, device=None, top_k=5,
                    progress=None, source="backfill"):
    """Score a bundle's decisions into its analysis cache.

    Returns a summary dict with ``scored``/``alreadyCached`` counts and the
    scorer's model info. ``progress(done, total)`` is called every few
    decisions so a UI can narrate long runs. Raises on scorer errors —
    misaligned model output is a bug, never silently skipped.
    """
    bundle_dir = os.path.abspath(os.path.expanduser(bundle_dir))
    decisions_path = os.path.join(bundle_dir, "decisions.jsonl")
    if not os.path.isfile(decisions_path):
        raise IOError("no decisions.jsonl in %s" % bundle_dir)
    card_cache = os.path.join(bundle_dir, "card_cache.json")
    scorer = load_checkpoint_scorer(
        checkpoint, device=device or None,
        card_cache=card_cache if os.path.isfile(card_cache) else None)
    cache = AnalysisCache(os.path.join(bundle_dir, "analysis.jsonl"))

    records = list(_iter_raw_jsonl(decisions_path))
    scored, cached = 0, 0
    for index, record in enumerate(records):
        key = analysis_cache_key(record, scorer.model_info)
        if cache.get(key) is not None:
            cached += 1
        else:
            started = time.perf_counter()
            scores = scorer.score(record)
            latency = int((time.perf_counter() - started) * 1000)
            value_method = getattr(scorer, "state_value", None)
            cache.add(make_analysis_record(
                record, scores, scorer.model_info, top_k=top_k,
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


def _iter_raw_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
