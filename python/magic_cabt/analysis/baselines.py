"""Deterministic non-learned controls for the common analysis schema."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

from .backfill import backfill_bundle
from .cache import AnalysisCache
from .compare import build_comparison, render_comparison_html
from .schema import analysis_cache_key, decision_fingerprint, make_analysis_record


class FirstLegalScorer:
    name = "first-legal"

    @property
    def model_info(self):
        return {"modelId": self.name, "checkpointId": "baseline:first-legal",
                "trainingState": "nonlearned"}

    def score(self, record):
        return [1.0 if index == 0 else 0.0
                for index, _option in enumerate(_options(record))]


class DeterministicRandomScorer:
    name = "deterministic-random"

    def __init__(self, seed=0):
        self.seed = int(seed)

    @property
    def model_info(self):
        return {"modelId": self.name,
                "checkpointId": "baseline:random:%d" % self.seed,
                "trainingState": "nonlearned", "seed": self.seed}

    def score(self, record):
        fingerprint = decision_fingerprint(record)
        result = []
        for index, _option in enumerate(_options(record)):
            digest = hashlib.sha256(
                ("%s|%d|%d" % (fingerprint, self.seed, index)).encode("utf-8")
            ).digest()
            result.append(int.from_bytes(digest[:8], "big") / float(2 ** 64))
        return result


class GenericActionHeuristicScorer:
    """Action-shape control with no card-name or rules-text strategy."""
    name = "generic-action-heuristic"

    @property
    def model_info(self):
        return {"modelId": self.name,
                "checkpointId": "baseline:generic-action-v1",
                "trainingState": "nonlearned"}

    def score(self, record):
        return [_generic_score(option) for option in _options(record)]


def make_baseline(name, seed=0):
    normalized = str(name).strip().lower()
    if normalized in ("first", "first-legal"):
        return FirstLegalScorer()
    if normalized in ("random", "deterministic-random"):
        return DeterministicRandomScorer(seed=seed)
    if normalized in ("heuristic", "generic-action", "generic-action-heuristic"):
        return GenericActionHeuristicScorer()
    raise ValueError("unknown baseline: %s" % name)


def backfill_scorer(bundle_dir, scorer, top_k=5, progress=None,
                    source="baseline"):
    bundle_dir = os.path.abspath(os.path.expanduser(bundle_dir))
    decisions_path = os.path.join(bundle_dir, "decisions.jsonl")
    if not os.path.isfile(decisions_path):
        raise IOError("no decisions.jsonl in %s" % bundle_dir)
    cache = AnalysisCache(os.path.join(bundle_dir, "analysis.jsonl"))
    records = list(_read_jsonl(decisions_path))
    scored = cached = 0
    for index, record in enumerate(records):
        key = analysis_cache_key(record, scorer.model_info)
        if cache.get(key) is not None:
            cached += 1
        else:
            started = time.perf_counter()
            scores = scorer.score(record)
            latency = int((time.perf_counter() - started) * 1000)
            cache.add(make_analysis_record(
                record, scores, scorer.model_info, top_k=top_k,
                latency_ms=latency, source=source), persist=True)
            scored += 1
        if progress and ((index + 1) % 25 == 0 or index + 1 == len(records)):
            progress(index + 1, len(records))
    return {"bundle": bundle_dir, "model": scorer.model_info,
            "scored": scored, "alreadyCached": cached}


def compare_suite(bundle_dir, entries, out_html, out_json=None,
                  device=None, top_k=5, seed=0, title=None, progress=None):
    summaries = []
    for display_name, spec in entries:
        if spec.startswith("baseline:"):
            scorer = make_baseline(spec.split(":", 1)[1], seed=seed)
            summary = backfill_scorer(
                bundle_dir, scorer, top_k=top_k,
                progress=(lambda done, total, label=display_name:
                          progress(label, done, total)) if progress else None,
                source="head-to-head-baseline")
            summaries.append({"name": display_name, "checkpoint": None,
                              "checkpointSha256": None,
                              "model": summary["model"],
                              "scored": summary["scored"],
                              "alreadyCached": summary["alreadyCached"]})
        else:
            checkpoint = os.path.abspath(os.path.expanduser(spec))
            summary = backfill_bundle(
                bundle_dir, checkpoint, device=device, top_k=top_k,
                progress=(lambda done, total, label=display_name:
                          progress(label, done, total)) if progress else None,
                source="head-to-head")
            summaries.append({"name": display_name, "checkpoint": checkpoint,
                              "checkpointSha256": _sha256_file(checkpoint),
                              "model": summary["model"],
                              "scored": summary["scored"],
                              "alreadyCached": summary["alreadyCached"]})
    report = build_comparison(bundle_dir, summaries, title=title)
    out_html = os.path.abspath(os.path.expanduser(out_html))
    out_json = os.path.abspath(os.path.expanduser(
        out_json or os.path.splitext(out_html)[0] + ".json"))
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    _atomic_write(out_json, json.dumps(
        report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    _atomic_write(out_html, render_comparison_html(report))
    return {"html": out_html, "json": out_json, "comparison": report}


def _generic_score(option):
    kind = str(option.get("type") or "").upper()
    label = str(option.get("label") or "").lower()
    if "CONCEDE" in kind or "concede" in label:
        return -100.0
    if "PASS" in kind or label == "pass":
        return -1.0
    for token, value in (("CAST", 3.0), ("PLAY_LAND", 2.5),
                         ("ACTIVATE", 2.0), ("ATTACK", 1.5),
                         ("BLOCK", 1.0), ("MULLIGAN", 0.0)):
        if token in kind:
            return value
    return 0.5


def _options(record):
    direct = record.get("select")
    select = direct if isinstance(direct, dict) else \
        (record.get("observation") or {}).get("select") or {}
    return select.get("option") or []


def _read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path, content):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(temporary, path)


def _parse_entry(value):
    if "=" not in value:
        raise ValueError("--model must use NAME=PATH_OR_BASELINE")
    name, spec = value.split("=", 1)
    if not name.strip() or not spec.strip():
        raise ValueError("--model must use NAME=PATH_OR_BASELINE")
    return name.strip(), spec.strip()


def build_parser():
    parser = argparse.ArgumentParser(prog="magic-cabt-compare-suite")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--model", action="append", required=True,
                        metavar="NAME=PATH_OR_BASELINE")
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = compare_suite(
        args.bundle, [_parse_entry(value) for value in args.model], args.out,
        out_json=args.json, device=args.device, top_k=args.top_k,
        seed=args.seed, title=args.title,
        progress=lambda name, done, total: print(
            "[%s] %d/%d" % (name, done, total), file=os.sys.stderr))
    print(json.dumps({"html": result["html"], "json": result["json"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
