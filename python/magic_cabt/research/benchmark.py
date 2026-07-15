"""Leakage-aware evaluation utilities for MTG research artifacts.

Decision metrics are clustered by game/match and match metrics are paired by a
stable scenario key.  The implementation is dependency-free so the statistical
contract can run in the repository's base installation.
"""
from __future__ import annotations

import itertools
import json
import math
import random
from collections import defaultdict
from typing import Mapping

from magic_cabt.analysis.schema import decision_fingerprint

SCHEMA_VERSION = 1


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError("%s:%d: invalid JSON: %s" %
                                 (path, line_number, exc))
            if not isinstance(row, Mapping):
                raise ValueError("%s:%d: JSONL row must be an object" %
                                 (path, line_number))
            rows.append(row)
    return rows


def cluster_bootstrap_interval(values, bootstrap_samples=1000,
                               confidence=0.95, seed=0):
    """Percentile interval for a mean while resampling whole clusters."""
    clusters = _clusters(values)
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if not _finite(confidence) or not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    flat = [value for rows in clusters.values() for value in rows]
    if not flat:
        return _interval(None, None, None, confidence, 0, 0,
                         bootstrap_samples)
    keys = sorted(clusters, key=str)
    rng = random.Random(seed)
    samples = []
    for _ in range(bootstrap_samples):
        selected = [keys[rng.randrange(len(keys))] for _unused in keys]
        sample = [value for key in selected for value in clusters[key]]
        samples.append(_mean(sample))
    alpha = (1.0 - float(confidence)) / 2.0
    return _interval(_mean(flat), _quantile(samples, alpha),
                     _quantile(samples, 1.0 - alpha), confidence,
                     len(keys), len(flat), bootstrap_samples)


def benchmark_analyses(decisions, analyses_by_name, checkpoint_by_name=None,
                       group_by=None, bootstrap_samples=1000,
                       confidence=0.95, seed=0):
    """Compare checkpoint analysis caches with held-out human decisions.

    Single-choice rows are scored. Canonical top-k metrics regard options with
    the same ``payload.canonicalKey`` as fungible; raw rank and MRR retain the
    exact selected-index metric already stored in ``analysis.jsonl``.
    """
    if not isinstance(analyses_by_name, Mapping) or not analyses_by_name:
        raise ValueError("analyses_by_name must contain at least one model")
    checkpoint_by_name = dict(checkpoint_by_name or {})
    unknown = sorted(set(checkpoint_by_name).difference(analyses_by_name))
    if unknown:
        raise ValueError("checkpoint selections have no analysis model: %s" %
                         unknown)
    group_by = tuple(group_by or
                     ("promptType", "optionCountBucket", "source"))
    decision_rows = [_decision_row(row, index)
                     for index, row in enumerate(decisions)]
    eligible = [row for row in decision_rows if row["eligible"]]

    reports = {}
    scored_by_model = {}
    for model_offset, (name, records) in enumerate(
            sorted(analyses_by_name.items())):
        index, selection = _analysis_index(
            records, checkpoint_by_name.get(name), name)
        scored = {}
        rows = []
        for decision in eligible:
            analysis = index.get(decision["fingerprint"])
            if analysis is None:
                continue
            metric = _analysis_metric(decision, analysis)
            rows.append(metric)
            scored[decision["occurrenceKey"]] = metric
        scored_by_model[name] = scored
        reports[name] = _analysis_report(
            name, selection, decision_rows, eligible, rows, group_by,
            bootstrap_samples, confidence, seed + model_offset * 1009)

    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "decision-analysis-benchmark-v1",
        "decisionRecords": len(decision_rows),
        "eligibleSingleChoiceRecords": len(eligible),
        "ineligibleRecords": len(decision_rows) - len(eligible),
        "groupBy": list(group_by),
        "models": reports,
        "comparisons": _analysis_comparisons(
            scored_by_model, bootstrap_samples, confidence, seed + 500003),
        "methodology": {
            "interval": "percentile cluster bootstrap over games/matches",
            "pairwiseTest": "paired cluster sign-flip",
            "multipleComparisonCorrection": "Holm",
            "canonicalTopK": (
                "options sharing payload.canonicalKey are fungible"),
        },
    }


def benchmark_matches(rows, bootstrap_samples=1000, confidence=0.95,
                      seed=0):
    """Evaluate long- or wide-format paired match-result records."""
    normalized = []
    for index, row in enumerate(rows):
        normalized.extend(_normalize_match(row, index))
    if not normalized:
        raise ValueError("match benchmark contains no rows")

    by_agent = defaultdict(lambda: defaultdict(list))
    pair_cluster = {}
    pair_metadata = {}
    for row in normalized:
        by_agent[row["agent"]][row["pairKey"]].append(row["score"])
        previous = pair_cluster.get(row["pairKey"])
        if previous is not None and previous != row["clusterId"]:
            raise ValueError("pairKey %s maps to multiple clusters" %
                             row["pairKey"])
        pair_cluster[row["pairKey"]] = row["clusterId"]
        pair_metadata.setdefault(row["pairKey"], row["metadata"])

    scores = {}
    agent_reports = {}
    for offset, agent in enumerate(sorted(by_agent)):
        agent_scores = {key: _mean(values)
                        for key, values in by_agent[agent].items()}
        scores[agent] = agent_scores
        clusters = _group_pair_values(agent_scores, pair_cluster)
        agent_reports[agent] = {
            "pairedUnits": len(agent_scores),
            "meanScore": cluster_bootstrap_interval(
                clusters, bootstrap_samples, confidence, seed + offset * 997),
            "bySuite": _match_suites(
                agent_scores, pair_metadata, pair_cluster,
                bootstrap_samples, confidence, seed + offset * 137),
        }

    comparisons = []
    pvalues = []
    for offset, (left, right) in enumerate(
            itertools.combinations(sorted(scores), 2)):
        common = sorted(set(scores[left]).intersection(scores[right]))
        deltas = defaultdict(list)
        outcomes = {"leftBetter": 0, "rightBetter": 0, "ties": 0}
        for key in common:
            delta = scores[left][key] - scores[right][key]
            deltas[pair_cluster[key]].append(delta)
            if delta > 0:
                outcomes["leftBetter"] += 1
            elif delta < 0:
                outcomes["rightBetter"] += 1
            else:
                outcomes["ties"] += 1
        pvalue = _sign_flip(deltas, bootstrap_samples,
                            seed + 20000 + offset)
        comparisons.append({
            "left": left,
            "right": right,
            "commonPairedUnits": len(common),
            "scoreDelta": cluster_bootstrap_interval(
                deltas, bootstrap_samples, confidence,
                seed + 10000 + offset),
            "pairedOutcomes": outcomes,
            "pValueRaw": pvalue,
            "pValueHolm": None,
        })
        pvalues.append(pvalue)
    for row, adjusted in zip(comparisons, _holm(pvalues)):
        row["pValueHolm"] = adjusted

    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "paired-match-benchmark-v1",
        "inputRows": len(rows),
        "normalizedRows": len(normalized),
        "agents": agent_reports,
        "comparisons": comparisons,
        "methodology": {
            "pairing": "common pairKey values",
            "interval": "percentile cluster bootstrap",
            "pairwiseTest": "paired cluster sign-flip",
            "multipleComparisonCorrection": "Holm",
        },
    }


def _decision_row(record, index):
    if not isinstance(record, Mapping):
        raise ValueError("decision row %d must be an object" % index)
    observation = record.get("observation") or {}
    select = (record.get("select") if isinstance(record.get("select"), Mapping)
              else observation.get("select") or {})
    options = select.get("option") or []
    chosen = record.get("selectedIndices")
    if chosen is None and isinstance(record.get("select"), list):
        chosen = record.get("select")
    chosen = chosen or record.get("selected") or []
    eligible = (isinstance(options, list) and options and
                isinstance(chosen, list) and len(chosen) == 1 and
                isinstance(chosen[0], int) and
                0 <= chosen[0] < len(options))
    fingerprint = decision_fingerprint(record)
    current = observation.get("current") or {}
    cluster = _text(record.get("matchId"), record.get("gameId"),
                    current.get("gameInstance"), "decision-%d" % index)
    chosen_index = chosen[0] if eligible else None
    return {
        "index": index,
        "fingerprint": fingerprint,
        "occurrenceKey": "%s#%d" % (fingerprint, index),
        "clusterId": cluster,
        "eligible": bool(eligible),
        "chosenGroupKey": (_group_key(options[chosen_index], chosen_index)
                           if eligible else None),
        "promptType": str(select.get("type") or "unknown"),
        "optionCount": len(options) if isinstance(options, list) else 0,
        "optionCountBucket": _option_bucket(
            len(options) if isinstance(options, list) else 0),
        "source": str(record.get("source") or "unknown"),
    }


def _analysis_index(records, checkpoint, model_name):
    available = sorted(set(
        _checkpoint(row) for row in records
        if isinstance(row, Mapping) and _checkpoint(row)))
    if checkpoint is None and len(available) > 1:
        raise ValueError(
            "analysis %s contains multiple checkpoints; select one: %s" %
            (model_name, available))
    selected = checkpoint or (available[0] if available else None)
    index = {}
    duplicates = ignored = malformed = 0
    for row in records:
        if not isinstance(row, Mapping):
            malformed += 1
            continue
        if selected is not None and _checkpoint(row) != selected:
            ignored += 1
            continue
        fingerprint = row.get("decisionFingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            malformed += 1
            continue
        if fingerprint in index:
            duplicates += 1
        index[fingerprint] = row
    return index, {
        "checkpointId": selected,
        "availableCheckpointIds": available,
        "selectedRecords": len(index),
        "duplicateRecords": duplicates,
        "ignoredOtherCheckpointRecords": ignored,
        "malformedRecords": malformed,
    }


def _analysis_metric(decision, record):
    analysis = record.get("analysis") or {}
    top = analysis.get("topK") or []
    raw_rank = analysis.get("chosenRank")
    raw_rank = raw_rank if isinstance(raw_rank, int) and raw_rank > 0 else None
    canonical_rank = None
    seen = set()
    group_count = 0
    for row in top:
        if not isinstance(row, Mapping):
            continue
        key = row.get("canonicalGroupKey")
        if not isinstance(key, str) or not key:
            key = "index:%s" % row.get("optionIndex")
        if key in seen:
            continue
        seen.add(key)
        group_count += 1
        if key == decision["chosenGroupKey"]:
            canonical_rank = group_count
            break
    return {
        "clusterId": decision["clusterId"],
        "fingerprint": decision["fingerprint"],
        "occurrenceKey": decision["occurrenceKey"],
        "promptType": decision["promptType"],
        "optionCountBucket": decision["optionCountBucket"],
        "source": decision["source"],
        "rawRank": raw_rank,
        "rawTop1": _at_k(raw_rank, 1),
        "rawTop3": _at_k(raw_rank, 3),
        "reciprocalRank": 1.0 / raw_rank if raw_rank else None,
        "canonicalTop1": (
            1.0 if canonical_rank == 1 else
            0.0 if group_count >= 1 else None),
        "canonicalTop3": (
            1.0 if canonical_rank is not None and canonical_rank <= 3 else
            0.0 if group_count >= 3 else None),
        "topKRows": len(top),
    }


def _analysis_report(name, selection, decisions, eligible, rows, group_by,
                     bootstrap_samples, confidence, seed):
    grouped = {field: defaultdict(list) for field in group_by}
    for row in rows:
        for field in group_by:
            grouped[field][str(row.get(field, "unknown"))].append(row)
    by_group = {}
    group_seed = seed + 100000
    for field in group_by:
        by_group[field] = {
            value: _analysis_metrics(
                group_rows, bootstrap_samples, confidence,
                group_seed + offset)
            for offset, (value, group_rows) in enumerate(
                sorted(grouped[field].items()))
        }
        group_seed += 10000
    return {
        "name": name,
        "selection": dict(selection),
        "decisionRecords": len(decisions),
        "eligibleSingleChoiceRecords": len(eligible),
        "matchedEligibleRecords": len(rows),
        "matchedUniqueFingerprints": len(set(
            row["fingerprint"] for row in rows)),
        "coverage": len(rows) / len(eligible) if eligible else None,
        "metrics": _analysis_metrics(
            rows, bootstrap_samples, confidence, seed),
        "byGroup": by_group,
    }


def _analysis_metrics(rows, bootstrap_samples, confidence, seed):
    output = {"examples": len(rows)}
    fields = (("rawTop1", "rawTop1Accuracy"),
              ("rawTop3", "rawTop3Accuracy"),
              ("reciprocalRank", "rawMRR"),
              ("canonicalTop1", "canonicalTop1Accuracy"),
              ("canonicalTop3", "canonicalTop3Accuracy"))
    for offset, (field, label) in enumerate(fields):
        clusters = defaultdict(list)
        for row in rows:
            if _finite(row.get(field)):
                clusters[row["clusterId"]].append(float(row[field]))
        output[label] = cluster_bootstrap_interval(
            clusters, bootstrap_samples, confidence, seed + offset * 101)
    ranks = [float(row["rawRank"]) for row in rows
             if _finite(row.get("rawRank"))]
    output["meanRawChosenRank"] = _mean(ranks) if ranks else None
    output["rawRankCoverage"] = len(ranks) / len(rows) if rows else None
    output["meanAnalysisTopKRows"] = (
        _mean([float(row["topKRows"]) for row in rows]) if rows else None)
    return output


def _analysis_comparisons(scored, bootstrap_samples, confidence, seed):
    comparisons = []
    top1_pvalues = []
    mrr_pvalues = []
    for offset, (left, right) in enumerate(
            itertools.combinations(sorted(scored), 2)):
        common = sorted(set(scored[left]).intersection(scored[right]))
        top1 = defaultdict(list)
        mrr = defaultdict(list)
        for key in common:
            left_row, right_row = scored[left][key], scored[right][key]
            cluster = left_row["clusterId"]
            if (_finite(left_row.get("canonicalTop1")) and
                    _finite(right_row.get("canonicalTop1"))):
                top1[cluster].append(
                    left_row["canonicalTop1"] - right_row["canonicalTop1"])
            if (_finite(left_row.get("reciprocalRank")) and
                    _finite(right_row.get("reciprocalRank"))):
                mrr[cluster].append(
                    left_row["reciprocalRank"] - right_row["reciprocalRank"])
        top1_p = _sign_flip(top1, bootstrap_samples, seed + offset * 307 + 1)
        mrr_p = _sign_flip(mrr, bootstrap_samples, seed + offset * 307 + 2)
        comparisons.append({
            "left": left,
            "right": right,
            "commonDecisions": len(common),
            "canonicalTop1PairedObservations": sum(map(len, top1.values())),
            "rawMRRPairedObservations": sum(map(len, mrr.values())),
            "canonicalTop1Delta": cluster_bootstrap_interval(
                top1, bootstrap_samples, confidence,
                seed + offset * 307 + 3),
            "rawMRRDelta": cluster_bootstrap_interval(
                mrr, bootstrap_samples, confidence,
                seed + offset * 307 + 4),
            "canonicalTop1PValueRaw": top1_p,
            "canonicalTop1PValueHolm": None,
            "rawMRRPValueRaw": mrr_p,
            "rawMRRPValueHolm": None,
        })
        top1_pvalues.append(top1_p)
        mrr_pvalues.append(mrr_p)
    for row, top1, mrr in zip(
            comparisons, _holm(top1_pvalues), _holm(mrr_pvalues)):
        row["canonicalTop1PValueHolm"] = top1
        row["rawMRRPValueHolm"] = mrr
    return comparisons


def _normalize_match(row, index):
    if not isinstance(row, Mapping):
        raise ValueError("match row %d must be an object" % index)
    pair_key = _pair_key(row, index)
    cluster = _text(row.get("clusterId"), row.get("scenarioCluster"),
                    pair_key)
    metadata = dict(row.get("metadata") or {}) \
        if isinstance(row.get("metadata"), Mapping) else {}
    for field in ("suite", "scenarioId", "deck", "opponent", "seat", "seed"):
        if field in row:
            metadata[field] = row[field]
    if all(key in row for key in
           ("testAgent", "referenceAgent", "testScore", "referenceScore")):
        return [
            _match_row(row["testAgent"], row["testScore"], pair_key,
                       cluster, metadata),
            _match_row(row["referenceAgent"], row["referenceScore"],
                       pair_key, cluster, metadata),
        ]
    agent = _text(row.get("agent"), row.get("model"), row.get("policy"))
    if not agent:
        raise ValueError("match row %d needs agent/model/policy" % index)
    if "score" in row:
        score = row["score"]
    elif "outcome" in row:
        score = row["outcome"]
    else:
        raise ValueError("match row %d needs score or outcome" % index)
    return [_match_row(agent, score, pair_key, cluster, metadata)]


def _match_row(agent, score, pair_key, cluster, metadata):
    return {"agent": str(agent), "score": _score(score),
            "pairKey": pair_key, "clusterId": cluster,
            "metadata": metadata}


def _pair_key(row, index):
    explicit = _text(row.get("pairKey"), row.get("pairId"),
                     row.get("episodeKey"))
    if explicit:
        return explicit
    scenario = _text(row.get("scenarioId"), row.get("scenario"),
                     row.get("suiteCase"))
    if not scenario or row.get("seed") is None:
        raise ValueError(
            "match row %d needs pairKey/pairId or scenarioId and seed" % index)
    return "|".join((scenario, str(row["seed"]),
                     str(row.get("replicate", 0)),
                     str(row.get("seat", "any")),
                     str(row.get("deck", "any")),
                     str(row.get("opponent", "any")),
                     str(row.get("gameNumber", "any"))))


def _match_suites(scores, metadata, pair_cluster, bootstrap_samples,
                  confidence, seed):
    suites = defaultdict(lambda: defaultdict(list))
    for pair_key, value in scores.items():
        suite = str((metadata.get(pair_key) or {}).get("suite", "unknown"))
        suites[suite][pair_cluster[pair_key]].append(value)
    return {
        name: cluster_bootstrap_interval(
            clusters, bootstrap_samples, confidence, seed + offset)
        for offset, (name, clusters) in enumerate(sorted(suites.items()))
    }


def _group_pair_values(values, pair_cluster):
    grouped = defaultdict(list)
    for key, value in values.items():
        grouped[pair_cluster[key]].append(value)
    return grouped


def _sign_flip(clusters, samples, seed):
    means = [_mean(values) for _key, values in sorted(clusters.items())
             if values]
    if not means:
        return None
    observed = abs(_mean(means))
    if observed == 0.0:
        return 1.0
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        value = _mean([item if rng.random() < 0.5 else -item
                       for item in means])
        extreme += abs(value) >= observed - 1e-15
    return (extreme + 1.0) / (samples + 1.0)


def _holm(pvalues):
    adjusted = [None] * len(pvalues)
    valid = sorted((index, float(value)) for index, value in enumerate(pvalues)
                   if value is not None)
    running = 0.0
    for rank, (index, value) in enumerate(valid):
        running = max(running, min(1.0, (len(valid) - rank) * value))
        adjusted[index] = running
    return adjusted


def _clusters(values):
    output = defaultdict(list)
    if isinstance(values, Mapping):
        iterator = values.items()
    else:
        iterator = enumerate(list(values or []))
    for key, raw in iterator:
        rows = raw if isinstance(raw, (list, tuple)) else [raw]
        for value in rows:
            if not _finite(value):
                raise ValueError("bootstrap values must be finite")
            output[str(key)].append(float(value))
    return dict(output)


def _checkpoint(row):
    model = row.get("model") or {}
    value = (model.get("checkpointId") or model.get("modelId")) \
        if isinstance(model, Mapping) else None
    return str(value) if value is not None else None


def _group_key(option, index):
    if isinstance(option, Mapping):
        payload = option.get("payload")
        if isinstance(payload, Mapping):
            key = payload.get("canonicalKey")
            if isinstance(key, str) and key:
                return key
    return "index:%d" % index


def _option_bucket(count):
    if count <= 1:
        return "0-1"
    if count <= 4:
        return "2-4"
    if count <= 8:
        return "5-8"
    if count <= 16:
        return "9-16"
    return "17+"


def _at_k(rank, k):
    if rank is None:
        return None
    return 1.0 if rank <= k else 0.0


def _score(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("win", "won", "w"):
            return 1.0
        if normalized in ("draw", "tie", "d"):
            return 0.5
        if normalized in ("loss", "lost", "lose", "l"):
            return 0.0
    if not _finite(value):
        raise ValueError("match score must be finite or win/draw/loss")
    return float(value)


def _interval(estimate, lower, upper, confidence, clusters, observations,
              samples):
    return {"estimate": estimate, "lower": lower, "upper": upper,
            "confidence": float(confidence), "clusters": clusters,
            "observations": observations, "bootstrapSamples": int(samples)}


def _quantile(values, probability):
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(float(probability), 0.0), 1.0) * (len(ordered) - 1)
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean(values):
    return sum(float(value) for value in values) / len(values)


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(float(value)))


def _text(*values):
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


__all__ = [
    "SCHEMA_VERSION", "benchmark_analyses", "benchmark_matches",
    "cluster_bootstrap_interval", "decision_fingerprint", "read_jsonl",
]
