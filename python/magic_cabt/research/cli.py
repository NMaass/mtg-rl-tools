"""Command-line entry point for the MTG research framework."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

from .benchmark import benchmark_analyses, benchmark_matches, read_jsonl
from .experiment import plan_report
from .expert_cost import (
    ExpertCostModel,
    evaluate_preferences,
    file_sha256,
    fit_cost_model,
    load_factor_specs,
    load_preferences,
    split_preferences_by_context,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="magic-cabt-research",
        description=(
            "Fit expert cost models, benchmark checkpoint analyses, and "
            "validate reproducible MTG learning plans."))
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser(
        "fit-cost", help="fit a monotone Bradley-Terry expert cost model")
    fit.add_argument("--factors", required=True, help="factor ontology JSON")
    fit.add_argument("--preferences", required=True, help="expert comparisons JSONL")
    fit.add_argument("--out", required=True, help="output model JSON")
    fit.add_argument("--iterations", type=int, default=2500)
    fit.add_argument("--learning-rate", type=float, default=0.05)
    fit.add_argument("--l2", type=float, default=0.01)
    fit.add_argument("--temperature", type=float, default=1.0)
    fit.add_argument("--holdout-fraction", type=float, default=0.2)
    fit.add_argument("--seed", type=int, default=0)

    score = subparsers.add_parser(
        "score-cost", help="score and explain one candidate factor JSON object")
    score.add_argument("--model", required=True, help="expert cost model JSON")
    score.add_argument("--candidate", required=True,
                       help="candidate JSON file containing factors or a factors object")
    score.add_argument("--out", default=None, help="optional result JSON")

    analysis = subparsers.add_parser(
        "benchmark-analysis",
        help="evaluate one or more analysis.jsonl checkpoint caches")
    analysis.add_argument("--decisions", required=True, help="DecisionRecord JSONL")
    analysis.add_argument(
        "--analysis", action="append", required=True, metavar="NAME=PATH",
        help="repeat for each model/checkpoint analysis cache")
    analysis.add_argument(
        "--checkpoint", action="append", default=[], metavar="NAME=CHECKPOINT_ID",
        help="select a checkpoint when one cache contains multiple checkpoints")
    analysis.add_argument(
        "--group-by", default="promptType,optionCountBucket,source",
        help="comma-separated decision fields")
    analysis.add_argument("--bootstrap-samples", type=int, default=1000)
    analysis.add_argument("--confidence", type=float, default=0.95)
    analysis.add_argument("--seed", type=int, default=0)
    analysis.add_argument("--out", required=True, help="report JSON")

    matches = subparsers.add_parser(
        "benchmark-matches", help="evaluate standardized match-result JSONL")
    matches.add_argument("--input", required=True, help="match results JSONL")
    matches.add_argument("--bootstrap-samples", type=int, default=1000)
    matches.add_argument("--confidence", type=float, default=0.95)
    matches.add_argument("--seed", type=int, default=0)
    matches.add_argument("--out", required=True, help="report JSON")

    validate = subparsers.add_parser(
        "validate-plan", help="validate a declarative experiment plan")
    validate.add_argument("plan", help="experiment plan JSON")
    validate.add_argument("--out", default=None, help="optional report JSON")
    validate.add_argument(
        "--strict", action="store_true", help="treat warnings as a non-zero exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "fit-cost":
            return _fit_cost(args)
        if args.command == "score-cost":
            return _score_cost(args)
        if args.command == "benchmark-analysis":
            return _benchmark_analysis(args)
        if args.command == "benchmark-matches":
            return _benchmark_matches(args)
        if args.command == "validate-plan":
            return _validate_plan(args)
        parser.error("unknown command")
    except (OSError, ValueError) as exc:
        sys.stderr.write("research command failed: %s\n" % exc)
        return 2
    return 2


def _fit_cost(args: argparse.Namespace) -> int:
    specs = load_factor_specs(args.factors)
    preferences = load_preferences(args.preferences, specs)
    train, holdout = split_preferences_by_context(
        preferences, args.holdout_fraction, args.seed)
    metadata = {
        "data": {
            "factorFile": os.path.abspath(args.factors),
            "factorSha256": file_sha256(args.factors),
            "preferenceFile": os.path.abspath(args.preferences),
            "preferenceSha256": file_sha256(args.preferences),
            "holdoutFraction": args.holdout_fraction,
            "splitUnit": "contextId",
            "splitSeed": args.seed,
        }
    }
    model = fit_cost_model(
        specs, train, iterations=args.iterations,
        learning_rate=args.learning_rate, l2=args.l2,
        temperature=args.temperature, metadata=metadata)
    payload = model.to_dict()
    payload["diagnostics"] = {
        "train": evaluate_preferences(model, train),
        "holdout": evaluate_preferences(model, holdout),
    }
    _write_json(args.out, payload)
    sys.stdout.write(json.dumps({
        "out": os.path.abspath(args.out),
        "trainExamples": len(train),
        "holdoutExamples": len(holdout),
        "weights": payload["weights"],
        "holdout": payload["diagnostics"]["holdout"],
    }, sort_keys=True) + "\n")
    return 0


def _score_cost(args: argparse.Namespace) -> int:
    with open(args.model, encoding="utf-8") as handle:
        model = ExpertCostModel.from_dict(json.load(handle))
    with open(args.candidate, encoding="utf-8") as handle:
        candidate = json.load(handle)
    if isinstance(candidate, Mapping) and isinstance(candidate.get("factors"), Mapping):
        factors = candidate["factors"]
    elif isinstance(candidate, Mapping):
        factors = candidate
    else:
        raise ValueError("candidate file must contain a JSON object")
    result = model.explain(factors)
    if args.out:
        _write_json(args.out, result)
    else:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


def _benchmark_analysis(args: argparse.Namespace) -> int:
    analyses = _name_path_map(args.analysis, "analysis")
    checkpoints = _name_path_map(args.checkpoint, "checkpoint")
    unknown = sorted(set(checkpoints).difference(analyses))
    if unknown:
        raise ValueError("checkpoint selection names have no analysis: %s" % unknown)
    decisions = read_jsonl(args.decisions)
    analysis_records = {name: read_jsonl(path) for name, path in analyses.items()}
    group_by = [field.strip() for field in args.group_by.split(",") if field.strip()]
    report = benchmark_analyses(
        decisions, analysis_records, checkpoint_by_name=checkpoints,
        group_by=group_by, bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence, seed=args.seed)
    report["inputs"] = {
        "decisions": {
            "path": os.path.abspath(args.decisions),
            "sha256": file_sha256(args.decisions),
        },
        "analyses": {
            name: {"path": os.path.abspath(path), "sha256": file_sha256(path)}
            for name, path in analyses.items()
        },
    }
    _write_json(args.out, report)
    return 0


def _benchmark_matches(args: argparse.Namespace) -> int:
    report = benchmark_matches(
        read_jsonl(args.input), bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence, seed=args.seed)
    report["input"] = {
        "path": os.path.abspath(args.input),
        "sha256": file_sha256(args.input),
    }
    _write_json(args.out, report)
    return 0


def _validate_plan(args: argparse.Namespace) -> int:
    with open(args.plan, encoding="utf-8") as handle:
        payload = json.load(handle)
    report = plan_report(payload)
    report["plan"] = {
        "path": os.path.abspath(args.plan),
        "sha256": file_sha256(args.plan),
    }
    if args.out:
        _write_json(args.out, report)
    else:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    if not report["valid"]:
        return 2
    if args.strict and report["warnings"]:
        return 1
    return 0


def _name_path_map(values: Sequence[str], label: str) -> Dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("%s must use NAME=VALUE: %s" % (label, value))
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError("%s must use non-empty NAME=VALUE" % label)
        if name in result:
            raise ValueError("duplicate %s name: %s" % (label, name))
        result[name] = path
    return result


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
