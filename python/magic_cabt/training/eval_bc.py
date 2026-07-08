"""Evaluate baseline behavior-cloning policies on compiled IL JSONL.

Metrics reported by the CLI are the contract future model-backed evaluators
should preserve: examples, top-1 accuracy, top-3 accuracy for prompts with at
least three options, mean reciprocal rank, accuracy by prompt type, accuracy
by chosen option type, pass-choice rate, and average option count.
"""

import argparse
import json
import random

from magic_cabt.agents.baseline_policy import rank_options
from magic_cabt.dataset import read_dataset

__all__ = [
    "evaluate",
    "format_metrics",
    "main",
]


def evaluate(examples, policy="first", seed=0):
    rng = random.Random(seed)
    rows = list(examples)
    totals = {
        "examples": len(rows),
        "top1_correct": 0,
        "top3_correct": 0,
        "top3_examples": 0,
        "reciprocal_rank_sum": 0.0,
        "option_count_sum": 0,
        "pass_choices": 0,
    }
    prompt_stats = {}
    chosen_type_stats = {}

    for example in rows:
        option_count = len(example.get("optionTexts") or [])
        chosen = example.get("chosenIndex")
        if option_count <= 0:
            continue
        ranking = rank_options(example, policy=policy, rng=rng)
        _assert_legal_ranking(ranking, option_count)
        rank = ranking.index(chosen) + 1 if chosen in ranking else None
        top1 = rank == 1
        top3 = rank is not None and rank <= 3

        totals["top1_correct"] += 1 if top1 else 0
        if option_count >= 3:
            totals["top3_examples"] += 1
            totals["top3_correct"] += 1 if top3 else 0
        if rank is not None:
            totals["reciprocal_rank_sum"] += 1.0 / rank
        totals["option_count_sum"] += option_count

        option_types = example.get("optionTypes") or []
        chosen_type = option_types[chosen] if isinstance(chosen, int) and chosen < len(option_types) else "UNKNOWN"
        if "PASS" in str(chosen_type).upper():
            totals["pass_choices"] += 1

        _add_group(prompt_stats, example.get("promptType") or "UNKNOWN", top1)
        _add_group(chosen_type_stats, chosen_type, top1)

    count = totals["examples"]
    return {
        "examples": count,
        "top1Accuracy": _ratio(totals["top1_correct"], count),
        "top3Accuracy": _ratio(totals["top3_correct"], totals["top3_examples"]),
        "top3Examples": totals["top3_examples"],
        "meanReciprocalRank": _ratio(totals["reciprocal_rank_sum"], count),
        "accuracyByPromptType": _finish_groups(prompt_stats),
        "accuracyByChosenOptionType": _finish_groups(chosen_type_stats),
        "passChoiceRate": _ratio(totals["pass_choices"], count),
        "averageOptionCount": _ratio(totals["option_count_sum"], count),
    }


def format_metrics(metrics):
    return json.dumps(metrics, sort_keys=True, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--policy", default="first",
                        choices=("first", "random"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    metrics = evaluate(read_dataset(args.input), policy=args.policy, seed=args.seed)
    print(format_metrics(metrics))
    return 0


def _assert_legal_ranking(ranking, option_count):
    for index in ranking:
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("policy produced a non-int index")
        if index < 0 or index >= option_count:
            raise ValueError("policy produced illegal option index %s" % index)


def _add_group(groups, name, correct):
    group = groups.setdefault(str(name), {"examples": 0, "correct": 0})
    group["examples"] += 1
    group["correct"] += 1 if correct else 0


def _finish_groups(groups):
    result = {}
    for name, group in sorted(groups.items()):
        result[name] = {
            "examples": group["examples"],
            "accuracy": _ratio(group["correct"], group["examples"]),
        }
    return result


def _ratio(numerator, denominator):
    if not denominator:
        return None
    return float(numerator) / float(denominator)


if __name__ == "__main__":
    raise SystemExit(main())

