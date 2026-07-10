"""Train a dependency-free behavior-cloning option scorer.

The first trainer is intentionally small: it reads compiled single-choice IL
JSONL, fits a bag-of-words legal-option ranker, and writes a JSON checkpoint
that can be loaded by ``BCAgent``. It is a smoke-testable baseline before adding
PyTorch/JAX policies.
"""

import argparse
import json
import os
import sys

from magic_cabt.dataset import read_dataset
from magic_cabt.models import BagOfWordsBCPolicy

__all__ = ["evaluate_policy", "main"]


def evaluate_policy(policy, examples):
    rows = list(examples)
    total = len(rows)
    top1 = 0
    top3 = 0
    top3_total = 0
    rr = 0.0
    by_prompt = {}
    for example in rows:
        chosen = example.get("chosenIndex")
        option_count = len(example.get("optionTexts") or [])
        if not isinstance(chosen, int) or chosen < 0 or chosen >= option_count:
            continue
        ranking = policy.rank_example(example)
        rank = ranking.index(chosen) + 1 if chosen in ranking else None
        correct = rank == 1
        if correct:
            top1 += 1
        if option_count >= 3:
            top3_total += 1
            if rank is not None and rank <= 3:
                top3 += 1
        if rank is not None:
            rr += 1.0 / rank
        prompt = str(example.get("promptType") or "UNKNOWN")
        group = by_prompt.setdefault(prompt, {"examples": 0, "top1": 0})
        group["examples"] += 1
        group["top1"] += 1 if correct else 0
    return {
        "examples": total,
        "top1Accuracy": _ratio(top1, total),
        "top3Accuracy": _ratio(top3, top3_total),
        "top3Examples": top3_total,
        "meanReciprocalRank": _ratio(rr, total),
        "accuracyByPromptType": {
            key: {
                "examples": value["examples"],
                "top1Accuracy": _ratio(value["top1"], value["examples"]),
            }
            for key, value in sorted(by_prompt.items())
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m magic_cabt.training.train_bc",
        description="Train a lightweight behavior-cloning policy from "
                    "compiled single-choice IL JSONL.",
    )
    parser.add_argument("--input", required=True,
                        help="single-choice IL JSONL from compile_il")
    parser.add_argument("--out", required=True,
                        help="output directory for checkpoint.json and metrics.json")
    parser.add_argument("--min-token-count", type=int, default=1)
    args = parser.parse_args(argv)

    examples = list(read_dataset(args.input))
    policy = BagOfWordsBCPolicy.train(examples,
                                      min_token_count=args.min_token_count)
    metrics = evaluate_policy(policy, examples)
    metrics["modelType"] = "bag_of_words_bc"
    metrics["minTokenCount"] = args.min_token_count

    os.makedirs(args.out, exist_ok=True)
    checkpoint = os.path.join(args.out, "checkpoint.json")
    metrics_path = os.path.join(args.out, "metrics.json")
    policy.save(checkpoint)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    sys.stderr.write("wrote %s and %s\n" % (checkpoint, metrics_path))
    return 0


def _ratio(numerator, denominator):
    if not denominator:
        return None
    return float(numerator) / float(denominator)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
