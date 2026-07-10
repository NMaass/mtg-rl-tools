"""Train the PyTorch option-ranker on canonical DecisionRecords.

    magic-cabt-train-ranker --input decisions.jsonl --out runs/ranker-small

Trains over canonical action GROUPS (``training.action_dedup``): the loss for
a decision is the negative log-probability of the *group* containing the
human-chosen option, i.e. ``-log(sum of softmax mass over fungible members)``.
A player who clicked "the second identical token" trains the same action as
one who clicked the first, and the model is never rewarded for telling
interchangeable instances apart.

Fail-fast by design: records whose chosen index cannot be mapped are counted
and reported, never silently patched into a different label.
"""

import argparse
import json
import os
import random
import sys

from magic_cabt.models.configs import get_model_config
from magic_cabt.models.torch_ranker import (
    TORCH_AVAILABLE,
    OptionRanker,
    hash_features,
)
from magic_cabt.training.action_dedup import canonical_groups, group_index_of
from magic_cabt.training.features import (
    option_text,
    prompt_type,
    state_text,
)
from magic_cabt.training.io import iter_decision_records

__all__ = ["build_examples", "train", "build_parser", "main"]


def build_examples(records, stats=None):
    """Turn DecisionRecords into ranker examples (pure, no torch).

    Each example: ``{stateVec, optionVecs, groups, chosenGroup, gameId}``
    where ``groups`` is a list of index-lists (fungible options share one).
    """
    stats = stats if stats is not None else {}

    def bump(key):
        stats[key] = stats.get(key, 0) + 1

    examples = []
    for record in records:
        select = record.get("select") or {}
        options = select.get("option") or []
        selected = record.get("selectedIndices") or []
        if not options:
            bump("skipped_no_options")
            continue
        if len(selected) != 1:
            bump("skipped_multi_or_no_select")
            continue
        chosen = selected[0]
        if not isinstance(chosen, int) or not 0 <= chosen < len(options):
            bump("skipped_bad_index")
            continue
        groups = canonical_groups(select)
        chosen_group = group_index_of(groups, chosen)
        if chosen_group is None:
            bump("skipped_ungrouped_index")
            continue
        examples.append({
            "stateText": state_text(record),
            "promptType": prompt_type(record),
            "optionTexts": [option_text(option) for option in options],
            "groups": [group["indices"] for group in groups],
            "chosenGroup": chosen_group,
            "gameId": record.get("gameId") or record.get("matchId") or "",
        })
        bump("compiled")
    return examples


def _split_by_game(examples, eval_fraction, seed):
    """Split train/eval on game boundaries so states never leak across."""
    game_ids = sorted(set(example["gameId"] for example in examples))
    rng = random.Random(seed)
    rng.shuffle(game_ids)
    eval_count = max(1, int(len(game_ids) * eval_fraction)) if len(game_ids) > 1 else 0
    eval_games = set(game_ids[:eval_count])
    train = [e for e in examples if e["gameId"] not in eval_games]
    holdout = [e for e in examples if e["gameId"] in eval_games]
    return train, holdout


def _batch_tensors(batch, state_dim, option_dim, device):
    import torch

    max_options = max(len(example["optionTexts"]) for example in batch)
    states, options, masks = [], [], []
    for example in batch:
        state_line = "%s | %s" % (example["promptType"], example["stateText"])
        states.append(hash_features(state_line, state_dim))
        vecs = [hash_features(text, option_dim)
                for text in example["optionTexts"]]
        mask = [True] * len(vecs)
        while len(vecs) < max_options:
            vecs.append([0.0] * option_dim)
            mask.append(False)
        options.append(vecs)
        masks.append(mask)
    return (torch.tensor(states, dtype=torch.float32, device=device),
            torch.tensor(options, dtype=torch.float32, device=device),
            torch.tensor(masks, dtype=torch.bool, device=device))


def _group_loss(logits, batch):
    """-log P(chosen group) with group mass = logsumexp over members."""
    import torch

    losses = []
    denominator = torch.logsumexp(logits, dim=1)                 # [B]
    for row, example in enumerate(batch):
        members = example["groups"][example["chosenGroup"]]
        numerator = torch.logsumexp(logits[row, members], dim=0)
        losses.append(denominator[row] - numerator)
    return torch.stack(losses).mean()


def _evaluate(model, examples, state_dim, option_dim, device, batch_size):
    import torch

    if not examples:
        return {"examples": 0}
    top1 = top3 = 0
    reciprocal_rank_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            states, options, masks = _batch_tensors(
                batch, state_dim, option_dim, device)
            logits = model(states, options, masks)
            probs = torch.softmax(logits, dim=1)
            for row, example in enumerate(batch):
                group_mass = [float(probs[row, members].sum())
                              for members in example["groups"]]
                ranking = sorted(range(len(group_mass)),
                                 key=lambda g: (-group_mass[g], g))
                rank = ranking.index(example["chosenGroup"]) + 1
                top1 += 1 if rank == 1 else 0
                top3 += 1 if rank <= 3 else 0
                reciprocal_rank_sum += 1.0 / rank
    count = len(examples)
    return {
        "examples": count,
        "groupTop1": top1 / count,
        "groupTop3": top3 / count,
        "groupMRR": reciprocal_rank_sum / count,
    }


def train(examples, config, epochs=10, lr=1e-3, batch_size=64,
          eval_fraction=0.1, seed=0, device=None, log=None):
    """Train an ``OptionRanker``; returns ``(model, metrics)``."""
    if not TORCH_AVAILABLE:
        raise ImportError("training requires PyTorch: pip install torch")
    import torch

    if not examples:
        raise ValueError("no trainable examples")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = int(config["stateFeatureDim"])
    option_dim = int(config["optionFeatureDim"])

    train_set, holdout = _split_by_game(examples, eval_fraction, seed)
    if not train_set:
        train_set, holdout = examples, []

    torch.manual_seed(seed)
    model = OptionRanker(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)

    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_set)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(train_set), batch_size):
            batch = train_set[start:start + batch_size]
            states, options, masks = _batch_tensors(
                batch, state_dim, option_dim, device)
            loss = _group_loss(model(states, options, masks), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            batches += 1
        if log:
            log("epoch %d/%d loss=%.4f" % (
                epoch + 1, epochs, total_loss / max(1, batches)))

    model.eval()
    metrics = {
        "config": config["name"],
        "device": device,
        "epochs": epochs,
        "trainExamples": len(train_set),
        "train": _evaluate(model, train_set, state_dim, option_dim,
                           device, batch_size),
        "holdout": _evaluate(model, holdout, state_dim, option_dim,
                             device, batch_size),
    }
    return model, metrics


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-train-ranker",
        description="Train the PyTorch option-ranker on DecisionRecord JSONL.")
    parser.add_argument("--input", required=True, action="append",
                        help="DecisionRecord JSONL (repeatable)")
    parser.add_argument("--out", required=True,
                        help="output dir for checkpoint.pt and metrics.json")
    parser.add_argument("--config", default="small",
                        help="model config name (small|full)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: auto)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = get_model_config(args.config)

    stats = {}
    examples = []
    for path in args.input:
        examples.extend(build_examples(iter_decision_records(path), stats))
    sys.stderr.write("examples: %s\n" % json.dumps(stats, sort_keys=True))
    if not examples:
        sys.stderr.write("no trainable examples found\n")
        return 2

    model, metrics = train(
        examples, config, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, eval_fraction=args.eval_fraction,
        seed=args.seed, device=args.device,
        log=lambda message: sys.stderr.write(message + "\n"))
    metrics["exampleStats"] = stats

    os.makedirs(args.out, exist_ok=True)
    checkpoint_path = os.path.join(args.out, "checkpoint.pt")
    metrics_path = os.path.join(args.out, "metrics.json")
    model.save_checkpoint(checkpoint_path)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    sys.stderr.write("wrote %s and %s\n" % (checkpoint_path, metrics_path))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
