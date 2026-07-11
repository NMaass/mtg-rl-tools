"""Train the card-selection model on limited-play datasets.

    magic-cabt-train-draft --input runs/draft-data --out runs/draft-model

Consumes the JSONL files written by ``magic-cabt-build-draft-dataset`` and
turns every record kind into one example shape: a context card set, candidate
cards, and the human-chosen positives. Deck builds are unrolled into
sequential inclusion decisions (deck-so-far vs remaining pool), which is also
what powers mid-draft deck outlook.

Fail-fast by design: a record whose chosen cards cannot be aligned with its
candidates is counted and skipped, never patched to a different label.
"""

import argparse
import json
import os
import random
import sys

from magic_cabt.models.draft_model import (
    MODES,
    TORCH_AVAILABLE,
    CardSelectionModel,
    DraftCardResolver,
    DraftModelConfig,
    DraftTensorizer,
)
from magic_cabt.models.structured_config import CardTextResolver

__all__ = ["build_selection_examples", "train", "build_parser", "main"]

DATASET_FILES = ("draft_picks.jsonl", "deck_builds.jsonl", "sideboards.jsonl")


def build_selection_examples(records, stats=None, max_deck_build_steps=None):
    """Turn dataset records into unified selection examples (pure)."""
    stats = stats if stats is not None else {}

    def bump(key):
        stats[key] = stats.get(key, 0) + 1

    examples = []
    for record in records:
        kind = record.get("kind")
        if kind == "draftPick":
            example = _pick_example(record)
            if example is None:
                bump("skipped_unaligned_pick")
            else:
                examples.append(example)
                bump("examples_draftPick")
        elif kind == "deckBuild":
            if not record.get("pool"):
                bump("skipped_build_without_pool")
                continue
            steps = _deck_build_examples(record, max_deck_build_steps)
            if steps is None:
                bump("skipped_unaligned_build")
            else:
                examples.extend(steps)
                stats["examples_deckBuild"] = stats.get(
                    "examples_deckBuild", 0) + len(steps)
        elif kind == "sideboard":
            example = _sideboard_example(record)
            if example is None:
                bump("skipped_unaligned_sideboard")
            else:
                examples.append(example)
                bump("examples_sideboard")
        else:
            bump("skipped_unknown_kind")
    return examples


def _align_positives(candidates, chosen):
    """Indices in ``candidates`` matching the ``chosen`` multiset, or None."""
    used, positives = set(), []
    for card in chosen:
        found = None
        for index, candidate in enumerate(candidates):
            if index not in used and candidate == card:
                found = index
                break
        if found is None:
            return None
        used.add(found)
        positives.append(found)
    return positives


def _pick_example(record):
    pack = list(record.get("pack") or [])
    positives = _align_positives(pack, record.get("picked") or [])
    if not pack or not positives:
        return None
    return {
        "mode": "draftPick",
        "contextIds": list(record.get("pool") or []),
        "candidateIds": pack,
        "positives": positives,
        "packNumber": record.get("packNumber"),
        "pickNumber": record.get("pickNumber"),
        "groupKey": "%s:%s" % (record.get("source"), record.get("draftId")),
    }


def _deck_build_examples(record, max_steps=None):
    remaining = list(record.get("pool") or [])
    deck_left = list(record.get("mainDeck") or [])
    built, steps = [], []
    for card in record.get("mainDeck") or []:
        if card in remaining:
            positives = _align_positives(
                remaining, [item for item in deck_left if item in remaining])
            if positives is None:
                return None
            steps.append({
                "mode": "deckBuild",
                "contextIds": list(built),
                "candidateIds": list(remaining),
                "positives": positives,
                "groupKey": "%s:%s" % (record.get("source"),
                                       record.get("draftId")),
            })
            remaining.remove(card)
        # Cards granted outside the pool (basic lands) still join the
        # context so later decisions see the mana base.
        built.append(card)
        deck_left.remove(card)
        if max_steps is not None and len(steps) >= max_steps:
            break
    return steps


def _sideboard_example(record):
    candidates = list(record.get("offeredDeck") or []) + \
        list(record.get("offeredSideboard") or [])
    positives = _align_positives(candidates, record.get("chosenDeck") or [])
    if not candidates or not positives:
        return None
    return {
        "mode": "sideboard",
        "contextIds": list(record.get("offeredDeck") or []),
        "candidateIds": candidates,
        "positives": positives,
        "groupKey": str(record.get("matchId")),
    }


def _split_by_group(examples, eval_fraction, seed):
    """Split on draft/match boundaries so decisions never leak across."""
    group_keys = sorted(set(example["groupKey"] for example in examples))
    rng = random.Random(seed)
    rng.shuffle(group_keys)
    eval_count = max(1, int(len(group_keys) * eval_fraction)) \
        if len(group_keys) > 1 else 0
    eval_groups = set(group_keys[:eval_count])
    train = [e for e in examples if e["groupKey"] not in eval_groups]
    holdout = [e for e in examples if e["groupKey"] in eval_groups]
    return train, holdout


def _multi_positive_loss(logits, batch):
    """Mean over positives of -log P(candidate), per example."""
    import torch

    denominator = torch.logsumexp(logits, dim=1)
    losses = []
    for row, example in enumerate(batch):
        positives = torch.tensor(example["positives"],
                                 dtype=torch.long, device=logits.device)
        losses.append((denominator[row] - logits[row, positives]).mean())
    return torch.stack(losses).mean()


def _evaluate(model, tensorizer, examples, device, batch_size):
    import torch

    if not examples:
        return {"examples": 0}
    top1 = 0
    recall_sum = 0.0
    reciprocal_rank_sum = 0.0
    by_mode = {}
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            tensors = tensorizer.batch_examples(batch, device=device)
            logits = model.score_candidates(*tensors)
            for row, example in enumerate(batch):
                count = len(example["candidateIds"])
                positives = set(example["positives"])
                ranking = sorted(range(count),
                                 key=lambda i: (-float(logits[row, i]), i))
                hit = 1 if ranking[0] in positives else 0
                top1 += hit
                top_p = set(ranking[:len(positives)])
                recall = len(top_p & positives) / float(len(positives))
                recall_sum += recall
                best_rank = min(ranking.index(p) for p in positives) + 1
                reciprocal_rank_sum += 1.0 / best_rank
                mode_bucket = by_mode.setdefault(
                    example["mode"], {"examples": 0, "top1": 0, "recall": 0.0})
                mode_bucket["examples"] += 1
                mode_bucket["top1"] += hit
                mode_bucket["recall"] += recall
    count = len(examples)
    for bucket in by_mode.values():
        bucket["top1"] = bucket["top1"] / bucket["examples"]
        bucket["recall"] = bucket["recall"] / bucket["examples"]
    return {
        "examples": count,
        "top1": top1 / count,
        "recallAtChosen": recall_sum / count,
        "mrr": reciprocal_rank_sum / count,
        "byMode": by_mode,
    }


def train(examples, config, resolver, epochs=20, lr=3e-4, batch_size=16,
          eval_fraction=0.15, seed=0, device=None, log=None,
          resume_path=None):
    """Train a ``CardSelectionModel``; returns ``(model, tensorizer, metrics)``."""
    if not TORCH_AVAILABLE:
        raise ImportError("training requires PyTorch: pip install torch")
    import torch

    if not examples:
        raise ValueError("no trainable examples")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_set, holdout = _split_by_group(examples, eval_fraction, seed)
    if not train_set:
        train_set, holdout = examples, []

    torch.manual_seed(seed)
    if resume_path:
        model, _extra = CardSelectionModel.load_checkpoint(resume_path)
        config = model.config
    else:
        model = CardSelectionModel(config)
    model = model.to(device)
    tensorizer = DraftTensorizer(config, resolver)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)

    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_set)
        total_loss, batches = 0.0, 0
        for start in range(0, len(train_set), batch_size):
            batch = train_set[start:start + batch_size]
            tensors = tensorizer.batch_examples(batch, device=device)
            loss = _multi_positive_loss(
                model.score_candidates(*tensors), batch)
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
        "device": device,
        "epochs": epochs,
        "trainExamples": len(train_set),
        "train": _evaluate(model, tensorizer, train_set, device, batch_size),
        "holdout": _evaluate(model, tensorizer, holdout, device, batch_size),
    }
    return model, tensorizer, metrics


def load_dataset_records(path):
    """Yield records from a dataset dir or a single JSONL file."""
    if os.path.isdir(path):
        for filename in DATASET_FILES:
            target = os.path.join(path, filename)
            if os.path.exists(target):
                for record in load_dataset_records(target):
                    yield record
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def make_resolver(arena_db_path=None, card_cache_path=None):
    """Best-available card resolver: MTGA SQLite, JSON cache, or bare ids."""
    database = None
    try:
        from magic_cabt.arena_mirror.cards import CardDatabase
        database = CardDatabase(db_path=arena_db_path,
                                cache_path=card_cache_path)
    except (IOError, OSError):
        database = None
    text_resolver = CardTextResolver(
        arena_db_path=arena_db_path or _default_arena_db())
    return DraftCardResolver(database, text_resolver)


def _default_arena_db():
    from magic_cabt.arena_mirror.cards import default_mtga_card_db_path
    return default_mtga_card_db_path()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-train-draft",
        description="Train the limited card-selection model on draft "
                    "datasets.")
    parser.add_argument("--input", required=True, action="append",
                        help="dataset directory or JSONL file (repeatable)")
    parser.add_argument("--out", required=True,
                        help="output dir for checkpoint.pt and metrics.json")
    parser.add_argument("--preset", default="local",
                        help="model preset (tiny|local)")
    parser.add_argument("--modes", default=",".join(MODES),
                        help="comma-separated record kinds to train on")
    parser.add_argument("--embedding-backend", default="hash",
                        help="hash (default) or sentence-transformer")
    parser.add_argument("--arena-card-db", default=None,
                        help="Raw_CardDatabase_*.mtga path (default: auto)")
    parser.add_argument("--card-cache", default=None,
                        help="card_cache.json fallback when MTGA is absent")
    parser.add_argument("--resume", default=None,
                        help="existing checkpoint.pt to continue from")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--max-deck-build-steps", type=int, default=None,
                        help="cap unrolled deck-build steps per deck")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: auto)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    modes = set(args.modes.split(","))

    stats = {}
    records = []
    for path in args.input:
        records.extend(record for record in load_dataset_records(path)
                       if record.get("kind") in modes)
    examples = build_selection_examples(
        records, stats, max_deck_build_steps=args.max_deck_build_steps)
    sys.stderr.write("examples: %s\n" % json.dumps(stats, sort_keys=True))
    if not examples:
        sys.stderr.write("no trainable examples found\n")
        return 2

    config = DraftModelConfig.preset(args.preset)
    config.embedding_backend = args.embedding_backend
    resolver = make_resolver(args.arena_card_db, args.card_cache)
    model, _tensorizer, metrics = train(
        examples, config, resolver, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, eval_fraction=args.eval_fraction,
        seed=args.seed, device=args.device, resume_path=args.resume,
        log=lambda message: sys.stderr.write(message + "\n"))
    metrics["exampleStats"] = stats

    os.makedirs(args.out, exist_ok=True)
    checkpoint_path = os.path.join(args.out, "checkpoint.pt")
    metrics_path = os.path.join(args.out, "metrics.json")
    model.save_checkpoint(checkpoint_path, extra={"metrics": metrics})
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    sys.stderr.write("wrote %s and %s\n" % (checkpoint_path, metrics_path))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
