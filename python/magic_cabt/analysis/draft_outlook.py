"""Mid-draft deck outlook: auto-build a deck from the current pool.

    magic-cabt-draft-outlook --checkpoint runs/draft-model/checkpoint.pt \
        --pool "97967,97932,97947,..."

Greedily selects a main deck from the pool with the card-selection model's
``deckBuild`` mode (the same decisions it trained on), then summarizes the
result — color spread, mana curve, and the ranked inclusion order — so a
drafter can ask "what does my deck look like right now?" after any pick.
"""

import argparse
import json
import sys

from magic_cabt.models.draft_model import (
    TORCH_AVAILABLE,
    CardSelectionModel,
    DraftTensorizer,
    mana_value,
)
from magic_cabt.training.train_draft import make_resolver

__all__ = ["deck_outlook", "build_parser", "main"]

DEFAULT_SPELL_TARGET = 23


def deck_outlook(model, tensorizer, pool, spell_target=DEFAULT_SPELL_TARGET):
    """Greedy deck build over ``pool`` grpIds; returns a summary dict."""
    if not TORCH_AVAILABLE:
        raise ImportError("deck outlook requires PyTorch")
    import torch

    resolver = tensorizer.resolver
    remaining = list(pool)
    built, inclusions = [], []
    model.eval()
    with torch.no_grad():
        while remaining and len(inclusions) < spell_target:
            example = {
                "mode": "deckBuild",
                "contextIds": list(built),
                "candidateIds": list(remaining),
                "positives": [0],  # unused at inference
            }
            tensors = tensorizer.batch_examples([example])
            logits = model.score_candidates(*tensors)[0]
            best = int(torch.argmax(logits))
            card = remaining.pop(best)
            built.append(card)
            inclusions.append({
                "grpId": card,
                "name": resolver.describe(card).get("name"),
                "score": float(logits[best]),
            })

    colors, curve = {}, {}
    for entry in inclusions:
        card = resolver.describe(entry["grpId"])
        for letter in str(card.get("colors") or "").upper():
            colors[letter] = colors.get(letter, 0) + 1
        value = mana_value(card.get("manaCost"))
        if value is not None:
            bucket = str(min(int(value), 7))
            curve[bucket] = curve.get(bucket, 0) + 1
    return {
        "poolSize": len(pool),
        "spellTarget": spell_target,
        "deck": inclusions,
        "cuts": [{"grpId": card, "name": resolver.describe(card).get("name")}
                 for card in remaining],
        "colors": dict(sorted(colors.items(), key=lambda kv: -kv[1])),
        "curve": {key: curve[key] for key in sorted(curve)},
    }


def _parse_pool(text):
    return [int(part) for part in str(text).replace(" ", "").split(",")
            if part.strip()]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-draft-outlook",
        description="Auto-build a deck from a draft pool with the "
                    "card-selection model.")
    parser.add_argument("--checkpoint", required=True,
                        help="card-selection checkpoint.pt")
    parser.add_argument("--pool", default=None,
                        help="comma-separated pool grpIds")
    parser.add_argument("--picks", default=None,
                        help="draft_picks.jsonl: use the latest pick's "
                             "pool+picked as the current pool")
    parser.add_argument("--spell-target", type=int,
                        default=DEFAULT_SPELL_TARGET)
    parser.add_argument("--arena-card-db", default=None)
    parser.add_argument("--card-cache", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.pool:
        pool = _parse_pool(args.pool)
    elif args.picks:
        last = None
        with open(args.picks, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = json.loads(line)
        if last is None:
            sys.stderr.write("no picks found in %s\n" % args.picks)
            return 2
        pool = list(last.get("pool") or []) + list(last.get("picked") or [])
    else:
        sys.stderr.write("pass --pool or --picks\n")
        return 2

    model, _extra = CardSelectionModel.load_checkpoint(args.checkpoint)
    resolver = make_resolver(args.arena_card_db, args.card_cache)
    tensorizer = DraftTensorizer(model.config, resolver)
    outlook = deck_outlook(model, tensorizer, pool,
                           spell_target=args.spell_target)
    print(json.dumps(outlook, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
