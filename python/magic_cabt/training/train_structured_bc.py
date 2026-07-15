"""Train the structured encoder and policy using behavior cloning only."""
from __future__ import annotations

import argparse
import sys

from magic_cabt.models.structured_jepa import CardTextResolver, StructuredJEPAConfig
from . import train_jepa as core
from .train_reliable import merge_best, write_outputs


def build_parser():
    parser = argparse.ArgumentParser(prog="magic-cabt-train-structured-bc")
    parser.add_argument("--input", required=True, action="append")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--preset", default="local",
                        choices=("tiny", "local", "large"))
    parser.add_argument("--embedding-backend", default="hash")
    parser.add_argument("--arena-card-db", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-decisions", type=int, default=100000)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", choices=("auto", "on", "off"),
                        default="auto")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    _transitions, decisions, cards = core.collect_training_data(
        args.input, max_transitions=0, max_decisions=args.max_decisions,
        seed=args.seed)
    if not decisions:
        raise SystemExit("no trainable single-choice decisions")
    config = StructuredJEPAConfig.preset(args.preset)
    config.embedding_backend = args.embedding_backend
    resolver = CardTextResolver(cards, arena_db_path=args.arena_card_db)
    previous_extra = {}
    if args.resume:
        from magic_cabt.models.structured_jepa import MagicJEPA
        _previous, previous_extra = MagicJEPA.load_checkpoint(
            args.resume, map_location="cpu")
    model, metrics = core.train(
        [], decisions, config=config, card_resolver=resolver,
        resume=args.resume, epochs=args.epochs, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed, device=args.device,
        causal_weight=0.0, value_weight=0.0, policy_weight=1.0,
        eval_fraction=args.eval_fraction, eval_seed=args.eval_seed,
        amp=args.amp, grad_accum_steps=args.grad_accum_steps,
        max_steps_per_epoch=args.max_steps_per_epoch,
        log=lambda text: sys.stderr.write(text + "\n"))
    model, metrics = merge_best(model, metrics, previous_extra)
    metrics["modelFamily"] = "structured-bc-control-v1"
    metrics["objective"] = {"policyWeight": 1.0, "jepaWeight": 0.0,
                            "causalWeight": 0.0, "valueWeight": 0.0}
    metrics["inputs"] = core._input_manifest(args.input)
    write_outputs(model, metrics, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
