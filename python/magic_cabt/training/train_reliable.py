"""Reliable public wrapper around the structured JEPA trainer."""
from __future__ import annotations

import json
import os
import sys

from magic_cabt.models.structured_jepa import (
    CardTextResolver, MagicJEPA, StructuredJEPAConfig)
from . import train_jepa as core


def merge_best(model, metrics, previous_extra=None):
    """Keep the lowest-loss selected state from previous and current runs."""
    previous_state = (previous_extra or {}).get("trainingState") or {}
    previous_metric = previous_state.get("bestSelectionMetric")
    previous_best = previous_state.get("bestStateDict")
    previous_epoch = previous_state.get("bestEpoch")
    current_metric = metrics.get("bestSelectionMetric")
    current_best = getattr(model, "_best_state_dict", None)
    use_previous = (
        previous_best is not None and previous_metric is not None and
        (current_best is None or current_metric is None or
         float(previous_metric) <= float(current_metric)))
    if use_previous:
        model._best_state_dict = previous_best
        model._best_epoch = previous_epoch
        metrics["bestEpoch"] = previous_epoch
        metrics["bestSelectionMetric"] = float(previous_metric)
    state = dict(getattr(model, "_training_state", {}) or {})
    state.update({
        "bestStateDict": getattr(model, "_best_state_dict", None),
        "bestEpoch": metrics.get("bestEpoch"),
        "bestSelectionMetric": metrics.get("bestSelectionMetric"),
    })
    model._training_state = state
    return model, metrics


def train(*args, resume=None, **kwargs):
    previous_extra = {}
    if resume:
        _previous, previous_extra = MagicJEPA.load_checkpoint(
            resume, map_location="cpu")
    model, metrics = core.train(*args, resume=resume, **kwargs)
    return merge_best(model, metrics, previous_extra)


def write_outputs(model, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = os.path.join(out_dir, "checkpoint.pt")
    model.save_checkpoint(checkpoint, extra={
        "metrics": metrics,
        "trainingState": getattr(model, "_training_state", {}),
    })
    best_checkpoint = None
    best_state = getattr(model, "_best_state_dict", None)
    if best_state is not None:
        best_model = MagicJEPA(model.config)
        best_model.load_state_dict(best_state)
        best_checkpoint = os.path.join(out_dir, "best.pt")
        best_model.save_checkpoint(best_checkpoint, extra={
            "metrics": metrics,
            "selection": {"epoch": metrics.get("bestEpoch"),
                          "metric": metrics.get("bestSelectionMetric")},
        })
    metrics["checkpoint"] = os.path.abspath(checkpoint)
    metrics["bestCheckpoint"] = os.path.abspath(best_checkpoint) \
        if best_checkpoint else None
    with open(os.path.join(out_dir, "metrics.json"), "w",
              encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return checkpoint, best_checkpoint


def main(argv=None):
    args = core.build_parser().parse_args(argv)
    transitions, decisions, cards = core.collect_training_data(
        args.input, max_transitions=args.max_transitions,
        max_decisions=args.max_decisions, seed=args.seed)
    sys.stderr.write("loaded %d transitions and %d decisions\n" %
                     (len(transitions), len(decisions)))
    config = StructuredJEPAConfig.preset(args.preset)
    config.embedding_backend = args.embedding_backend
    resolver = CardTextResolver(cards, arena_db_path=args.arena_card_db)
    model, metrics = train(
        transitions, decisions, config=config, card_resolver=resolver,
        resume=args.resume, epochs=args.epochs, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed, device=args.device,
        eval_fraction=args.eval_fraction, eval_seed=args.eval_seed,
        amp=args.amp, grad_accum_steps=args.grad_accum_steps,
        max_steps_per_epoch=args.max_steps_per_epoch,
        log=lambda text: sys.stderr.write(text + "\n"))
    metrics["inputs"] = core._input_manifest(args.input)
    checkpoint, best = write_outputs(model, metrics, args.out)
    sys.stderr.write("wrote %s\n" % checkpoint)
    if best:
        sys.stderr.write("wrote %s\n" % best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
