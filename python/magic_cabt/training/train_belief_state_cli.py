"""Public complete-game CLI for belief-state training."""
from __future__ import annotations

import json
import os
import sys

from magic_cabt.models.belief import BeliefInformationStateModel
from magic_cabt.models.structured_jepa import CardTextResolver, StructuredJEPAConfig
from . import train_belief_state as trainer
from . import train_jepa as core
from .sequence_data import collect_complete_decision_games
from .train_information_state import game_key


def main(argv=None):
    args = trainer.build_parser().parse_args(argv)
    labels, vocabulary = trainer.load_vocabulary(args.vocabulary)
    decisions, cards, collection = collect_complete_decision_games(
        args.input, game_key, max_decisions=args.max_decisions)
    compiled, label_summary = trainer.compile_belief_records(decisions, labels)
    if not compiled:
        raise SystemExit("no complete trainable decision games")
    if not label_summary["labeledCells"]:
        raise SystemExit("no oracle-label-only belief targets")

    config = StructuredJEPAConfig.preset(args.preset)
    config.embedding_backend = args.embedding_backend
    resolver = CardTextResolver(cards, arena_db_path=args.arena_card_db)
    model, metrics = trainer.train(
        compiled, labels, config=config, resolver=resolver,
        epochs=args.epochs, batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        eval_fraction=args.eval_fraction, seed=args.seed, device=args.device,
        lr=args.lr, weight_decay=args.weight_decay,
        belief_weight=args.belief_weight, amp=args.amp,
        grad_accum_steps=args.grad_accum_steps,
        memory_layers=args.memory_layers,
        log=lambda text: sys.stderr.write(text + "\n"))
    metrics["collection"] = collection
    metrics["labelSummary"] = label_summary
    metrics["vocabulary"] = {
        "path": os.path.abspath(args.vocabulary),
        "sha256": trainer.sha256_file(args.vocabulary),
        "schemaVersion": vocabulary["schemaVersion"],
    }
    metrics["inputs"] = core._input_manifest(args.input)

    os.makedirs(args.out, exist_ok=True)
    checkpoint = os.path.join(args.out, "checkpoint.pt")
    model.save_checkpoint(checkpoint, extra={
        "metrics": metrics, "trainingState": model._training_state})
    best = BeliefInformationStateModel(
        labels, config=model.config, memory_layers=model.memory_layers)
    best.load_state_dict(model._best_state_dict)
    best_path = os.path.join(args.out, "best.pt")
    best.save_checkpoint(best_path, extra={
        "metrics": metrics,
        "selection": {"epoch": metrics["bestEpoch"],
                      "metric": metrics["bestSelectionMetric"]}})
    metrics["checkpoint"] = os.path.abspath(checkpoint)
    metrics["bestCheckpoint"] = os.path.abspath(best_path)
    with open(os.path.join(args.out, "metrics.json"), "w",
              encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
