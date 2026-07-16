"""Public complete-game CLI for recurrent information-state training."""
from __future__ import annotations

import json
import sys

from magic_cabt.models.structured_jepa import CardTextResolver, StructuredJEPAConfig
from . import train_information_state as trainer
from . import train_jepa as core
from .sequence_data import collect_complete_decision_games


def main(argv=None):
    args = trainer.build_parser().parse_args(argv)
    decisions, cards, collection = collect_complete_decision_games(
        args.input, trainer.game_key, max_decisions=args.max_decisions)
    if not decisions:
        raise SystemExit("no complete trainable decision games")
    config = StructuredJEPAConfig.preset(args.preset)
    config.embedding_backend = args.embedding_backend
    resolver = CardTextResolver(cards, arena_db_path=args.arena_card_db)
    model, metrics = trainer.train(
        decisions, config=config, resolver=resolver, epochs=args.epochs,
        batch_size=args.batch_size, sequence_length=args.sequence_length,
        eval_fraction=args.eval_fraction, seed=args.seed, device=args.device,
        lr=args.lr, weight_decay=args.weight_decay, amp=args.amp,
        grad_accum_steps=args.grad_accum_steps,
        memory_layers=args.memory_layers,
        log=lambda text: sys.stderr.write(text + "\n"))
    metrics["collection"] = collection
    metrics["inputs"] = core._input_manifest(args.input)
    trainer._write_outputs(model, metrics, args.out)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
