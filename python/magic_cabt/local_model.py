"""One-command local model lab for the Arena mirror."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from magic_cabt.training.local_evolve import LocalEvolver

DEFAULT_MODEL_DIR = "~/.magic-cabt/local-model"


def _add_common(parser):
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--preset", default="local", choices=("tiny", "local", "large"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--embedding-backend", default="hash")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-local-model",
        description="Run, inspect, or train the local evolving Magic model.")
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init")
    _add_common(initialize)
    initialize.add_argument("--force", action="store_true")

    train = commands.add_parser("train")
    _add_common(train)
    train.add_argument("bundle")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--replay-bundles", type=int, default=20)
    train.add_argument("--arena-card-db", default=None)

    gui = commands.add_parser("gui")
    _add_common(gui)
    gui.add_argument("--epochs-per-game", type=int, default=1)
    gui.add_argument("--replay-bundles", type=int, default=20)
    gui.add_argument("--top-k", type=int, default=5)
    gui.add_argument("--no-auto-train", action="store_true")

    replay = commands.add_parser("replay")
    _add_common(replay)
    replay.add_argument("bundle")

    analyze = commands.add_parser(
        "analyze",
        help="score a recorded bundle's decisions into its analysis.jsonl "
             "cache, so replays show ranked choices without a live session")
    _add_common(analyze)
    analyze.add_argument("bundle")
    analyze.add_argument("--checkpoint", default=None,
                         help="checkpoint.pt (default: the local model's)")
    analyze.add_argument("--top-k", type=int, default=5)

    status = commands.add_parser("status")
    status.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    return parser


def _make_evolver(args, epochs=1, replay_bundles=20,
                  arena_card_db=None):
    return LocalEvolver(
        model_dir=args.model_dir,
        preset=getattr(args, "preset", "local"),
        epochs=epochs,
        replay_bundles=replay_bundles,
        device=getattr(args, "device", None),
        embedding_backend=getattr(args, "embedding_backend", "hash"),
        arena_card_db=arena_card_db,
        log=lambda text: sys.stderr.write("[local-model] %s\n" % text))


def _configure_environment(args, auto_train):
    evolver = _make_evolver(
        args, epochs=getattr(args, "epochs_per_game", 1),
        replay_bundles=getattr(args, "replay_bundles", 20))
    checkpoint = evolver.ensure_checkpoint()
    os.environ.update({
        "MAGIC_CABT_MODEL_CHECKPOINT": checkpoint,
        "MAGIC_CABT_MODEL_DIR": os.path.abspath(
            os.path.expanduser(args.model_dir)),
        "MAGIC_CABT_MODEL_PRESET": args.preset,
        "MAGIC_CABT_MODEL_DEVICE": args.device or "",
        "MAGIC_CABT_EMBEDDING_BACKEND": args.embedding_backend,
        "MAGIC_CABT_AUTO_TRAIN": "1" if auto_train else "0",
        "MAGIC_CABT_MODEL_EPOCHS": str(
            getattr(args, "epochs_per_game", 1)),
        "MAGIC_CABT_MODEL_REPLAY_BUNDLES": str(
            getattr(args, "replay_bundles", 20)),
        "MAGIC_CABT_ANALYSIS_TOP_K": str(getattr(args, "top_k", 5)),
    })


def _analyze_bundle(args):
    """Backfill ``analysis.jsonl`` for a recorded bundle, offline.

    Every decision is scored with the checkpoint and cached under its
    decision fingerprint + checkpoint id — the exact identity the replay
    overlays look up — so already-analyzed decisions are free and a new
    checkpoint re-analyzes cleanly alongside the old records.
    """
    import time

    from magic_cabt.analysis import (AnalysisCache, analysis_cache_key,
                                     load_checkpoint_scorer,
                                     make_analysis_record)

    bundle = os.path.abspath(os.path.expanduser(args.bundle))
    decisions_path = os.path.join(bundle, "decisions.jsonl")
    if not os.path.isfile(decisions_path):
        sys.stderr.write("no decisions.jsonl in %s\n" % bundle)
        return 2
    checkpoint = args.checkpoint or _make_evolver(args).ensure_checkpoint()
    card_cache = os.path.join(bundle, "card_cache.json")
    scorer = load_checkpoint_scorer(
        checkpoint, device=args.device or None,
        card_cache=card_cache if os.path.isfile(card_cache) else None)
    cache = AnalysisCache(os.path.join(bundle, "analysis.jsonl"))

    # Read the decisions raw, exactly as the replay overlay does: the cache
    # is looked up by the fingerprint of the *recorded* decision, so any
    # normalization here would orphan the records we are writing.
    scored, cached = 0, 0
    for record in _iter_raw_jsonl(decisions_path):
        key = analysis_cache_key(record, scorer.model_info)
        if cache.get(key) is not None:
            cached += 1
            continue
        started = time.perf_counter()
        scores = scorer.score(record)
        latency = int((time.perf_counter() - started) * 1000)
        value_method = getattr(scorer, "state_value", None)
        cache.add(make_analysis_record(
            record, scores, scorer.model_info, top_k=args.top_k,
            latency_ms=latency,
            value=value_method(record) if value_method else None,
            source="backfill"), persist=True)
        scored += 1
    summary = {"bundle": bundle, "checkpoint": checkpoint,
               "scored": scored, "alreadyCached": cached,
               "model": scorer.model_info}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _iter_raw_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv=None):
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)

    if args.command == "init":
        evolver = _make_evolver(args)
        if args.force and os.path.isdir(evolver.model_dir):
            shutil.rmtree(evolver.model_dir)
            os.makedirs(evolver.model_dir)
        print(evolver.ensure_checkpoint())
        return 0

    if args.command == "train":
        evolver = _make_evolver(
            args, epochs=args.epochs,
            replay_bundles=args.replay_bundles,
            arena_card_db=args.arena_card_db)
        evolver.train_once(args.bundle)
        return 0

    if args.command == "analyze":
        return _analyze_bundle(args)

    if args.command == "status":
        path = os.path.join(os.path.abspath(
            os.path.expanduser(args.model_dir)), "status.json")
        if not os.path.exists(path):
            print("not initialized")
            return 1
        with open(path, "r", encoding="utf-8") as handle:
            print(json.dumps(json.load(handle), indent=2, sort_keys=True))
        return 0

    _configure_environment(
        args, auto_train=(args.command == "gui" and
                          not args.no_auto_train))
    from magic_cabt.arena_mirror.local_model_hooks import (
        install_local_model_gui_hooks, install_local_model_hooks)
    install_local_model_hooks()
    if args.command == "gui":
        install_local_model_gui_hooks()

    from magic_cabt.arena_mirror.__main__ import main as mirror_main
    if args.command == "gui":
        return mirror_main(["gui"] + passthrough)
    return mirror_main(["replay", args.bundle] + passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
