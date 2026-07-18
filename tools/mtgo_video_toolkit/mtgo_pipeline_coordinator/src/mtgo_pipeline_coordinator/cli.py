"""One local command for corpus extraction, XMage validation, and inventory."""
from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path

from mtgo_video_parser.carddb import CardNameResolver
from mtgo_video_parser.layout import LayoutProfile
from mtgo_video_parser.ocr import make_ocr_backend
from mtgo_video_parser.pipeline import VideoExtractionPipeline
from mtgo_video_parser.structured_vision import SecondaryVisionPolicy
from mtg_state_contract.jsonl import read_states
from xmage_state_follower.follower import (
    FollowConfig, SubprocessReplayBackend, XmageFollower)
from xmage_state_follower.protocol import ReplayManifest

from .batch import CorpusRunner
from .bundle import PipelineBundle
from .doctor import run_doctor


def build_parser():
    parser = argparse.ArgumentParser(prog="mtgo-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")

    extract = sub.add_parser("extract")
    extract.add_argument("--video", required=True)
    extract.add_argument("--bundle", required=True)
    _add_extraction_options(extract)

    batch = sub.add_parser("batch")
    batch.add_argument("--source-manifest", required=True)
    batch.add_argument("--out-root", required=True)
    batch.add_argument("--replace", action="store_true")
    _add_extraction_options(batch)

    follow = sub.add_parser("follow")
    follow.add_argument("--bundle", required=True)
    follow.add_argument("--manifest", required=True)
    _add_follow_options(follow)

    run = sub.add_parser("run")
    run.add_argument("--video", required=True)
    run.add_argument("--bundle", required=True)
    _add_extraction_options(run)
    run.add_argument("--replay-manifest", required=True)
    _add_follow_options(run)

    inventory = sub.add_parser("inventory")
    inventory.add_argument("--bundle", required=True)
    return parser


def _add_extraction_options(parser):
    parser.add_argument("--layout", default=None)
    parser.add_argument("--ocr", choices=("tesseract", "paddle", "openrouter",
                                         "local-vlm"), default="tesseract")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--change-threshold", type=float, default=0.018)
    parser.add_argument("--max-interval-seconds", type=float, default=3.0)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--model", default="qwen/qwen3-vl-8b-instruct")
    parser.add_argument("--endpoint",
                        default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--paddle-device", default="gpu:0")
    parser.add_argument("--paddle-version", default="PP-OCRv6")
    parser.add_argument("--card-db", default=None)
    parser.add_argument("--secondary-vlm", choices=("none", "openrouter", "local-vlm"),
                        default="none")
    parser.add_argument("--secondary-model", default="qwen/qwen3-vl-8b-instruct")
    parser.add_argument("--secondary-endpoint",
                        default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--secondary-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--secondary-every", type=int, default=0)
    parser.add_argument("--secondary-threshold", type=float, default=0.48)
    parser.add_argument("--secondary-max-calls", type=int, default=None)
    parser.add_argument("--training-quality-threshold", type=float, default=0.65)
    parser.add_argument("--keep-player-names", action="store_true")


def _add_follow_options(parser):
    parser.add_argument("--classpath", default=None)
    parser.add_argument("--xmage-command", default=None)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--maximum-implicit-steps", type=int, default=8)
    parser.add_argument("--max-hard-mismatches", type=int, default=0)
    parser.add_argument("--ambiguity-is-error", action="store_true")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        result = run_doctor()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["readyForBaseExtraction"] else 1
    if args.command == "inventory":
        result = PipelineBundle(args.bundle).inventory()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "extract":
        extraction = _extract(args.video, args.bundle, args)
        print(json.dumps(extraction, indent=2, sort_keys=True))
        return 0
    if args.command == "batch":
        runner = CorpusRunner(lambda video, destination:
                              _extract(video, destination, args))
        result = runner.run(args.source_manifest, args.out_root, args.replace)
        print(json.dumps({
            "completed": len(result["completed"]),
            "skipped": len(result["skipped"]),
            "failed": len(result["failed"]),
            "report": str(Path(args.out_root).resolve() /
                          "corpus_run_report.json"),
        }, indent=2))
        return 0 if not result["failed"] else 2
    if args.command == "run":
        extraction = _extract(args.video, args.bundle, args)
        follow = _follow(
            args.bundle, args.replay_manifest, args.classpath,
            args.xmage_command, args.beam_width, args.candidates,
            args.maximum_implicit_steps, args.max_hard_mismatches,
            args.ambiguity_is_error)
        bundle = PipelineBundle(args.bundle)
        manifest = {
            "schemaVersion": 1,
            "kind": "mtgo-video-xmage-pipeline",
            "extraction": extraction,
            "follow": follow,
            "inventory": bundle.inventory(),
        }
        bundle.write_manifest(manifest)
        print(json.dumps({"bundle": str(Path(args.bundle).resolve()),
                          "followPassed": follow["passed"],
                          "verified": follow.get("verified", False)}, indent=2))
        return 0 if follow.get("verified") else 2
    if args.command == "follow":
        result = _follow(
            args.bundle, args.manifest, args.classpath,
            args.xmage_command, args.beam_width, args.candidates,
            args.maximum_implicit_steps, args.max_hard_mismatches,
            args.ambiguity_is_error)
        print(json.dumps({"passed": result["passed"],
                          "verified": result.get("verified", False),
                          "steps": len(result["steps"])}, indent=2))
        return 0 if result.get("verified") else 2
    return 2


def _extract(video_path, bundle_path, args):
    layout = LayoutProfile.load(args.layout)
    if args.ocr == "tesseract":
        ocr = make_ocr_backend("tesseract")
    elif args.ocr == "paddle":
        ocr = make_ocr_backend("paddle", device=args.paddle_device,
                               ocr_version=args.paddle_version)
    else:
        ocr = make_ocr_backend(
            args.ocr, model=args.model, endpoint=args.endpoint,
            api_key=os.environ.get(args.api_key_env))
    resolver = CardNameResolver.from_json(args.card_db) if args.card_db else None
    secondary = None
    policy = None
    if args.secondary_vlm != "none":
        secondary = make_ocr_backend(
            args.secondary_vlm, model=args.secondary_model,
            endpoint=args.secondary_endpoint,
            api_key=os.environ.get(args.secondary_api_key_env))
        policy = SecondaryVisionPolicy(
            every_n_samples=max(0, args.secondary_every),
            minimum_primary_confidence=args.secondary_threshold,
            max_calls=args.secondary_max_calls)
    return VideoExtractionPipeline(
        layout, ocr, card_resolver=resolver,
        secondary_vision=secondary, secondary_policy=policy,
        pseudonymize_players=not args.keep_player_names).run(
            video_path, bundle_path, fps=args.fps,
            change_threshold=args.change_threshold,
            max_interval_seconds=args.max_interval_seconds,
            save_frames=args.save_frames,
            training_quality_threshold=args.training_quality_threshold)


def _follow(bundle_path, manifest_path, classpath, xmage_command,
            beam_width, candidates, maximum_implicit_steps,
            max_hard_mismatches, ambiguity_is_error):
    bundle = Path(bundle_path)
    manifest = ReplayManifest.load(manifest_path)
    command = shlex.split(xmage_command) if xmage_command else None
    backend = SubprocessReplayBackend(
        manifest, command=command,
        classpath=classpath or os.environ.get("MAGIC_CABT_CLASSPATH"))
    actions = _read_jsonl(bundle / "observed_actions.jsonl")
    states = list(read_states(bundle / "canonical_states.jsonl"))
    report = XmageFollower(
        backend, manifest,
        FollowConfig(
            beam_width=max(1, beam_width),
            candidates_per_step=max(1, candidates),
            maximum_implicit_steps=max(0, maximum_implicit_steps),
            max_hard_mismatches=max(0, max_hard_mismatches),
            ambiguity_is_error=ambiguity_is_error)).follow(actions, states)
    target = bundle / "xmage_follow_report.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report.to_dict()


def _read_jsonl(path: Path):
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                values.append(json.loads(line))
    return values


if __name__ == "__main__":
    raise SystemExit(main())
