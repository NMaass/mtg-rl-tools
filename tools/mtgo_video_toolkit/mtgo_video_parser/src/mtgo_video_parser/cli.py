"""CLI for the headless MTGO video parser."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .calibration import write_calibration
from .carddb import CardNameResolver, download_scryfall_bulk
from .layout import LayoutProfile
from .ocr import make_ocr_backend
from .pipeline import VideoExtractionPipeline
from .structured_vision import SecondaryVisionPolicy
from .video import FrameSampler, write_frame


def build_parser():
    parser = argparse.ArgumentParser(prog="mtgo-video")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="extract canonical states from a local video")
    extract.add_argument("--video", required=True)
    extract.add_argument("--out", required=True)
    extract.add_argument("--layout", default=None)
    extract.add_argument("--ocr", choices=("tesseract", "paddle", "openrouter",
                                           "local-vlm"), default="tesseract")
    extract.add_argument("--fps", type=float, default=2.0)
    extract.add_argument("--change-threshold", type=float, default=0.018)
    extract.add_argument("--max-interval-seconds", type=float, default=3.0)
    extract.add_argument("--start-seconds", type=float, default=0.0)
    extract.add_argument("--end-seconds", type=float, default=None)
    extract.add_argument("--save-frames", action="store_true")
    extract.add_argument("--model", default="qwen/qwen3-vl-8b-instruct")
    extract.add_argument("--endpoint",
                         default="https://openrouter.ai/api/v1/chat/completions")
    extract.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    extract.add_argument("--paddle-device", default="gpu:0")
    extract.add_argument("--paddle-version", default="PP-OCRv6")
    extract.add_argument("--tesseract-executable", default=None)
    extract.add_argument("--card-db", default=None,
                         help="local Scryfall bulk JSON used to validate card title OCR")
    extract.add_argument("--secondary-vlm", choices=("none", "openrouter", "local-vlm"),
                         default="none")
    extract.add_argument("--secondary-model",
                         default="qwen/qwen3-vl-8b-instruct")
    extract.add_argument("--secondary-endpoint",
                         default="https://openrouter.ai/api/v1/chat/completions")
    extract.add_argument("--secondary-api-key-env", default="OPENROUTER_API_KEY")
    extract.add_argument("--secondary-every", type=int, default=0,
                         help="run structured VLM every N sampled frames; 0 disables periodic calls")
    extract.add_argument("--secondary-threshold", type=float, default=0.48,
                         help="run VLM when mean primary field confidence is below this")
    extract.add_argument("--secondary-max-calls", type=int, default=None)
    extract.add_argument("--training-quality-threshold", type=float, default=0.65)
    extract.add_argument("--keep-player-names", action="store_true",
                         help="retain OCR/native player names; default pseudonymizes them")

    sample = sub.add_parser("sample", help="save adaptive keyframes")
    sample.add_argument("--video", required=True)
    sample.add_argument("--out", required=True)
    sample.add_argument("--fps", type=float, default=1.0)
    sample.add_argument("--change-threshold", type=float, default=0.018)
    sample.add_argument("--max-frames", type=int, default=100)

    layout = sub.add_parser("print-layout", help="write the default profile")
    layout.add_argument("--out", required=True)

    calibrate = sub.add_parser(
        "calibrate", help="draw layout regions over one representative video frame")
    calibrate.add_argument("--video", required=True)
    calibrate.add_argument("--out-image", required=True)
    calibrate.add_argument("--out-layout", default=None)
    calibrate.add_argument("--layout", default=None)
    calibrate.add_argument("--time", type=float, default=0.0)
    calibrate.add_argument("--crops", default=None)

    fetch = sub.add_parser("fetch-card-db", help="download a Scryfall bulk JSON")
    fetch.add_argument("--out", required=True)
    fetch.add_argument("--type", default="oracle_cards")

    resolve = sub.add_parser("resolve-card")
    resolve.add_argument("--db", required=True)
    resolve.add_argument("text")

    sub.add_parser("doctor")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        result = {
            "ffmpeg": shutil.which("ffmpeg"),
            "ffprobe": shutil.which("ffprobe"),
            "tesseract": shutil.which("tesseract"),
            "nvidiaSmi": shutil.which("nvidia-smi"),
            "openrouterKey": bool(os.environ.get("OPENROUTER_API_KEY")),
            "baseReady": bool(shutil.which("ffmpeg")),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["baseReady"] else 1
    if args.command == "print-layout":
        LayoutProfile.load().save(args.out)
        print(args.out)
        return 0
    if args.command == "calibrate":
        profile = LayoutProfile.load(args.layout)
        if args.out_layout:
            profile.save(args.out_layout)
        result = write_calibration(
            args.video, profile, args.out_image,
            timestamp_seconds=args.time, crops_dir=args.crops)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "fetch-card-db":
        print(download_scryfall_bulk(args.out, args.type))
        return 0
    if args.command == "resolve-card":
        value = CardNameResolver.from_json(args.db).resolve(args.text)
        print(json.dumps(value.__dict__, indent=2, sort_keys=True))
        return 0 if value.name else 2
    if args.command == "sample":
        target = Path(args.out)
        target.mkdir(parents=True, exist_ok=True)
        count = 0
        for sample in FrameSampler(args.video, fps=args.fps,
                                   change_threshold=args.change_threshold):
            write_frame(str(target / f"{sample.frame_index:08d}.jpg"), sample.image)
            count += 1
            if count >= args.max_frames:
                break
        print(json.dumps({"frames": count, "out": str(target)}, indent=2))
        return 0
    if args.command == "extract":
        layout = LayoutProfile.load(args.layout)
        ocr = _make_primary(args)
        resolver = CardNameResolver.from_json(args.card_db) if args.card_db else None
        secondary = None
        secondary_policy = None
        if args.secondary_vlm != "none":
            secondary = make_ocr_backend(
                args.secondary_vlm, model=args.secondary_model,
                endpoint=args.secondary_endpoint,
                api_key=os.environ.get(args.secondary_api_key_env))
            secondary_policy = SecondaryVisionPolicy(
                every_n_samples=max(0, args.secondary_every),
                minimum_primary_confidence=args.secondary_threshold,
                max_calls=args.secondary_max_calls)
        manifest = VideoExtractionPipeline(
            layout, ocr, card_resolver=resolver,
            secondary_vision=secondary,
            secondary_policy=secondary_policy,
            pseudonymize_players=not args.keep_player_names).run(
                args.video, args.out, fps=args.fps,
                change_threshold=args.change_threshold,
                max_interval_seconds=args.max_interval_seconds,
                save_frames=args.save_frames, start_seconds=args.start_seconds,
                end_seconds=args.end_seconds,
                training_quality_threshold=args.training_quality_threshold)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    return 2


def _make_primary(args):
    if args.ocr == "tesseract":
        return make_ocr_backend("tesseract",
                                executable=args.tesseract_executable)
    if args.ocr == "paddle":
        return make_ocr_backend("paddle", device=args.paddle_device,
                                ocr_version=args.paddle_version)
    return make_ocr_backend(
        args.ocr, model=args.model, endpoint=args.endpoint,
        api_key=os.environ.get(args.api_key_env))


if __name__ == "__main__":
    raise SystemExit(main())
