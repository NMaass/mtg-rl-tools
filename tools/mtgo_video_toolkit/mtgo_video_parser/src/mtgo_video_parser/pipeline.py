"""End-to-end headless extraction pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import hashlib
import json
import os

from .carddb import CardNameResolver
from .layout import LayoutProfile
from .ocr import OCRBackend
from .perception import FramePerceiver
from .structured_vision import SecondaryVisionPolicy
from .tracker import MTGOStateTracker
from .video import FrameSampler, write_frame


class VideoExtractionPipeline:
    def __init__(self, layout: LayoutProfile, ocr: OCRBackend,
                 local_seat: str = "1", opponent_seat: str = "2",
                 card_resolver: Optional[CardNameResolver] = None,
                 card_ocr: Optional[OCRBackend] = None,
                 secondary_vision: Optional[OCRBackend] = None,
                 secondary_policy: Optional[SecondaryVisionPolicy] = None,
                 pseudonymize_players: bool = True):
        self.layout = layout
        self.ocr = ocr
        self.perceiver = FramePerceiver(
            layout, ocr, card_resolver=card_resolver, card_ocr=card_ocr,
            secondary_vision=secondary_vision,
            secondary_policy=secondary_policy)
        self.tracker = MTGOStateTracker(
            local_seat, opponent_seat,
            pseudonymize_players=pseudonymize_players)
        self.pseudonymize_players = bool(pseudonymize_players)

    def run(self, video_path: str, output_dir: str, fps: float = 2.0,
            change_threshold: float = 0.018,
            max_interval_seconds: float = 3.0,
            save_frames: bool = False,
            start_seconds: float = 0.0,
            end_seconds: Optional[float] = None,
            training_quality_threshold: float = 0.65) -> Dict[str, object]:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        frames_dir = target / "frames"
        handles = {
            "perceived": (target / "perceived_frames.jsonl").open("w", encoding="utf-8"),
            "states": (target / "canonical_states.jsonl").open("w", encoding="utf-8"),
            "actions": (target / "observed_actions.jsonl").open("w", encoding="utf-8"),
            "training": (target / "training_candidates.jsonl").open("w", encoding="utf-8"),
            "quarantine": (target / "quarantine.jsonl").open("w", encoding="utf-8"),
            "errors": (target / "extraction_errors.jsonl").open("w", encoding="utf-8"),
        }
        counts = {
            "sampledFrames": 0, "emittedStates": 0, "observedActions": 0,
            "trainingCandidates": 0, "quarantinedStates": 0,
            "extractionErrors": 0, "identifiedCards": 0,
            "unknownCards": 0, "secondaryVisionCalls": 0,
            "secondaryVisionErrors": 0,
        }
        quality_values = []
        try:
            sampler = FrameSampler(
                video_path, fps=fps, change_threshold=change_threshold,
                max_interval_seconds=max_interval_seconds,
                start_seconds=start_seconds, end_seconds=end_seconds)
            for sample in sampler:
                counts["sampledFrames"] += 1
                try:
                    perceived = self.perceiver.perceive(
                        sample.frame_index, sample.timestamp_ms,
                        sample.change_score, sample.image)
                except Exception as error:
                    counts["extractionErrors"] += 1
                    handles["errors"].write(_json({
                        "frameIndex": sample.frame_index,
                        "timestampMs": sample.timestamp_ms,
                        "error": f"{type(error).__name__}: {error}",
                    }) + "\n")
                    continue
                handles["perceived"].write(_json(perceived.to_dict()) + "\n")
                if save_frames:
                    write_frame(str(frames_dir / f"{sample.frame_index:08d}.jpg"),
                                sample.image)
                secondary = perceived.raw_regions.get("secondaryVision") or {}
                if secondary.get("used"):
                    counts["secondaryVisionCalls"] += 1
                if secondary.get("error"):
                    counts["secondaryVisionErrors"] += 1
                counts["identifiedCards"] += sum(
                    1 for card in perceived.cards if card.name)
                counts["unknownCards"] += sum(
                    1 for card in perceived.cards if not card.name)
                state, actions = self.tracker.update(perceived)
                for action in actions:
                    handles["actions"].write(_json(action.to_dict()) + "\n")
                    counts["observedActions"] += 1
                if state is not None:
                    quality = state_quality(state)
                    quality_values.append(quality["score"])
                    state.provenance["extractionQuality"] = quality
                    handles["states"].write(_json(state.to_dict()) + "\n")
                    counts["emittedStates"] += 1
                    if quality["score"] >= training_quality_threshold and \
                            not quality["criticalUnknownPaths"]:
                        handles["training"].write(_json(state.to_dict()) + "\n")
                        counts["trainingCandidates"] += 1
                    else:
                        handles["quarantine"].write(_json({
                            "reason": "extraction-quality-gate",
                            "quality": quality,
                            "state": state.to_dict(),
                        }) + "\n")
                        counts["quarantinedStates"] += 1
        finally:
            for handle in handles.values():
                handle.close()
        quality_summary = {
            "threshold": float(training_quality_threshold),
            "mean": sum(quality_values) / len(quality_values)
                if quality_values else None,
            "minimum": min(quality_values) if quality_values else None,
            "maximum": max(quality_values) if quality_values else None,
        }
        manifest = {
            "schemaVersion": 1,
            "kind": "mtgo-video-extraction",
            "toolVersion": "0.1.0",
            "video": {
                "path": os.path.abspath(video_path),
                "sha256": _sha256(video_path),
            },
            "layout": self.layout.to_dict(),
            "ocrBackend": getattr(self.ocr, "name", type(self.ocr).__name__),
            "settings": {
                "fps": fps, "changeThreshold": change_threshold,
                "maxIntervalSeconds": max_interval_seconds,
                "startSeconds": start_seconds, "endSeconds": end_seconds,
                "trainingQualityThreshold": training_quality_threshold,
                "pseudonymizePlayers": self.pseudonymize_players,
            },
            "counts": counts,
            "quality": quality_summary,
            "files": {
                "perceivedFrames": "perceived_frames.jsonl",
                "canonicalStates": "canonical_states.jsonl",
                "observedActions": "observed_actions.jsonl",
                "trainingCandidates": "training_candidates.jsonl",
                "quarantine": "quarantine.jsonl",
                "extractionErrors": "extraction_errors.jsonl",
            },
        }
        with (target / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with (target / "quality_report.json").open("w", encoding="utf-8") as handle:
            json.dump({"counts": counts, "quality": quality_summary}, handle,
                      indent=2, sort_keys=True)
            handle.write("\n")
        return manifest


def state_quality(state) -> Dict[str, object]:
    seats = [str(player.seat) for player in state.players[:2]]
    critical = [f"/players/{seat}/life" for seat in seats]
    critical.extend([
        "/turn_number", "/phase", "/zones/battlefield/count",
    ])
    confidence_values = [float(value) for value in state.confidence.values()
                         if isinstance(value, (int, float))]
    mean_confidence = sum(confidence_values) / len(confidence_values) \
        if confidence_values else 0.0
    critical_unknown = [path for path in critical
                        if path in state.unknown_paths or
                        state.confidence_at(path, 0.0) < 0.55]
    known_objects = sum(1 for rows in state.zones.values()
                        for value in rows if value.name)
    all_objects = sum(len(rows) for rows in state.zones.values())
    identity_coverage = known_objects / all_objects if all_objects else 1.0
    # State numeric/public structure matters more than card identity.  Unknown
    # card names can later be repaired from the game log or XMage hypothesis.
    score = max(0.0, min(1.0,
                         0.78 * mean_confidence + 0.22 * identity_coverage))
    return {
        "score": score,
        "meanFieldConfidence": mean_confidence,
        "cardIdentityCoverage": identity_coverage,
        "criticalUnknownPaths": critical_unknown,
        "unknownPaths": list(state.unknown_paths),
    }


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
