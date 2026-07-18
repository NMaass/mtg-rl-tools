"""Temporal MTGO perception tracker producing canonical symbolic states."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple
import json

from mtg_state_contract import (
    CanonicalEvent, CanonicalObject, CanonicalPlayer, CanonicalState,
)

from .actions import (
    MTGOLogActionParser, ObservedAction, merge_spans_to_lines)
from .types import OCRSpan, PerceivedFrame


class MTGOStateTracker:
    """Fuse repeated OCR readings and rectangle detections over time.

    The tracker does not pretend to recover hidden cards. It emits unknown
    fields explicitly and records confidence per canonical JSON path.
    """

    def __init__(self, local_seat: str = "1", opponent_seat: str = "2",
                 consensus_window: int = 5,
                 emit_duplicate_states: bool = False,
                 maximum_history_events: int = 256,
                 pseudonymize_players: bool = True):
        self.local_seat = str(local_seat)
        self.opponent_seat = str(opponent_seat)
        self.consensus_window = max(1, int(consensus_window))
        self.emit_duplicate_states = emit_duplicate_states
        self.maximum_history_events = max(1, int(maximum_history_events))
        self.pseudonymize_players = bool(pseudonymize_players)
        self.actor_aliases: Dict[str, str] = {}
        self.values: Dict[str, Deque[Tuple[Any, float]]] = defaultdict(
            lambda: deque(maxlen=self.consensus_window))
        self.action_parser = MTGOLogActionParser()
        self.actions: List[ObservedAction] = []
        self.last_log_lines: List[str] = []
        self.sequence = 0
        self.last_fingerprint = None

    def update(self, frame: PerceivedFrame) -> Tuple[Optional[CanonicalState],
                                                    List[ObservedAction]]:
        for key, value in frame.fields.items():
            score = frame.field_confidence.get(key, 0.0)
            if value is not None:
                self.values[key].append((value, score))
        new_spans = self._new_log_spans(frame.log_lines)
        new_actions = self.action_parser.parse(new_spans, frame.timestamp_ms)
        if self.pseudonymize_players:
            for action in new_actions:
                original = action.actor
                action.actor = self._actor_alias(original)
                if original and action.actor != original:
                    action.metadata["actorPseudonymized"] = True
        self.actions.extend(new_actions)
        if len(self.actions) > self.maximum_history_events:
            self.actions = self.actions[-self.maximum_history_events:]
        self.sequence += 1
        state = self._state(frame)
        fingerprint = state.semantic_fingerprint()
        if not self.emit_duplicate_states and fingerprint == self.last_fingerprint:
            return None, new_actions
        self.last_fingerprint = fingerprint
        return state, new_actions

    def _new_log_spans(self, spans: Sequence[OCRSpan]) -> List[OCRSpan]:
        """Recover newly appended lines from a scrolling MTGO log window.

        A global set is incorrect because repeated actions such as drawing a
        card or playing the same card must remain distinct. We instead find the
        longest suffix/prefix overlap between consecutive visible windows.
        """
        lines = merge_spans_to_lines(spans)
        normalized = [_normalize_line(text) for text, _score in lines]
        overlap = _longest_overlap(self.last_log_lines, normalized)
        self.last_log_lines = normalized
        return [OCRSpan(text=text, confidence=score)
                for text, score in lines[overlap:]]

    def _state(self, frame: PerceivedFrame) -> CanonicalState:
        local_life, local_life_conf = self._consensus("local_life")
        opponent_life, opponent_life_conf = self._consensus("opponent_life")
        local_hand, local_hand_conf = self._consensus("local_hand_count")
        opponent_hand, opponent_hand_conf = self._consensus("opponent_hand_count")
        phase, phase_conf = self._consensus("phase")
        turn, turn_conf = self._consensus("turn_number")
        players = [
            CanonicalPlayer(seat=self.local_seat,
                            life=_as_int(local_life),
                            hand_count=_as_int(local_hand)),
            CanonicalPlayer(seat=self.opponent_seat,
                            life=_as_int(opponent_life),
                            hand_count=_as_int(opponent_hand)),
        ]
        zones: Dict[str, List[CanonicalObject]] = defaultdict(list)
        for index, card in enumerate(frame.cards):
            zones[card.zone].append(CanonicalObject(
                object_key=f"video:{frame.frame_index}:{index}",
                name=card.name,
                oracle_id=card.metadata.get("oracleId"),
                controller=card.controller,
                owner=card.controller,
                zone=card.zone,
                tapped=card.tapped,
                face_down=card.name is None,
                metadata={"bbox": list(card.bbox),
                          "confidence": card.confidence,
                          **card.metadata},
            ))
        history = [CanonicalEvent(
            sequence=index, timestamp_ms=action.timestamp_ms,
            actor=action.actor, action_type=action.action_type,
            card_name=action.card_name, targets=list(action.targets),
            text=action.text, confidence=action.confidence,
            metadata=dict(action.metadata))
            for index, action in enumerate(self.actions[-64:])]
        confidence = {
            "/players/%s/life" % self.local_seat: local_life_conf,
            "/players/%s/life" % self.opponent_seat: opponent_life_conf,
            "/players/%s/hand_count" % self.local_seat: local_hand_conf,
            "/players/%s/hand_count" % self.opponent_seat: opponent_hand_conf,
            "/phase": phase_conf,
            "/turn_number": turn_conf,
        }
        for zone, value in frame.zone_confidence.items():
            confidence[f"/zones/{zone}/count"] = value
        state = CanonicalState(
            source="mtgo-video", source_id=f"frame:{frame.frame_index}",
            timestamp_ms=frame.timestamp_ms, sequence=self.sequence,
            perspective_seat=self.local_seat,
            turn_number=_as_int(turn), phase=str(phase).upper() if phase else None,
            players=players, zones=dict(zones), public_history=history,
            confidence=confidence,
            provenance={
                "frameIndex": frame.frame_index,
                "changeScore": frame.change_score,
                "parser": "mtgo-video-parser/0.1.0",
                "secondaryVision": frame.raw_regions.get("secondaryVision"),
            },
        )
        for path, value in (
            (f"/players/{self.local_seat}/life", local_life),
            (f"/players/{self.opponent_seat}/life", opponent_life),
            (f"/players/{self.local_seat}/hand_count", local_hand),
            (f"/players/{self.opponent_seat}/hand_count", opponent_hand),
            ("/phase", phase), ("/turn_number", turn),
        ):
            if value is None:
                state.mark_unknown(path)
        # A configured zone with weak detection evidence is explicitly unknown;
        # the state comparator will not turn it into a false empty-zone match.
        for zone, score in frame.zone_confidence.items():
            if score < 0.55:
                state.mark_unknown(f"/zones/{zone}/count")
        return state

    def _actor_alias(self, actor: Optional[str]) -> Optional[str]:
        if actor is None:
            return None
        text = " ".join(str(actor).split())
        if not text:
            return None
        normalized = text.casefold()
        if normalized in {"you", "your", "local player"}:
            return f"seat:{self.local_seat}"
        if normalized in {"opponent", "your opponent"}:
            return f"seat:{self.opponent_seat}"
        if normalized not in self.actor_aliases:
            self.actor_aliases[normalized] = f"player:{len(self.actor_aliases) + 1}"
        return self.actor_aliases[normalized]

    def _consensus(self, key: str):
        rows = list(self.values.get(key) or [])
        if not rows:
            return None, 0.0
        buckets: Dict[str, List[Tuple[Any, float]]] = defaultdict(list)
        for value, score in rows:
            buckets[json.dumps(value, sort_keys=True, default=str)].append((value, score))
        winner = max(buckets.values(), key=lambda group: (
            sum(score for _, score in group), len(group)))
        value = winner[-1][0]
        support = len(winner) / len(rows)
        mean_evidence = sum(score for _, score in winner) / len(winner)
        confidence = min(1.0, 0.55 * mean_evidence + 0.45 * support)
        return value, confidence


def _as_int(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_line(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def _longest_overlap(previous: Sequence[str], current: Sequence[str]) -> int:
    maximum = min(len(previous), len(current))
    for size in range(maximum, 0, -1):
        if list(previous[-size:]) == list(current[:size]):
            return size
    return 0
