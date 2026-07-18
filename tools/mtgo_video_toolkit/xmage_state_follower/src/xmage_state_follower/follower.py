"""Beam-search replay follower with canonical-state verification.

The current bridge has no cheap clone/import-state operation.  Branching is
therefore implemented by restarting a deterministic session and replaying the
selection prefix.  This is intentionally slower but correct for proof-of-method.
A future bridge clone API can implement the same ``ReplayBackend`` interface.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import copy
import json
import math

from mtg_state_contract import (
    CanonicalState,
    CanonicalStateFormatter,
    ComparisonPolicy,
    compare_states,
)

from .matcher import OptionMatch, rank_options
from .protocol import ReplayManifest, XmageProtocolClient


@dataclass
class ReplayResult:
    response: Dict[str, Any]
    state: CanonicalState
    selections: List[List[int]]
    finished: bool = False
    result: Optional[Dict[str, Any]] = None


class ReplayBackend:
    def run(self, selections: Sequence[Sequence[int]]) -> ReplayResult:
        raise NotImplementedError


class SubprocessReplayBackend(ReplayBackend):
    def __init__(self, manifest: ReplayManifest,
                 command: Optional[Sequence[str]] = None,
                 classpath: Optional[str] = None,
                 cwd: Optional[str] = None):
        self.manifest = manifest
        self.command = list(command) if command else None
        self.classpath = classpath
        self.cwd = cwd
        self.formatter = CanonicalStateFormatter()

    def run(self, selections: Sequence[Sequence[int]]) -> ReplayResult:
        with XmageProtocolClient(command=self.command, classpath=self.classpath,
                                cwd=self.cwd) as client:
            response = client.start(self.manifest)
            full_prefix = list(self.manifest.bootstrap_selections) + \
                [list(row) for row in selections]
            for selection in full_prefix:
                if response.get("finished"):
                    break
                response = client.select(selection)
            if response.get("finished"):
                raw_state = (response.get("result") or {}).get("finalState") or {}
                state = self.formatter.format(raw_state, source="xmage")
                return ReplayResult(response, state,
                                    [list(row) for row in selections], True,
                                    response.get("result"))
            current = (response.get("observation") or {}).get("current")
            if not isinstance(current, dict):
                snapshot = client.snapshot()
                current = snapshot.get("state") or {}
            state = self.formatter.format({
                "sequence": response.get("sequence"),
                "observation": {"current": current},
            }, source="xmage")
            return ReplayResult(response, state,
                                [list(row) for row in selections], False, None)


@dataclass
class FollowConfig:
    beam_width: int = 4
    candidates_per_step: int = 4
    minimum_action_score: float = 45.0
    mismatch_penalty: float = 18.0
    unknown_penalty: float = 0.5
    max_hard_mismatches: int = 0
    maximum_implicit_steps: int = 8
    implicit_candidates_per_prompt: int = 2
    implicit_step_penalty: float = 1.5
    maximum_replays_per_action: int = 96
    state_time_tolerance_ms: int = 8000
    ambiguity_is_error: bool = False
    comparison_policy: ComparisonPolicy = field(default_factory=lambda:
        ComparisonPolicy(min_confidence=0.60, ignore_priority=True))


@dataclass
class FollowStep:
    step_index: int
    timestamp_ms: Optional[int]
    action: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    surviving_hypotheses: int
    best_score: Optional[float]
    best_selection_prefix: List[List[int]]
    comparison: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class FollowReport:
    manifest: Dict[str, Any]
    config: Dict[str, Any]
    steps: List[FollowStep]
    final_hypotheses: List[Dict[str, Any]]

    @property
    def passed(self) -> bool:
        return bool(self.final_hypotheses) and all(
            step.error is None for step in self.steps)

    @property
    def verified(self) -> bool:
        """True only when every consumed action was synchronized to a state.

        A successful action replay without a nearby perceived state is useful
        but is not verification.  This prevents a missing/empty state stream
        from being reported as an engine-validated video.
        """
        if not self.passed or not self.steps:
            return False
        for step in self.steps:
            if step.comparison is None:
                return False
            if not step.comparison.get("passed", False):
                return False
        return True

    def to_dict(self):
        return {
            "passed": self.passed,
            "verified": self.verified,
            "manifest": self.manifest,
            "config": self.config,
            "steps": [asdict(row) for row in self.steps],
            "finalHypotheses": self.final_hypotheses,
        }


@dataclass
class _Hypothesis:
    selections: List[List[int]]
    score: float
    replay: ReplayResult
    comparisons: List[Dict[str, Any]] = field(default_factory=list)


class XmageFollower:
    def __init__(self, backend: ReplayBackend, manifest: ReplayManifest,
                 config: Optional[FollowConfig] = None):
        self.backend = backend
        self.manifest = manifest
        self.config = config or FollowConfig()

    def follow(self, actions: Sequence[Dict[str, Any]],
               observed_states: Sequence[CanonicalState]) -> FollowReport:
        initial = self.backend.run([])
        beam = [_Hypothesis([], 0.0, initial)]
        steps: List[FollowStep] = []
        states = sorted(observed_states,
                        key=lambda row: row.timestamp_ms or -1)
        for step_index, action in enumerate(actions):
            timestamp = action.get("timestamp_ms") or action.get("timestampMs")
            target_state = _state_at_or_after(
                states, timestamp, self.config.state_time_tolerance_ms)
            expanded: List[_Hypothesis] = []
            candidate_log: List[Dict[str, Any]] = []
            for hypothesis in beam:
                consumed, log_rows = self._consume_observed_action(
                    hypothesis, action, target_state)
                expanded.extend(consumed)
                candidate_log.extend(log_rows)
            expanded.sort(key=lambda row: (-row.score, len(row.selections)))
            beam = expanded[:self.config.beam_width]
            if not beam:
                steps.append(FollowStep(
                    step_index, timestamp, dict(action), candidate_log, 0, None,
                    [], error="no XMage hypothesis survived action/state validation"))
                break
            best = beam[0]
            error = None
            if self.config.ambiguity_is_error and len(beam) > 1:
                error = "multiple XMage hypotheses remain after synchronization"
            steps.append(FollowStep(
                step_index=step_index, timestamp_ms=timestamp,
                action=dict(action), candidates=candidate_log,
                surviving_hypotheses=len(beam), best_score=best.score,
                best_selection_prefix=copy.deepcopy(best.selections),
                comparison=best.comparisons[-1] if best.comparisons else None,
                error=error))
            if error:
                break
        final = [{
            "score": row.score,
            "selections": row.selections,
            "finished": row.replay.finished,
            "state": row.replay.state.to_dict(),
            "comparisons": row.comparisons,
        } for row in beam]
        config_dict = asdict(self.config)
        config_dict["comparison_policy"] = asdict(self.config.comparison_policy)
        return FollowReport(self.manifest.to_dict(), config_dict, steps, final)

    def _consume_observed_action(
            self, hypothesis: _Hypothesis, action: Dict[str, Any],
            target_state: Optional[CanonicalState]
            ) -> Tuple[List[_Hypothesis], List[Dict[str, Any]]]:
        """Match one perceived action, inserting unobserved pass/continue choices.

        MTGO videos rarely expose every priority pass.  The follower explores a
        bounded set of semantically safe implicit choices before consuming the
        next observed action.  Every resulting branch is still validated
        against the perceived canonical state.
        """
        queue: List[Tuple[_Hypothesis, int]] = [(hypothesis, 0)]
        consumed: List[_Hypothesis] = []
        log_rows: List[Dict[str, Any]] = []
        seen = {json.dumps(hypothesis.selections, separators=(",", ":"))}
        replays = 0
        while queue and replays < self.config.maximum_replays_per_action:
            current, implicit_depth = queue.pop(0)
            options = (((current.replay.response.get("observation") or {})
                        .get("select") or {}).get("option") or [])
            matches = rank_options(
                action, options,
                minimum_score=self.config.minimum_action_score)
            for match in matches[:self.config.candidates_per_step]:
                row = match.to_dict()
                row.update({"implicit": False, "implicitDepth": implicit_depth})
                log_rows.append(row)
                selections = current.selections + [[match.option_index]]
                try:
                    replay = self.backend.run(selections)
                    replays += 1
                except Exception as error:
                    row["replayError"] = f"{type(error).__name__}: {error}"
                    continue
                score = current.score + match.score
                comparisons = list(current.comparisons)
                if target_state is not None:
                    comparison = compare_states(
                        replay.state, target_state,
                        self.config.comparison_policy)
                    score -= comparison.mismatches * self.config.mismatch_penalty
                    score -= comparison.unknowns * self.config.unknown_penalty
                    comparisons.append(comparison.to_dict())
                    if comparison.mismatches > self.config.max_hard_mismatches:
                        continue
                consumed.append(_Hypothesis(
                    selections, score, replay, comparisons))

            if implicit_depth >= self.config.maximum_implicit_steps:
                continue
            implicit = [row for row in options if _is_implicit_option(row)]
            implicit = implicit[:self.config.implicit_candidates_per_prompt]
            for option in implicit:
                index = option.get("index")
                if not isinstance(index, int):
                    try:
                        index = options.index(option)
                    except ValueError:
                        continue
                selections = current.selections + [[index]]
                key = json.dumps(selections, separators=(",", ":"))
                if key in seen:
                    continue
                seen.add(key)
                row = {
                    "option_index": index,
                    "score": -self.config.implicit_step_penalty,
                    "option": dict(option),
                    "reasons": ["implicit pass/continue between visible actions"],
                    "implicit": True,
                    "implicitDepth": implicit_depth + 1,
                }
                log_rows.append(row)
                try:
                    replay = self.backend.run(selections)
                    replays += 1
                except Exception as error:
                    row["replayError"] = f"{type(error).__name__}: {error}"
                    continue
                queue.append((_Hypothesis(
                    selections,
                    current.score - self.config.implicit_step_penalty,
                    replay, list(current.comparisons)), implicit_depth + 1))
        return consumed, log_rows


def _state_at_or_after(states: Sequence[CanonicalState], timestamp_ms: Any,
                       tolerance_ms: int) -> Optional[CanonicalState]:
    if not states:
        return None
    if timestamp_ms is None:
        return states[0]
    try:
        target = int(timestamp_ms)
    except (TypeError, ValueError):
        return states[0]
    after = [row for row in states
             if row.timestamp_ms is not None and row.timestamp_ms >= target]
    if after:
        candidate = min(after, key=lambda row: row.timestamp_ms - target)
        if candidate.timestamp_ms - target <= max(0, int(tolerance_ms)):
            return candidate
    nearest = min(states, key=lambda row: abs((row.timestamp_ms or target) - target))
    if abs((nearest.timestamp_ms or target) - target) <= max(0, int(tolerance_ms)):
        return nearest
    return None


def _is_implicit_option(option: Dict[str, Any]) -> bool:
    payload = option.get("payload") or {}
    text = " ".join(str(value) for value in (
        option.get("type"), option.get("label"), payload.get("canonicalKey"),
        payload.get("actionType")) if value is not None).upper()
    safe = (
        "PASS_PRIORITY", "PASS", "CONTINUE", "RESOLVE", "OK", "DONE",
        "SKIP", "NO_ACTION", "DECLINE_OPTIONAL",
    )
    return any(token in text for token in safe)
