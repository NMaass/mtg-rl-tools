"""Auditable expert-preference models for MTG action costs.

The module intentionally fits a small monotone Bradley-Terry model instead of
asking an expert to hand tune one opaque scalar reward.  Experts compare legal
candidate actions in context; candidate consequences are expressed through a
versioned factor ontology; the fitted model is evaluated on held-out contexts.

The learned quantity is a *utility* (higher is better).  ``cost`` is its
negative.  For online RL the model should normally be used as an auxiliary
head, an offline reranker, or as a potential ``Phi`` in
``gamma * Phi(next) - Phi(current)``.  It is not a replacement for terminal
win/loss reward.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = 1
_MODEL_KIND = "expert-cost-model-v1"
_FACTOR_KIND = "expert-factor-ontology-v1"
_ALLOWED_DIRECTIONS = frozenset(("higher_better", "lower_better"))
_ALLOWED_TRANSFORMS = frozenset((
    "identity", "log1p", "signed_log1p", "sqrt", "tanh", "binary"))


@dataclass(frozen=True)
class FactorSpec:
    """Definition of one expert-visible causal/tactical factor.

    ``scale`` controls numerical conditioning, not strategic importance.
    Strategic importance is learned as a non-negative weight after the factor
    is oriented so that larger values always mean better outcomes.
    """

    name: str
    direction: str = "higher_better"
    transform: str = "identity"
    scale: float = 1.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    default: float = 0.0
    group: str = "general"
    description: str = ""
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("factor name must be non-empty")
        if self.direction not in _ALLOWED_DIRECTIONS:
            raise ValueError(
                "factor %s direction must be one of %s" %
                (self.name, sorted(_ALLOWED_DIRECTIONS)))
        if self.transform not in _ALLOWED_TRANSFORMS:
            raise ValueError(
                "factor %s transform must be one of %s" %
                (self.name, sorted(_ALLOWED_TRANSFORMS)))
        if not _finite(self.scale) or float(self.scale) <= 0.0:
            raise ValueError("factor %s scale must be finite and positive" % self.name)
        if not _finite(self.default):
            raise ValueError("factor %s default must be finite" % self.name)
        if self.minimum is not None and not _finite(self.minimum):
            raise ValueError("factor %s minimum must be finite" % self.name)
        if self.maximum is not None and not _finite(self.maximum):
            raise ValueError("factor %s maximum must be finite" % self.name)
        if (self.minimum is not None and self.maximum is not None and
                float(self.minimum) > float(self.maximum)):
            raise ValueError("factor %s minimum exceeds maximum" % self.name)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorSpec":
        if not isinstance(payload, Mapping):
            raise ValueError("factor specification must be an object")
        return cls(
            name=str(payload.get("name") or "").strip(),
            direction=str(payload.get("direction") or "higher_better"),
            transform=str(payload.get("transform") or "identity"),
            scale=float(payload.get("scale", 1.0)),
            minimum=_optional_float(payload.get("minimum")),
            maximum=_optional_float(payload.get("maximum")),
            default=float(payload.get("default", 0.0)),
            group=str(payload.get("group") or "general"),
            description=str(payload.get("description") or ""),
            unit=str(payload.get("unit") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def oriented(self, value: Any) -> float:
        """Return a bounded/scaled value where larger always means better."""
        numeric = self.default if value is None else _numeric(value, self.name)
        if self.minimum is not None:
            numeric = max(float(self.minimum), numeric)
        if self.maximum is not None:
            numeric = min(float(self.maximum), numeric)
        transformed = _transform(numeric, self.transform)
        oriented = transformed if self.direction == "higher_better" else -transformed
        return oriented / float(self.scale)


@dataclass(frozen=True)
class CandidateFactors:
    candidate_id: str
    factors: Mapping[str, float]
    label: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
            cls, payload: Mapping[str, Any], specs: Sequence[FactorSpec]) -> "CandidateFactors":
        if not isinstance(payload, Mapping):
            raise ValueError("candidate must be an object")
        candidate_id = str(
            payload.get("candidateId") or payload.get("id") or payload.get("label") or ""
        ).strip()
        if not candidate_id:
            raise ValueError("candidate needs candidateId, id, or label")
        raw_factors = payload.get("factors")
        if not isinstance(raw_factors, Mapping):
            raise ValueError("candidate %s factors must be an object" % candidate_id)
        spec_names = {spec.name for spec in specs}
        unknown = sorted(set(str(name) for name in raw_factors).difference(spec_names))
        if unknown:
            raise ValueError(
                "candidate %s contains unknown factors: %s" % (candidate_id, unknown))
        factors = {
            str(name): _numeric(value, str(name))
            for name, value in raw_factors.items()
        }
        metadata = payload.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise ValueError("candidate %s metadata must be an object" % candidate_id)
        return cls(
            candidate_id=candidate_id,
            label=str(payload.get("label") or candidate_id),
            factors=factors,
            metadata=dict(metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidateId": self.candidate_id,
            "label": self.label,
            "factors": dict(self.factors),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PreferenceExample:
    context_id: str
    left: CandidateFactors
    right: CandidateFactors
    preferred: str
    expert_id: str = "anonymous"
    confidence: float = 1.0
    weight: float = 1.0
    rationale: str = ""
    assumptions: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.context_id:
            raise ValueError("preference context_id must be non-empty")
        if self.preferred not in ("left", "right", "tie"):
            raise ValueError("preferred must be left, right, or tie")
        if not _finite(self.confidence) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not _finite(self.weight) or float(self.weight) <= 0.0:
            raise ValueError("weight must be finite and positive")

    @property
    def target_probability(self) -> float:
        if self.preferred == "left":
            return 1.0
        if self.preferred == "right":
            return 0.0
        return 0.5

    @property
    def effective_weight(self) -> float:
        # Confidence zero preserves the annotation for audit but excludes it from fitting.
        return float(self.weight) * float(self.confidence)

    @classmethod
    def from_dict(
            cls, payload: Mapping[str, Any], specs: Sequence[FactorSpec]) -> "PreferenceExample":
        if not isinstance(payload, Mapping):
            raise ValueError("preference must be an object")
        context_id = str(
            payload.get("contextId") or payload.get("decisionId") or payload.get("id") or ""
        ).strip()
        left = CandidateFactors.from_dict(payload.get("left") or {}, specs)
        right = CandidateFactors.from_dict(payload.get("right") or {}, specs)
        preferred = _parse_preferred(payload.get("preferred"))
        assumptions = payload.get("assumptions") or []
        tags = payload.get("tags") or []
        if not isinstance(assumptions, list) or not all(
                isinstance(value, str) for value in assumptions):
            raise ValueError("assumptions must be a list of strings")
        if not isinstance(tags, list) or not all(isinstance(value, str) for value in tags):
            raise ValueError("tags must be a list of strings")
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("preference metadata must be an object")
        return cls(
            context_id=context_id,
            left=left,
            right=right,
            preferred=preferred,
            expert_id=str(payload.get("expertId") or "anonymous"),
            confidence=float(payload.get("confidence", 1.0)),
            weight=float(payload.get("weight", 1.0)),
            rationale=str(payload.get("rationale") or ""),
            assumptions=tuple(assumptions),
            tags=tuple(tags),
            metadata=dict(metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contextId": self.context_id,
            "expertId": self.expert_id,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "preferred": self.preferred,
            "confidence": self.confidence,
            "weight": self.weight,
            "rationale": self.rationale,
            "assumptions": list(self.assumptions),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExpertCostModel:
    specs: Tuple[FactorSpec, ...]
    weights: Mapping[str, float]
    temperature: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("expert cost model needs at least one factor")
        names = [spec.name for spec in self.specs]
        if len(set(names)) != len(names):
            raise ValueError("expert cost model factor names must be unique")
        if set(self.weights) != set(names):
            raise ValueError("model weights must exactly match factor names")
        for name, value in self.weights.items():
            if not _finite(value) or float(value) < 0.0:
                raise ValueError("weight %s must be finite and non-negative" % name)
        if not _finite(self.temperature) or float(self.temperature) <= 0.0:
            raise ValueError("temperature must be finite and positive")

    @property
    def spec_by_name(self) -> Dict[str, FactorSpec]:
        return {spec.name: spec for spec in self.specs}

    def feature_vector(self, factors: Mapping[str, Any]) -> Dict[str, float]:
        if not isinstance(factors, Mapping):
            raise ValueError("factors must be an object")
        unknown = sorted(set(str(name) for name in factors).difference(self.spec_by_name))
        if unknown:
            raise ValueError("unknown factors: %s" % unknown)
        return {
            spec.name: spec.oriented(factors.get(spec.name, spec.default))
            for spec in self.specs
        }

    def utility(self, factors: Mapping[str, Any]) -> float:
        vector = self.feature_vector(factors)
        return sum(float(self.weights[name]) * vector[name] for name in vector)

    def cost(self, factors: Mapping[str, Any]) -> float:
        return -self.utility(factors)

    def preference_probability(
            self, left_factors: Mapping[str, Any], right_factors: Mapping[str, Any]) -> float:
        delta = (self.utility(left_factors) - self.utility(right_factors)) / self.temperature
        return _sigmoid(delta)

    def explain(self, factors: Mapping[str, Any]) -> Dict[str, Any]:
        vector = self.feature_vector(factors)
        total_weight = sum(float(value) for value in self.weights.values())
        contributions = []
        for spec in self.specs:
            raw = factors.get(spec.name, spec.default)
            contribution = float(self.weights[spec.name]) * vector[spec.name]
            contributions.append({
                "factor": spec.name,
                "group": spec.group,
                "rawValue": raw,
                "orientedValue": vector[spec.name],
                "weight": float(self.weights[spec.name]),
                "normalizedWeight": (
                    float(self.weights[spec.name]) / total_weight if total_weight else 0.0),
                "utilityContribution": contribution,
                "costContribution": -contribution,
                "description": spec.description,
            })
        contributions.sort(key=lambda row: (-abs(row["utilityContribution"]), row["factor"]))
        utility = sum(row["utilityContribution"] for row in contributions)
        return {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "expert-cost-explanation-v1",
            "utility": utility,
            "cost": -utility,
            "contributions": contributions,
            "warning": (
                "This score is an expert-preference model, not a substitute for "
                "terminal game outcome."),
        }

    def potential_shaping_reward(
            self, before: Mapping[str, Any], after: Mapping[str, Any], gamma: float) -> float:
        if not _finite(gamma) or not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        return float(gamma) * self.utility(after) - self.utility(before)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "kind": _MODEL_KIND,
            "factors": [spec.to_dict() for spec in self.specs],
            "weights": {name: float(value) for name, value in self.weights.items()},
            "temperature": float(self.temperature),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpertCostModel":
        if not isinstance(payload, Mapping):
            raise ValueError("model payload must be an object")
        if payload.get("schemaVersion") != SCHEMA_VERSION:
            raise ValueError("unsupported expert cost schemaVersion")
        if payload.get("kind") != _MODEL_KIND:
            raise ValueError("not an expert cost model")
        factors = payload.get("factors")
        if not isinstance(factors, list) or not factors:
            raise ValueError("model factors must be a non-empty list")
        weights = payload.get("weights")
        if not isinstance(weights, Mapping):
            raise ValueError("model weights must be an object")
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("model metadata must be an object")
        return cls(
            specs=tuple(FactorSpec.from_dict(item) for item in factors),
            weights={str(name): float(value) for name, value in weights.items()},
            temperature=float(payload.get("temperature", 1.0)),
            metadata=dict(metadata),
        )


def fit_cost_model(
        specs: Sequence[FactorSpec],
        preferences: Sequence[PreferenceExample],
        iterations: int = 2500,
        learning_rate: float = 0.05,
        l2: float = 0.01,
        temperature: float = 1.0,
        metadata: Optional[Mapping[str, Any]] = None) -> ExpertCostModel:
    """Fit non-negative factor weights with projected gradient descent.

    Every factor is oriented by ``FactorSpec`` so non-negative weights encode
    monotonicity.  This makes the learned score inspectable and prevents a
    small/noisy dataset from silently reversing expert-declared semantics.
    """
    specs = tuple(specs)
    preferences = tuple(preferences)
    _validate_specs(specs)
    if not preferences:
        raise ValueError("cannot fit expert cost model without preferences")
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not _finite(learning_rate) or float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not _finite(l2) or float(l2) < 0.0:
        raise ValueError("l2 must be finite and non-negative")
    if not _finite(temperature) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")

    names = [spec.name for spec in specs]
    # Equal small positive initialization keeps all ontology factors available
    # while allowing unsupported factors to shrink to zero under regularization.
    weights = {name: 0.1 for name in names}
    vectors = []
    for example in preferences:
        left = _vector(specs, example.left.factors)
        right = _vector(specs, example.right.factors)
        vectors.append((
            [left[name] - right[name] for name in names],
            example.target_probability,
            example.effective_weight,
        ))

    total_example_weight = sum(item[2] for item in vectors)
    if total_example_weight <= 0.0:
        raise ValueError("at least one preference must have positive confidence")
    for iteration in range(iterations):
        gradients = {name: 0.0 for name in names}
        for differences, target, sample_weight in vectors:
            margin = sum(
                weights[name] * differences[index]
                for index, name in enumerate(names)) / float(temperature)
            probability = _sigmoid(margin)
            error = (probability - target) * sample_weight / float(temperature)
            for index, name in enumerate(names):
                gradients[name] += error * differences[index]
        step = float(learning_rate) / math.sqrt(1.0 + iteration / 250.0)
        denominator = max(total_example_weight, 1e-12)
        for name in names:
            gradient = gradients[name] / denominator + float(l2) * weights[name]
            weights[name] = max(0.0, weights[name] - step * gradient)

    output_metadata = dict(metadata or {})
    output_metadata.update({
        "fit": {
            "algorithm": "projected-gradient-bradley-terry",
            "monotoneWeights": True,
            "iterations": int(iterations),
            "learningRate": float(learning_rate),
            "l2": float(l2),
            "examples": len(preferences),
            "contexts": len(set(item.context_id for item in preferences)),
            "experts": sorted(set(item.expert_id for item in preferences)),
        }
    })
    return ExpertCostModel(
        specs=specs,
        weights=weights,
        temperature=float(temperature),
        metadata=output_metadata,
    )


def evaluate_preferences(
        model: ExpertCostModel,
        preferences: Sequence[PreferenceExample]) -> Dict[str, Any]:
    examples = list(preferences)
    if not examples:
        return {
            "examples": 0,
            "nonTieExamples": 0,
            "pairwiseAccuracy": None,
            "weightedLogLoss": None,
            "weightedBrier": None,
            "meanPreferredMargin": None,
            "byExpert": {},
        }
    rows = []
    for example in examples:
        probability = model.preference_probability(
            example.left.factors, example.right.factors)
        target = example.target_probability
        weight = example.effective_weight
        if example.preferred == "left":
            preferred_margin = model.utility(example.left.factors) - model.utility(
                example.right.factors)
            correct = probability > 0.5
        elif example.preferred == "right":
            preferred_margin = model.utility(example.right.factors) - model.utility(
                example.left.factors)
            correct = probability < 0.5
        else:
            preferred_margin = -abs(
                model.utility(example.left.factors) - model.utility(example.right.factors))
            correct = None
        clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
        rows.append({
            "expert": example.expert_id,
            "target": target,
            "probability": probability,
            "weight": weight,
            "correct": correct,
            "logLoss": -(target * math.log(clipped) +
                         (1.0 - target) * math.log(1.0 - clipped)),
            "brier": (probability - target) ** 2,
            "preferredMargin": preferred_margin,
        })
    return _preference_metrics(rows)


def expert_agreement(preferences: Sequence[PreferenceExample]) -> Dict[str, Any]:
    """Summarize inter-rater agreement for contexts annotated by multiple experts."""
    by_context: Dict[str, List[PreferenceExample]] = {}
    for example in preferences:
        by_context.setdefault(example.context_id, []).append(example)
    comparable = []
    disagreements = []
    for context_id, rows in sorted(by_context.items()):
        if len(rows) < 2:
            continue
        votes = {"left": 0.0, "right": 0.0, "tie": 0.0}
        for row in rows:
            votes[row.preferred] += row.effective_weight
        total = sum(votes.values())
        majority = max(votes, key=lambda value: (votes[value], value))
        agreement = votes[majority] / total if total else 0.0
        comparable.append(agreement)
        if agreement < 0.75:
            disagreements.append({
                "contextId": context_id,
                "agreement": agreement,
                "votes": votes,
                "experts": sorted(set(row.expert_id for row in rows)),
            })
    return {
        "multiExpertContexts": len(comparable),
        "meanMajorityAgreement": (
            sum(comparable) / len(comparable) if comparable else None),
        "disagreementContexts": disagreements,
    }


def split_preferences_by_context(
        preferences: Sequence[PreferenceExample],
        holdout_fraction: float = 0.2,
        seed: int = 0) -> Tuple[List[PreferenceExample], List[PreferenceExample]]:
    """Deterministically split whole contexts, never individual comparisons."""
    examples = list(preferences)
    if not examples:
        return [], []
    if not _finite(holdout_fraction) or not 0.0 <= float(holdout_fraction) < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    contexts = sorted(set(example.context_id for example in examples))
    if len(contexts) == 1 or holdout_fraction == 0.0:
        return examples, []
    ranked = sorted(
        contexts,
        key=lambda value: hashlib.sha256(
            (str(seed) + "\x00" + value).encode("utf-8")).hexdigest())
    holdout_count = int(round(len(contexts) * float(holdout_fraction)))
    holdout_count = min(max(1, holdout_count), len(contexts) - 1)
    holdout_contexts = set(ranked[:holdout_count])
    train = [item for item in examples if item.context_id not in holdout_contexts]
    holdout = [item for item in examples if item.context_id in holdout_contexts]
    return train, holdout


def load_factor_specs(path: str) -> List[FactorSpec]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        factors = payload
    elif isinstance(payload, Mapping):
        version = payload.get("schemaVersion")
        if version is not None and version != SCHEMA_VERSION:
            raise ValueError("unsupported factor ontology schemaVersion")
        kind = payload.get("kind")
        if kind is not None and kind != _FACTOR_KIND:
            raise ValueError("not an expert factor ontology")
        factors = payload.get("factors")
    else:
        factors = None
    if not isinstance(factors, list) or not factors:
        raise ValueError("factor ontology must contain a non-empty factors list")
    specs = [FactorSpec.from_dict(item) for item in factors]
    _validate_specs(specs)
    return specs


def load_preferences(
        path: str, specs: Sequence[FactorSpec]) -> List[PreferenceExample]:
    examples = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                examples.append(PreferenceExample.from_dict(payload, specs))
            except (TypeError, ValueError) as exc:
                raise ValueError("%s:%d: %s" % (path, line_number, exc))
    if not examples:
        raise ValueError("preference file contains no examples")
    return examples


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _preference_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total_weight = sum(float(row["weight"]) for row in rows)
    non_ties = [row for row in rows if row["correct"] is not None]
    non_tie_weight = sum(float(row["weight"]) for row in non_ties)
    by_expert: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_expert.setdefault(str(row["expert"]), []).append(row)
    result = {
        "examples": len(rows),
        "nonTieExamples": len(non_ties),
        "pairwiseAccuracy": (
            sum(float(row["weight"]) for row in non_ties if row["correct"]) /
            non_tie_weight if non_tie_weight else None),
        "weightedLogLoss": (
            sum(float(row["weight"]) * float(row["logLoss"]) for row in rows) /
            total_weight if total_weight else None),
        "weightedBrier": (
            sum(float(row["weight"]) * float(row["brier"]) for row in rows) /
            total_weight if total_weight else None),
        "meanPreferredMargin": (
            sum(float(row["weight"]) * float(row["preferredMargin"]) for row in rows) /
            total_weight if total_weight else None),
        "byExpert": {},
    }
    for expert, expert_rows in sorted(by_expert.items()):
        weight = sum(float(row["weight"]) for row in expert_rows)
        expert_non_ties = [row for row in expert_rows if row["correct"] is not None]
        expert_non_tie_weight = sum(float(row["weight"]) for row in expert_non_ties)
        result["byExpert"][expert] = {
            "examples": len(expert_rows),
            "pairwiseAccuracy": (
                sum(float(row["weight"]) for row in expert_non_ties if row["correct"]) /
                expert_non_tie_weight if expert_non_tie_weight else None),
            "weightedLogLoss": (
                sum(float(row["weight"]) * float(row["logLoss"])
                    for row in expert_rows) / weight if weight else None),
        }
    return result


def _vector(specs: Sequence[FactorSpec], factors: Mapping[str, Any]) -> Dict[str, float]:
    if not isinstance(factors, Mapping):
        raise ValueError("candidate factors must be an object")
    names = {spec.name for spec in specs}
    unknown = sorted(set(str(name) for name in factors).difference(names))
    if unknown:
        raise ValueError("unknown factors: %s" % unknown)
    return {
        spec.name: spec.oriented(factors.get(spec.name, spec.default))
        for spec in specs
    }


def _validate_specs(specs: Sequence[FactorSpec]) -> None:
    if not specs:
        raise ValueError("at least one factor is required")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        raise ValueError("duplicate factor names: %s" % duplicates)


def _parse_preferred(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        aliases = {
            "left": "left", "l": "left", "a": "left", "1": "left",
            "right": "right", "r": "right", "b": "right", "-1": "right",
            "tie": "tie", "equal": "tie", "draw": "tie", "0": "tie",
        }
        if normalized in aliases:
            return aliases[normalized]
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if value > 0:
            return "left"
        if value < 0:
            return "right"
        return "tie"
    raise ValueError("preferred must identify left, right, or tie")


def _transform(value: float, transform: str) -> float:
    if transform == "identity":
        return value
    if transform == "log1p":
        if value < 0.0:
            raise ValueError("log1p factor values must be non-negative")
        return math.log1p(value)
    if transform == "signed_log1p":
        return math.copysign(math.log1p(abs(value)), value)
    if transform == "sqrt":
        if value < 0.0:
            raise ValueError("sqrt factor values must be non-negative")
        return math.sqrt(value)
    if transform == "tanh":
        return math.tanh(value)
    if transform == "binary":
        return 1.0 if value > 0.0 else 0.0
    raise ValueError("unsupported factor transform: %s" % transform)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponent = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(max(value, -700.0))
    return exponent / (1.0 + exponent)


def _numeric(value: Any, name: str) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float)) or not _finite(value):
        raise ValueError("factor %s must be a finite number" % name)
    return float(value)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(float(value))) or isinstance(value, bool)


__all__ = [
    "CandidateFactors",
    "ExpertCostModel",
    "FactorSpec",
    "PreferenceExample",
    "SCHEMA_VERSION",
    "evaluate_preferences",
    "expert_agreement",
    "file_sha256",
    "fit_cost_model",
    "load_factor_specs",
    "load_preferences",
    "split_preferences_by_context",
]
