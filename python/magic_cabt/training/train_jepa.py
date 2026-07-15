"""Train the local structured Magic JEPA from replay bundles.

The trainer mixes four signals:

* action-conditioned future-latent prediction at 1/4/16 state horizons;
* generic causal before/after deltas supplied by the recorded trajectory;
* terminal value labels when a bundle has one unambiguous game result;
* imitation loss over canonical legal-option groups.

Training and evaluation are split on whole-game identities. The run records
throughput, collapse diagnostics, held-out losses, and enough optimizer/RNG
state to resume interrupted experiments without silently starting a new run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from itertools import cycle

from magic_cabt.models.embeddings import make_embedding_provider
from magic_cabt.models.structured_jepa import (
    TORCH_AVAILABLE,
    CardTextResolver,
    MagicJEPA,
    StructuredJEPAConfig,
    StructuredTensorizer,
    causal_delta_vector,
    model_parameter_count,
)

try:
    from magic_cabt.training.action_dedup import canonical_groups, group_index_of
except ImportError:  # pragma: no cover - standalone development fallback
    canonical_groups = group_index_of = None
try:
    from magic_cabt.training.io import iter_decision_records
except ImportError:  # pragma: no cover
    iter_decision_records = None

__all__ = [
    "collect_training_data",
    "split_training_data",
    "train",
    "build_parser",
    "main",
]

_HORIZONS = (1, 4, 16)


def collect_training_data(inputs, max_transitions=200000,
                          max_decisions=100000, seed=0):
    """Return bounded transition/decision samples plus merged card metadata."""
    transitions = _reservoir(
        _iter_all_transitions(inputs), max_transitions, seed)
    decisions = _reservoir(
        _iter_all_decisions(inputs), max_decisions, seed + 1)
    cards = {}
    for path in inputs:
        if not os.path.isdir(path):
            continue
        cache = os.path.join(path, "card_cache.json")
        if not os.path.exists(cache):
            continue
        try:
            with open(cache, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                cards.update(payload)
        except (OSError, ValueError):
            pass
    return transitions, decisions, cards


def split_training_data(transitions, decisions, eval_fraction=0.1, seed=0):
    """Split both supervision streams on shared whole-game identities.

    Examples without a recoverable game identity remain in training. They are
    never randomly split by row because doing so can place adjacent states from
    one trajectory on both sides of the evaluation boundary.
    """
    eval_fraction = float(eval_fraction)
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in [0, 1)")
    groups = sorted({
        group for group in
        [_example_group(item) for item in transitions] +
        [_example_group(item) for item in decisions]
        if group is not None
    })
    rng = random.Random(seed)
    rng.shuffle(groups)
    eval_count = 0
    if eval_fraction > 0.0 and len(groups) > 1:
        eval_count = max(1, int(round(len(groups) * eval_fraction)))
        eval_count = min(eval_count, len(groups) - 1)
    eval_groups = set(groups[:eval_count])

    def partition(items):
        train_rows, eval_rows = [], []
        unknown = 0
        for item in items:
            group = _example_group(item)
            if group is None:
                unknown += 1
                train_rows.append(item)
            elif group in eval_groups:
                eval_rows.append(item)
            else:
                train_rows.append(item)
        return train_rows, eval_rows, unknown

    train_transitions, eval_transitions, unknown_transitions = partition(
        transitions)
    train_decisions, eval_decisions, unknown_decisions = partition(decisions)
    metadata = {
        "unit": "game",
        "seed": int(seed),
        "evalFraction": eval_fraction,
        "knownGroups": len(groups),
        "evalGroups": len(eval_groups),
        "trainGroups": len(groups) - len(eval_groups),
        "unknownTransitionGroups": unknown_transitions,
        "unknownDecisionGroups": unknown_decisions,
        "evalGroupIds": sorted(eval_groups),
    }
    return (train_transitions, eval_transitions,
            train_decisions, eval_decisions, metadata)


def train(transitions, decisions, config=None, embedding_provider=None,
          card_resolver=None, resume=None, epochs=3, batch_size=32,
          lr=3e-4, weight_decay=1e-4, tau=0.996, seed=0,
          device=None, causal_weight=0.25, value_weight=0.25,
          policy_weight=1.0, eval_fraction=0.1, eval_seed=None,
          amp="auto", grad_accum_steps=1, max_steps_per_epoch=None,
          eval_batch_size=None, collapse_sample_limit=2048, log=None):
    """Train and return ``(model, metrics)``.

    The returned model is the final resumable checkpoint. The best held-out
    state is attached as ``model._best_state_dict`` so the CLI can also write a
    separate inference checkpoint without making optimizer state inconsistent.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "JEPA training requires PyTorch: pip install -e 'python[jepa]'")
    import torch
    import torch.nn.functional as F

    if not transitions and not decisions:
        raise ValueError("no training examples")
    epochs = max(1, int(epochs))
    batch_size = max(1, int(batch_size))
    grad_accum_steps = max(1, int(grad_accum_steps))
    eval_batch_size = max(1, int(eval_batch_size or batch_size))
    if max_steps_per_epoch is not None:
        max_steps_per_epoch = max(1, int(max_steps_per_epoch))

    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = _resolve_amp(amp, device)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    previous_extra = {}
    if resume:
        model, previous_extra = MagicJEPA.load_checkpoint(
            resume, map_location=device)
        config = model.config
    else:
        config = config or StructuredJEPAConfig.preset("local")
        model = MagicJEPA(config)
    model.to(device)
    provider = embedding_provider or make_embedding_provider(
        config.embedding_backend, dimension=config.text_dim,
        device=device if device.type == "cuda" else None)
    tensorizer = StructuredTensorizer(
        config, provider, card_resolver or CardTextResolver())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    previous_state = previous_extra.get("trainingState") \
        if isinstance(previous_extra, dict) else None
    previous_state = previous_state if isinstance(previous_state, dict) else {}
    if previous_state.get("optimizer"):
        optimizer.load_state_dict(previous_state["optimizer"])
    if amp_enabled and previous_state.get("gradScaler"):
        scaler.load_state_dict(previous_state["gradScaler"])
    _restore_rng_state(previous_state, torch, device)

    split_seed = seed if eval_seed is None else int(eval_seed)
    (train_transitions, eval_transitions,
     train_decisions, eval_decisions, split_metadata) = split_training_data(
         transitions, decisions, eval_fraction=eval_fraction, seed=split_seed)

    if not train_transitions and not train_decisions:
        raise ValueError("evaluation split consumed every trainable example")

    rng = random.Random()
    if previous_state.get("orderRandomState") is not None:
        try:
            rng.setstate(previous_state["orderRandomState"])
        except (TypeError, ValueError):
            rng.seed(seed)
    else:
        rng.seed(seed)
    history = []
    started = time.perf_counter()
    completed_before = int(previous_state.get("completedEpochs") or 0)
    best_metric = math.inf
    best_epoch = None
    best_state = None
    optimizer_steps = 0
    examples_seen = 0

    for epoch in range(epochs):
        model.train()
        transition_order = list(range(len(train_transitions)))
        decision_order = list(range(len(train_decisions)))
        rng.shuffle(transition_order)
        rng.shuffle(decision_order)
        transition_batches = list(_index_batches(
            transition_order, batch_size))
        decision_batches = list(_index_batches(decision_order, batch_size))
        transition_cycle = cycle(transition_batches) \
            if transition_batches else None
        decision_cycle = cycle(decision_batches) if decision_batches else None
        step_count = max(len(transition_batches), len(decision_batches), 1)
        if max_steps_per_epoch is not None:
            step_count = min(step_count, max_steps_per_epoch)
        totals = {key: 0.0 for key in
                  ("loss", "jepa", "causal", "value", "policy")}
        counts = {key: 0 for key in ("transition", "decision", "value")}
        optimizer.zero_grad(set_to_none=True)

        for step in range(step_count):
            with _autocast_context(torch, device, amp_enabled):
                loss = torch.zeros((), device=device)
                if transition_cycle is not None:
                    indices = next(transition_cycle)
                    batch = [train_transitions[index] for index in indices]
                    pieces = _transition_losses(
                        model, tensorizer, batch, device, config,
                        causal_weight, value_weight, F)
                    loss = loss + pieces["weightedLoss"]
                    totals["jepa"] += pieces["jepa"]
                    totals["causal"] += pieces["causal"]
                    totals["value"] += pieces["value"]
                    counts["transition"] += len(batch)
                    counts["value"] += pieces["valueExamples"]
                    examples_seen += len(batch)

                if decision_cycle is not None:
                    indices = next(decision_cycle)
                    batch = [train_decisions[index] for index in indices]
                    policy_loss = _policy_loss(
                        model, tensorizer, batch, device)
                    loss = loss + policy_weight * policy_loss
                    totals["policy"] += float(policy_loss.detach())
                    counts["decision"] += len(batch)
                    examples_seen += len(batch)

            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    "non-finite training loss at epoch %d step %d" %
                    (completed_before + epoch + 1, step + 1))
            totals["loss"] += float(loss.detach())
            scaler.scale(loss / grad_accum_steps).backward()
            should_step = ((step + 1) % grad_accum_steps == 0 or
                           step + 1 == step_count)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                model.update_target(tau=tau)
                optimizer_steps += 1

        train_metrics = {
            key: value / max(1, step_count)
            for key, value in totals.items()
        }
        train_metrics.update({
            "transitionExamples": counts["transition"],
            "decisionExamples": counts["decision"],
            "valueExamples": counts["value"],
            "steps": step_count,
        })
        eval_metrics = _evaluate(
            model, tensorizer, eval_transitions, eval_decisions,
            device=device, config=config, batch_size=eval_batch_size,
            causal_weight=causal_weight, value_weight=value_weight,
            policy_weight=policy_weight,
            collapse_sample_limit=collapse_sample_limit)
        selection = _selection_metric(eval_metrics, train_metrics)
        absolute_epoch = completed_before + epoch + 1
        epoch_metrics = {
            "epoch": absolute_epoch,
            "train": train_metrics,
            "eval": eval_metrics,
            "selectionMetric": selection,
        }
        history.append(epoch_metrics)
        if selection < best_metric:
            best_metric = selection
            best_epoch = absolute_epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if log:
            log(_format_epoch_log(
                absolute_epoch, completed_before + epochs,
                train_metrics, eval_metrics, selection))

    elapsed = max(time.perf_counter() - started, 1e-9)
    peak_memory = None
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))

    training_state = {
        "optimizer": optimizer.state_dict(),
        "gradScaler": scaler.state_dict() if amp_enabled else None,
        "completedEpochs": completed_before + epochs,
        "pythonRandomState": random.getstate(),
        "orderRandomState": rng.getstate(),
        "torchRngState": torch.get_rng_state(),
        "cudaRngStateAll": torch.cuda.get_rng_state_all()
            if device.type == "cuda" else None,
    }
    model._training_state = training_state
    model._best_state_dict = best_state
    model._best_epoch = best_epoch

    metrics = {
        "kind": "magic-structured-jepa-training-v2",
        "device": str(device),
        "amp": {
            "requested": amp,
            "enabled": amp_enabled,
            "dtype": "float16" if amp_enabled else "float32",
        },
        "parameters": model_parameter_count(model),
        "trainableParameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad),
        "transitionExamples": len(transitions),
        "decisionExamples": len(decisions),
        "trainTransitionExamples": len(train_transitions),
        "evalTransitionExamples": len(eval_transitions),
        "trainDecisionExamples": len(train_decisions),
        "evalDecisionExamples": len(eval_decisions),
        "epochsThisRun": epochs,
        "completedEpochs": completed_before + epochs,
        "history": history,
        "bestEpoch": best_epoch,
        "bestSelectionMetric": best_metric,
        "split": split_metadata,
        "batchSize": batch_size,
        "evalBatchSize": eval_batch_size,
        "gradientAccumulationSteps": grad_accum_steps,
        "optimizerSteps": optimizer_steps,
        "examplesSeen": examples_seen,
        "wallSeconds": elapsed,
        "examplesPerSecond": examples_seen / elapsed,
        "peakCudaMemoryBytes": peak_memory,
        "resumed": bool(resume),
        "resumeCheckpoint": os.path.abspath(resume) if resume else None,
        "previousCompletedEpochs": completed_before,
    }
    return model, metrics


def _transition_losses(model, tensorizer, batch, device, config,
                       causal_weight, value_weight, F):
    import torch

    previous = [item["prev"] for item in batch]
    following = [item["next"] for item in batch]
    actions = [item.get("action") for item in batch]
    horizons = torch.tensor(
        [int(item.get("horizon") or 1) for item in batch],
        dtype=torch.long, device=device)
    prev_rows, prev_mask = tensorizer.batch_states(previous, device)
    next_rows, next_mask = tensorizer.batch_states(following, device)
    action_vectors = tensorizer.batch_actions(actions, device)
    state = model.encode(prev_rows, prev_mask)
    predicted, log_scale = model.predict_distribution(
        state, action_vectors, horizons)
    with torch.no_grad():
        target = model.encode_target(next_rows, next_mask)
    jepa_loss, pieces = model.jepa_loss(predicted, target, log_scale)
    causal_target = torch.tensor([
        causal_delta_vector(
            item["prev"], item["next"],
            dimension=config.causal_dim)
        for item in batch
    ], dtype=torch.float32, device=device)
    causal_loss = F.smooth_l1_loss(
        model.causal_delta(state, action_vectors, horizons),
        causal_target)
    weighted = jepa_loss + causal_weight * causal_loss
    value_loss = state.new_tensor(0.0)
    labeled = [
        (row, item.get("outcome"))
        for row, item in enumerate(batch)
        if item.get("outcome") is not None
    ]
    if labeled:
        rows = torch.tensor(
            [row for row, _value in labeled],
            dtype=torch.long, device=device)
        targets = torch.tensor(
            [float(value) for _row, value in labeled],
            dtype=torch.float32, device=device)
        values = model.value(state.index_select(0, rows)).squeeze(-1)
        value_loss = F.mse_loss(values, targets)
        weighted = weighted + value_weight * value_loss
    return {
        "weightedLoss": weighted,
        "jepaTensor": jepa_loss,
        "causalTensor": causal_loss,
        "valueTensor": value_loss,
        "predicted": predicted,
        "target": target,
        "jepa": float(jepa_loss.detach()),
        "causal": float(causal_loss.detach()),
        "value": float(value_loss.detach()),
        "alignment": float(pieces["alignment"]),
        "variancePenalty": float(pieces["variance"]),
        "uncertainty": float(pieces["uncertainty"]),
        "valueExamples": len(labeled),
    }


def _policy_loss(model, tensorizer, records, device):
    import torch

    rows, row_mask = tensorizer.batch_states(records, device)
    option_vectors, option_mask = tensorizer.batch_options(records, device)
    logits = model.score_options(
        rows, row_mask, option_vectors, option_mask)
    losses = []
    for row, record in enumerate(records):
        groups = record["_groups"]
        chosen_group = record["_chosenGroup"]
        denominator = torch.logsumexp(logits[row], dim=0)
        numerator = torch.logsumexp(
            logits[row, groups[chosen_group]], dim=0)
        losses.append(denominator - numerator)
    return torch.stack(losses).mean()


def _evaluate(model, tensorizer, transitions, decisions, device, config,
              batch_size, causal_weight, value_weight, policy_weight,
              collapse_sample_limit):
    import torch
    import torch.nn.functional as F

    model.eval()
    totals = {key: 0.0 for key in
              ("loss", "jepa", "causal", "value", "policy")}
    transition_batches = decision_batches = 0
    value_examples = 0
    representations = []
    policy_top1 = policy_top3 = 0
    reciprocal_rank = 0.0
    policy_examples = 0

    with torch.no_grad():
        for start in range(0, len(transitions), batch_size):
            batch = transitions[start:start + batch_size]
            pieces = _transition_losses(
                model, tensorizer, batch, device, config,
                causal_weight, value_weight, F)
            totals["loss"] += float(pieces["weightedLoss"])
            totals["jepa"] += pieces["jepa"]
            totals["causal"] += pieces["causal"]
            totals["value"] += pieces["value"]
            transition_batches += 1
            value_examples += pieces["valueExamples"]
            if sum(row.shape[0] for row in representations) < \
                    collapse_sample_limit:
                representations.append(pieces["predicted"].detach().float().cpu())

        for start in range(0, len(decisions), batch_size):
            batch = decisions[start:start + batch_size]
            rows, row_mask = tensorizer.batch_states(batch, device)
            option_vectors, option_mask = tensorizer.batch_options(
                batch, device)
            logits = model.score_options(
                rows, row_mask, option_vectors, option_mask)
            policy_loss = _policy_loss(
                model, tensorizer, batch, device)
            totals["policy"] += float(policy_loss)
            totals["loss"] += policy_weight * float(policy_loss)
            decision_batches += 1
            for row, record in enumerate(batch):
                probabilities = torch.softmax(logits[row], dim=0)
                masses = [
                    float(probabilities[members].sum())
                    for members in record["_groups"]
                ]
                ranking = sorted(
                    range(len(masses)), key=lambda index: (-masses[index], index))
                rank = ranking.index(record["_chosenGroup"]) + 1
                policy_top1 += 1 if rank == 1 else 0
                policy_top3 += 1 if rank <= 3 else 0
                reciprocal_rank += 1.0 / rank
                policy_examples += 1

    divisor = max(1, transition_batches + decision_batches)
    result = {
        "examples": len(transitions) + len(decisions),
        "transitionExamples": len(transitions),
        "decisionExamples": len(decisions),
        "valueExamples": value_examples,
        "loss": totals["loss"] / divisor,
        "jepa": totals["jepa"] / max(1, transition_batches),
        "causal": totals["causal"] / max(1, transition_batches),
        "value": totals["value"] / max(1, transition_batches),
        "policy": totals["policy"] / max(1, decision_batches),
        "policyTop1": policy_top1 / policy_examples
            if policy_examples else None,
        "policyTop3": policy_top3 / policy_examples
            if policy_examples else None,
        "policyMRR": reciprocal_rank / policy_examples
            if policy_examples else None,
    }
    result["collapse"] = _collapse_diagnostics(
        representations, collapse_sample_limit)
    model.train()
    return result


def _collapse_diagnostics(representations, limit):
    if not representations:
        return {
            "examples": 0,
            "meanDimensionStd": None,
            "minimumDimensionStd": None,
            "effectiveRank": None,
        }
    import torch

    values = torch.cat(representations, dim=0)[:limit]
    if values.shape[0] < 2:
        return {
            "examples": int(values.shape[0]),
            "meanDimensionStd": 0.0,
            "minimumDimensionStd": 0.0,
            "effectiveRank": 1.0,
        }
    std = values.std(dim=0, unbiased=False)
    centered = values - values.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return {
        "examples": int(values.shape[0]),
        "meanDimensionStd": float(std.mean()),
        "minimumDimensionStd": float(std.min()),
        "effectiveRank": float(entropy.exp()),
    }


def _selection_metric(eval_metrics, train_metrics):
    if eval_metrics.get("examples"):
        return float(eval_metrics["loss"])
    return float(train_metrics["loss"])


def _format_epoch_log(epoch, final_epoch, train_metrics, eval_metrics,
                      selection):
    eval_text = "n/a"
    if eval_metrics.get("examples"):
        eval_text = "%.4f" % eval_metrics["loss"]
    return (
        "epoch %d/%d train=%.4f eval=%s jepa=%.4f causal=%.4f "
        "value=%.4f policy=%.4f select=%.4f"
        % (epoch, final_epoch, train_metrics["loss"], eval_text,
           train_metrics["jepa"], train_metrics["causal"],
           train_metrics["value"], train_metrics["policy"], selection)
    )


def _resolve_amp(requested, device):
    value = str(requested or "auto").lower()
    if value not in ("auto", "on", "off"):
        raise ValueError("amp must be auto, on, or off")
    if value == "on" and device.type != "cuda":
        raise ValueError("AMP was requested but the selected device is not CUDA")
    return device.type == "cuda" and value != "off"


def _autocast_context(torch, device, enabled):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _restore_rng_state(state, torch, device):
    try:
        if state.get("pythonRandomState") is not None:
            random.setstate(state["pythonRandomState"])
        if state.get("torchRngState") is not None:
            torch.set_rng_state(state["torchRngState"])
        if device.type == "cuda" and state.get("cudaRngStateAll") is not None:
            torch.cuda.set_rng_state_all(state["cudaRngStateAll"])
    except (RuntimeError, TypeError, ValueError):
        pass


def _example_group(item):
    current = (item.get("prev") or item.get("current") or
               (item.get("observation") or {}).get("current") or {})
    match = item.get("matchId") or item.get("gameId") or current.get("matchId")
    game = item.get("gameNumber")
    if game is None:
        game = current.get("gameNumber")
    instance = item.get("gameInstance") or current.get("gameInstance")
    if match is None and game is None and instance is None:
        return None
    return json.dumps(
        [match, game, instance], sort_keys=False, separators=(",", ":"))


def _iter_all_transitions(inputs):
    for path in inputs:
        outcome = _bundle_outcome(path) if os.path.isdir(path) else None
        if os.path.isdir(path):
            states_path = os.path.join(path, "mirror_states.jsonl")
            decisions_path = os.path.join(path, "decisions.jsonl")
            explicit = os.path.join(path, "transitions.jsonl")
            generated = False
            if os.path.exists(states_path):
                generated = True
                yield from _horizon_transitions(
                    _read_jsonl(states_path), outcome=outcome,
                    source="mirror_states")
            if os.path.exists(decisions_path):
                generated = True
                records = list(_decision_reader(decisions_path))
                yield from _decision_transitions(records, outcome=outcome)
            if not generated and os.path.exists(explicit):
                for item in _read_jsonl(explicit):
                    if outcome is not None and item.get("outcome") is None:
                        item["outcome"] = outcome
                    item.setdefault("horizon", 1)
                    yield item
            continue
        name = os.path.basename(path)
        if name == "mirror_states.jsonl":
            yield from _horizon_transitions(
                _read_jsonl(path), source="mirror_states")
        elif name == "transitions.jsonl":
            for item in _read_jsonl(path):
                item.setdefault("horizon", 1)
                yield item
        else:
            yield from _decision_transitions(
                list(_decision_reader(path)))


def _iter_all_decisions(inputs):
    for path in inputs:
        if os.path.isdir(path):
            path = os.path.join(path, "decisions.jsonl")
            if not os.path.exists(path):
                continue
        elif os.path.basename(path) in (
                "mirror_states.jsonl", "transitions.jsonl"):
            continue
        for record in _decision_reader(path):
            compiled = _compile_decision(record)
            if compiled is not None:
                yield compiled


def _decision_reader(path):
    if iter_decision_records is not None:
        yield from iter_decision_records(path)
        return
    for raw in _read_jsonl(path):
        if "observation" not in raw:
            continue
        record = dict(raw)
        if "selectedIndices" not in record:
            record["selectedIndices"] = raw.get("select") or \
                raw.get("selected") or []
        if not isinstance(record.get("select"), dict):
            record["select"] = (raw.get("observation") or {}).get(
                "select") or {}
        yield record


def _compile_decision(record):
    select = _select(record)
    options = select.get("option") or []
    selected = _selected(record)
    if len(selected) != 1 or not options:
        return None
    chosen = selected[0]
    if not isinstance(chosen, int) or not 0 <= chosen < len(options):
        return None
    if canonical_groups is None:
        groups = [[index] for index in range(len(options))]
        chosen_group = chosen
    else:
        raw_groups = canonical_groups(select)
        groups = [group["indices"] for group in raw_groups]
        chosen_group = group_index_of(raw_groups, chosen)
        if chosen_group is None:
            return None
    value = dict(record)
    value["select"] = select
    value["selectedIndices"] = selected
    value["_groups"] = groups
    value["_chosenGroup"] = chosen_group
    return value


def _decision_transitions(records, outcome=None):
    entries = []
    for record in records:
        current = (record.get("observation") or {}).get("current") or \
            record.get("current")
        if isinstance(current, dict):
            entries.append((record, current))
    for start, (record, previous) in enumerate(entries):
        for horizon in _HORIZONS:
            target_index = start + horizon
            if target_index >= len(entries):
                continue
            target_record, following = entries[target_index]
            if _game_key(previous, record) != \
                    _game_key(following, target_record):
                continue
            yield {
                "source": "decisions",
                "matchId": record.get("matchId") or record.get("gameId"),
                "gameNumber": record.get("gameNumber"),
                "gameInstance": previous.get("gameInstance"),
                "horizon": horizon,
                "prev": previous,
                "next": following,
                "action": _selected_action(record),
                "outcome": outcome,
            }


def _horizon_transitions(states, outcome=None, source="mirror_states"):
    states = [state for state in states if isinstance(state, dict)]
    for start, previous in enumerate(states):
        for horizon in _HORIZONS:
            target_index = start + horizon
            if target_index >= len(states):
                continue
            following = states[target_index]
            if _game_key(previous) != _game_key(following):
                continue
            yield {
                "source": source,
                "matchId": following.get("matchId"),
                "gameNumber": following.get("gameNumber"),
                "gameInstance": following.get("gameInstance"),
                "horizon": horizon,
                "prev": previous,
                "next": following,
                "action": None,
                "outcome": outcome,
            }


def _selected_action(record):
    select = _select(record)
    options = select.get("option") or []
    selected = _selected(record)
    selected_options = [
        options[index] for index in selected
        if isinstance(index, int) and 0 <= index < len(options)
    ]
    return {
        "promptType": select.get("type"),
        "selectedIndices": selected,
        "selectedOption": selected_options[0]
            if len(selected_options) == 1 else None,
        "selectedOptions": selected_options,
    }


def _game_key(state, record=None):
    record = record or {}
    return (
        state.get("matchId") or record.get("matchId") or record.get("gameId"),
        state.get("gameNumber") or record.get("gameNumber"),
        state.get("gameInstance"),
    )


def _bundle_outcome(path):
    """Return a value label only for an unambiguous one-game bundle."""
    summary = os.path.join(path, "summary.json")
    if not os.path.exists(summary):
        return None
    try:
        with open(summary, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    matches = payload.get("matches")
    games = payload.get("games")
    match_ids = payload.get("matchIds")
    if isinstance(matches, list) and len(matches) != 1:
        return None
    if isinstance(games, list) and len(games) > 1:
        return None
    if isinstance(match_ids, list) and len(set(match_ids)) > 1:
        return None
    result = (payload.get("result") or "").lower()
    return 1.0 if result == "win" else (-1.0 if result == "loss" else None)


def _select(record):
    direct = record.get("select")
    if isinstance(direct, dict):
        return direct
    return (record.get("observation") or {}).get("select") or {}


def _selected(record):
    selected = record.get("selectedIndices") or record.get("selected")
    if selected is None and isinstance(record.get("select"), list):
        selected = record.get("select")
    return selected or []


def _read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def _reservoir(iterable, limit, seed):
    if not limit or limit < 0:
        return list(iterable)
    rng = random.Random(seed)
    result = []
    for seen, item in enumerate(iterable, start=1):
        if len(result) < limit:
            result.append(item)
        else:
            replacement = rng.randrange(seen)
            if replacement < limit:
                result[replacement] = item
    return result


def _index_batches(indices, batch_size):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _input_manifest(paths):
    manifest = []
    names = (
        "decisions.jsonl", "mirror_states.jsonl", "transitions.jsonl",
        "summary.json", "card_cache.json")
    for raw in paths:
        path = os.path.abspath(os.path.expanduser(raw))
        if os.path.isdir(path):
            files = []
            for name in names:
                candidate = os.path.join(path, name)
                if os.path.isfile(candidate):
                    files.append({
                        "name": name,
                        "bytes": os.path.getsize(candidate),
                        "sha256": _file_sha256(candidate),
                    })
            manifest.append({"path": path, "kind": "bundle", "files": files})
        elif os.path.isfile(path):
            manifest.append({
                "path": path,
                "kind": "file",
                "bytes": os.path.getsize(path),
                "sha256": _file_sha256(path),
            })
        else:
            manifest.append({"path": path, "kind": "missing"})
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-train-jepa",
        description="Train the structured JEPA with whole-game evaluation.")
    parser.add_argument(
        "--input", required=True, action="append",
        help="bundle directory or JSONL file; repeatable")
    parser.add_argument(
        "--out", required=True,
        help="output directory for checkpoint.pt, best.pt, and metrics.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--preset", default="local", choices=("tiny", "local", "large"))
    parser.add_argument(
        "--embedding-backend", default="hash",
        help="hash or sentence-transformers:<model>")
    parser.add_argument(
        "--arena-card-db", default=None,
        help="optional MTGA Raw_CardDatabase_*.mtga for rules text")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-transitions", type=int, default=200000)
    parser.add_argument("--max-decisions", type=int, default=100000)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--amp", choices=("auto", "on", "off"), default="auto",
        help="mixed precision; auto enables it only on CUDA")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    transitions, decisions, cards = collect_training_data(
        args.input, max_transitions=args.max_transitions,
        max_decisions=args.max_decisions, seed=args.seed)
    sys.stderr.write("loaded %d transitions and %d decisions\n" %
                     (len(transitions), len(decisions)))
    config = StructuredJEPAConfig.preset(args.preset)
    config.embedding_backend = args.embedding_backend
    resolver = CardTextResolver(
        cards, arena_db_path=args.arena_card_db)
    model, metrics = train(
        transitions, decisions, config=config, card_resolver=resolver,
        resume=args.resume, epochs=args.epochs, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
        device=args.device, eval_fraction=args.eval_fraction,
        eval_seed=args.eval_seed, amp=args.amp,
        grad_accum_steps=args.grad_accum_steps,
        max_steps_per_epoch=args.max_steps_per_epoch,
        log=lambda text: sys.stderr.write(text + "\n"))
    metrics["inputs"] = _input_manifest(args.input)
    os.makedirs(args.out, exist_ok=True)

    checkpoint = os.path.join(args.out, "checkpoint.pt")
    model.save_checkpoint(checkpoint, extra={
        "metrics": metrics,
        "trainingState": getattr(model, "_training_state", {}),
    })

    best_checkpoint = None
    best_state = getattr(model, "_best_state_dict", None)
    if best_state is not None:
        best_model = MagicJEPA(model.config)
        best_model.load_state_dict(best_state)
        best_checkpoint = os.path.join(args.out, "best.pt")
        best_model.save_checkpoint(best_checkpoint, extra={
            "metrics": metrics,
            "selection": {
                "epoch": metrics.get("bestEpoch"),
                "metric": metrics.get("bestSelectionMetric"),
            },
        })

    metrics["checkpoint"] = os.path.abspath(checkpoint)
    metrics["bestCheckpoint"] = os.path.abspath(best_checkpoint) \
        if best_checkpoint else None
    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    sys.stderr.write("wrote %s\n" % checkpoint)
    if best_checkpoint:
        sys.stderr.write("wrote %s\n" % best_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
