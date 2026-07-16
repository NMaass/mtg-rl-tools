"""Train recurrent policy and calibrated belief probes without label leakage."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext

from magic_cabt.models.belief import (
    TORCH_AVAILABLE, BeliefInformationStateModel)
from magic_cabt.models.structured_jepa import StructuredJEPAConfig
from magic_cabt.models.visibility import VisibilitySafeTensorizer
from .train_information_state import (
    _batch_windows, build_game_sequences, split_sequences, window_sequences)

_FORBIDDEN_OBSERVATION_KEYS = frozenset({
    "traininglabels", "oraclelabels", "belieflabels", "trueopponenthand",
    "opponenthandtruth", "hiddenstatetruth", "oraclehiddenstate",
})


def load_vocabulary(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schemaVersion") != 1:
        raise ValueError("belief vocabulary schemaVersion must be 1")
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("belief vocabulary requires labels")
    normalized = [str(label).strip() for label in labels]
    if any(not label for label in normalized):
        raise ValueError("belief labels must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("belief labels must be unique")
    return normalized, payload


def _normalized_key(value):
    return "".join(character for character in str(value).lower()
                   if character.isalnum())


def _contains_forbidden_key(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if _normalized_key(key) in _FORBIDDEN_OBSERVATION_KEYS:
                return True
            if _contains_forbidden_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def compile_belief_record(record, labels):
    """Attach masked targets while leaving the observation unchanged.

    Unlabeled decisions remain in the sequence with an all-false belief mask so
    the recurrent history and policy objective are not punctured.
    """
    observation = record.get("observation") or {}
    if _contains_forbidden_key(observation):
        raise ValueError("oracle belief labels must not appear in observation")
    payload = record.get("trainingLabels")
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("trainingLabels must be an object")
    belief = payload.get("belief")
    if belief is not None:
        if payload.get("visibility") != "oracle-label-only":
            raise ValueError(
                "trainingLabels.visibility must be oracle-label-only")
        if not payload.get("source"):
            raise ValueError("trainingLabels.source is required")
        if not isinstance(belief, dict):
            raise ValueError("trainingLabels.belief must be an object")
    else:
        belief = {}
    targets, mask = [], []
    for label in labels:
        value = belief.get(label)
        if value is None:
            targets.append(0.0)
            mask.append(False)
        elif value in (0, 1, False, True):
            targets.append(float(value))
            mask.append(True)
        else:
            raise ValueError("belief target %s must be binary" % label)
    compiled = dict(record)
    compiled["_beliefTarget"] = targets
    compiled["_beliefMask"] = mask
    compiled["_beliefSource"] = str(payload.get("source") or "unlabeled")
    return compiled


def compile_belief_records(records, labels):
    compiled = []
    labeled_records = 0
    labeled_cells = 0
    sources = {}
    for record in records:
        value = compile_belief_record(record, labels)
        count = sum(value["_beliefMask"])
        if count:
            labeled_records += 1
            labeled_cells += count
            source = value["_beliefSource"]
            sources[source] = sources.get(source, 0) + 1
        compiled.append(value)
    return compiled, {
        "records": len(compiled),
        "labeledRecords": labeled_records,
        "unlabeledRecords": len(compiled) - labeled_records,
        "labeledCells": labeled_cells,
        "sources": dict(sorted(sources.items())),
    }


def calibration_metrics(probabilities, targets, bins=10):
    if len(probabilities) != len(targets):
        raise ValueError("calibration probabilities/targets length mismatch")
    if not probabilities:
        return {"examples": 0, "brier": None, "logLoss": None,
                "expectedCalibrationError": None, "bins": []}
    eps = 1e-7
    clipped = [max(eps, min(1.0 - eps, float(value)))
               for value in probabilities]
    truth = [float(value) for value in targets]
    brier = sum((p - y) ** 2 for p, y in zip(clipped, truth)) / len(truth)
    log_loss = -sum(
        y * math.log(p) + (1.0 - y) * math.log(1.0 - p)
        for p, y in zip(clipped, truth)) / len(truth)
    rows = []
    ece = 0.0
    for index in range(max(1, int(bins))):
        low = index / bins
        high = (index + 1) / bins
        members = [(p, y) for p, y in zip(clipped, truth)
                   if low <= p < high or (index == bins - 1 and p == 1.0)]
        if not members:
            rows.append({"low": low, "high": high, "count": 0,
                         "confidence": None, "frequency": None})
            continue
        confidence = sum(row[0] for row in members) / len(members)
        frequency = sum(row[1] for row in members) / len(members)
        ece += len(members) / len(truth) * abs(confidence - frequency)
        rows.append({"low": low, "high": high, "count": len(members),
                     "confidence": confidence, "frequency": frequency})
    return {"examples": len(truth), "brier": brier, "logLoss": log_loss,
            "expectedCalibrationError": ece, "bins": rows}


def calibration_report(labels, per_label_probabilities, per_label_targets):
    per_label = {}
    all_probabilities = []
    all_targets = []
    for index, label in enumerate(labels):
        probabilities = per_label_probabilities[index]
        targets = per_label_targets[index]
        per_label[label] = calibration_metrics(probabilities, targets)
        all_probabilities.extend(probabilities)
        all_targets.extend(targets)
    return {
        "aggregate": calibration_metrics(all_probabilities, all_targets),
        "perLabel": per_label,
    }


def _run_epoch(model, tensorizer, windows, labels, device, batch_size,
               belief_weight=1.0, optimizer=None, scaler=None,
               amp_enabled=False, grad_accum_steps=1):
    import torch
    import torch.nn.functional as F

    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "policy": 0.0, "belief": 0.0,
              "decisions": 0, "beliefCells": 0,
              "top1": 0, "top3": 0, "mrr": 0.0}
    probabilities = [[] for _label in labels]
    targets = [[] for _label in labels]
    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, start in enumerate(range(0, len(windows), batch_size)):
        batch = windows[start:start + batch_size]
        context = torch.autocast(device_type="cuda", dtype=torch.float16) \
            if amp_enabled else nullcontext()
        with context:
            rows, masks, previous, options, option_masks, sequence_mask = \
                _batch_windows(batch, tensorizer, device)
            memories, _hidden = model.information_states(
                rows, masks, previous, sequence_mask=sequence_mask)
            policy_logits = model.score_from_memory(
                memories, options, option_masks)
            belief_logits = model.belief_logits(memories)
            policy_losses = []
            belief_losses = []
            rankings = []
            batch_belief_cells = 0
            for row, window in enumerate(batch):
                for step, record in enumerate(window["records"]):
                    denominator = torch.logsumexp(
                        policy_logits[row, step], dim=0)
                    members = record["_groups"][record["_chosenGroup"]]
                    numerator = torch.logsumexp(
                        policy_logits[row, step, members], dim=0)
                    policy_losses.append(denominator - numerator)
                    softmax = torch.softmax(policy_logits[row, step], dim=0)
                    masses = [float(softmax[group].sum().detach())
                              for group in record["_groups"]]
                    ranking = sorted(range(len(masses)),
                                     key=lambda index: (-masses[index], index))
                    rankings.append(ranking.index(record["_chosenGroup"]) + 1)

                    target = torch.tensor(record["_beliefTarget"],
                                          dtype=torch.float32, device=device)
                    mask = torch.tensor(record["_beliefMask"],
                                        dtype=torch.bool, device=device)
                    if bool(mask.any()):
                        selected_logits = belief_logits[row, step][mask]
                        selected_target = target[mask]
                        belief_losses.append(
                            F.binary_cross_entropy_with_logits(
                                selected_logits, selected_target,
                                reduction="sum"))
                        batch_belief_cells += int(mask.sum())
                        predicted = torch.sigmoid(
                            belief_logits[row, step]).detach().float().cpu()
                        target_cpu = target.detach().float().cpu()
                        mask_cpu = mask.detach().cpu()
                        for label_index in range(len(labels)):
                            if bool(mask_cpu[label_index]):
                                probabilities[label_index].append(
                                    float(predicted[label_index]))
                                targets[label_index].append(
                                    float(target_cpu[label_index]))
            policy_loss = torch.stack(policy_losses).mean()
            if belief_losses:
                belief_loss = torch.stack(belief_losses).sum() / \
                    max(1, batch_belief_cells)
            else:
                belief_loss = policy_loss.new_tensor(0.0)
            loss = policy_loss + float(belief_weight) * belief_loss

        if training:
            scaler.scale(loss / max(1, grad_accum_steps)).backward()
            should_step = ((batch_index + 1) % grad_accum_steps == 0 or
                           start + batch_size >= len(windows))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        count = len(rankings)
        totals["loss"] += float(loss.detach()) * count
        totals["policy"] += float(policy_loss.detach()) * count
        totals["belief"] += float(belief_loss.detach()) * batch_belief_cells
        totals["decisions"] += count
        totals["beliefCells"] += batch_belief_cells
        totals["top1"] += sum(rank == 1 for rank in rankings)
        totals["top3"] += sum(rank <= 3 for rank in rankings)
        totals["mrr"] += sum(1.0 / rank for rank in rankings)

    count = totals["decisions"]
    belief_cells = totals["beliefCells"]
    return {
        "examples": count,
        "beliefCells": belief_cells,
        "loss": totals["loss"] / max(1, count),
        "policyLoss": totals["policy"] / max(1, count),
        "beliefLoss": totals["belief"] / max(1, belief_cells),
        "policyTop1": totals["top1"] / count if count else None,
        "policyTop3": totals["top3"] / count if count else None,
        "policyMRR": totals["mrr"] / count if count else None,
        "calibration": calibration_report(labels, probabilities, targets),
    }


def train(records, labels, config=None, resolver=None, epochs=5,
          batch_size=8, sequence_length=32, eval_fraction=0.1, seed=0,
          device=None, lr=3e-4, weight_decay=1e-4, belief_weight=1.0,
          amp="auto", grad_accum_steps=1, memory_layers=1, log=None):
    if not TORCH_AVAILABLE:
        raise ImportError("belief training requires magic-cabt[jepa]")
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = device.type == "cuda" and amp != "off"
    if amp == "on" and device.type != "cuda":
        raise ValueError("AMP was requested but the selected device is not CUDA")

    sequences = build_game_sequences(records)
    train_sequences, eval_sequences = split_sequences(
        sequences, eval_fraction=eval_fraction, seed=seed)
    train_windows = window_sequences(train_sequences, sequence_length)
    eval_windows = window_sequences(eval_sequences, sequence_length)
    if not train_windows:
        raise ValueError("no belief training sequences")

    config = config or StructuredJEPAConfig.preset("local")
    model = BeliefInformationStateModel(
        labels, config=config, memory_layers=memory_layers).to(device)
    tensorizer = VisibilitySafeTensorizer(config, card_resolver=resolver)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    rng = random.Random(seed)
    history = []
    best_metric, best_epoch, best_state = math.inf, None, None
    started = time.perf_counter()

    for epoch in range(1, max(1, int(epochs)) + 1):
        rng.shuffle(train_windows)
        train_metrics = _run_epoch(
            model, tensorizer, train_windows, labels, device,
            max(1, int(batch_size)), belief_weight=belief_weight,
            optimizer=optimizer, scaler=scaler, amp_enabled=amp_enabled,
            grad_accum_steps=max(1, int(grad_accum_steps)))
        with torch.no_grad():
            eval_metrics = _run_epoch(
                model, tensorizer, eval_windows, labels, device,
                max(1, int(batch_size)), belief_weight=belief_weight) \
                if eval_windows else {
                    "examples": 0, "beliefCells": 0, "loss": None,
                    "policyLoss": None, "beliefLoss": None,
                    "policyTop1": None, "policyTop3": None,
                    "policyMRR": None,
                    "calibration": calibration_report(
                        labels, [[] for _ in labels], [[] for _ in labels])}
        selection = eval_metrics["loss"] if eval_metrics["examples"] \
            else train_metrics["loss"]
        history.append({"epoch": epoch, "train": train_metrics,
                        "eval": eval_metrics,
                        "selectionMetric": selection})
        if selection < best_metric:
            best_metric, best_epoch = float(selection), epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if log:
            log("epoch %d/%d train=%.4f eval=%s belief=%.4f" % (
                epoch, epochs, train_metrics["loss"],
                "n/a" if not eval_metrics["examples"] else
                "%.4f" % eval_metrics["loss"],
                train_metrics["beliefLoss"]))

    elapsed = max(time.perf_counter() - started, 1e-9)
    metrics = {
        "kind": "magic-belief-information-state-training-v1",
        "modelFamily": "belief-information-state-v1",
        "decisionExamples": len(records),
        "beliefLabels": list(labels),
        "beliefWeight": float(belief_weight),
        "history": history,
        "bestEpoch": best_epoch,
        "bestSelectionMetric": best_metric,
        "trainGames": len(train_sequences),
        "evalGames": len(eval_sequences),
        "sequenceLength": int(sequence_length),
        "memoryLayers": int(memory_layers),
        "split": {
            "unit": "game",
            "seed": int(seed),
            "evalFraction": float(eval_fraction),
            "trainGameIds": [item["gameKey"] for item in train_sequences],
            "evalGameIds": [item["gameKey"] for item in eval_sequences],
        },
        "wallSeconds": elapsed,
        "visibilityPolicy": "public-history-and-perspective-state-v1",
        "amp": {"requested": amp, "enabled": amp_enabled},
        "device": str(device),
    }
    model._best_state_dict = best_state
    model._training_state = {
        "optimizer": optimizer.state_dict(),
        "completedEpochs": len(history),
        "bestStateDict": best_state,
        "bestEpoch": best_epoch,
        "bestSelectionMetric": best_metric,
    }
    return model, metrics


def build_parser():
    parser = argparse.ArgumentParser(prog="magic-cabt-train-belief-state")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preset", choices=("tiny", "local", "large"),
                        default="local")
    parser.add_argument("--embedding-backend", default="hash")
    parser.add_argument("--arena-card-db", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--memory-layers", type=int, default=1)
    parser.add_argument("--max-decisions", type=int, default=100000)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--belief-weight", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", choices=("auto", "on", "off"),
                        default="auto")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
