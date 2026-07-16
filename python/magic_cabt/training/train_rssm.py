"""Train a structured stochastic RSSM on complete Magic trajectories."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext

from magic_cabt.models.rssm import TORCH_AVAILABLE, StructuredRSSM
from magic_cabt.models.structured_jepa import (
    CardTextResolver, StructuredJEPAConfig, causal_delta_vector)
from magic_cabt.models.visibility import VisibilitySafeTensorizer
from . import train_jepa as core


def transition_game_key(item):
    previous = item.get("prev") or {}
    match = item.get("matchId") or item.get("gameId") or previous.get("matchId")
    game = item.get("gameNumber")
    if game is None:
        game = previous.get("gameNumber")
    instance = item.get("gameInstance") or previous.get("gameInstance")
    if match is None and game is None and instance is None:
        return None
    return json.dumps([match, game, instance], separators=(",", ":"))


def transition_sequence(item, fallback=0):
    previous = item.get("prev") or {}
    for value in (item.get("sequenceNumber"), item.get("sequence"),
                  previous.get("seq"), previous.get("sequenceNumber")):
        if isinstance(value, (int, float)):
            return float(value)
    return float(fallback)


def build_transition_sequences(transitions):
    grouped = defaultdict(list)
    unknown = []
    for index, item in enumerate(transitions):
        if int(item.get("horizon") or 1) != 1:
            continue
        if not isinstance(item.get("prev"), dict) or \
                not isinstance(item.get("next"), dict):
            continue
        key = transition_game_key(item)
        if key is None:
            unknown.append(item)
        else:
            grouped[key].append((transition_sequence(item, index), index, item))
    sequences = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda value: (value[0], value[1]))
        sequences.append({"gameKey": key,
                          "transitions": [value[2] for value in rows]})
    for index, item in enumerate(unknown):
        sequences.append({"gameKey": "unknown:%d" % index,
                          "transitions": [item]})
    return sequences


def split_sequences(sequences, eval_fraction=0.1, seed=0):
    if not 0.0 <= float(eval_fraction) < 1.0:
        raise ValueError("eval_fraction must be in [0, 1)")
    rows = list(sequences)
    rng = random.Random(seed)
    rng.shuffle(rows)
    eval_count = 0
    if eval_fraction > 0.0 and len(rows) > 1:
        eval_count = max(1, int(round(len(rows) * float(eval_fraction))))
        eval_count = min(eval_count, len(rows) - 1)
    return rows[eval_count:], rows[:eval_count]


def window_sequences(sequences, sequence_length):
    length = max(1, int(sequence_length))
    windows = []
    for sequence in sequences:
        transitions = sequence["transitions"]
        for start in range(0, len(transitions), length):
            chunk = transitions[start:start + length]
            if chunk:
                windows.append({"gameKey": sequence["gameKey"],
                                "start": start, "transitions": chunk})
    return windows


def collect_transition_data(inputs, max_transitions=200000):
    """Collect horizon-one transitions without truncating an accepted game."""
    limit = int(max_transitions) if max_transitions is not None else -1
    accepted = []
    current = []
    current_key = None
    unknown = 0
    stopped = False

    def flush():
        nonlocal stopped
        if not current:
            return
        if limit > 0 and accepted and len(accepted) + len(current) > limit:
            stopped = True
            return
        accepted.extend(current)

    for item in core._iter_all_transitions(inputs):
        if int(item.get("horizon") or 1) != 1:
            continue
        if not isinstance(item.get("prev"), dict) or \
                not isinstance(item.get("next"), dict):
            continue
        key = transition_game_key(item)
        if key is None:
            key = "unknown:%d" % unknown
            unknown += 1
        if current_key is not None and key != current_key:
            flush()
            if stopped:
                break
            current = []
        current_key = key
        current.append(item)
    if not stopped:
        flush()

    cards = {}
    for path in inputs:
        if not os.path.isdir(path):
            continue
        cache = os.path.join(path, "card_cache.json")
        if not os.path.isfile(cache):
            continue
        try:
            with open(cache, encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                cards.update(payload)
        except (OSError, ValueError):
            pass
    metadata = {
        "unit": "complete-game",
        "softLimit": limit,
        "acceptedTransitions": len(accepted),
        "truncatedAtGameBoundary": stopped,
        "unknownGameTransitions": unknown,
    }
    return accepted, cards, metadata


def _batch_windows(windows, tensorizer, device, causal_dim):
    import torch

    batch = len(windows)
    steps = max(len(window["transitions"]) for window in windows)
    states_per_window = steps + 1
    states = []
    actions = []
    state_mask = torch.zeros((batch, states_per_window), dtype=torch.bool,
                             device=device)
    transition_mask = torch.zeros((batch, steps), dtype=torch.bool,
                                  device=device)
    causal_targets = torch.zeros((batch, steps, causal_dim),
                                 dtype=torch.float32, device=device)
    value_targets = torch.zeros((batch, steps), dtype=torch.float32,
                                device=device)
    value_mask = torch.zeros((batch, steps), dtype=torch.bool, device=device)

    for row, window in enumerate(windows):
        transitions = window["transitions"]
        row_states = [transitions[0]["prev"]] + [
            item["next"] for item in transitions]
        row_actions = [item.get("action") for item in transitions]
        state_mask[row, :len(row_states)] = True
        transition_mask[row, :len(transitions)] = True
        for step, item in enumerate(transitions):
            causal_targets[row, step] = torch.tensor(
                causal_delta_vector(item["prev"], item["next"],
                                    dimension=causal_dim),
                dtype=torch.float32, device=device)
            if item.get("outcome") is not None:
                value_targets[row, step] = float(item["outcome"])
                value_mask[row, step] = True
        row_states.extend([row_states[-1]] *
                          (states_per_window - len(row_states)))
        row_actions.extend([None] * (steps - len(row_actions)))
        states.extend(row_states)
        actions.extend(row_actions)

    rows, object_mask = tensorizer.batch_states(states, device)
    objects, features = rows.shape[1:]
    rows = rows.reshape(batch, states_per_window, objects, features)
    object_mask = object_mask.reshape(batch, states_per_window, objects)
    action_vectors = tensorizer.batch_actions(actions, device).reshape(
        batch, steps, -1)
    return (rows, object_mask, action_vectors, state_mask, transition_mask,
            causal_targets, value_targets, value_mask)


def _masked_mean(values, mask):
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.to(values.dtype)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def _run_batch(model, tensorizer, windows, device, free_nats,
               kl_weight, kl_balance, causal_weight, value_weight,
               open_loop_horizon, sample):
    import torch
    import torch.nn.functional as F

    (rows, object_mask, actions, state_mask, transition_mask,
     causal_targets, value_targets, value_mask) = _batch_windows(
        windows, tensorizer, device, model.config.causal_dim)
    batch, state_count = state_mask.shape
    steps = state_count - 1
    flat_rows = rows.reshape(batch * state_count, *rows.shape[2:])
    flat_mask = object_mask.reshape(batch * state_count, object_mask.shape[-1])
    observations = model.encode_observation(flat_rows, flat_mask).reshape(
        batch, state_count, -1)

    deterministic, stochastic = model.initial(batch, device)
    priors = []
    posteriors = []
    reconstruction = []
    kl_terms = []
    prior_nll = []
    standardized_residual = []

    zero_action = torch.zeros(
        batch, actions.shape[-1], dtype=actions.dtype, device=device)
    for state_index in range(state_count):
        action = zero_action if state_index == 0 else actions[:, state_index - 1]
        prior, posterior = model.posterior_step(
            observations[:, state_index], action,
            deterministic, stochastic, sample=sample)
        deterministic = posterior["deterministic"]
        stochastic = posterior["stochastic"]
        priors.append(prior)
        posteriors.append(posterior)
        decoded = model.decode_observation(posterior["feature"])
        reconstruction.append((decoded - observations[:, state_index].detach())
                              .pow(2).mean(dim=-1))

        dynamic_kl = model.diagonal_kl(
            posterior["mean"].detach(), posterior["logScale"].detach(),
            prior["mean"], prior["logScale"]).sum(dim=-1)
        representation_kl = model.diagonal_kl(
            posterior["mean"], posterior["logScale"],
            prior["mean"].detach(), prior["logScale"].detach()).sum(dim=-1)
        balanced = (float(kl_balance) * dynamic_kl +
                    (1.0 - float(kl_balance)) * representation_kl)
        kl_terms.append(torch.clamp(balanced, min=float(free_nats)))

        scale = prior["logScale"].exp().clamp_min(1e-5)
        residual = (posterior["mean"].detach() - prior["mean"]) / scale
        standardized_residual.append(residual.pow(2).mean(dim=-1))
        prior_nll.append((0.5 * residual.pow(2) + prior["logScale"] +
                          0.5 * math.log(2.0 * math.pi)).mean(dim=-1))

    reconstruction_values = torch.stack(reconstruction, dim=1)
    kl_values = torch.stack(kl_terms, dim=1)
    nll_values = torch.stack(prior_nll, dim=1)
    residual_values = torch.stack(standardized_residual, dim=1)
    reconstruction_loss = _masked_mean(reconstruction_values, state_mask)
    kl_loss = _masked_mean(kl_values, state_mask)

    one_step = []
    causal = []
    value = []
    for step in range(steps):
        prior = priors[step + 1]
        prediction = model.decode_observation(prior["feature"])
        one_step.append((prediction - observations[:, step + 1].detach())
                        .pow(2).mean(dim=-1))
        causal.append(F.smooth_l1_loss(
            model.causal_delta(posteriors[step]["feature"], actions[:, step]),
            causal_targets[:, step], reduction="none").mean(dim=-1))
        value.append((model.state_value(posteriors[step]["feature"]) -
                      value_targets[:, step]).pow(2))
    one_step_values = torch.stack(one_step, dim=1)
    causal_values = torch.stack(causal, dim=1)
    value_values = torch.stack(value, dim=1)
    one_step_loss = _masked_mean(one_step_values, transition_mask)
    causal_loss = _masked_mean(causal_values, transition_mask)
    value_loss = _masked_mean(value_values, value_mask) \
        if bool(value_mask.any()) else one_step_loss.new_tensor(0.0)

    loss = (reconstruction_loss + one_step_loss +
            float(kl_weight) * kl_loss +
            float(causal_weight) * causal_loss +
            float(value_weight) * value_loss)

    open_loop = defaultdict(list)
    maximum = min(max(1, int(open_loop_horizon)), steps)
    for start in range(steps):
        deterministic = posteriors[start]["deterministic"]
        stochastic = posteriors[start]["stochastic"]
        for horizon in range(1, maximum + 1):
            target_state = start + horizon
            if target_state >= state_count:
                break
            valid = transition_mask[:, start:target_state].all(dim=1) & \
                state_mask[:, target_state]
            imagined = model.prior_step(
                actions[:, target_state - 1], deterministic, stochastic,
                sample=False)
            deterministic = imagined["deterministic"]
            stochastic = imagined["stochastic"]
            if bool(valid.any()):
                error = (model.decode_observation(imagined["feature"]) -
                         observations[:, target_state].detach()).pow(2)
                error = error.mean(dim=-1)
                open_loop[horizon].append(error[valid].detach().cpu())

    posterior_means = []
    for index, posterior in enumerate(posteriors):
        valid = state_mask[:, index]
        if bool(valid.any()):
            posterior_means.append(
                posterior["mean"][valid].detach().float().cpu())
    means = torch.cat(posterior_means, dim=0) if posterior_means \
        else torch.empty((0, model.latent_dim))
    return loss, {
        "stateExamples": int(state_mask.sum()),
        "transitionExamples": int(transition_mask.sum()),
        "loss": float(loss.detach()),
        "reconstruction": float(reconstruction_loss.detach()),
        "oneStepPrediction": float(one_step_loss.detach()),
        "kl": float(kl_loss.detach()),
        "causal": float(causal_loss.detach()),
        "value": float(value_loss.detach()),
        "priorNll": float(_masked_mean(nll_values, state_mask).detach()),
        "standardizedResidualRms": float(torch.sqrt(
            _masked_mean(residual_values, state_mask)).detach()),
        "openLoop": {horizon: torch.cat(values)
                     for horizon, values in open_loop.items() if values},
        "posteriorMeans": means,
    }


def collapse_metrics(values):
    import torch
    if values.numel() == 0:
        return {"examples": 0, "meanDimensionStd": None,
                "minimumDimensionStd": None, "effectiveRank": None}
    if values.shape[0] < 2:
        return {"examples": int(values.shape[0]),
                "meanDimensionStd": 0.0,
                "minimumDimensionStd": 0.0,
                "effectiveRank": 1.0}
    std = values.std(dim=0, unbiased=False)
    centered = values - values.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return {"examples": int(values.shape[0]),
            "meanDimensionStd": float(std.mean()),
            "minimumDimensionStd": float(std.min()),
            "effectiveRank": float(entropy.exp())}


def _run_epoch(model, tensorizer, windows, device, batch_size,
               free_nats, kl_weight, kl_balance, causal_weight,
               value_weight, open_loop_horizon, optimizer=None,
               scaler=None, amp_enabled=False, grad_accum_steps=1):
    import torch

    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    states = transitions = 0
    open_loop = defaultdict(list)
    posterior_means = []
    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, start in enumerate(range(0, len(windows), batch_size)):
        batch = windows[start:start + batch_size]
        context = torch.autocast(device_type="cuda", dtype=torch.float16) \
            if amp_enabled else nullcontext()
        with context:
            loss, metrics = _run_batch(
                model, tensorizer, batch, device, free_nats, kl_weight,
                kl_balance, causal_weight, value_weight,
                open_loop_horizon, sample=training)
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

        weight = max(1, metrics["transitionExamples"])
        for key in ("loss", "reconstruction", "oneStepPrediction", "kl",
                    "causal", "value", "priorNll",
                    "standardizedResidualRms"):
            totals[key] += metrics[key] * weight
        states += metrics["stateExamples"]
        transitions += metrics["transitionExamples"]
        for horizon, values in metrics["openLoop"].items():
            open_loop[int(horizon)].append(values)
        posterior_means.append(metrics["posteriorMeans"])

    denominator = max(1, transitions)
    result = {key: totals[key] / denominator for key in totals}
    result.update({"stateExamples": states,
                   "transitionExamples": transitions})
    result["openLoopMseByHorizon"] = {
        str(horizon): float(torch.cat(values).mean())
        for horizon, values in sorted(open_loop.items()) if values}
    means = torch.cat(posterior_means, dim=0) if posterior_means \
        else torch.empty((0, model.latent_dim))
    result["collapse"] = collapse_metrics(means)
    return result


def train(transitions, config=None, resolver=None, epochs=5, batch_size=8,
          sequence_length=16, eval_fraction=0.1, seed=0, device=None,
          lr=3e-4, weight_decay=1e-4, latent_dim=None, free_nats=1.0,
          kl_weight=0.1, kl_balance=0.8, causal_weight=0.25,
          value_weight=0.25, open_loop_horizon=8, amp="auto",
          grad_accum_steps=1, log=None):
    if not TORCH_AVAILABLE:
        raise ImportError("RSSM training requires magic-cabt[jepa]")
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = device.type == "cuda" and amp != "off"
    if amp == "on" and device.type != "cuda":
        raise ValueError("AMP was requested but the selected device is not CUDA")

    sequences = build_transition_sequences(transitions)
    train_sequences, eval_sequences = split_sequences(
        sequences, eval_fraction=eval_fraction, seed=seed)
    train_windows = window_sequences(train_sequences, sequence_length)
    eval_windows = window_sequences(eval_sequences, sequence_length)
    if not train_windows:
        raise ValueError("no horizon-one RSSM training sequences")

    config = config or StructuredJEPAConfig.preset("local")
    model = StructuredRSSM(config, latent_dim=latent_dim).to(device)
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
            model, tensorizer, train_windows, device, max(1, int(batch_size)),
            free_nats, kl_weight, kl_balance, causal_weight, value_weight,
            open_loop_horizon, optimizer=optimizer, scaler=scaler,
            amp_enabled=amp_enabled,
            grad_accum_steps=max(1, int(grad_accum_steps)))
        with torch.no_grad():
            eval_metrics = _run_epoch(
                model, tensorizer, eval_windows, device,
                max(1, int(batch_size)), free_nats, kl_weight, kl_balance,
                causal_weight, value_weight, open_loop_horizon) \
                if eval_windows else {
                    "stateExamples": 0, "transitionExamples": 0,
                    "loss": None, "openLoopMseByHorizon": {},
                    "collapse": collapse_metrics(
                        torch.empty((0, model.latent_dim)))}
        selection = eval_metrics["loss"] \
            if eval_metrics["transitionExamples"] else train_metrics["loss"]
        history.append({"epoch": epoch, "train": train_metrics,
                        "eval": eval_metrics,
                        "selectionMetric": selection})
        if selection < best_metric:
            best_metric, best_epoch = float(selection), epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if log:
            log("epoch %d/%d train=%.4f eval=%s one-step=%.4f kl=%.4f" % (
                epoch, epochs, train_metrics["loss"],
                "n/a" if not eval_metrics["transitionExamples"] else
                "%.4f" % eval_metrics["loss"],
                train_metrics["oneStepPrediction"], train_metrics["kl"]))

    elapsed = max(time.perf_counter() - started, 1e-9)
    metrics = {
        "kind": "magic-structured-rssm-training-v1",
        "modelFamily": "structured-rssm-v1",
        "transitionExamples": len(transitions),
        "trainGames": len(train_sequences),
        "evalGames": len(eval_sequences),
        "trainWindows": len(train_windows),
        "evalWindows": len(eval_windows),
        "sequenceLength": int(sequence_length),
        "latentDim": model.latent_dim,
        "objective": {
            "freeNats": float(free_nats),
            "klWeight": float(kl_weight),
            "klBalance": float(kl_balance),
            "causalWeight": float(causal_weight),
            "valueWeight": float(value_weight),
        },
        "openLoopHorizon": int(open_loop_horizon),
        "history": history,
        "bestEpoch": best_epoch,
        "bestSelectionMetric": best_metric,
        "split": {
            "unit": "game", "seed": int(seed),
            "evalFraction": float(eval_fraction),
            "trainGameIds": [item["gameKey"] for item in train_sequences],
            "evalGameIds": [item["gameKey"] for item in eval_sequences],
        },
        "visibilityPolicy": "public-history-and-perspective-state-v1",
        "device": str(device),
        "amp": {"requested": amp, "enabled": amp_enabled},
        "wallSeconds": elapsed,
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
    parser = argparse.ArgumentParser(prog="magic-cabt-train-rssm")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preset", choices=("tiny", "local", "large"),
                        default="local")
    parser.add_argument("--embedding-backend", default="hash")
    parser.add_argument("--arena-card-db", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--max-transitions", type=int, default=200000)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--free-nats", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--kl-balance", type=float, default=0.8)
    parser.add_argument("--causal-weight", type=float, default=0.25)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--open-loop-horizon", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", choices=("auto", "on", "off"),
                        default="auto")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    transitions, cards, collection = collect_transition_data(
        args.input, max_transitions=args.max_transitions)
    if not transitions:
        raise SystemExit("no complete horizon-one transition games")
    config = StructuredJEPAConfig.preset(args.preset)
    config.embedding_backend = args.embedding_backend
    resolver = CardTextResolver(cards, arena_db_path=args.arena_card_db)
    model, metrics = train(
        transitions, config=config, resolver=resolver,
        epochs=args.epochs, batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        eval_fraction=args.eval_fraction, seed=args.seed, device=args.device,
        lr=args.lr, weight_decay=args.weight_decay,
        latent_dim=args.latent_dim, free_nats=args.free_nats,
        kl_weight=args.kl_weight, kl_balance=args.kl_balance,
        causal_weight=args.causal_weight, value_weight=args.value_weight,
        open_loop_horizon=args.open_loop_horizon, amp=args.amp,
        grad_accum_steps=args.grad_accum_steps,
        log=lambda text: sys.stderr.write(text + "\n"))
    metrics["collection"] = collection
    metrics["inputs"] = core._input_manifest(args.input)

    os.makedirs(args.out, exist_ok=True)
    checkpoint = os.path.join(args.out, "checkpoint.pt")
    model.save_checkpoint(checkpoint, extra={
        "metrics": metrics, "trainingState": model._training_state})
    best = StructuredRSSM(model.config, latent_dim=model.latent_dim)
    best.load_state_dict(model._best_state_dict)
    best_path = os.path.join(args.out, "best.pt")
    best.save_checkpoint(best_path, extra={
        "metrics": metrics,
        "selection": {"epoch": metrics["bestEpoch"],
                      "metric": metrics["bestSelectionMetric"]}})
    metrics["checkpoint"] = os.path.abspath(checkpoint)
    metrics["bestCheckpoint"] = os.path.abspath(best_path)
    with open(os.path.join(args.out, "metrics.json"), "w",
              encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
