"""Train recurrent information-state behavior cloning on whole games."""
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

from magic_cabt.models.information_state import (
    TORCH_AVAILABLE, RecurrentInformationStateModel)
from magic_cabt.models.structured_jepa import (
    CardTextResolver, StructuredJEPAConfig, StructuredTensorizer)
from . import train_jepa as core
from .train_structured_bc import collect_decision_data


def game_key(record):
    current = (record.get("observation") or {}).get("current") or \
        record.get("current") or {}
    match = record.get("matchId") or record.get("gameId") or current.get("matchId")
    game = record.get("gameNumber")
    if game is None:
        game = current.get("gameNumber")
    instance = record.get("gameInstance") or current.get("gameInstance")
    if match is None and game is None and instance is None:
        return None
    return json.dumps([match, game, instance], separators=(",", ":"))


def sequence_number(record, fallback=0):
    current = (record.get("observation") or {}).get("current") or \
        record.get("current") or {}
    for value in (record.get("sequenceNumber"), record.get("sequence"),
                  current.get("seq"), current.get("sequenceNumber")):
        if isinstance(value, (int, float)):
            return float(value)
    return float(fallback)


def build_game_sequences(decisions):
    grouped = defaultdict(list)
    unknown = []
    for index, record in enumerate(decisions):
        key = game_key(record)
        if key is None:
            unknown.append([record])
        else:
            grouped[key].append((sequence_number(record, index), index, record))
    sequences = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda item: (item[0], item[1]))
        sequences.append({"gameKey": key,
                          "records": [item[2] for item in rows]})
    for index, records in enumerate(unknown):
        sequences.append({"gameKey": "unknown:%d" % index, "records": records})
    return sequences


def split_sequences(sequences, eval_fraction=0.1, seed=0):
    if not 0.0 <= float(eval_fraction) < 1.0:
        raise ValueError("eval_fraction must be in [0, 1)")
    rows = list(sequences)
    rng = random.Random(seed)
    rng.shuffle(rows)
    eval_count = 0
    if eval_fraction > 0.0 and len(rows) > 1:
        eval_count = max(1, round(len(rows) * float(eval_fraction)))
        eval_count = min(eval_count, len(rows) - 1)
    return rows[eval_count:], rows[:eval_count]


def window_sequences(sequences, sequence_length):
    length = max(1, int(sequence_length))
    result = []
    for sequence in sequences:
        records = sequence["records"]
        for start in range(0, len(records), length):
            chunk = records[start:start + length]
            if chunk:
                previous = selected_action(records[start - 1]) if start else None
                result.append({"gameKey": sequence["gameKey"],
                               "start": start, "previousAction": previous,
                               "records": chunk})
    return result


def selected_action(record):
    select = record.get("select") or \
        (record.get("observation") or {}).get("select") or {}
    options = select.get("option") or []
    selected = record.get("selectedIndices") or record.get("selected") or []
    if len(selected) != 1 or not isinstance(selected[0], int):
        return None
    index = selected[0]
    if not 0 <= index < len(options):
        return None
    return {"promptType": select.get("type"), "selectedOption": options[index]}


def _batch_windows(windows, tensorizer, device):
    import torch
    batch = len(windows)
    time_steps = max(len(window["records"]) for window in windows)
    flat = []
    for window in windows:
        rows = list(window["records"])
        rows.extend([rows[-1]] * (time_steps - len(rows)))
        flat.extend(rows)
    state_rows, state_mask = tensorizer.batch_states(flat, device)
    option_vectors, option_mask = tensorizer.batch_options(flat, device)
    objects, features = state_rows.shape[1:]
    option_count, option_features = option_vectors.shape[1:]
    state_rows = state_rows.reshape(batch, time_steps, objects, features)
    state_mask = state_mask.reshape(batch, time_steps, objects)
    option_vectors = option_vectors.reshape(
        batch, time_steps, option_count, option_features)
    option_mask = option_mask.reshape(batch, time_steps, option_count)
    sequence_mask = torch.zeros((batch, time_steps), dtype=torch.bool,
                                device=device)
    previous = []
    for row, window in enumerate(windows):
        records = window["records"]
        sequence_mask[row, :len(records)] = True
        actions = [window.get("previousAction")]
        actions.extend(selected_action(record) for record in records[:-1])
        actions.extend([None] * (time_steps - len(actions)))
        previous.extend(actions)
    previous_actions = tensorizer.batch_actions(previous, device).reshape(
        batch, time_steps, -1)
    return (state_rows, state_mask, previous_actions, option_vectors,
            option_mask, sequence_mask)


def _loss_and_metrics(model, tensorizer, windows, device, batch_size,
                      optimizer=None, amp_enabled=False, scaler=None,
                      grad_accum_steps=1):
    import torch
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "examples": 0, "top1": 0,
              "top3": 0, "reciprocalRank": 0.0}
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, start in enumerate(range(0, len(windows), batch_size)):
        batch = windows[start:start + batch_size]
        context = torch.autocast(device_type="cuda", dtype=torch.float16) \
            if amp_enabled else nullcontext()
        with context:
            (rows, masks, previous, options, option_masks,
             sequence_mask) = _batch_windows(batch, tensorizer, device)
            memories, _hidden = model.information_states(
                rows, masks, previous, sequence_mask=sequence_mask)
            logits = model.score_from_memory(memories, options, option_masks)
            losses = []
            rankings = []
            for row, window in enumerate(batch):
                for step, record in enumerate(window["records"]):
                    denominator = torch.logsumexp(logits[row, step], dim=0)
                    members = record["_groups"][record["_chosenGroup"]]
                    numerator = torch.logsumexp(logits[row, step, members], dim=0)
                    losses.append(denominator - numerator)
                    probabilities = torch.softmax(logits[row, step], dim=0)
                    masses = [float(probabilities[group].sum().detach())
                              for group in record["_groups"]]
                    ranking = sorted(range(len(masses)),
                                     key=lambda index: (-masses[index], index))
                    rankings.append(ranking.index(record["_chosenGroup"]) + 1)
            loss = torch.stack(losses).mean()
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
        totals["loss"] += float(loss.detach()) * len(rankings)
        totals["examples"] += len(rankings)
        totals["top1"] += sum(rank == 1 for rank in rankings)
        totals["top3"] += sum(rank <= 3 for rank in rankings)
        totals["reciprocalRank"] += sum(1.0 / rank for rank in rankings)
    examples = totals["examples"]
    return {
        "examples": examples,
        "loss": totals["loss"] / max(1, examples),
        "policyTop1": totals["top1"] / examples if examples else None,
        "policyTop3": totals["top3"] / examples if examples else None,
        "policyMRR": totals["reciprocalRank"] / examples if examples else None,
    }


def train(decisions, config=None, resolver=None, epochs=5, batch_size=8,
          sequence_length=32, eval_fraction=0.1, seed=0, device=None,
          lr=3e-4, weight_decay=1e-4, amp="auto", grad_accum_steps=1,
          memory_layers=1, log=None):
    if not TORCH_AVAILABLE:
        raise ImportError("information-state training requires magic-cabt[jepa]")
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = device.type == "cuda" and amp != "off"
    if amp == "on" and device.type != "cuda":
        raise ValueError("AMP was requested but the selected device is not CUDA")
    sequences = build_game_sequences(decisions)
    train_sequences, eval_sequences = split_sequences(
        sequences, eval_fraction=eval_fraction, seed=seed)
    train_windows = window_sequences(train_sequences, sequence_length)
    eval_windows = window_sequences(eval_sequences, sequence_length)
    if not train_windows:
        raise ValueError("no training sequences")
    config = config or StructuredJEPAConfig.preset("local")
    model = RecurrentInformationStateModel(config, memory_layers=memory_layers)
    model.to(device)
    tensorizer = StructuredTensorizer(config, card_resolver=resolver)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history = []
    best_metric = math.inf
    best_state = None
    best_epoch = None
    started = time.perf_counter()
    rng = random.Random(seed)
    for epoch in range(1, max(1, int(epochs)) + 1):
        rng.shuffle(train_windows)
        train_metrics = _loss_and_metrics(
            model, tensorizer, train_windows, device, max(1, batch_size),
            optimizer=optimizer, amp_enabled=amp_enabled, scaler=scaler,
            grad_accum_steps=max(1, grad_accum_steps))
        with torch.no_grad():
            eval_metrics = _loss_and_metrics(
                model, tensorizer, eval_windows, device, max(1, batch_size)) \
                if eval_windows else {"examples": 0, "loss": None,
                                      "policyTop1": None, "policyTop3": None,
                                      "policyMRR": None}
        selection = eval_metrics["loss"] if eval_metrics["examples"] else train_metrics["loss"]
        history.append({"epoch": epoch, "train": train_metrics,
                        "eval": eval_metrics, "selectionMetric": selection})
        if selection < best_metric:
            best_metric = float(selection)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if log:
            log("epoch %d/%d train=%.4f eval=%s" % (
                epoch, epochs, train_metrics["loss"],
                "n/a" if not eval_metrics["examples"] else
                "%.4f" % eval_metrics["loss"]))
    elapsed = max(time.perf_counter() - started, 1e-9)
    metrics = {
        "kind": "magic-recurrent-information-state-training-v1",
        "modelFamily": "recurrent-information-state-bc-v1",
        "device": str(device),
        "amp": {"requested": amp, "enabled": amp_enabled},
        "decisionExamples": len(decisions),
        "trainGames": len(train_sequences),
        "evalGames": len(eval_sequences),
        "trainWindows": len(train_windows),
        "evalWindows": len(eval_windows),
        "sequenceLength": int(sequence_length),
        "memoryLayers": int(memory_layers),
        "history": history,
        "bestEpoch": best_epoch,
        "bestSelectionMetric": best_metric,
        "wallSeconds": elapsed,
        "examplesPerSecond": sum(item["train"]["examples"] for item in history) / elapsed,
        "split": {"unit": "game", "seed": seed,
                  "evalFraction": float(eval_fraction),
                  "evalGameIds": [item["gameKey"] for item in eval_sequences]},
    }
    model._best_state_dict = best_state
    model._training_state = {"optimizer": optimizer.state_dict(),
                             "completedEpochs": len(history),
                             "bestStateDict": best_state,
                             "bestEpoch": best_epoch,
                             "bestSelectionMetric": best_metric}
    return model, metrics


def build_parser():
    parser = argparse.ArgumentParser(prog="magic-cabt-train-information-state")
    parser.add_argument("--input", action="append", required=True)
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
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser


def _write_outputs(model, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = os.path.join(out_dir, "checkpoint.pt")
    model.save_checkpoint(checkpoint, extra={
        "metrics": metrics, "trainingState": model._training_state})
    best_path = os.path.join(out_dir, "best.pt")
    best_model = RecurrentInformationStateModel(
        model.config, memory_layers=model.memory_layers)
    best_model.load_state_dict(model._best_state_dict)
    best_model.save_checkpoint(best_path, extra={
        "metrics": metrics,
        "selection": {"epoch": metrics["bestEpoch"],
                      "metric": metrics["bestSelectionMetric"]}})
    metrics["checkpoint"] = os.path.abspath(checkpoint)
    metrics["bestCheckpoint"] = os.path.abspath(best_path)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv=None):
    args = build_parser().parse_args(argv)
    decisions, cards = collect_decision_data(
        args.input, max_decisions=args.max_decisions, seed=args.seed)
    config = StructuredJEPAConfig.preset(args.preset)
    config.embedding_backend = args.embedding_backend
    resolver = CardTextResolver(cards, arena_db_path=args.arena_card_db)
    model, metrics = train(
        decisions, config=config, resolver=resolver, epochs=args.epochs,
        batch_size=args.batch_size, sequence_length=args.sequence_length,
        eval_fraction=args.eval_fraction, seed=args.seed, device=args.device,
        lr=args.lr, weight_decay=args.weight_decay, amp=args.amp,
        grad_accum_steps=args.grad_accum_steps,
        memory_layers=args.memory_layers,
        log=lambda text: sys.stderr.write(text + "\n"))
    metrics["inputs"] = core._input_manifest(args.input)
    _write_outputs(model, metrics, args.out)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
