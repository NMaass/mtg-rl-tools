"""Train the local structured Magic JEPA from replay bundles.

The trainer mixes four signals:

* action-conditioned future-latent prediction at 1/4/16 state horizons;
* generic causal before/after deltas supplied by the recorded trajectory;
* terminal value labels when a bundle has one unambiguous game result;
* imitation loss over canonical legal-option groups.

No card-specific effect vocabulary is required. Card semantics enter through
frozen text embeddings and the structured fields already captured by CABT.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
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

__all__ = ["collect_training_data", "train", "build_parser", "main"]

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


def train(transitions, decisions, config=None, embedding_provider=None,
          card_resolver=None, resume=None, epochs=3, batch_size=32,
          lr=3e-4, weight_decay=1e-4, tau=0.996, seed=0,
          device=None, causal_weight=0.25, value_weight=0.25,
          policy_weight=1.0, log=None):
    """Train and return ``(model, metrics)``."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "JEPA training requires PyTorch: pip install -e 'python[jepa]'")
    import torch
    import torch.nn.functional as F

    if not transitions and not decisions:
        raise ValueError("no training examples")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)

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
        device=device if str(device).startswith("cuda") else None)
    tensorizer = StructuredTensorizer(
        config, provider, card_resolver or CardTextResolver())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=lr, weight_decay=weight_decay)

    rng = random.Random(seed)
    history = []
    for epoch in range(int(epochs)):
        model.train()
        transition_order = list(range(len(transitions)))
        decision_order = list(range(len(decisions)))
        rng.shuffle(transition_order)
        rng.shuffle(decision_order)
        transition_batches = list(_index_batches(
            transition_order, batch_size))
        decision_batches = list(_index_batches(decision_order, batch_size))
        transition_cycle = cycle(transition_batches) \
            if transition_batches else None
        decision_cycle = cycle(decision_batches) if decision_batches else None
        step_count = max(len(transition_batches), len(decision_batches), 1)
        totals = {key: 0.0 for key in
                  ("loss", "jepa", "causal", "value", "policy")}

        for _step in range(step_count):
            loss = torch.zeros((), device=device)
            if transition_cycle is not None:
                indices = next(transition_cycle)
                batch = [transitions[index] for index in indices]
                previous = [item["prev"] for item in batch]
                following = [item["next"] for item in batch]
                actions = [item.get("action") for item in batch]
                horizons = torch.tensor(
                    [int(item.get("horizon") or 1) for item in batch],
                    dtype=torch.long, device=device)
                prev_rows, prev_mask = tensorizer.batch_states(
                    previous, device)
                next_rows, next_mask = tensorizer.batch_states(
                    following, device)
                action_vectors = tensorizer.batch_actions(actions, device)
                state = model.encode(prev_rows, prev_mask)
                predicted, log_scale = model.predict_distribution(
                    state, action_vectors, horizons)
                with torch.no_grad():
                    target = model.encode_target(next_rows, next_mask)
                jepa_loss, _pieces = model.jepa_loss(
                    predicted, target, log_scale)
                causal_target = torch.tensor([
                    causal_delta_vector(
                        item["prev"], item["next"],
                        dimension=config.causal_dim)
                    for item in batch
                ], dtype=torch.float32, device=device)
                causal_loss = F.smooth_l1_loss(
                    model.causal_delta(state, action_vectors, horizons),
                    causal_target)
                loss = loss + jepa_loss + causal_weight * causal_loss
                totals["jepa"] += float(jepa_loss.detach())
                totals["causal"] += float(causal_loss.detach())

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
                    values = model.value(
                        state.index_select(0, rows)).squeeze(-1)
                    value_loss = F.mse_loss(values, targets)
                    loss = loss + value_weight * value_loss
                    totals["value"] += float(value_loss.detach())

            if decision_cycle is not None:
                indices = next(decision_cycle)
                batch = [decisions[index] for index in indices]
                policy_loss = _policy_loss(
                    model, tensorizer, batch, device)
                loss = loss + policy_weight * policy_loss
                totals["policy"] += float(policy_loss.detach())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target(tau=tau)
            totals["loss"] += float(loss.detach())

        epoch_metrics = {
            key: value / max(1, step_count)
            for key, value in totals.items()
        }
        epoch_metrics["epoch"] = epoch + 1
        history.append(epoch_metrics)
        if log:
            log("epoch %d/%d loss=%.4f jepa=%.4f causal=%.4f "
                "value=%.4f policy=%.4f" % (
                    epoch + 1, epochs, epoch_metrics["loss"],
                    epoch_metrics["jepa"], epoch_metrics["causal"],
                    epoch_metrics["value"], epoch_metrics["policy"]))

    model.eval()
    metrics = {
        "kind": "magic-structured-jepa-v1",
        "device": str(device),
        "parameters": model_parameter_count(model),
        "trainableParameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad),
        "transitionExamples": len(transitions),
        "decisionExamples": len(decisions),
        "epochs": int(epochs),
        "history": history,
        "resumed": bool(resume),
        "previousExtra": previous_extra,
    }
    return model, metrics


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
        with open(summary, "r", encoding="utf-8") as handle:
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
    with open(path, "r", encoding="utf-8") as handle:
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


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-train-jepa",
        description="Train the local structured JEPA from replay bundles.")
    parser.add_argument(
        "--input", required=True, action="append",
        help="bundle directory or JSONL file; repeatable")
    parser.add_argument(
        "--out", required=True,
        help="output directory for checkpoint.pt and metrics.json")
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-transitions", type=int, default=200000)
    parser.add_argument("--max-decisions", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
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
        lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
        device=args.device,
        log=lambda text: sys.stderr.write(text + "\n"))
    os.makedirs(args.out, exist_ok=True)
    checkpoint = os.path.join(args.out, "checkpoint.pt")
    model.save_checkpoint(checkpoint, extra={"metrics": metrics})
    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    sys.stderr.write("wrote %s\n" % checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
