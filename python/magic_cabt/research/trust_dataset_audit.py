"""Dependency-light dataset gates for the training trust audit."""
from __future__ import annotations

from collections import Counter, defaultdict

from magic_cabt.analysis.schema import decision_fingerprint
from magic_cabt.training.io import iter_decision_records
from magic_cabt.training.records import validate_record

from .trust_audit_common import (
    files_of_kind,
    forbidden_paths,
    game_key,
    iter_jsonl,
    reopened_segments,
    select_of,
    selected_of,
    sequence_of,
)


def audit_decisions(report, inputs, max_records=0):
    paths = files_of_kind(inputs, "decisions")
    total = 0
    unknown_game = 0
    invalid_selected = []
    invalid_records = []
    empty_options = []
    duplicate_option_indices = []
    hidden_paths = []
    generic_history = 0
    public_history = 0
    fingerprints = Counter()
    canonical_type_conflicts = []
    sequence_issues = []
    last_sequence = {}
    ordered_games = []
    truncated = False

    for path in paths:
        for record in iter_decision_records(path):
            if max_records and total >= max_records:
                truncated = True
                break
            index = total
            total += 1
            errors = validate_record(record)
            if errors:
                invalid_records.append({
                    "record": index,
                    "line": record.get("__source_line"),
                    "messages": errors,
                })

            game = game_key(record)
            ordered_games.append(game)
            if game is None:
                unknown_game += 1
            else:
                sequence = sequence_of(record, index)
                previous = last_sequence.get(game)
                if previous is not None and sequence < previous:
                    sequence_issues.append({
                        "game": game,
                        "previous": previous,
                        "current": sequence,
                        "record": index,
                    })
                last_sequence[game] = sequence

            select = select_of(record)
            options = select.get("option") or []
            if not isinstance(options, list) or not options:
                empty_options.append(index)
                continue
            option_indices = [
                option.get("index", position)
                if isinstance(option, dict) else position
                for position, option in enumerate(options)
            ]
            if len(set(map(str, option_indices))) != len(option_indices):
                duplicate_option_indices.append(index)
            selected = selected_of(record)
            if not selected or any(
                    not isinstance(value, int) or isinstance(value, bool) or
                    not 0 <= value < len(options) for value in selected):
                invalid_selected.append({
                    "record": index,
                    "selected": selected,
                    "optionCount": len(options),
                })

            canonical_types = defaultdict(set)
            for option in options:
                payload = option.get("payload") if isinstance(option, dict) else None
                key = payload.get("canonicalKey") if isinstance(payload, dict) else None
                if key is not None:
                    canonical_types[str(key)].add(str(option.get("type") or ""))
            for key, option_types in canonical_types.items():
                if len(option_types) > 1:
                    canonical_type_conflicts.append({
                        "record": index,
                        "canonicalKey": key,
                        "types": sorted(option_types),
                    })

            observation = record.get("observation") or {}
            hidden_paths.extend(forbidden_paths(observation))
            if isinstance(observation, dict) and observation.get("history") is not None:
                generic_history += 1
            if isinstance(observation, dict) and \
                    observation.get("publicHistory") is not None:
                public_history += 1
            fingerprints[decision_fingerprint(record)] += 1
        if truncated:
            break

    reopened = reopened_segments(ordered_games)
    duplicate_rows = sum(count - 1 for count in fingerprints.values()
                         if count > 1)
    present = "pass" if total else ("fail" if paths else "not-applicable")
    report.add(
        "dataset.decisions.present", present,
        "Decision records are available" if total else
        ("Decision files contained no records" if paths else
         "No decision files were supplied"),
        {"files": len(paths), "records": total, "truncated": truncated},
        scope="dataset")
    report.add(
        "dataset.decisions.schema", "fail" if invalid_records else "pass",
        "Decision records satisfy the canonical validator"
        if not invalid_records else "Some decision records are invalid",
        {"count": len(invalid_records), "examples": invalid_records[:20]},
        scope="dataset")
    report.add(
        "dataset.decisions.selected-index",
        "fail" if invalid_selected else "pass",
        "Selected option indices are legal" if not invalid_selected else
        "Some decisions select missing or out-of-range options",
        {"count": len(invalid_selected), "examples": invalid_selected[:20]},
        scope="dataset")
    report.add(
        "dataset.decisions.options",
        "fail" if empty_options or duplicate_option_indices else "pass",
        "Decision rows have non-empty, uniquely indexed options"
        if not empty_options and not duplicate_option_indices else
        "Some decisions have empty or ambiguously indexed option sets",
        {"emptyOptionRows": empty_options[:20],
         "duplicateIndexRows": duplicate_option_indices[:20]},
        scope="dataset")
    report.add(
        "dataset.hidden-information.observation",
        "fail" if hidden_paths else "pass",
        "No oracle/private label keys occur in observations"
        if not hidden_paths else "Oracle/private fields occur in observations",
        {"count": len(hidden_paths),
         "paths": sorted(set(hidden_paths))[:50]}, scope="dataset")
    report.add(
        "dataset.sequence.game-identity", "warn" if unknown_game else "pass",
        "All decision rows have recoverable game identities"
        if not unknown_game else
        "Some decisions cannot participate in leakage-safe game splits",
        {"unknownRows": unknown_game, "records": total}, scope="dataset")
    report.add(
        "dataset.sequence.order",
        "fail" if sequence_issues or reopened else "pass",
        "Decision streams are monotone and game-contiguous"
        if not sequence_issues and not reopened else
        "Decision streams contain out-of-order or reopened games",
        {"nonMonotone": sequence_issues[:20],
         "reopenedGames": reopened[:20]}, scope="dataset")
    report.add(
        "dataset.history.visibility", "warn" if generic_history else "pass",
        "Temporal context uses explicitly public history"
        if not generic_history else
        "Generic history fields are present and must remain excluded",
        {"genericHistoryRows": generic_history,
         "publicHistoryRows": public_history}, scope="dataset")
    report.add(
        "dataset.options.canonical-types",
        "warn" if canonical_type_conflicts else "pass",
        "Canonical action keys are type-consistent within prompts"
        if not canonical_type_conflicts else
        "Some canonical keys merge different option types",
        {"count": len(canonical_type_conflicts),
         "examples": canonical_type_conflicts[:20]}, scope="dataset")
    report.add(
        "dataset.decisions.duplicates", "warn" if duplicate_rows else "pass",
        "No duplicate public decision fingerprints were detected"
        if not duplicate_rows else
        "Duplicate public decision fingerprints may overweight repeated rows",
        {"duplicateRows": duplicate_rows,
         "uniqueFingerprints": len(fingerprints)}, scope="dataset")
    return {"records": total,
            "games": set(key for key in last_sequence if key is not None),
            "truncated": truncated}


def audit_transitions(report, inputs, max_records=0):
    paths = files_of_kind(inputs, "transitions")
    total = 0
    invalid = []
    unknown_game = 0
    sequence_issues = []
    last_sequence = {}
    ordered_games = []
    horizons = Counter()
    truncated = False

    for path in paths:
        for item in iter_jsonl(path):
            if max_records and total >= max_records:
                truncated = True
                break
            index = total
            total += 1
            horizon = item.get("horizon", 1)
            if not isinstance(horizon, int) or isinstance(horizon, bool):
                horizon = -1
            horizons[horizon] += 1
            if horizon < 1 or not isinstance(item.get("prev"), dict) or \
                    not isinstance(item.get("next"), dict):
                invalid.append(index)
            pseudo = {
                "gameId": item.get("matchId"),
                "gameNumber": item.get("gameNumber"),
                "sequenceNumber": item.get("sequenceNumber", index),
                "observation": {"current": {
                    "gameInstance": item.get("gameInstance"),
                }},
            }
            game = game_key(pseudo)
            ordered_games.append(game)
            if game is None:
                unknown_game += 1
            else:
                sequence = sequence_of(pseudo, index)
                previous = last_sequence.get(game)
                if previous is not None and sequence < previous:
                    sequence_issues.append({
                        "game": game,
                        "previous": previous,
                        "current": sequence,
                        "record": index,
                    })
                last_sequence[game] = sequence
        if truncated:
            break

    reopened = reopened_segments(ordered_games)
    report.add(
        "dataset.transitions.structure",
        "fail" if invalid else ("pass" if total else "not-applicable"),
        "Transitions contain valid before/after states and horizons"
        if total and not invalid else
        ("Some transitions are malformed" if invalid else
         "No transition rows were supplied"),
        {"records": total, "invalidRows": invalid[:20],
         "horizons": dict(sorted(horizons.items())),
         "truncated": truncated}, scope="dataset")
    if total:
        report.add(
            "dataset.transitions.game-identity",
            "warn" if unknown_game else "pass",
            "All transitions have game identities" if not unknown_game else
            "Some transitions cannot participate in game-level splits",
            {"unknownRows": unknown_game}, scope="dataset")
        report.add(
            "dataset.transitions.order",
            "fail" if sequence_issues or reopened else "pass",
            "Transition streams are monotone and game-contiguous"
            if not sequence_issues and not reopened else
            "Transition streams contain out-of-order or reopened games",
            {"nonMonotone": sequence_issues[:20],
             "reopenedGames": reopened[:20]}, scope="dataset")
    return {"records": total,
            "games": set(key for key in last_sequence if key is not None),
            "truncated": truncated}


def audit_macro_actions(report, inputs, max_records=0):
    paths = files_of_kind(inputs, "macro-actions")
    total = 0
    incomplete = []
    orphaned = []
    heuristic = []
    truncated = False
    for path in paths:
        for item in iter_jsonl(path):
            if max_records and total >= max_records:
                truncated = True
                break
            index = total
            total += 1
            if item.get("complete") is not True:
                incomplete.append(index)
            if item.get("orphanedParameterGroup") is True:
                orphaned.append(index)
            if item.get("classificationConfidence") == "heuristic":
                heuristic.append(index)
        if truncated:
            break
    status = "not-applicable" if not total else \
        ("warn" if incomplete or orphaned or heuristic else "pass")
    report.add(
        "dataset.macro-actions.completeness", status,
        "Macro actions are complete and use known prompt families"
        if total and status == "pass" else
        ("Macro-action capture has incomplete, orphaned, or heuristic groups"
         if total else "No macro-action rows were supplied"),
        {"records": total, "incomplete": incomplete[:20],
         "orphaned": orphaned[:20], "heuristic": heuristic[:20],
         "truncated": truncated}, scope="dataset")
