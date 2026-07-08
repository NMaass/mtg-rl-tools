"""Score recorded decisions with a policy and emit replay annotations.

    python -m magic_cabt.replay.annotate \
        --input <decision-records-or-bundle> \
        --policy first \
        --out annotations.jsonl \
        --top-k 5

Reads any decision-record stream ``magic_cabt.training.records`` understands
(Java transition dataset, self-play / human replay frames, Arena bundles),
scores every option of each decision with the chosen policy, and writes one
annotation per decision::

    {"gameId": ..., "sequenceNumber": 123, "policy": "first",
     "topK": [{"index": 2, "score": 1.0, "label": "Play Mountain"}],
     "chosenRank": 1, "chosenScore": 1.0}

Model-agnostic: the policy is anything exposing ``name`` and
``score(select) -> [float]`` (higher = more preferred). The built-in agents
supply that today; Agent 2's model scorer can drop in unchanged.
"""

import argparse
import glob
import json
import os
import sys

from magic_cabt.agents import make_agent, options_of, select_block

# Primary dependency: Agent 1's canonical DecisionRecord reader. It may not be
# merged yet (issue #12 develops in parallel with #10), so we import it
# defensively and fall back to a local minimal reader -- the "temporary local
# helper that is easy to replace" the issue calls for. When the canonical API
# lands this import simply starts winning and the fallback goes unused.
try:
    from magic_cabt.training.records import iter_decision_records \
        as _canonical_iter_records
except Exception:  # pragma: no cover - only when Agent 1's API is absent
    _canonical_iter_records = None

__all__ = [
    "annotate_record",
    "annotate_stream",
    "build_parser",
    "main",
]


def annotate_record(record, scorer, top_k=5):
    """Return the annotation dict for one decision record.

    ``scorer`` exposes ``name`` and ``score(observation) -> [float]``. The
    chosen option is taken from ``selectedIndices`` (its first entry for a
    multi-select prompt); ``chosenRank`` is 1-based over the score ordering
    (ties broken by option index).
    """
    select = select_block(record)
    options = options_of(select)
    scores = scorer.score(select)
    # Defensive: keep score/option lists aligned even if a scorer misbehaves.
    if len(scores) != len(options):
        scores = (list(scores) + [0.0] * len(options))[:len(options)]

    ranking = sorted(range(len(options)),
                     key=lambda index: (-scores[index], index))
    rank_of = dict((index, position + 1)
                   for position, index in enumerate(ranking))

    top = []
    for index in ranking[:top_k if top_k and top_k > 0 else len(ranking)]:
        top.append({
            "index": index,
            "score": scores[index],
            "label": options[index].get("label"),
        })

    chosen = record.get("selectedIndices") or []
    chosen_index = chosen[0] if chosen else None
    if isinstance(chosen_index, int) and 0 <= chosen_index < len(options):
        chosen_rank = rank_of[chosen_index]
        chosen_score = scores[chosen_index]
    else:
        chosen_rank = None
        chosen_score = None

    return {
        "gameId": record.get("gameId"),
        "sequenceNumber": record.get("sequenceNumber"),
        "policy": scorer.name,
        "topK": top,
        "chosenRank": chosen_rank,
        "chosenScore": chosen_score,
    }


def annotate_stream(records, scorer, top_k=5):
    """Yield an annotation for every record with at least one option."""
    for record in records:
        if not options_of(select_block(record)):
            continue
        yield annotate_record(record, scorer, top_k=top_k)


def _read_records(path, source_hint=None):
    if _canonical_iter_records is not None:
        for record in _canonical_iter_records(path, source_hint=source_hint):
            yield record
    else:
        for record in _local_iter_records(path):
            yield record


def _iter_input_records(input_path, source_hint=None):
    """Iterate decision records from a file or a directory of ``*.jsonl``."""
    if os.path.isdir(input_path):
        for path in sorted(glob.glob(os.path.join(input_path, "*.jsonl"))):
            for record in _read_records(path, source_hint=source_hint):
                yield record
    else:
        for record in _read_records(input_path, source_hint=source_hint):
            yield record


def _local_iter_records(path_or_file):
    """Dependency-free stand-in for Agent 1's ``iter_decision_records``.

    Handles the self-play / human replay frames this package writes and
    canonical / Java transition records, enough for annotation to run before
    the canonical reader is available. The trailing ``{result}`` self-play
    line carries no decision and is skipped. Replace with the canonical API
    once it lands (see the guarded import at the top of this module).
    """
    own_handle = not hasattr(path_or_file, "read")
    handle = open(path_or_file, "r", encoding="utf-8") if own_handle \
        else path_or_file
    try:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                continue
            record = _local_normalize(raw)
            if record is not None:
                yield record
    finally:
        if own_handle:
            handle.close()


def _local_normalize(raw):
    """Map one raw line to a minimal record (``select`` + ``selectedIndices``)."""
    # self-play / human replay frame: {sequence, player, observation, selected}
    if "observation" in raw and "selected" in raw:
        observation = raw.get("observation") or {}
        return {
            "gameId": raw.get("gameId"),
            "sequenceNumber": raw.get("sequence", raw.get("sequenceNumber")),
            "select": observation.get("select") or {},
            "selectedIndices": raw.get("selected") or [],
        }
    top_select = raw.get("select")
    # canonical / Java transition record: top-level select spec (a dict)
    if isinstance(top_select, dict):
        return {
            "gameId": raw.get("gameId"),
            "sequenceNumber": raw.get("sequenceNumber"),
            "select": top_select,
            "selectedIndices": raw.get("selectedIndices") or [],
        }
    # arena decision: top-level select is the chosen index list, prompt nested
    if isinstance(top_select, list):
        observation = raw.get("observation") or {}
        return {
            "gameId": raw.get("gameId"),
            "sequenceNumber": raw.get("sequence", raw.get("sequenceNumber")),
            "select": observation.get("select") or {},
            "selectedIndices": top_select,
        }
    # trailing self-play {result} line (or anything else): no decision to score
    return None


# --- CLI --------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic_cabt.replay.annotate",
        description="Annotate recorded decisions with a policy's option scores.",
    )
    parser.add_argument("--input", required=True,
                        help="decision-record JSONL file or directory of them")
    parser.add_argument("--policy", default="first",
                        help="scoring policy (random|first)")
    parser.add_argument("--out", default=None,
                        help="output annotations.jsonl (default: stdout)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for stochastic policies (e.g. random)")
    parser.add_argument("--source-hint", default=None,
                        help="override the record source label")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    scorer = make_agent(args.policy, seed=args.seed)
    records = _iter_input_records(args.input, source_hint=args.source_hint)
    annotations = annotate_stream(records, scorer, top_k=args.top_k)

    handle = open(args.out, "w") if args.out else sys.stdout
    count = 0
    try:
        for annotation in annotations:
            handle.write(json.dumps(annotation) + "\n")
            count += 1
    finally:
        if args.out:
            handle.close()
    if args.out:
        sys.stderr.write("wrote %d annotations to %s\n" % (count, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
