"""Action-space compressibility diagnostics.

These helpers measure whether action choices are predictable under coarse or
fine abstractions. They are analysis tools, not legality filters: the engine
still supplies the full legal option list.
"""

import math

from .action_abstraction import abstract_record

__all__ = [
    "entropy",
    "analyze_action_entropy",
]


def entropy(counts):
    """Return Shannon entropy in bits for a mapping of label -> count."""
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        value -= p * math.log(p, 2)
    return value


def analyze_action_entropy(records, profile="small"):
    """Return global and prompt-conditional action entropy statistics."""
    global_counts = {}
    prompt_counts = {}
    legal_count_hist = {}
    examples = 0
    selected = 0

    for record in records:
        examples += 1
        summary = abstract_record(record, profile=profile)
        prompt = summary.get("promptType") or "UNKNOWN"
        legal_count = len(summary.get("optionBuckets") or [])
        legal_count_hist[str(legal_count)] = legal_count_hist.get(str(legal_count), 0) + 1
        for bucket in summary.get("selectedBuckets") or []:
            selected += 1
            global_counts[bucket] = global_counts.get(bucket, 0) + 1
            per_prompt = prompt_counts.setdefault(prompt, {})
            per_prompt[bucket] = per_prompt.get(bucket, 0) + 1

    conditional = {}
    for prompt, counts in sorted(prompt_counts.items()):
        total = sum(counts.values())
        most_common = _most_common(counts)
        conditional[prompt] = {
            "total": total,
            "entropyBits": entropy(counts),
            "mostCommon": most_common,
            "mostCommonRate": (counts[most_common] / float(total)) if total else None,
            "counts": counts,
        }

    return {
        "profile": profile,
        "records": examples,
        "selectedActions": selected,
        "global": {
            "entropyBits": entropy(global_counts),
            "mostCommon": _most_common(global_counts),
            "counts": global_counts,
        },
        "byPromptType": conditional,
        "legalActionCount": legal_count_hist,
    }


def _most_common(counts):
    if not counts:
        return None
    return max(sorted(counts), key=lambda key: counts[key])
