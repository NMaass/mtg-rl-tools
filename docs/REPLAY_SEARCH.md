# Exact counterfactual search by deterministic replay

Native cloned-game search remains the desired high-performance backend. The
replay backend provides an exact, slower path that can generate counterfactual
training labels before native cloning exists.

For each candidate at a recorded decision root it:

1. starts a fresh XMage bridge;
2. starts the original decks with the original seed;
3. replays every recorded selection before the root;
4. verifies the reconstructed public state and semantic legal-option list;
5. applies one alternative legal selection;
6. records the exact immediate successor;
7. optionally continues with rollout agents toward a terminal result.

The cost is approximately:

```text
candidate count x replay-prefix length
```

This is intended for offline label generation, small tactical suites, and
regression tests—not low-latency live play.

## Command

```bash
magic-cabt-replay-search \
  --input target/cabt-selfplay/game-0000.jsonl \
  --decision-index 42 \
  --deck0 examples/deck-a.txt \
  --deck1 examples/deck-b.txt \
  --seed 7 \
  --out target/search/game-0000-42.json \
  --transitions-out target/search/transitions.jsonl
```

The game seed and decks must match the recording. Use `--game-id` when the
input contains multiple games.

By default, branches stop after the candidate's immediate successor. To attach
rollout outcomes:

```bash
--rollout-agent0 first \
--rollout-agent1 first \
--max-rollout-decisions 64
```

Rollout policy quality is separate from transition correctness. Immediate
successor states remain exact engine outputs even when rollout values are weak.

## Supported automatic candidates

The CLI automatically enumerates prompts whose legal envelope selects at most
one option:

```text
exactly one: [0], [1], ...
optional one: [], [0], [1], ...
```

It refuses to guess attacker, blocker, pile, or other multi-select
combinations. Library callers may pass an explicit candidate list for those
prompts. This prevents an exponential or strategically incomplete candidate
set from being mislabeled as exhaustive search.

## Replay verification

Before branching, the reconstructed root is compared with the recorded root.
The signature includes:

- the public current-state JSON after removing volatile sequence/timestamp and
  per-process object identifiers;
- active and priority player semantics mapped to stable player indices;
- prompt type and count envelope;
- acting player;
- ordered option semantics using type, canonical key, stable name, and scrubbed
  label.

A mismatch raises `ReplayDivergenceError` before any branch result is accepted.
The exception carries both signatures for diagnosis.

## Output

The JSON result records:

```json
{
  "recordType": "ReplaySearchResult",
  "backend": "deterministic-replay",
  "rootSignature": {"sha256": "..."},
  "prefixLength": 42,
  "branches": [
    {
      "selectedIndices": [0],
      "selectedOptions": [],
      "nextObservation": {},
      "finished": false,
      "reward": null,
      "rolloutDecisions": 0,
      "rolloutTruncated": false
    }
  ]
}
```

`--transitions-out` converts successful immediate successors into the existing
transition format with `source: search`. Those rows can train action-conditioned
dynamics or act as exact comparison labels for a learned model.

## Limitations

- Replay assumes XMage is deterministic for the supplied seed and action
  prefix.
- Hidden random choices must be controlled by the game seed.
- The current CLI requires the full prefix from game start.
- Automatic enumeration is intentionally limited to at-most-one prompts.
- Fresh processes make deep or wide search expensive.
- A terminal branch has no nonterminal `nextObservation`; it still records the
  exact result and reward in the search artifact.

These limitations are explicit so the replay backend can serve as a correctness
oracle while native `search_begin/search_step/search_release` is developed.
