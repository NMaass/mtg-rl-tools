# DecisionRecord schema v1

`DecisionRecord` is the canonical, source-agnostic record that every gameplay
capture pipeline in this repo normalizes to before it lands in a training or
evaluation dataset. All four sources share the same shape:

- `arena_human` — recordings from MTG Arena `Player.log` ladders (human games),
  emitted by the arena-mirror recorder.
- `engine_selfplay` — XMage games driven by local agents on the bridge
  (`examples/run_selfplay.py` and the Java `CabtDatasetWriter`).
- `engine_human` — XMage games driven live by a human from the bridge.
- `search` — future search/lookahead rollouts.

This is the **infrastructure** contract. It does not constrain how a model
reads / encodes the observation; it only guarantees that downstream consumers
(record→batch, replay viewers, dataset validators, evaluators) all see one
field layout no matter where the data came from.

## Canonical record

```
{
  "schemaVersion": 1,
  "source": "arena_human|engine_selfplay|engine_human|search",
  "gameId": "...",                       // opaque game/session id or null
  "sequenceNumber": 123,                 // monotonic per game; 0-based is fine
  "playerIndex": 0,                      // acting seat index; null if unknown
  "observation": {},                     // arbitrary JSON: the pre-decision state
                                         // snapshot + log buffer the agent saw.
                                         // Source-specific. Always excludes
                                         // hidden information the acting seat
                                         // could not see.
  "select": {                            // the legal indexed action list
    "type": "...",                       // UPPER_SNAKE prompt kind (PRIORITY,
                                         // MULLIGAN, DECLAREATTACKERSREQ, ...)
    "minCount": 1,                       // inclusive minimum selections
    "maxCount": 1,                       // inclusive maximum selections
    "option": [                          // ordered list of legal choices
      {
        "index": 0,                      // 0-based ordinal; canonical value the
                                         // agent commits
        "type": "...",                   // UPPER_SNAKE option kind (ACTION,
                                         // TARGET, PAY, FIND, KEEP, ...)
        "label": "...",                  // human-readable tracing label
        "payload": {}                    // source-specific action handle
      }
    ]
  },
  "selectedIndices": [4],                // the agent's choice: list of option
                                         // indices that were committed. Empty
                                         // means "no choice recorded yet".
  "nextObservation": null,              // post-decision observation. null when
                                         // unknown (self-play replay, search)
                                         // or when the record is terminal.
  "terminal": false,                    // true when the decision ended the
                                         // game (resolved to result/reward).
  "reward": null,                        // float / int / null; from the
                                         // playerIndex perspective. null when
                                         // unknown.
  "result": null,                        // game result envelope the writer
                                         // produced, e.g. {"winner": 0}. null
                                         // away from terminal transitions.
  "metadata": {                          // source-specific provenance block.
                                         // Only keys the normalizer knows about
                                         // live here; canonical fields do not.
    "captureConfidence": "exact|mirror|partial"
                                         // exact: engine ground truth
                                         // mirror: reconstructed from public log
                                         // partial: best-effort reconstruction
                                         // with known gaps
  }
}
```

The Python representation is plain `dict` objects plus validation helpers in
`python/magic_cabt/training/records.py`. No heavy schema library is introduced
because there is no project convention for one; the import surface stays
optional.

## Canonical vs source-specific fields

| Field            | Canonical?    | Notes |
| ---              | ---           | --- |
| `schemaVersion`  | yes           | Always `1` for v1. |
| `source`         | yes           | One of the four source labels above. |
| `gameId`         | yes           | May be `null` when the source has no concept of a game id (early self-play replay frames). |
| `sequenceNumber` | yes           | Per-game monotonic integer; resets across games within a match. |
| `playerIndex`    | yes           | The acting seat. `null` if the source cannot tell us. |
| `observation`    | yes (shell)   | The container is canonical; **the contents are source-specific**. |
| `select`         | yes           | Always normalized to `{type, minCount, maxCount, option[]}`. |
| `selectedIndices`| yes           | Always a list of zero-based option indices (possibly empty). |
| `nextObservation`| yes           | `null` when not captured. |
| `terminal`       | yes           | `true`/`false`/`null` (record unknown). `true` on the decision that ended the game. |
| `reward`         | yes           | Numeric or `null`. Always from the `playerIndex` perspective. |
| `result`         | yes           | Game-end envelope or `null`. |
| `metadata`       | yes (block)   | Any-field bucket for source-specific provenance. Only well-known entries documented here are checked. |
| `metadata.captureConfidence` | recommended | `exact` / `mirror` / `partial`. Defaults: `exact` for engine sources, `mirror` for Arena. |
| `metadata.xmageVersion`, `metadata.deck0Id`, `metadata.deck1Id`, `metadata.decisionMethod` | Java transition dataset | Preserved unchanged. |
| `metadata.matchId`, `metadata.gameNumber`, `metadata.seat`, `metadata.player`, `metadata.promptTimestamp`, `metadata.responseTimestamp`, `metadata.selectionMatched`, `metadata.promptMessageType`, `metadata.responseMessageType`, `metadata.responsePayload` | Arena decisions.jsonl | Preserved unchanged. Source metadata lives under `metadata`, not at the top level — the canonical top-level surface stays clean. |

Anything else the writer wants to preserve goes under `metadata` and is passed
through verbatim. Unknown top-level keys are *also* preserved by
`normalize_record` (defensive — never silently drop provenance), but the
canonical contract above is the only thing downstream code may rely on.

## Hidden-information rule

`DecisionRecord` does **not** attempt to infer hidden cards. Unknown opponent
hand cards / face-down permanents must remain face-down placeholders: the
`observation` carried into the record is whatever the source already produced
(`GameStateTracker.snapshot()` for Arena, the engine's own public view for the
XMage bridge, etc.). `validate_record` only fails on hidden-info leakage that
is *detectable* (e.g. an explicit `"leaked": true` marker if a future writer
opts in to declaring it). It does not have to inspect every card payload.

## Per-source normalization

`python/magic_cabt/training/records.py:normalize_record(raw, source_hint=None)`
accepts one record from any of the three input formats and returns the
canonical shape. Detection is best-effort by field presence:

1. **Java transition dataset** (`python/magic_cabt/dataset.py` format).
   Output already carries `schemaVersion`, `sequenceNumber`, `observation`,
   top-level `select`, `selectedIndices`, `nextObservation`, `terminal`,
   `reward`, `metadata`. The normalizer lifts `select.playerIndex` to a true
   top-level `playerIndex`, copies `decisionMethod` into `metadata` if absent,
   and fills `source` (default `engine_selfplay`) plus
   `metadata.captureConfidence = "exact"`.

2. **`examples/run_selfplay.py` replay lines** (`replay.jsonl`).
   Each line is `{sequence, player, observation, selected}` plus a trailing
   `{result}` line. The normalizer renames `sequence`→`sequenceNumber`,
   `player`→`playerIndex`, `selected`→`selectedIndices`, lifts
   `observation.select` to a top-level `select`, marks the trailing result
   line's game as `terminal` on the last preceding decision (its `result` is
   attached there), and stamps `source = engine_selfplay` /
   `metadata.captureConfidence = "exact"`. Pure-result lines are not yielded
   as separate records — they only annotate the game's final decision.

3. **Arena `decisions.jsonl` bundle records** (arena-mirror recorder).
   Each line already wraps the prompt spec as `observation.select` and stores
   the chosen indices in a top-level `select` (the recorder's variable name),
   which collides with the canonical `select`. The normalizer:
   - moves `observation.select` (prompt spec) into canonical top-level `select`,
   - renames the Arena top-level `select` → `selectedIndices`,
   - lifts `seat` → `playerIndex`,
   - renames `sequence` → `sequenceNumber`,
   - moves `matchId`, `gameNumber`, `selectionMatched`, `promptMessageType`,
     `responseMessageType`, `responsePayload`, and `prompt`/`response`
     timestamps under `metadata`,
   - stamps `source = arena_human` and `metadata.captureConfidence = "mirror"`
     unless the source explicitly says otherwise.

## Validation rules

`validate_record(record)` returns a list of human-readable error strings
(empty list == valid). It checks:

- the record is a JSON **object** (a `dict`).
- `schemaVersion == 1`.
- a `select` block exists either at the top level or nested under
  `observation` (the latter only in source records; after normalization, it is
  always top level). The normalizer always produces the top-level form.
- `selectedIndices` is a list of integers. Python `bool` is rejected (bools are
  a subtype of int but never a valid selection).
- each selected index satisfies `0 <= index < len(select.option)`.
- `len(selectedIndices) >= select.minCount` (when `minCount` is present).
- `len(selectedIndices) <= select.maxCount` (when `maxCount` is present and
  positive).
- no duplicate selected indices (zero selections trivially satisfies this).
- `playerIndex` is present when the source made it available. The validator
  will not fail a record whose source genuinely cannot supply a player index
  (e.g. an Arena concede decision before seat assignment); but it will fail a
  record that *drops* a known player index (the normalizer never does this).
- when `terminal == true` the record is expected to carry `result` and/or
  `reward`; missing both produces a warning, not a hard failure.
- `metadata` present (the normalizer always adds at least
  `captureConfidence`); source-specific keys are preserved untouched.
- optional leakage check: if a future writer sets `metadata.knownHiddenLeak`
  to a truthy marker the record is *detectably* leaked and fails. The default
  records in this repo never set it.

A record that fails one rule is still reported for all rules; the CLI surfaces
the **first** violations per line so a writer can fix them in one pass.

`validate_records(records)` aggregates per-record errors across an iterable and
returns a summary dict:

```
{
  "total": int,
  "valid": int,
  "invalid": int,
  "errors": [
    {"record": int, "line": int, "messages": ["...", ...]},
    ...
  ],
  "selectTypes": {"PRIORITY": 12, "MULLIGAN": 2, ...},
  "optionTypes": {"ACTION": 12, "PASS": 3, ...},
  "selectedCount": {"0": 1, "1": 13, ...},
}
```

`line` is `1`-based from the input file when known; `-1` when validating a
record in isolation (no file context).

## CLI

```
python -m magic_cabt.training.validate_dataset <jsonl>
```

Reads the path with `iter_decision_records` (normalized) and prints:

- total / valid / invalid record counts,
- `select.type` distribution (when present),
- `option.type` distribution (when present),
- `selectedIndices` length distribution,
- the first few validation errors with line numbers.

Exit code is `1` whenever one or more records are invalid; `0` otherwise.