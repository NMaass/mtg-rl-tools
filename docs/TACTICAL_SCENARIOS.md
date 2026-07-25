# Replay-root tactical scenario suites

The tactical benchmark evaluates a model on fixed, annotated decision roots captured from real Arena or XMage trajectories.

It does **not** synthesize an arbitrary board inside XMage. The current bridge can start games and replay action prefixes, but it cannot load a complete hand-authored engine state. Until native state loading exists, a scenario must preserve the recorded public observation and legal-option set at the decision boundary.

## Command

```bash
magic-cabt-research benchmark-scenarios \
  --suite examples/research/core_tactics_v1.jsonl \
  --model first=baseline:first-legal \
  --model heuristic=baseline:heuristic \
  --model structured=runs/structured-bc/best.pt \
  --top-k 3 \
  --out runs/tactical/core-tactics-v1.json
```

Checkpoint models use the existing analysis scorer factory. Baseline-only runs do not require PyTorch.

The command exits:

- `0` when every model scores every scenario;
- `1` when a scorer fails on one or more scenarios;
- `1` under `--strict` when the suite contains warnings, such as positional expectations;
- `2` for malformed input, missing checkpoints, or invalid command arguments.

## JSONL schema

Each line is one scenario:

```json
{
  "schemaVersion": 1,
  "suite": "core-tactics-v1",
  "scenarioId": "hold-removal-for-combat-trick-001",
  "tags": ["combat", "removal", "resource-preservation"],
  "history": [],
  "decision": {
    "schemaVersion": 1,
    "source": "arena_human",
    "gameId": "...",
    "sequenceNumber": 42,
    "playerIndex": 0,
    "observation": {
      "current": {},
      "publicHistory": [],
      "select": {}
    },
    "select": {
      "type": "PRIORITY",
      "minCount": 1,
      "maxCount": 1,
      "option": []
    },
    "selectedIndices": [0]
  },
  "acceptableActionKeys": [
    "CAST_SPELL|cast:lightning-strike:target:attacker-7"
  ],
  "prohibitedActionKeys": [
    "PASS_PRIORITY|pass"
  ],
  "annotation": {
    "reviewers": ["expert-1", "expert-2"],
    "agreement": "unanimous",
    "rationale": "Removal before damage prevents the known pump line."
  },
  "provenance": {
    "datasetSha256": "...",
    "decisionFingerprint": "sha256:..."
  }
}
```

Only `schemaVersion`, `suite`, `scenarioId`, `decision`, and a non-empty `acceptableActionKeys` list are required by the evaluator. `tags`, `history`, `prohibitedActionKeys`, `annotation`, and `provenance` are strongly recommended.

## Semantic action keys

A stable action key is:

```text
<option type>|<payload.canonicalKey>
```

Examples:

```text
CAST_SPELL|cast:lightning-bolt
TARGET|player:opponent
PASS_PRIORITY|pass
```

Fungible concrete options sharing the same option type and `canonicalKey` are evaluated as one semantic action group. The group's score is the maximum score among its concrete members.

When an option has no `canonicalKey`, the evaluator exposes a fallback such as `INDEX|3`. Positional expectations are permitted only for temporary investigation and produce a suite warning because option order is not a stable semantic contract.

## History and stateful models

`history` contains earlier public DecisionRecords from the same game, in order. For scorers implementing `reset()` and `observe(record)`, the evaluator:

1. resets before each scenario;
2. replays every history record through `observe`;
3. scores the tactical decision.

A stateful scorer with `reset()` but no `observe()` is advanced by calling `score()` on the history records. Stateless scorers ignore history.

History must contain only information available from the acting player's perspective. The same obvious oracle/private observation-key gate used by the training trust audit is applied to the scenario decision and its history.

## Metrics

For every model, the report contains:

- scenario coverage and scorer-error count;
- acceptable top-1 rate;
- acceptable top-k rate;
- mean reciprocal rank of the highest acceptable action;
- prohibited top-1 rate;
- per-tag metrics;
- per-prompt-type metrics;
- ranked semantic groups and concrete option indices for each scenario;
- model/checkpoint provenance and suite SHA-256.

Multiple actions may be acceptable. The metric uses the best rank among the acceptable set.

## Authoring policy

A committed benchmark scenario should:

1. originate from a retained Arena/XMage record or a deterministic replay root;
2. preserve the exact public observation and legal choices;
3. have at least two semantic legal choices;
4. use canonical semantic action keys rather than labels or raw indices;
5. include source hashes and the decision fingerprint;
6. be reviewed by at least one person other than the original player when used for promotion decisions;
7. mark genuine strategic ambiguity with multiple acceptable actions rather than forcing false consensus;
8. remain outside all training and model-selection data.

Scenarios copied from a model's own failures may be added to the next benchmark version, but the previous version must remain immutable so regression claims remain reviewable.

## Versioning

- `suite` names are immutable version identifiers, such as `core-tactics-v1`.
- One JSONL file may contain only one suite name.
- Editing an expectation, decision root, or public history requires a new suite version.
- Tags and annotation wording may be corrected only when the semantic expected set is unchanged; record the correction in version control.

## Limitations

This benchmark measures action ranking at recorded roots. It does not by itself establish:

- long-horizon gameplay strength;
- correctness under unseen legal-option generation;
- value calibration;
- robustness to UI capture errors;
- optimality of the human annotation;
- the quality of dependent target/payment selections after the root action.

Use it alongside whole-game splits, comparison reports, replay-search counterfactuals, and paired XMage games.
