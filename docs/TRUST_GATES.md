# Training trust audit

`magic-cabt-training-audit` is a fail-fast preflight for datasets and model checkpoints. It answers whether the supplied artifacts satisfy the repository's mechanical trust requirements. It does **not** establish that a model plays Magic well.

## Command

```bash
magic-cabt-training-audit \
  --input arena-mirror-runs/run-001 \
  --input runs/search/transitions.jsonl \
  --checkpoint runs/structured-bc/best.pt \
  --out runs/audits/run-001.json
```

Use `--strict` to promote every warning to a failure.

Dataset auditing has no PyTorch dependency. Checkpoints are audited when PyTorch is installed. Add `--require-checkpoint-audit` when the command must fail rather than warn if PyTorch is unavailable.

## Dataset gates

### Input identity

- every resolved file is hashed with SHA-256;
- empty, malformed, filename/shape-conflicting, and unknown JSONL inputs fail;
- arbitrary JSONL filenames are classified from their first object.

### Decisions

- canonical `DecisionRecord` validation;
- selected indices exist and remain in the legal option range;
- legal-option lists are non-empty and uniquely indexed;
- game identities are recoverable for whole-game splitting;
- sequence values are monotone and games do not reopen later in the stream;
- canonical action keys do not combine different option types;
- duplicate public decision fingerprints are reported;
- generic unverified `history` is reported separately from `publicHistory`.

### Hidden information

The audit recursively rejects obvious oracle/private training keys inside model observations, including belief/training labels, true opponent hands, hidden-state truth, private hands, full opponent decks, and oracle values.

This is a detectable-leakage gate, not proof that every capture source is visibility-safe. Capture code and tensorizers still require their own perspective-invariance tests.

### Transitions and macro actions

- transitions require dictionary `prev` and `next` states and a positive integer horizon;
- transition games must be identifiable, monotone, and contiguous;
- macro-action output reports incomplete groups, orphaned parameter prompts, and heuristic prompt classifications.

## Checkpoint gates

Checkpoint auditing checks:

- artifact existence and loadability;
- recognized model family;
- finite state-dictionary tensors;
- embedded training metrics;
- finite best-selection metric when recorded;
- whole-game split unit and train/evaluation disjointness;
- held-out game IDs against supplied dataset identities;
- recorded input hashes against files still available locally;
- visibility-policy attestation for recurrent, belief, and RSSM models;
- model-family-specific held-out evidence.

The evidence registry currently covers:

| Checkpoint kind | Required held-out evidence |
| --- | --- |
| `torch-option-ranker` | Generic checkpoint and split checks |
| `magic-structured-jepa-v1` | JEPA, causal, policy, and collapse diagnostics |
| `magic-recurrent-information-state-v1` | Loss, top-1, top-3, and MRR |
| `magic-belief-information-state-v1` | Brier score, log loss, ECE, and vocabulary hash |
| `magic-structured-rssm-v1` | Prior NLL, standardized residual RMS, open-loop MSE by horizon, and effective rank |

Unknown checkpoint families are warned and still receive generic tensor, metrics, split, and provenance checks.

## Status semantics

- `pass`: the mechanical requirement was met.
- `warn`: evidence is missing or suspicious, but the artifact may still be usable for exploratory work.
- `fail`: the artifact should not be used for a trusted result.
- `not-applicable`: the input did not include that artifact type.

The report's `summary.trusted` field is true only when no check failed. Under `--strict`, warnings become failures and retain `originalStatus: "warn"`.

## What passing does not mean

A passing audit does not prove:

- optimal or expert-quality labels;
- causal identification;
- absence of every hidden-information leak;
- calibration outside the measured holdout;
- generalization to new decks, cards, formats, or players;
- stronger gameplay than a cheaper baseline;
- safe automatic checkpoint promotion.

Those claims require the evaluation and promotion gates in [SCOPE.md](SCOPE.md).
