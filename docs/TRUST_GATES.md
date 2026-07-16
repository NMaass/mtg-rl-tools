# Training trust gates

## Purpose

`magic-cabt-training-audit` produces machine-readable evidence about the data,
split, checkpoint, and diagnostics behind a model result. It is intended to run
before expensive training, before publishing comparisons, and before promoting
a checkpoint into live play or planning experiments.

Passing the audit means the checked invariants were satisfied. It does not prove
that the model is strong, causal, calibrated outside the evaluation
population, or safe for imagined search.

## Command

```bash
magic-cabt-training-audit \
  --input arena-mirror-runs/run-001 \
  --input arena-mirror-runs/run-002 \
  --checkpoint runs/structured-bc/best.pt \
  --checkpoint runs/information-state/best.pt \
  --checkpoint runs/belief-state/best.pt \
  --checkpoint runs/rssm/best.pt \
  --out runs/audits/training-audit.json
```

Use `--strict` to promote every warning to a failure. `--max-records 0` audits
all available records. The command exits nonzero on any failed check.

## Dataset gates

### Decision legality

- at least one decision exists;
- every trainable decision has a non-empty legal-option set;
- option indices are unambiguous;
- the recorded single selection refers to a legal option;
- canonical keys do not silently merge different action types without a
  warning.

The audit reads raw decisions before the training compiler. This is necessary
because the compiler intentionally drops malformed and non-single-choice rows.

### Hidden-information boundary

- oracle labels and private-state keys are forbidden inside `observation`;
- generic unverified `history` is reported;
- hidden opponent-hand card identity and numeric characteristics are perturbed;
- the perturbation must not change `VisibilitySafeTensorizer` output.

This is an input-invariance test, not a semantic proof that every possible
private field has been enumerated.

### Sequence integrity

- game identities are recoverable;
- sequence values are monotone within each game;
- a completed game does not reappear later in the same stream;
- decision and transition collectors report when auditing was truncated.

Recurrent training commands separately preserve complete accepted games at
collection limits.

## Checkpoint gates

### Artifact integrity

- checkpoint loads successfully;
- model family is recognized;
- state dictionary exists;
- all tensor values are finite;
- best-selection metric is finite;
- embedded training metrics are present for research models.

### Split integrity

- split unit is `game`;
- training and evaluation game IDs are disjoint;
- held-out IDs are compared with supplied data when available;
- available input files are checked against recorded SHA-256 hashes.

A missing local source file produces a provenance warning because the audit
cannot recompute its hash. A mismatched available file is a failure.

### Model-specific evidence

**Structured JEPA** requires held-out JEPA, causal, policy, and collapse
metrics.

**Recurrent information state** requires held-out policy NLL, top-1, top-3,
and MRR, plus the visibility policy and complete-game collection metadata.

**Belief information state** additionally requires a hashed vocabulary and
held-out Brier score, log loss, and expected calibration error.

**Structured RSSM** additionally requires held-out prior NLL, standardized
residual RMS, open-loop error by horizon, and latent effective rank.

## Status levels

- `pass`: the available evidence satisfies the check;
- `warn`: evidence is incomplete or a suspicious but not necessarily invalid
  condition exists;
- `fail`: a hard invariant is violated;
- `not-applicable`: the relevant data or artifact was not supplied.

The JSON preserves `originalStatus` when strict mode promotes a warning to a
failure.

## Promotion policy

A checkpoint should not enter a published head-to-head result unless:

1. the audit has no failures;
2. the exact audit JSON is archived with the experiment;
3. all compared checkpoints use the same held-out games or an explicitly
   paired evaluation manifest;
4. the non-learned and capacity-matched controls are included;
5. model-specific interpretation gates in `BELIEF_SUPERVISION.md` and
   `RSSM_DIAGNOSTICS.md` are respected.
