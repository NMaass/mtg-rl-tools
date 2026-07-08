# MTG-Causal-RL parity notes

MTG-Causal-RL frames Magic as a reproducible causal-RL benchmark: masked
actions, partial observations, fixed archetypes, reward variants, causal
variables, per-factor credit traces, and run manifests. This repository has a
different core advantage — a real XMage rules engine and Arena-log capture — so
parity should mean adopting the benchmark discipline without replacing engine
legality.

## Adopted in this PR

- **Strategic causal variables**: `magic_cabt.training.causal` extracts stable
  public-state factors such as life buffer, card/hand/library/graveyard counts,
  battlefield/creature/land counts, stack count, active-player flag, and
  priority flag. Missing or hidden information is not inferred.
- **Per-factor traces**: `factor_credit_trace(record)` records before/after
  factors and deltas when `nextObservation` is available.
- **Dataset manifests**: `magic_cabt.training.manifest` summarizes source,
  prompt/action, option-count, reward/terminal, validation, capture-confidence,
  and causal-factor coverage for a DecisionRecord stream.
- **CLI**: `python -m magic_cabt.training.build_manifest --input <jsonl>` writes
  the manifest JSON that should be attached to training/evaluation artifacts.

## Still needed for full parity

- Fixed Standard/deck suites with versioned deck manifests.
- Reward-profile support: sparse terminal, potential-shaped, and dense tactical
  reward views.
- A full SCM specification over the selected strategic variables, including
  intervention propagation and factor-target sign agreement metrics.
- Paired-seed evaluation with confidence intervals and multiple-comparison
  correction for benchmark claims.
- Reference baselines beyond `random`/`first`: masked PPO, scalar-control PPO,
  and a causal auxiliary-head baseline.
- Stable observation tensorization for model training. This PR intentionally
  remains JSON/dict based so it can sit on top of both XMage self-play and
  Arena-recorded DecisionRecords.
