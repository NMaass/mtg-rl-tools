# Project scope

`mtg-rl-tools` exists to answer one research question:

> Given a hidden-information-safe Magic game state and the legal choices supplied by XMage, can a reproducible agent rank or select useful plays?

The repository may contain experiments around that question, but they are not all equally supported. This document defines the boundary.

## Supported core

The supported path is intentionally narrow.

### 1. Rules and legal actions

- XMage owns legality, priority, phases, the stack, costs, targets, combat, and hidden information.
- `CabtBridge` exposes each pending decision as an indexed legal-option set.
- Invalid or unsupported decisions fail closed.

### 2. Data capture and normalization

- MTG Arena `Player.log` capture and hidden-information-safe mirroring.
- XMage self-play and human-play records.
- Canonical `DecisionRecord` normalization and validation.
- Dataset manifests, strategic macro-action grouping, and state transitions.

### 3. Minimal policy baselines

- nonlearned first/random/generic-action controls;
- dependency-free behavior cloning;
- hashed option ranker;
- structured state-aware behavior cloning.

These are the required comparison ladder. A more complicated model is not promoted unless it beats the appropriate cheaper control on held-out data or gameplay.

### 4. Evaluation

- whole-game train/evaluation splits;
- top-k, MRR, negative log likelihood, coverage, calibration where applicable;
- replay annotation and side-by-side model comparison;
- fixed engine scenarios and paired agent games;
- legality, hidden-information, throughput, and failure reporting.

### 5. Counterfactual supervision

- deterministic replay search is the current correctness-first backend;
- native in-memory XMage cloning is a performance improvement, not a separate product direction;
- search outputs must preserve the same semantic-action and transition contracts used by the rest of the repository.

## Experimental research

The following code may remain in the repository, but it is not part of the default supported path until promotion evidence exists:

- action-conditioned JEPA training;
- recurrent information-state policies;
- calibrated belief supervision;
- RSSM or other learned dynamics;
- expert preference/cost models;
- causal-factor and statistical research utilities;
- generic masked-RL wrappers;
- deck-local search or self-play leagues.

Experimental modules must:

1. reuse the core data, visibility, semantic-action, and evaluation contracts;
2. include a cheaper capacity-matched control;
3. state what result would falsify the experiment;
4. report held-out downstream performance, not only training loss;
5. avoid becoming a required dependency of data capture or baseline evaluation.

## Adjacent or parked work

These areas are useful, but they are not required to build a play-suggestion model and should not expand inside the core package without a separate scope decision:

- draft, deck-construction, sideboarding, and metagame prediction;
- hosted policy services, public upload ingestion, and Cloudflare deployment;
- automatic between-game continual training and checkpoint swapping;
- broad product/lobby/session infrastructure;
- replay-viewer visual polish beyond what is required to inspect captured state.

Existing code is not deleted merely for being adjacent. It should be isolated, documented as parked, and considered for a separate package or repository when it next receives substantial work.

## Promotion gates

A feature may move from experimental to supported only when all applicable gates pass.

### Data gate

- valid canonical records;
- complete game identities and monotone sequence ordering;
- no detectable hidden-information leakage;
- known capture confidence and discard reasons;
- dataset and input hashes recorded.

### Baseline gate

- compared against first/random and the cheapest relevant learned baseline;
- compared at a shared whole-game split;
- capacity and compute reported;
- no improvement claim based only on training loss.

### Evaluation gate

- held-out primary metric declared before tuning;
- coverage and failure rate reported;
- uncertainty reported for gameplay comparisons;
- at least one fixed tactical or paired-game evaluation when claiming stronger play.

### Maintenance gate

- one documented entry point;
- focused tests;
- no duplicated trainer, tensorizer, checkpoint, or analysis infrastructure;
- no temporary synchronization workflows;
- clear owner and removal condition.

## Rules for new work

Before adding a new model family, product service, GUI, or cross-task head, answer:

1. Which supported milestone does it unblock?
2. What existing implementation can be reused?
3. What is the smallest falsifiable version?
4. Which cheaper baseline must it beat?
5. What will be removed or consolidated if it succeeds?
6. Where should it live if it does not directly serve play recommendation?

A proposal that cannot answer these questions should remain an issue or external experiment rather than becoming another first-class command.
