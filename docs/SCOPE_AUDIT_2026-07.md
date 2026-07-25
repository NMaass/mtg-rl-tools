# Scope audit — July 2026

## Conclusion

The repository has a strong core, but its public surface expanded faster than its evidence base. The main risk is no longer missing model code. It is that multiple research hypotheses, adjacent products, and convenience workflows now look equally supported even though the project has not yet established a single benchmarked play-suggestion path at useful data scale.

This audit does not recommend deleting research. It recommends making the core path explicit, parking adjacent work, and requiring promotion evidence before adding more first-class features.

## Evidence reviewed

- root and Python package READMEs;
- `python/pyproject.toml` console-script surface;
- the research architecture and model ladder;
- merged PRs through #42;
- open PRs #39–#41;
- current data, training, analysis, hosted-service, limited-play, and local-evolution entry points.

## Findings

### 1. Public documentation described an older repository

The root README still described cloned search and dataset tooling as future work after deterministic replay search, macro-action grouping, transition builders, model training, and comparison tooling had already landed.

Impact:

- contributors could choose work from an obsolete roadmap;
- implemented tools appeared unsupported or undiscoverable;
- the actual core milestone was obscured by historical setup detail.

Decision: rewrite the README around the current supported workflow and link the scope policy.

### 2. The console-command surface is too flat

The Python package exposes more than two dozen top-level commands. Capture, validation, baseline training, world models, belief models, hosted services, upload ingestion, draft models, continual training, and research statistics all appear at the same support level.

Impact:

- users cannot tell which path is expected to work end to end;
- every experimental command creates documentation, compatibility, and checkpoint obligations;
- incomplete features look promoted merely because they have an entry point.

Decision: classify commands as supported, experimental, or adjacent/parked. Do not add another first-class command without a scope and maintenance gate.

### 3. The model portfolio is ahead of the benchmark portfolio

The repository contains or documents:

- bag-of-words behavior cloning;
- hashed PyTorch ranking;
- structured behavior cloning;
- structured JEPA;
- recurrent information-state modeling;
- belief supervision;
- stochastic RSSM dynamics;
- expert preference/cost models;
- future MuZero, PUCT, and public-belief search directions.

The architecture document correctly calls these hypotheses, but the implementation sequence nevertheless produced several model families before a stable corpus-size report, tactical suite, and paired gameplay benchmark were established.

Impact:

- engineering effort can optimize infrastructure for models that may not outperform structured BC;
- duplicated trainer/checkpoint/analysis logic becomes likely;
- low training loss can be mistaken for progress toward stronger play.

Decision: freeze new model families. The next promoted work should be dataset auditability, tactical scenarios, replay-search labels, scaling curves, and matched comparisons.

### 4. Adjacent product work entered the core package

Hosted policy service, opt-in upload ingestion, Cloudflare deployment, draft/deck/sideboard models, and local automatic between-game training are not necessary for the central play-ranking research loop.

Impact:

- security, consent, deployment, GUI, and lifecycle concerns compete with game-model research;
- package dependencies and user expectations broaden;
- adjacent tasks encourage a shared-model platform before the gameplay representation is validated.

Decision: retain existing code but mark it parked. New substantial work in these areas should begin with an extraction plan to a separate package or repository.

### 5. Automatic continual training is premature

The local-evolution path can train between games and atomically swap checkpoints. This is technically useful but operationally ahead of the evidence gates.

Missing prerequisites include:

- trusted dataset quarantine;
- stable regression scenarios;
- automatic promotion/rejection based on held-out and gameplay metrics;
- rollback criteria tied to semantic behavior rather than only checkpoint validity;
- protection against one-session distribution collapse.

Decision: do not treat local auto-evolution as supported. Manual, reproducible training runs remain the default.

### 6. Open synchronization PRs were repository noise

PRs #40 and #41 existed only to manipulate ancestry for PR #39. PR #39 then mixed the desired trust audit with duplicated belief/RSSM code and temporary workflow patches.

Decision:

- close #40 and #41 as obsolete;
- close #39 as mixed/superseded;
- track a narrow trust-audit extraction in issue #43.

### 7. Some overlapping command pairs should converge

Examples:

- `compare-models` and `compare-suite`;
- raw transitions and macro-action transitions as separate user decisions;
- multiple training CLIs sharing most optimizer/checkpoint/split behavior;
- `local-model` and `local-evolve` lifecycle overlap.

Decision: no immediate breaking removal. Consolidate when one path can subsume the other with compatibility aliases and migration documentation.

### 8. Feature completeness is inconsistent

The RSSM implementation is present, but the public package script list does not expose the command described by its merged PR. Conversely, several adjacent features do have top-level commands despite lacking promotion evidence.

Decision: entry-point presence must not be used as the definition of support. The scope document and command catalog are authoritative until command consolidation is complete.

## Supported command path

### Capture and engine

- `magic-cabt-arena-mirror`
- `magic-cabt-play`
- `magic-cabt-eval-play`

### Validate and build datasets

- `magic-cabt-validate`
- `magic-cabt-build-manifest`
- `magic-cabt-compile-il`
- `magic-cabt-build-transitions`
- `magic-cabt-build-macro-actions`
- `magic-cabt-analyze-actions`

### Baselines and comparison

- `magic-cabt-train-bc`
- `magic-cabt-eval-bc`
- `magic-cabt-train-ranker`
- `magic-cabt-train-structured-bc`
- `magic-cabt-replay-annotate`
- `magic-cabt-compare-suite`

### Counterfactual data

- `magic-cabt-replay-search`

## Experimental command path

- `magic-cabt-train-jepa`
- `magic-cabt-train-information-state`
- `magic-cabt-train-belief-state`
- `magic-cabt-research`

RSSM remains experimental and should not be promoted merely by restoring a missing entry point. First decide whether it remains an active comparator and whether its diagnostics are maintained.

## Adjacent/parked command path

- `magic-cabt-upload`
- `magic-cabt-agent-service`
- `magic-cabt-local-evolve`
- `magic-cabt-local-model`
- `magic-cabt-build-draft-dataset`
- `magic-cabt-train-draft`
- `magic-cabt-draft-outlook`

## Next work, in order

1. Implement the narrow trust audit tracked by issue #43.
2. Produce a dataset inventory over real available bundles: games, root decisions, complete transitions, prompt coverage, duplicate rate, capture confidence, and card/deck coverage.
3. Build a versioned tactical scenario suite.
4. Run the same whole-game splits through first/random, bag-of-words BC, hashed ranker, and structured BC.
5. Generate replay-search labels for a bounded tactical subset.
6. Run data-scaling curves before resuming JEPA, belief, RSSM, MuZero, or PUCT expansion.
7. Extract or archive adjacent hosted, continual-training, and limited-play features when they next receive development.

## Stop conditions

Do not add another model family or product subsystem until at least one of the following is true:

- the structured BC baseline is demonstrably data-limited and the proposed model addresses the observed failure;
- exact search labels expose a specific value/policy approximation problem;
- a fixed benchmark shows a recurrent/belief/world-model experiment improves play or calibration;
- an adjacent subsystem has an explicit extraction owner and repository boundary.
