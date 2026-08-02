# mtg-rl-tools

[![python-mirror-tests](https://github.com/NMaass/mtg-rl-tools/actions/workflows/python-mirror-tests.yml/badge.svg)](https://github.com/NMaass/mtg-rl-tools/actions/workflows/python-mirror-tests.yml)

Research tooling for recommending Magic: The Gathering plays from structured, hidden-information-safe game states.

The supported path is deliberately narrow:

```text
MTG Arena logs or XMage games
        -> canonical DecisionRecords
        -> strategic macro actions and transitions
        -> legal-option policy baselines
        -> held-out and engine evaluation
        -> exact replay-search supervision
```

XMage owns legality. Models rank the legal choices that the engine supplies; they do not reproduce the rules or emit an unrestricted action vocabulary.

See [Project scope](docs/SCOPE.md) for the supported/experimental boundary, [the July 2026 scope audit](docs/SCOPE_AUDIT_2026-07.md) for the current cleanup decisions, [the July 2026 dataset feasibility reanalysis](docs/DATASET_FEASIBILITY_2026-07.md) for how the training corpus is acquired, and [the August 2026 IL training-run plan](docs/IL_TRAINING_RUN_2026-08.md) for the first run on the personal 17lands corpus.

## Project status

This is active research software, not a complete competitive Magic agent platform.

| Area | Status |
| --- | --- |
| XMage option-index bridge | Implemented. Real XMage callbacks expose indexed legal choices and fail closed on unsupported decisions. |
| Python live-game client | Implemented through `magic_cabt.CabtBridge`. |
| MTG Arena capture | Implemented. `Player.log` sessions can be followed, normalized, mirrored, and recorded. |
| MTGO video capture | Implemented. MTGO footage is OCR'd into a game log, reconstructed into board states, replayed in XMage, and verified three ways: against XMage's own rendering, against MTGO's on-screen life totals, and across captures of the same match at different resolutions. The UI is located by detection rather than fixed coordinates, so 720p through 1440p decode identically. See [`docs/MTGO_VIDEO.md`](docs/MTGO_VIDEO.md). |
| Canonical data contract | Implemented for Arena, XMage self-play, engine-human play, and search records. |
| Strategic action grouping | Implemented. Low-level target/mode/payment callbacks can be compiled into auditable macro actions. |
| Transition datasets | Implemented for state streams, decisions, and complete macro actions. |
| Corpus inventory | Implemented in dataset manifests: games, strategic decisions, semantic options, next-state coverage, duplicates, cards, decks, capture confidence, and input hashes. |
| Training trust audit | Implemented for raw datasets, whole-game assumptions, provenance, and optional checkpoint evidence. |
| Baseline policies | Implemented: first/random controls, behavior cloning, hashed ranking, and structured behavior cloning. |
| Model comparison | Implemented for recorded decisions and agent-vs-agent engine games. |
| Tactical benchmark evaluator | Implemented for versioned replay-root scenarios, semantic acceptable actions, public history, and grouped metrics. A curated expert-reviewed suite is not yet committed. |
| Counterfactual search | Implemented through deterministic replay search. It is exact but slower than native cloning. |
| Native cloned-game search | Not implemented. This is a throughput improvement, not a missing data contract. |
| Broad competitive agent | Not demonstrated. |

## Supported workflow

### Fast path: one command from saved Arena logs

`magic-cabt-training-run` chains the whole supported pipeline — batch
ingest of saved `Player.log` files, validation, trust audit, manifest,
whole-game splits, IL compilation, baselines, training, evaluation — into
one resumable run directory with a final report. It needs no XMage build.

```sh
cd python && python -m pip install -e .

# Smoke-test the pipeline end to end on a synthetic corpus (seconds):
magic-cabt-training-run --toy 30 --out runs/toy

# Real run: point it at saved Player.log files (or directories of them):
magic-cabt-training-run --log captures/Player.log --log captures/old-logs \
  --out runs/first-run --name arena-personal-v1
```

To pull your own 17Lands draft history (Mythic patronage backfills it in
full), see [docs/SEVENTEENLANDS_EXPORT.md](docs/SEVENTEENLANDS_EXPORT.md):

```sh
magic-cabt-17lands-export --cookie-file ~/.17lands-cookie --out 17lands-export
```

Re-running the same command resumes: stages whose inputs did not change are
skipped. Torch model stages run when `.[torch]` is installed and are
recorded as skipped otherwise. `--dry-run` prints the resolved plan;
`report.md` in the run directory summarizes corpus, splits, metrics, stage
timings, and warnings. The steps below remain the manual, stage-by-stage
version of the same path.

### 1. Build and verify the XMage overlay

This repository is an overlay for an XMage checkout, not a fork.

```sh
git clone https://github.com/NMaass/mtg-rl-tools.git
git clone https://github.com/magefree/mage.git
rsync -a --exclude .git --exclude README.md mtg-rl-tools/ mage/
cd mage
scripts/run-cabt-adapter-tests.sh
```

The verification script runs Java prompt/validation tests, full-engine smoke games, Python tests, and live protocol tests when the generated classpath is available.

### 2. Install the Python package

```sh
cd python
python -m pip install -e .
```

For the maintained PyTorch baselines:

```sh
python -m pip install -e ".[torch]"
```

### 3. Capture or generate games

Run XMage self-play after building the classpath:

```sh
export MAGIC_CABT_CLASSPATH="$(cat Mage.Server.Plugins/Mage.Player.AI/target/cabt-classpath.full.txt)"
python3 examples/run_selfplay.py --seed 42 --max-turns 15
```

Capture Arena games on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-arena-mirror.ps1 `
  -XmageDir C:\path\to\mage
powershell -ExecutionPolicy Bypass -File scripts\arena-mirror-gui.ps1
```

Detailed Arena instructions: [docs/RUNNING.md](docs/RUNNING.md).

### 4. Validate and compile data

```sh
magic-cabt-validate arena-mirror-runs/run-001/decisions.jsonl

magic-cabt-training-audit \
  --input arena-mirror-runs/run-001 \
  --out runs/run-001/trust-audit.json

magic-cabt-build-manifest \
  --input arena-mirror-runs/run-001 \
  --input arena-mirror-runs/run-002 \
  --name arena-standard-corpus \
  --out runs/arena-standard-manifest.json

magic-cabt-build-macro-actions \
  --input arena-mirror-runs/run-001/decisions.jsonl \
  --out runs/run-001/macro_actions.jsonl \
  --transitions-out runs/run-001/macro_transitions.jsonl
```

The audit records input hashes and checks raw decision legality, visible-state boundaries, game sequencing, transition and macro-action integrity, split assumptions, and optional checkpoint evidence. See [docs/TRUST_GATES.md](docs/TRUST_GATES.md).

The manifest accepts repeated JSONL files or bundle directories. Its `inventory` section reports whole games, strategic root decisions, deduplicated semantic choices, trainable single-choice rows, non-terminal next-state coverage, duplicate public fingerprints, observed card identifiers, deck identifiers, and capture-confidence distributions. Duplicate tracking is exact until its declared memory bound is reached and then reports `trackingTruncated: true`.

### 5. Establish the baseline ladder

Start cheap. A larger representation is not evidence of stronger play by itself.

```sh
magic-cabt-compile-il \
  --input arena-mirror-runs/run-001/decisions.jsonl \
  --out runs/run-001/single_choice.jsonl

magic-cabt-train-bc \
  --input runs/run-001/single_choice.jsonl \
  --out runs/bc

magic-cabt-train-ranker \
  --input arena-mirror-runs/run-001 \
  --out runs/ranker

magic-cabt-train-structured-bc \
  --input arena-mirror-runs/run-001 \
  --out runs/structured-bc \
  --preset local
```

### 6. Compare models on the same decisions

```sh
magic-cabt-compare-suite \
  --bundle arena-mirror-runs/run-001 \
  --model first=baseline:first-legal \
  --model random=baseline:random \
  --model ranker=runs/ranker/checkpoint.pt \
  --model structured-bc=runs/structured-bc/best.pt \
  --out runs/comparisons/run-001.html
```

### 7. Evaluate fixed tactical roots

```sh
magic-cabt-research benchmark-scenarios \
  --suite examples/research/core_tactics_v1.jsonl \
  --model first=baseline:first-legal \
  --model heuristic=baseline:heuristic \
  --model structured=runs/structured-bc/best.pt \
  --top-k 3 \
  --out runs/tactical/core-tactics-v1.json
```

Tactical scenarios preserve captured public observations and legal choices, annotate one or more acceptable semantic action groups, and may include prior public history for recurrent scorers. The evaluator reports coverage, top-1, top-k, MRR, prohibited top-1, tags, and prompt groups. The repository currently provides the schema and evaluator; curating the first immutable expert-reviewed suite remains work. See [docs/TACTICAL_SCENARIOS.md](docs/TACTICAL_SCENARIOS.md).

### 8. Generate exact counterfactual branches

```sh
magic-cabt-replay-search \
  --input target/cabt-selfplay/replay.jsonl \
  --decision-index 42 \
  --deck0 examples/basic_deck.txt \
  --deck1 examples/basic_deck.txt \
  --seed 42 \
  --out target/search/decision-42.json \
  --transitions-out target/search/transitions.jsonl
```

Replay search rebuilds the decision from the recorded action prefix, verifies the public state and legal choices, and evaluates alternatives in fresh engine processes. See [docs/REPLAY_SEARCH.md](docs/REPLAY_SEARCH.md).

## Scope levels

### Supported

- XMage bridge and legal-option protocol;
- Arena capture and replay inspection;
- DecisionRecord validation, trust auditing, and corpus manifests;
- macro actions and transitions;
- minimal policy baselines;
- whole-game, replay-root tactical, and comparison evaluation;
- deterministic replay search.

### Experimental

The repository also contains JEPA, recurrent information-state, belief, RSSM, causal-factor, expert-preference, and broader RL experiments. They remain research hypotheses and must beat cheaper controls before promotion.

### Adjacent or parked

Draft/deck/sideboard models, hosted policy/upload infrastructure, and automatic between-game continual training are retained but are not part of the supported play-recommendation path. Substantial future work in those areas should begin with an extraction plan.

## Repository layout

```text
Mage.Server.Plugins/Mage.Player.AI/
  src/main/java/mage/player/cabt/      XMage bridge and protocol
  src/test/java/mage/player/cabt/      engine and callback tests
  docs/                                Java-side design and verification

Mage.Client/
  src/main/java/mage/client/cabtmirror/ optional Arena replay display

python/
  magic_cabt/                           client, capture, datasets, models, evaluation
  tests/                                Python tests

examples/                               decks, self-play, research fixtures
scripts/                                setup and verification
docs/                                   user and research documentation
```

## Design rules

- **XMage owns legality.** Never replace engine legality with a static universal action enum.
- **Fail closed.** Unsupported decisions and invalid selections must remain visible failures.
- **Preserve perspective.** Deployment inputs may contain only information available to the acting player.
- **Use semantic actions.** Option indices are transport identifiers, not stable model meaning.
- **Split whole games.** Adjacent decisions from one trajectory must not cross train/evaluation boundaries.
- **Beat the baseline.** New model families require capacity-matched controls and downstream evidence.
- **Keep adjacent products separate.** Hosting, uploads, draft tools, and continual training do not become core by default.

## Current roadmap

1. Curate and freeze the first expert-reviewed `core-tactics-v1` scenario corpus from retained replay roots.
2. Run scaling curves for BC, hashed ranker, and structured BC using committed corpus manifests.
3. Generate replay-search labels for a bounded tactical subset.
4. Resume world-model or belief experiments only when a fixed benchmark identifies the failure they are intended to solve.
5. Implement native XMage cloning only when replay-search throughput becomes the measured bottleneck.

## Tests

```sh
cd python
python -m unittest discover -s tests -v
```

Full overlay verification:

```sh
scripts/run-cabt-adapter-tests.sh
```

## License

MIT — see [LICENSE](LICENSE). XMage is a separate project under its own license.
