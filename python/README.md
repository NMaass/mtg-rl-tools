# magic-cabt

Python tooling for CABT-style Magic: The Gathering reinforcement-learning,
imitation-learning, Arena-log recording, replay annotation, and local agent
experiments.

This package is the Python layer of `mtg-rl-tools`. The live XMage bridge still
requires the Java overlay to be copied into and built inside an XMage checkout;
see the repository root README for the full bridge setup.

## Editable install

From the repository root:

```sh
cd python
python -m pip install -e .
```

For PyTorch ranker and JEPA training:

```sh
python -m pip install -e ".[jepa]"
```

Then the import surface is available from any working directory:

```python
from magic_cabt import CabtBridge, load_decklist
```

## Console commands

The package exposes the common research/data commands as console scripts:

```sh
magic-cabt-validate <records.jsonl>
magic-cabt-compile-il --input <records.jsonl> --out <single_choice.jsonl>
magic-cabt-build-manifest --input <records.jsonl> --out <manifest.json>
magic-cabt-train-bc --input <single_choice.jsonl> --out runs/bc
magic-cabt-eval-bc --input <single_choice.jsonl> --policy first
magic-cabt-analyze-actions --input <records.jsonl>
magic-cabt-eval-play --deck0 ../examples/basic_deck.txt --deck1 ../examples/basic_deck.txt
magic-cabt-play --deck0 ../examples/basic_deck.txt --deck1 ../examples/basic_deck.txt --agent1 random
magic-cabt-replay-annotate --input <records-or-bundle> --policy first
magic-cabt-arena-mirror live --no-display
```

`magic-cabt-eval-play`, `CabtBridge`, and display-backed Arena mirror commands
need `MAGIC_CABT_CLASSPATH` or `--classpath` pointing at a built XMage+CABT
classpath.

## Reliable JEPA training

The structured trainer splits transitions and decisions on whole-game IDs,
reports held-out policy/world-model metrics and collapse diagnostics, supports
CUDA mixed precision and gradient accumulation, and writes both:

- `checkpoint.pt`: final model plus optimizer, scaler, RNG, and epoch state for
  exact continuation;
- `best.pt`: lowest held-out-loss model for analysis or deployment.

Example:

```sh
magic-cabt-train-jepa \
  --input ../arena-mirror-runs/run-001 \
  --input ../arena-mirror-runs/run-002 \
  --out runs/jepa-local \
  --preset local \
  --device cuda \
  --amp auto \
  --batch-size 16 \
  --grad-accum-steps 2 \
  --epochs 10 \
  --eval-fraction 0.1
```

Resume the final training state:

```sh
magic-cabt-train-jepa \
  --input ../arena-mirror-runs/run-001 \
  --out runs/jepa-local-resumed \
  --resume runs/jepa-local/checkpoint.pt \
  --device cuda \
  --epochs 5
```

`metrics.json` records whole-game split identities, train/eval losses, canonical
policy top-k/MRR, latent effective rank, throughput, peak CUDA allocation, input
hashes, and the selected best epoch.

## Head-to-head model analysis

Score the same recorded decisions with multiple checkpoints and generate a
self-contained HTML comparison plus a machine-readable JSON artifact:

```sh
magic-cabt-compare-models \
  --bundle ../arena-mirror-runs/run-001 \
  --model hashed-ranker=runs/ranker/checkpoint.pt \
  --model structured-jepa=runs/jepa/best.pt \
  --out runs/comparisons/run-001.html \
  --device cuda
```

Agreement and human-play rank use canonical action groups when XMage identifies
fungible options. Options without a canonical key retain concrete option
identity; equal display labels are never treated as strategically equivalent.

## Research experiment commands

`magic-cabt-research` provides dependency-free guardrails and reports shared by
imitation, JEPA/world-model, search, and causal-factor experiments:

```sh
# Reject split/hidden-information/statistical leakage in an experiment manifest.
magic-cabt-research validate-plan \
  ../examples/research/experiment_matrix_v1.json --strict

# Fit a monotone Bradley-Terry expert-preference cost model.
magic-cabt-research fit-cost \
  --factors ../examples/research/causal_factors_v1.json \
  --preferences ../examples/research/expert_preferences.example.jsonl \
  --out runs/expert-cost-v1.json

# Benchmark model analysis caches against held-out decisions.
magic-cabt-research benchmark-analysis \
  --decisions heldout.decisions.jsonl \
  --analysis ranker=runs/ranker/analysis.jsonl \
  --analysis jepa=runs/jepa/analysis.jsonl \
  --out runs/analysis-benchmark.json

# Benchmark paired match scenarios with clustered intervals and Holm correction.
magic-cabt-research benchmark-matches \
  --input ../examples/research/match_results.example.jsonl \
  --out runs/match-benchmark.json
```

The research rationale, model ladder, data layers, benchmark protocol, and
expert-annotation procedure are documented in:

- `../docs/RESEARCH_ARCHITECTURE.md`
- `../docs/EXPERT_COST_PROTOCOL.md`

## Tests

```sh
cd python
python -m unittest discover -s tests -v
```

The dedicated `python-training-tests` GitHub Actions workflow installs the
PyTorch extra and executes a real forward/backward/checkpoint smoke run. This
prevents the optional training path from passing CI only because torch-gated
tests were skipped.
