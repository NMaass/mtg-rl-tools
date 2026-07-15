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
