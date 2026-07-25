# magic-cabt

Python tooling for the supported `mtg-rl-tools` play-recommendation workflow.

The package also contains experimental and adjacent modules. A console script existing does not, by itself, mean the feature is promoted. See [`../docs/SCOPE.md`](../docs/SCOPE.md).

## Install

```sh
cd python
python -m pip install -e .
```

For maintained PyTorch baselines:

```sh
python -m pip install -e ".[torch]"
```

The XMage-backed commands require `MAGIC_CABT_CLASSPATH` or an explicit `--classpath` pointing at a built XMage+CABT classpath.

## Supported command path

### Capture and engine evaluation

```sh
magic-cabt-arena-mirror live --no-display
magic-cabt-play --deck0 ../examples/basic_deck.txt --deck1 ../examples/basic_deck.txt --agent1 random
magic-cabt-eval-play --deck0 ../examples/basic_deck.txt --deck1 ../examples/basic_deck.txt
```

### Validate and build datasets

```sh
magic-cabt-validate <records.jsonl>
magic-cabt-training-audit --input <bundle-or-jsonl> --out <audit.json>
magic-cabt-build-manifest \
  --input <bundle-or-jsonl> \
  --input <another-bundle-or-jsonl> \
  --out <manifest.json>
magic-cabt-compile-il --input <records.jsonl> --out <single_choice.jsonl>
magic-cabt-build-transitions --input <bundle-or-jsonl> --out <transitions.jsonl>
magic-cabt-build-macro-actions --input <records.jsonl> --out <macro_actions.jsonl>
magic-cabt-analyze-actions --input <records.jsonl>
```

The trust audit checks raw decision legality, hidden-information markers, sequence and split assumptions, transition/macro-action integrity, provenance hashes, and optional checkpoint evidence. See [`../docs/TRUST_GATES.md`](../docs/TRUST_GATES.md).

The manifest command accepts repeated files or bundle directories and records each input's byte size and SHA-256. Its `inventory` section reports games, strategic and trainable decisions, semantic-option coverage, next-observation coverage, bounded duplicate tracking, observed card identifiers, deck identifiers, and the existing source/prompt/capture-confidence distributions.

### Baselines

```sh
magic-cabt-train-bc --input <single_choice.jsonl> --out runs/bc
magic-cabt-eval-bc --input <single_choice.jsonl> --policy first
magic-cabt-train-ranker --input <bundle-or-jsonl> --out runs/ranker
magic-cabt-train-structured-bc --input <bundle-or-jsonl> --out runs/structured-bc
```

### Analysis and comparison

```sh
magic-cabt-replay-annotate --input <records-or-bundle> --policy first

magic-cabt-compare-suite \
  --bundle <bundle> \
  --model first=baseline:first-legal \
  --model random=baseline:random \
  --model ranker=runs/ranker/checkpoint.pt \
  --model structured-bc=runs/structured-bc/best.pt \
  --out runs/comparison.html
```

`magic-cabt-compare-models` remains as a compatibility alias to the same suite entry point. New scripts and documentation should use `magic-cabt-compare-suite`.

### Exact counterfactual data

```sh
magic-cabt-replay-search \
  --input <selfplay-records.jsonl> \
  --decision-index 42 \
  --deck0 ../examples/basic_deck.txt \
  --deck1 ../examples/basic_deck.txt \
  --seed 7 \
  --out runs/search/decision-42.json \
  --transitions-out runs/search/transitions.jsonl
```

## Experimental commands

These share the core data and evaluation contracts but are not promoted as stronger-play solutions:

```text
magic-cabt-train-jepa
magic-cabt-train-information-state
magic-cabt-train-belief-state
magic-cabt-research
```

Use them only with capacity-matched baselines, whole-game splits, held-out downstream metrics, and explicit promotion gates. Training or latent loss alone is not sufficient evidence.

The RSSM implementation is also experimental. Its merged implementation and documentation should be reviewed before restoring or advertising a public console entry point.

## Adjacent or parked commands

These are retained for existing users and experiments but are outside the supported play-recommendation path:

```text
magic-cabt-upload
magic-cabt-agent-service
magic-cabt-local-evolve
magic-cabt-local-model
magic-cabt-build-draft-dataset
magic-cabt-train-draft
magic-cabt-draft-outlook
```

Substantial work in these areas should begin with an extraction or package-boundary plan.

## Model promotion rule

Before adding or promoting a model family, compare it on the same whole-game split against:

1. first/random controls;
2. the dependency-free BC baseline where applicable;
3. the hashed ranker;
4. structured behavior cloning at matched capacity.

Report coverage, failures, compute, and a downstream metric. World-model, belief, or recurrent objectives are auxiliary hypotheses until they improve action ranking, calibration, search consistency, tactical scenarios, or paired engine games.

## Tests

```sh
cd python
python -m unittest discover -s tests -v
```

The root repository README describes the full XMage overlay setup and supported end-to-end workflow.
