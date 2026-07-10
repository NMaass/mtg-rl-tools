# JEPA world-model plan and sizing

## Why JEPA fits this project

A JEPA (joint-embedding predictive architecture) learns an encoder by
predicting the *embedding* of a future state from the current state (and
optionally the action) — no reconstruction, no negatives. The decisive
property for us: **pretraining consumes trajectories, not human decisions.**
Any legal game produces training signal, so the scarce resource (recorded
human choices) is only needed for the small policy head on top.

Data sources, cheapest first:

1. **Self-play transitions** — `magic-cabt-eval-play` with random/first
   agents through XMage generates legal trajectories at machine speed;
   an overnight CPU run yields hundreds of thousands of transitions.
2. **Arena mirror states** — ~800 snapshots per recorded match, human-quality
   trajectories (`mirror_states.jsonl`).
3. **Human decisions** — ~200 per match; reserved for the policy head.

`magic-cabt-build-transitions` folds sources 1–3 into one
`transitions.jsonl` (consecutive states within a game, with the acting
choice attached when the source is DecisionRecords).

## Target size: 10–30M parameters

Sized for the local machine (RTX 3070 Ti Laptop, 8 GB VRAM) and realistic
data volume, not for maximum capacity:

| Component | Shape | Params |
| --- | --- | --- |
| Object/state encoder | set-transformer over board objects, 4–6 layers, d_model 256–384 | 5–15M |
| Predictor | 2–3 layers, action-conditioned | 2–5M |
| Card identity | frozen text-embedding of card text + learned adapter | ~1M learned |
| **Total learned** | | **~10–20M** |

Notes:

- A learned embedding table over ~25k Arena card ids would cost ~6.4M params
  alone and generalize poorly to unseen cards; encoding card *text* with a
  frozen pretrained encoder is cheaper and compositional.
- 20M params ≈ 320 MB with AdamW state; a board is <100 objects, so batches
  of 128–256 in bf16 fit comfortably in 8 GB. Wall-clock is hours per run.
- EMA target encoder (BYOL/I-JEPA style) to avoid collapse; predict masked
  future latents at k ∈ {1, 4, 16} steps to cover priority-pass noise vs
  turn-scale dynamics.

## Data thresholds

- **< 100k transitions**: don't pretrain; the hashed-feature
  `OptionRanker` (`magic-cabt-train-ranker`, 0.23M params) is the right
  model and validates the exact state/option feature path the JEPA encoder
  later replaces.
- **~0.5–1M transitions** (one or two overnight self-play batches plus
  mirrored matches): 10–20M JEPA is worth training.
- **Policy head**: needs thousands of human decisions — ~50 recorded matches
  is a floor, ~500 comfortable. Keep mirroring.

## Invariants the encoder must respect

Carried over from docs/TARGET_FUNGIBILITY.md:

- Inputs are canonical: no per-game instance ids reach features
  (`features.canonical_text`); board presence enters as per-object rows, so
  two identical tokens are two *rows* (multiplicity preserved) that are
  *permutation-symmetric* (set encoder, no positional identity).
- Action conditioning uses canonical action groups
  (`training.action_dedup`), never raw option indices.
- Hidden zones stay hidden: encode opponent hand/library as counts only,
  exactly as the observation already does.

## Roadmap

1. **Done** — transition dataset builder (`magic-cabt-build-transitions`),
   canonical features, dedup view, `OptionRanker` baseline trainer.
2. Generate self-play volume: tournament runs writing replays, folded into
   `transitions.jsonl`.
3. Structured state tensorizer: per-object feature rows (card text embedding,
   controller, P/T, tapped, counters, zone) replacing hashed bag-of-words.
4. JEPA pretrain (encoder + predictor, EMA target) on transitions.
5. Policy head fine-tune on human decisions over canonical groups; compare
   against the `OptionRanker` and BC baselines on held-out games.
