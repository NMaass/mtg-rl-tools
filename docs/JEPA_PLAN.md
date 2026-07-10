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

Two caveats keep this honest:

- **Actions are recorded semantically, never positionally.** A transition's
  `action.selectedOptions` holds the full chosen option dicts (payloads with
  `canonicalKey` and target references); indices are provenance only. An
  option index has no stable meaning across states, so an action-conditioned
  predictor must never condition on it.
- **Random self-play teaches rules, not strategy.** It supplies legality,
  zone/stack/life dynamics, and transition volume — not sequencing, card
  importance, or lethal recognition. The trajectory curriculum should climb:
  random → heuristic/first → option ranker → BC agents → self-play against
  checkpoints → counterfactual branching once clone-based search exists.

## Target size: 10–30M parameters

Sized for the local machine (RTX 3070 Ti Laptop, 8 GB VRAM) and realistic
data volume, not for maximum capacity:

| Component | Shape | Params |
| --- | --- | --- |
| Object/state encoder | set-transformer over board objects, 4–6 layers, d_model 256–384 | 5–15M |
| Predictor | 2–3 layers, action-conditioned | 2–5M |
| Card identity | frozen text-embedding of card text + learned adapter | ~1M learned |
| **Total learned** | | **~10–20M** |

The first structured JEPA should sit at the bottom of that range —
**10–16M learned params** (5-layer width-256 set transformer, 2-layer action
encoder, 3-layer predictor, small heads) — and grow toward 20–30M only with
event history, belief tokens, or millions of diverse transitions in hand.

Notes:

- Card identity is a composite, not a choice: frozen text embedding (for
  transfer to unseen cards) + exact structured fields (types, P/T, mana
  cost, modes — a language encoder cannot be trusted to preserve "any
  target" vs "target creature" or damage vs destroy) + a **small** learned
  identity embedding (32–64d over ~25k cards is under 2M params and lets the
  model memorize strategically load-bearing facts). What to avoid is only
  making a *large* ID table the primary representation.
- 20M params ≈ 320 MB with AdamW state; a board is <100 objects, so batches
  of 128–256 in bf16 fit comfortably in 8 GB. Wall-clock is hours per run.
- EMA target encoder (BYOL/I-JEPA style) to avoid collapse; predict masked
  future latents at k ∈ {1, 4, 16} steps to cover priority-pass noise vs
  turn-scale dynamics.
- **The predictor must be stochastic.** A deterministic squared-error
  predictor averages over draws, hidden opponent hands, and opponent
  responses — exactly where Magic is most interesting. Use discrete latent
  codes, a mixture head, or particle-conditioned predictions; keep a
  deterministic branch only for public rules-driven consequences.
- **Anchor the latent with exact auxiliary heads.** The engine provides
  exact per-transition labels for free (per-player life delta, terminal
  flag, zone changes, counters/damage marked). Supervising small heads on
  these forces the representation to carry the concepts that matter instead
  of hoping a generic latent loss discovers them. `build_transitions`
  already emits `deltas` (life, terminal) for this.

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
