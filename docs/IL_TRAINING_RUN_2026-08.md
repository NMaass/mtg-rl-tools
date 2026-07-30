# First imitation-learning run on the personal 17lands corpus — plan

## Decision context

The corpus decision from the [feasibility reanalysis](DATASET_FEASIBILITY_2026-07.md)
is made: the primary gameplay corpus is the owner's complete 17lands history,
obtained via Mythic-patron personal export (10k+ games). Public 17lands replay
aggregates are demoted to supporting roles — value/belief priors and
mulligan/draft/deck-build modeling — because per-turn aggregates lose phase,
order, and sequence, which matter for policy training. PD logs are demoted for
the same reason from the other side: hands are hidden. Both demotions are
recorded in the feasibility doc's addendum.

Scope note that shapes everything below: 17lands computes statistics for
**limited events only**, so this is a *limited-format* gameplay corpus (draft
and sealed, back to ~2021). Constructed remains future work on other sources.

## Step zero — pin down the export's fidelity before building anything

The single load-bearing unknown: whether the personal export is
**event-level** (the data behind the site's action-by-action replay viewer,
which proves event-level data exists server-side) or the **per-turn aggregate
schema** of the public replay files. Everything forks on this:

| Export turns out to be | Consequence |
| --- | --- |
| Event-level (GRE-derived actions) | Full-fidelity policy corpus. Ingestion is a normalizer in the shape of `arena_log.py` (which already maps GRE decision prompts to DecisionRecords via the mirror) — moderate, well-understood work. |
| Per-turn aggregates | The owner's own history has the same fidelity ceiling as public approach A. The historical corpus then serves value/belief/mulligan training, and the *policy* corpus is built going forward at full fidelity by running the already-implemented Arena mirror recorder during play. |

Actions: ask 17lands support for the replay-level JSON explicitly (the replay
viewer is the evidence it exists); regardless of the answer, **start the Arena
mirror recorder now** for all future play — the export covers the past at
whatever fidelity it has, the recorder covers the future at full fidelity, and
the two-tier corpus (large/lower-fidelity + growing/exact) is the realistic
shape either way. Both tiers carry `captureConfidence` so trainers and
manifests can weight or filter by tier.

## Corpus preparation

Standard gates, in repo terms, before any training claim:

1. `magic-cabt-validate` on every converted bundle; fail-closed on
   unresolved cards and unsupported callbacks.
2. `magic-cabt-training-audit` for perspective/leakage checks — especially
   important for converted third-party data.
3. `magic-cabt-build-manifest` inventory: games, root decisions, dedup,
   card/deck coverage, confidence distribution, hashes.
4. `magic-cabt-build-macro-actions` to compile strategic decisions.

Splits (per the research architecture: never split decisions within a game):

- **whole-game** train/val/test;
- **time split**: newest N months held out (metagame drift);
- **set holdout**: the most recent set entirely held out, to measure transfer
  to unseen cards — the card-generalization axis the benchmark stack already
  calls for;
- per-format strata (draft vs sealed; Bo1 vs Bo3).

At ~25–40 strategic decisions per limited game, 10k games ≈ 250–400k
decisions: this covers the 10k/30k/100k/300k points of the architecture doc's
scaling grid exactly. The scaling curve *is* the first experiment; it also
answers how much the next 10k games would be worth.

## The IL run itself — how to get the most out of it

Follow the M0→M2 ladder with the existing commands, then add leverage in this
order (each item is an ablation, not a default):

1. **Baselines first**: `first`/`random`/heuristic and the hashed ranker
   (M1) on the same splits — the floor every later claim is measured against.
2. **Structured BC (M2)** at 3 capacity points with matched controls;
   listwise loss over the canonical legal-option groups; fungible options
   share credit via the existing canonical grouping rather than label
   smoothing invented per-run.
3. **Auxiliary heads on the same forward pass** (shared-representation
   protocol from the architecture doc): terminal win probability, own next
   draw (deck is known — free supervision), opponent-next-action-class, and
   the causal-factor head from the parity notes. Multi-task on 10k games
   substitutes for data the corpus doesn't have; report the 4-way sharing
   ablation rather than assuming sharing helps.
4. **Skill/outcome conditioning, not filtering.** One player's corpus is
   consistent (good for BC — low multimodality) but includes their mistakes.
   Don't delete losses: condition on outcome (win/loss token) at train time
   and decode conditioned on "win" at eval (return-conditioning, Decision
   Transformer-style), and separately try advantage-weighted BC
   (AWR/AWAC-style, weights from the value head) as an ablation. Both beat
   hard filtering, which shrinks an already-small corpus and biases it
   toward games the opponent lost.
5. **Covariate shift is the known failure mode of BC** (DAgger's lesson):
   the cheap corrective here is engine-in-the-loop — roll the BC policy in
   XMage self-play, collect states it actually visits, and label a sample of
   drifted states with shallow `magic-cabt-replay-search` (the engine as the
   DAgger expert). This is Phase C's "DAgger-like querying" with search
   standing in for expert time.
6. **Calibration and per-slice reporting** from the benchmark stack
   (prompt type, option count, turn, archetype), because the BC policy's
   next job is proposal/prior for search — a miscalibrated prior poisons
   PUCT even when top-1 looks fine.
7. **Keep the BC checkpoint as the anchor.** Any later RL or search layer
   should be KL-regularized toward it (AlphaStar's league and
   Cicero/piKL's planning both rest on this); the BC run's value outlives
   its own win rate.

Cheap first target worth doing in week one: a **mulligan model**. Opening
hand, play/draw, and outcome are present even in aggregate-fidelity data;
the decision is isolated, high-value, and exercises the whole
ingest→validate→train→evaluate loop end to end before the main run.

## Causal-function upgrades (mtg-causal-rl parity gaps)

The parity notes list what's missing; the corpus decision lets several gaps
close with data rather than by hand:

1. **Reward profiles.** Implement the three declared views over
   DecisionRecords: sparse terminal; potential-shaped with a *learned*
   Φ; dense tactical. Φ = a calibrated win-probability model over the
   causal factors (and later the structured state), trained on personal
   states and on public per-turn aggregates — WP training only needs
   turn-level state snapshots plus outcome, which is exactly what the public
   replay files contain at millions-of-games scale. Ordering infidelity that
   disqualifies the public data for policy is harmless for Φ. Shaping stays
   policy-invariant in the potential form (Ng et al.) and is always reported
   as an ablation, per the architecture doc's rule.
2. **Interventional validation, not observational claims.** The repo's
   distinctive asset is exact do-operations: replay-search branch pairs
   differing in one action give causal effects on WP/outcome proxies.
   Use them to (a) validate each factor's sign (the missing
   "factor-target sign agreement" metric), (b) audit ΔΦ attribution on a
   sampled decision set, and (c) stamp factor values with the causal
   provenance classes the expert-cost protocol already defines.
3. **The dataset contains real randomization — use it.** Three natural
   experiments are built into Magic and identify causal effects from purely
   observational logs: the play/draw assignment (randomized seat effect),
   the opening hand dealt (randomized given decklist — identifies
   hand-feature effects and calibrates the mulligan model), and the card
   drawn each turn (randomized given the remaining library — same-state,
   different-topdeck comparisons identify card-in-context value). These
   anchor the observational corners of the SCM that engine interventions
   don't reach. Action choices themselves stay confounded (by hidden hands
   and skill) — that boundary should be stated wherever a factor effect is
   reported.
4. **Per-factor credit over time.** The existing `factor_credit_trace`
   gives per-decision deltas; add a return-decomposition alternative
   (RUDDER-style: predict terminal outcome from the trajectory, attribute
   per-step contributions) and audit both against replay-search exact
   counterfactuals on a small set. Whichever attribution survives the audit
   becomes the dense-tactical reward view.
5. **Factor ontology v2** (monotone orientation preserved): mana
   efficiency (untapped-vs-curve), evasion/threat-weighted board power,
   turns-to-lethal in both directions, legal-option richness. Fit weights
   with the monotone Bradley–Terry expert model *and* with the WP
   regression; disagreement between the two fits is itself a finding.

## Research base — imitation learning in long-horizon, hidden-information games

What each strand contributes to this run:

| Work | Takeaway for this project |
| --- | --- |
| AlphaGo SL policy (Silver et al. 2016) | BC on human games as the prior/proposal for search — the role this run's checkpoint is being trained for. |
| AlphaStar (Vinyals et al. 2019) | ~971k replays, conditioning on strategy statistics (*z*), and KL-to-human during RL. At 10k games, the transferable ideas are conditioning tokens and the KL anchor, not the scale. |
| Maia chess (McIlroy-Young et al. 2020) | Human-move prediction from modest per-band corpora; per-player fine-tuning from a population prior works. Argues: population prior (public data) → personal fine-tune (this corpus). |
| Suphx (Li et al. 2020) | Hidden information handled by oracle guiding: train with full info as a scaffold, anneal it away; global reward prediction to de-noise terminal variance. Both map directly (engine self-play provides the oracle; Φ is the reward predictor). |
| Cicero / piKL (FAIR 2022; Jacob et al. 2022) | Planning regularized toward a BC anchor beats both raw BC and unanchored search in a game where staying human-compatible matters. The strongest argument for treating this BC run as permanent infrastructure. |
| VPT (Baker et al. 2022) | Inverse-dynamics pseudo-labeling turns unlabeled demonstrations into training data. The pattern for later upgrading public/PD corpora using a small exact-labeled set. |
| ReBeL (Brown et al. 2020); Player of Games (Schmid et al. 2021) | Belief-state search is the principled endgame for imperfect info; the architecture doc already gates it behind a validated belief model — nothing here changes that ordering. |
| DeepNash (Perolat et al. 2022) | Model-free equilibrium learning at scale without search — the contrasting data point; not the near-term path here. |
| Decision Transformer (Chen et al. 2021); AWR/AWAC (Peng 2019; Nair 2020) | Outcome conditioning and advantage weighting: the two principled ways to learn from a corpus containing mistakes without discarding it. |
| RUDDER / Align-RUDDER (Arjona-Medina et al. 2019/2020) | Return decomposition for long-horizon credit — the observational complement to engine counterfactuals in the causal plan; Align-RUDDER specifically targets few-demonstration regimes. |
| DAgger (Ross et al. 2011) | Compounding error under distribution shift is BC's failure mode; querying the expert on visited states fixes it — here the engine's search is the queryable expert. |
| Stochastic MuZero (Antonoglou et al. 2022) | Learned models with chance nodes — the right template *if* amortizing engine search ever becomes the bottleneck (M5-C); not before. |

## Is there a world model in this data?

Yes — three of them, and none is a learned dynamics model:

1. **The engine is the dynamics model.** XMage already provides exact,
   legal, action-conditioned public transitions. Learning neural dynamics at
   10k games would reproduce a worse copy of an asset the repo owns; the
   architecture doc's M5-C framing (learn dynamics only to *amortize* search
   speed, gated on measured need) stands.
2. **The belief model is the world model worth training** (Layer 5): the
   part of the world the engine cannot simulate is the opponent's hidden
   state. Supervision comes from three places at three scales: engine
   self-play with oracle labels (exact, unlimited, synthetic); weak labels
   from human games (a card cast later was held earlier —
   positive-unlabeled learning over the public corpus); and hold/priors
   mined from public per-turn aggregates (what opponents at this rank, in
   these colors, hold given open mana — millions of games, unaffected by
   ordering infidelity). This is where the demoted public data does its
   best work.
3. **Φ, the calibrated value model, is the third** — turn-level state →
   win probability, trained across personal and public data as in the
   causal section. It is simultaneously the potential function, the
   advantage-weighting source, the reward-decomposition baseline, and the
   evaluation diagnostic.

The user's instinct about mulligan, draft, deck-build, and sideboard modeling
fits here with one sharpening: those decisions are **fully observable in the
public data** — draft picks (pack contents + pick), opening hands +
keep/mull, submitted decks are all public information with no ordering loss —
so they should train on the *public* millions, not spend the personal corpus.
The existing (parked) `build-draft-dataset`/`train-draft` code is the starting
point; promoting it to experimental status in service of belief/value context
is a recorded scope decision, not product work. These side models feed the
gameplay model as context (archetype priors, deck embeddings), which is the
architecture doc's Layer 8 exactly.

## Sequence

| When | What | Gate (from the architecture doc) |
| --- | --- | --- |
| Week 1 | Export request + fidelity verification; Arena recorder running on all play; mulligan model end-to-end | Phase A: runs replay, no leakage, CIs automatic |
| Weeks 2–4 | Ingest converter for whichever format arrives; corpus gates; M0/M1 baselines; scaling-curve points at 10k/30k/100k decisions | Baselines reproducible on manifests |
| Month 2 | Structured BC sweep + auxiliary heads + conditioning/weighting ablations; DAgger-lite engine correction pass | Phase C: held-out decision or gameplay improvement, not loss |
| Month 2–3 | Φ + reward profiles + intervention-validated factor signs; belief model v1 from self-play oracle + public priors | Phase B/C gates; sign-agreement metric reported |

Everything heavier (JEPA, RSSM, MuZero-style dynamics, PUCT leagues) stays
behind the scope audit's stop conditions: the scaling curve and these
benchmarks have to show a data- or representation-limited failure first.
