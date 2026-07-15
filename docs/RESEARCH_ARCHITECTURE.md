# MTG learning architecture and research plan

## Executive decision

The next useful step is **not** to choose one winning architecture. It is to make several architectural claims falsifiable on the same data and evaluation contract.

The recommended order is:

1. freeze a hidden-information-safe semantic data contract;
2. establish cheap imitation and heuristic baselines;
3. build fixed tactical, decision-ranking, belief, and paired-match benchmarks;
4. add expert preference/cost supervision as an auditable auxiliary signal;
5. expose cloned XMage search and distill its targets;
6. test JEPA, RSSM/Dreamer-style dynamics, and deck-local PUCT as separate hypotheses.

This ordering uses the strongest asset of `mtg-rl-tools`: a complete rules engine owns legality while Arena logs and self-play share one decision format. It also prevents a large representation experiment from becoming the only available explanation for every improvement or failure.

## Evidence labels used here

- **Established**: directly supported by a released implementation or a result in the cited work.
- **Supported design choice**: follows from the mechanics of the available data or from methods with evidence in related domains, but is not yet demonstrated for full MTG.
- **Hypothesis**: plausible and worth testing; no MTG-specific evidence should be implied.

Model widths, dataset thresholds, and training-time estimates in this repository are therefore experiment presets, not empirical laws.

## What the public approaches contribute

| Approach | Strongest contribution | What it demonstrates | Main boundary |
| --- | --- | --- | --- |
| `mtg-rl-tools` | XMage legality, canonical legal-option decisions, Arena log capture, common replay/data records | A path from real human MTGA actions and full-engine self-play into one training surface | Search throughput, benchmark coverage, and competitive baselines are still immature |
| MTG-Causal-RL | A compact Gymnasium benchmark with masked actions, partial observations, hand-specified SCM factors, reward variants, transfer tests, and a statistical protocol | Causal-factor diagnostics and paired evaluation can be first-class MTG benchmark outputs | The released environment intentionally covers a bounded card pool/action abstraction rather than the full rules and card possibility space |
| MageZero | Deck-local AlphaZero-style policy/value learning with XMage MCTS and reported self-play improvement over greedy opponents | Restricting the deck/matchup can make search-based RL tractable and can discover non-obvious lines | Fixed/local action heads, XMage search cost, hidden-information treatment, and external benchmark comparability remain open |
| Game-data/JEPA proposal | A broad case for game trajectories as temporally aligned action-consequence data and for a modular world model, actor, memory, and cost system | Games are a useful controlled source of action-conditioned trajectories | It is a research agenda, not evidence for the sample complexity or best model size of a structured MTG JEPA |

### MTG-Causal-RL: what to adopt

The 2026 paper specifies a 3,077-dimensional partial observation, a 478-action masked space, five Standard archetypes, three reward schemes, and a hand-authored SCM. Its strongest contribution for this repository is methodological:

- expose factor values and intervention predictions;
- compare scalar and factorized controls at matched capacity;
- use leave-one-archetype-out transfer;
- use paired seeds and scenario-level uncertainty;
- separate headline game outcome from calibration and factor diagnostics.

The paper is unusually candid about evidence strength: its full protocol calls for seven paired seeds and 300 evaluation episodes per cell, while the submitted broad sweep has two seeds and is explicitly described as exploratory. This is the correct standard for this project: preserve useful negative and mixed results, and do not convert a small run into a general claim.

### MageZero: what to adopt

MageZero's deck-local decomposition is practical. A universal MTG action vocabulary is not required when XMage provides the legal set at every decision, but constraining training to one deck or matchup still reduces state distribution and makes self-play/search much cheaper. Adopt:

- deck-local curricula and opponent pools;
- PUCT/search visit distributions as policy targets;
- value targets richer than terminal-only labels;
- staged progression from scripted opponents to checkpoint leagues;
- throughput and search-budget reporting.

Do not inherit fixed output heads as the cross-deck interface. `mtg-rl-tools` should continue to score the **current semantic legal-option set**. This retains transfer between cards/actions and avoids allocating a permanent logit to every possible engine callback.

### JEPA: what is and is not established

I-JEPA and V-JEPA demonstrate that latent prediction can learn transferable visual representations. V-JEPA 2 pretraining uses internet-scale video and later action-conditioned robot data. That scale does not transfer directly to a symbolic MTG state model: the input entropy, invariances, supervision density, and downstream task are different.

For MTG, JEPA is best treated as this narrower hypothesis:

> Action-conditioned prediction of future **public-state representations**, anchored by exact engine-derived auxiliary labels, will produce a state encoder that transfers better to held-out action ranking, value estimation, belief prediction, or search than the same encoder trained only by imitation.

JEPA loss improving by itself is not sufficient evidence. The experiment needs frozen-encoder probes and downstream comparisons against the same structured encoder without JEPA pretraining.

## System decomposition

Do not bind state representation, learning target, and planning method into one model name. Treat them as independent axes.

```text
XMage / Arena logs
      |
      v
canonical perspective record
(public state + own private state + event history + semantic legal actions)
      |
      +--> card/rules encoder
      +--> object/set state encoder
      +--> event-memory encoder
      +--> belief encoder
      |
      v
shared information-state representation
      |
      +--> legal-option policy head
      +--> terminal/value head
      +--> belief/calibration head
      +--> expert-factor/cost head
      +--> future-latent or RSSM dynamics head
      +--> draft/deck/sideboard selection head
      |
      +--> optional XMage search / learned-model search
```

The shared representation is an empirical choice, not an article of faith. Every multi-task result should include:

1. independent task encoders;
2. shared frozen card encoder only;
3. shared card and state encoder;
4. shared encoder with task adapters.

This directly tests the intuition that a draft model benefits from gameplay knowledge without assuming that draft, deck construction, sideboarding, and game play are one homogeneous process.

## Canonical data layers

### Layer 0: static card and rules semantics

Store exact structured fields and text-derived semantics separately:

- canonical card identity and face identity;
- mana cost, colors, types, supertypes, subtypes;
- power/toughness/loyalty/defense where applicable;
- keyword and ability structure when the engine exposes it;
- oracle/rules text embedding;
- format, set, and rebalancing provenance.

A text embedding is useful for transfer to unseen cards but must not replace exact fields. “Any target,” “target creature,” damage, destroy, sacrifice, and exile are strategically and legally distinct even when their language embeddings are close.

### Layer 1: public state

Use one perspective-stable representation of:

- turn, step, active player, priority holder;
- life and public counters;
- battlefield objects, attachments, counters, damage, tapped state;
- stack objects, modes, targets, payments, and ordering;
- public graveyard/exile/command information;
- hand/library counts;
- public continuous/replacement effects and delayed triggers where available.

Objects form a multiset: identical tokens are separate rows, but row order and per-game object IDs are not semantic features.

### Layer 2: acting player's private state

Include only information legally available to the acting player:

- own hand and privately viewed cards;
- known top/bottom/revealed library information with expiry/provenance;
- face-down objects whose identity is known to that player;
- sideboard/companion information allowed by the event.

Every record needs `perspectiveSeat`, visibility provenance, and a capture-confidence field.

### Layer 3: semantic action

An option index is transport, never meaning. A semantic action should identify:

- prompt/callback type;
- pass/cast/activate/play/special-action class;
- source card or ability by canonical identity;
- origin zone;
- selected modes and alternative/additional costs;
- mana/payment semantics;
- target roles and canonical target groups;
- amount distributions;
- ordering and yes/no semantics.

Canonical action groups make fungible choices explicit. “Bolt either copy of the same 3/3 token” may be one strategic group while preserving the exact object selection for engine replay.

### Layer 4: transition and event history

Record both decision-to-decision and engine-event transitions:

- previous public/perspective state;
- semantic action;
- immediate deterministic consequences;
- intervening opponent/chance events;
- next decision state;
- horizon definition;
- exact deltas and terminal outcome.

Decision-count horizons are not equivalent to engine-event or turn horizons. Store the unit.

### Layer 5: belief state

A policy must not receive the true opponent hand. A separately supervised belief model may use true hidden state as a training label for engine self-play, provided that its **inputs** contain only information available at that point.

Useful outputs include:

- per-card or card-class probability in opponent hand;
- remaining deck/archetype posterior;
- probability of strategically defined holdings, such as removal, counterspell, combat trick, or sweeper;
- calibrated particle/belief samples for search.

Belief quality requires log loss, Brier score, calibration curves, and decision-conditioned value—not only card-recovery accuracy.

### Layer 6: outcome and search supervision

Keep distinct labels for:

- terminal game/match outcome;
- bootstrapped value target;
- engine-search root value;
- visit distribution over canonical actions;
- rollout horizon and simulation budget;
- opponent/search policy;
- determinization or belief sample used.

This makes it possible to compare human imitation, search distillation, and self-play without silently merging their targets.

### Layer 7: expert factors and preferences

Experts should annotate comparisons between legal candidate consequences, not assign an unexplained scalar to a card name. Each annotation records:

- exact decision context and perspective;
- candidate actions and consequence horizon;
- factor vectors and their provenance;
- preferred candidate or tie;
- confidence, rationale, assumptions, tags, and expert identity;
- whether consequences are exact engine interventions, rollouts, learned predictions, or human estimates.

The implementation in `magic_cabt.research.expert_cost` fits a monotone Bradley-Terry preference model over a versioned factor ontology and evaluates it on held-out contexts.

### Layer 8: limited and metagame context

Draft, deck construction, sideboarding, and gameplay should share card semantics but retain task-specific context:

- draft: pack, pool, pick/pack number, observed colors/signals, event/set;
- deck build: pool, selected deck, basic lands, curve, constraints;
- sideboard: submitted main/sideboard, opponent evidence, game number, play/draw;
- gameplay: information state and legal action set;
- metagame: time-bounded archetype and matchup priors.

Metagame data must be time-split. A model should not be credited for predicting an old draft or sideboard decision using statistics published after that event.

## Model portfolio

### M0: nonlearned controls

- random legal;
- first legal;
- heuristic policy;
- simple material/tempo evaluator;
- human action-frequency baseline.

These expose action ordering bugs and benchmark difficulty.

### M1: hashed legal-option ranker

Use the existing small ranker as the minimum learned baseline. It validates ingestion, canonical grouping, game-level splits, scoring, and checkpoint analysis. It should remain in every experiment family even after larger models exist.

### M2: structured set-transformer behavior cloning

Use the same object tensorizer intended for world modeling, but train only legal-option imitation and optional terminal value. This is the essential control for a JEPA claim: if structured representation alone explains the gain, latent prediction is not the cause.

Suggested sweep—not a data requirement:

- 2–4M parameters;
- 8–12M parameters;
- 20–30M parameters.

Report train/holdout curves against unique games, unique decisions, wall-clock, and tokens/objects processed.

### M3: recurrent information-state behavior cloning

Add event history or recurrent memory plus a belief head. This tests whether hidden-information memory is the missing ingredient before introducing a learned dynamics model.

Ablate:

- current state only;
- public event history;
- history plus calibrated beliefs;
- history plus true hidden information **as an oracle-only upper bound**, never as a deployable result.

### M4: search distillation with the exact engine

A cloned-game API is the highest-leverage engine feature. Use XMage to branch legal candidates, then store root values and visit distributions. Start with shallow tactical scenarios and deterministic/public consequences before attempting broad MCTS.

Search can supervise:

- policy ranking;
- state value;
- tactical consequence heads;
- hard-negative expert comparisons;
- counterfactual JEPA/RSSM transitions.

Because full XMage search is expensive, use the learned policy for proposal/pruning and batch neural evaluations. Always report simulations, nodes, wall-clock, heap use, and search failures.

### M5-A: structured action-conditioned JEPA

Use an online encoder, EMA target encoder, semantic action encoder, horizon token, and stochastic/multimodal predictor. Exact public deltas provide auxiliary anchors.

Required diagnostics:

- latent per-dimension variance and covariance/effective rank;
- predictor-target cosine/L1 loss by horizon;
- public life/zone/counter delta accuracy;
- deterministic-rule versus stochastic-event errors;
- frozen linear probes;
- held-out policy/value transfer;
- model-size and dataset-size scaling curves.

Do not infer “world understanding” from one low latent loss. Hidden draws and opponent responses make the future multimodal; a deterministic squared-error predictor can learn an average latent that is useless for planning.

### M5-B: RSSM/Dreamer-style latent dynamics

Run an RSSM or Dreamer-style alternative on the same state/action tensors. It explicitly represents stochastic latent dynamics and provides imagined rollouts. This is a stronger direct comparator for control than JEPA alone.

Compare at matched encoder width and training examples:

- one-step and multi-step public prediction;
- value/policy transfer;
- imagined-rollout consistency;
- compute and memory;
- calibration under hidden information.

### M5-C: MuZero-style value-equivalent dynamics

MuZero predicts reward, policy, and value quantities needed for planning rather than reconstructing the full environment. In this repository, exact XMage remains available, so the value of MuZero-style dynamics is **amortized fast search**, not replacing rules correctness.

A useful experiment is to distill exact shallow XMage search into a learned latent dynamics/search model and measure value error as horizon grows.

### M6: deck-local PUCT/self-play

Implement a MageZero-comparable arm using the semantic legal-option scorer. Restrict by deck and opponent pool first. Train against checkpoint leagues and fixed scripted anchors; then test cross-matchup initialization and shared card-encoder transfer.

This arm answers a different question from JEPA:

> Can exact-engine search plus self-play improve a deck-local policy under a practical simulation budget?

It should not be used as evidence for universal cross-card generalization unless evaluated on held-out decks/cards.

## Hidden-information strategy

Perfect-information MCTS on a sampled opponent hand is vulnerable to strategy fusion and determinization bias. The progression should be:

1. **Policy-only information state**: no hidden-state sampling.
2. **Belief-conditioned policy/value**: recurrent public history and calibrated beliefs.
3. **Root determinization baseline**: sample hidden states from the belief model; report exploitability-like pathologies and sensitivity.
4. **Re-determinizing/information-set search**: prevent a player from conditioning on hidden information unavailable at their decision nodes.
5. **Public-belief-state methods**: investigate ReBeL-like search only after the compact benchmark and belief model are validated.

ReBeL is evidence that search and learning can be combined in two-player zero-sum imperfect-information games; it is not a drop-in solution for full MTG. MTG's chance events, enormous card/state space, variable legal actions, and long callback sequences make the public belief state difficult to represent.

## Expert knowledge as a cost/preference model

### Do not start from a dense scalar reward

An ad hoc score such as

```text
3 * card_advantage + 2 * board_power + life_difference
```

will be brittle, matchup-dependent, and easy to exploit. It also confuses three separate objects:

- an expert's strategic preference;
- an SCM/intervention effect;
- the environment's terminal objective.

The implementation therefore learns an interpretable preference model over declared factors. Higher oriented factor values are always better and factor weights are constrained non-negative. This prevents a small dataset from silently learning that “more lethal risk is good,” while still allowing unsupported factors to shrink toward zero.

### Lightning Bolt and Abrade example

The stable concept is not the card name. It is the consequence vector.

“Bolt the 3/3” can be represented by:

- threat value removed;
- tempo gained;
- expected life preserved;
- card/mana spent;
- face reach sacrificed;
- future legal options retained;
- lethal risk after the line.

“Bolt face” has a different vector. At three opponent life, terminal probability dominates. Outside a lethal/race context, removing the 3/3 may dominate.

Abrade and Lightning Strike can have identical immediate damage but different future coverage. Which spell to spend depends on whether artifact coverage or face reach is more valuable in the matchup and state. That difference belongs in `specialized_answer_coverage_retained`, `face_reach_retained`, and `future_option_value_delta`, not in a rule saying “Abrade is worth X.”

### Safe deployment roles

Use an expert model as one or more of:

- an auxiliary prediction head;
- an offline replay annotation;
- a reranker for shallow/tied search lines;
- a source of active-learning queries;
- a potential function `gamma * Phi(next) - Phi(current)` tested against terminal-only reward;
- an evaluation diagnostic.

Do not replace win/loss with the expert score. Potential shaping should be reported as an ablation because partial observability, approximation error, and finite horizons can still change practical learning behavior even when the ideal theoretical form is policy-preserving.

## Benchmark stack

### 1. Data and legality integrity

Report:

- parser/capture confidence;
- unresolved cards and unsupported callbacks;
- illegal or fail-closed selections;
- replay-to-engine state agreement;
- hidden-information leakage checks;
- instance-ID feature checks;
- dataset hashes/manifests.

A model result is not valid if its action surface or state reconstruction silently omits difficult decisions.

### 2. Held-out human decision ranking

Split whole games, matches, sessions, drafts, or time blocks. Never split individual decisions from one game.

Metrics:

- canonical-group top-1/top-3;
- exact selected-index top-k;
- MRR and negative log likelihood;
- coverage and discard reasons;
- prompt type, option-count, archetype, turn, and source slices;
- expert disagreement ceiling;
- clustered confidence intervals by game/session.

Human action agreement is not identical to optimal play. It is evidence of imitation and representation quality, not final strength.

### 3. Tactical scenario suite

Build versioned engine states covering:

- lethal and avoid-lethal;
- removal target selection;
- modal spell and payment choices;
- stack/counterspell sequencing;
- attacks, blocks, tricks, and damage assignment;
- replacement effects and triggers;
- card advantage versus tempo;
- role assignment and racing;
- information probes;
- hidden-information branches.

For each scenario store legal canonical groups, acceptable actions, expert rationales, exact short-horizon consequences, and adversarial near-misses. Measure acceptable-action rate and regret relative to engine search/expert consensus.

### 4. Paired gameplay benchmark

Use fixed deck manifests, opponent versions, seeds, seats, play/draw, mulligan seed, and search budget. Rotate seats and pair the same scenario across agents.

Report:

- game and match score;
- confidence intervals clustered by seed/scenario;
- paired differences;
- Holm correction within pre-registered comparison families;
- throughput and failure rate;
- results by matchup and seat, not only pooled win rate.

Five training seeds are a useful minimum gate for a benchmark run, not a guarantee of precision. Increase runs when intervals remain decision-relevant.

### 5. Hidden-information benchmark

Evaluate:

- hand/card-class log loss;
- Brier score and reliability diagrams;
- archetype posterior accuracy over time;
- value of belief information in policy ablations;
- performance under belief perturbation;
- oracle-hidden-state upper bound clearly labeled as nondeployable.

### 6. World-model benchmark

At horizons 1, 4, 16, and turn-scale:

- exact public variable delta error;
- zone membership and object-count error;
- terminal and reward prediction;
- latent variance/effective rank;
- action sensitivity and counterfactual ordering;
- stochastic calibration/coverage;
- downstream policy/value probe quality;
- search value consistency.

Use game-level and card-set holdouts. A model that memorizes card IDs but fails on new cards should not be credited with semantic transfer.

### 7. Draft, deck construction, and sideboarding

Draft metrics:

- held-out pick top-k/NLL;
- regret against later expert review;
- pool/deck consistency;
- set/time holdout;
- downstream deck and match outcome only as a noisy secondary metric.

Deck construction:

- inclusion ranking;
- land/color/curve constraint violations;
- exact deck distance and expert acceptable-set agreement;
- gameplay value under a fixed pilot.

Sideboarding:

- card-in/card-out pair accuracy;
- submitted-deck likelihood;
- matchup/time holdout;
- post-board paired match score.

## Scaling experiments instead of guessed thresholds

For every model family, construct learning curves over both **unique games** and **training examples**. Suggested grid:

- data: 10k, 30k, 100k, 300k, 1M, 3M transitions/decisions where available;
- model: small, medium, large presets with roughly 4× parameter spacing;
- source mixtures: random self-play, heuristic self-play, human trajectories, search branches;
- card coverage: in-card, held-out printing, held-out card, held-out set;
- horizon: immediate, decision-scale, turn-scale.

Predefine promotion gates. For example, a larger JEPA proceeds to search integration only if it beats the matched structured-BC encoder on at least one held-out downstream metric without regressing collapse diagnostics or throughput beyond the declared budget.

## Recommended implementation sequence

### Phase A — benchmark first

- finish canonical action coverage and dataset validation;
- add tactical scenario manifests;
- generate checkpoint-aware analysis caches;
- use `magic-cabt-research benchmark-analysis` and `benchmark-matches`;
- establish paired heuristic, ranker, and current structured-JEPA reports.

**Gate:** all runs replay, no hidden leakage, and confidence intervals are produced automatically.

### Phase B — expert preference proof of method

- freeze factor ontology v1;
- annotate 100–300 deliberately diverse decision contexts;
- double-annotate at least 20%; resolve low-agreement factor definitions;
- fit the monotone model and evaluate by held-out context, expert, archetype, and time;
- compare against equal weights and simple hand weights.

**Gate:** held-out pairwise accuracy/calibration exceeds simple controls and explanations are strategically coherent under review.

### Phase C — structured imitation and beliefs

- train hashed and structured BC at matched splits;
- add public history and belief supervision;
- run DAgger-like expert querying only on states the learned policy actually visits, if expert time permits.

**Gate:** structured/recurrent models improve held-out decisions or fixed gameplay, not only training loss.

### Phase D — cloned XMage search

- implement branch lifecycle and deterministic scenario tests;
- produce shallow root targets;
- benchmark throughput and replay identity;
- distill policy/value and compare with human BC.

**Gate:** search improves tactical acceptable-action rate and paired match score within a usable compute budget.

### Phase E — world-model portfolio

- run matched structured BC, JEPA, and RSSM/Dreamer-style models;
- scale data/model size only after small-run diagnostics pass;
- add counterfactual search transitions;
- test whether learned dynamics can prune/amortize exact search.

**Gate:** downstream or search benefit, not latent loss alone.

### Phase F — self-play leagues and cross-task transfer

- deck-local PUCT leagues;
- shared versus independent card/state encoders for gameplay and limited tasks;
- time/card/deck holdouts;
- publish manifests and negative results.

## Claim checklist

Before describing a result as an improvement, require:

- same legal-action and observation contract;
- same train/holdout split and no duplicated game across partitions;
- same card/deck/time availability;
- paired evaluation seeds/scenarios where possible;
- at least one cheap baseline and one capacity-matched control;
- uncertainty interval and comparison-family correction;
- failure/coverage/throughput report;
- predeclared primary metric;
- model and dataset hashes;
- no true opponent hidden state at inference;
- no claim that a strategic factor is causal unless its intervention semantics are identified.

## References

- da Costa Cunha, C., Mian, A., French, T., and Liu, W. *Causal Reinforcement Learning for Complex Card Games: A Magic The Gathering Benchmark*. arXiv:2605.06066, 2026. https://arxiv.org/abs/2605.06066
- Wroble, W. *MageZero: A Deck-Local AI Framework for Magic: The Gathering*. https://github.com/WillWroble/MageZero
- Silver, D. et al. *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*. Science, 2018.
- Schrittwieser, J. et al. *Mastering Atari, Go, chess and shogi by planning with a learned model*. arXiv:1911.08265.
- Hafner, D. et al. *Mastering Diverse Domains through World Models*. arXiv:2301.04104.
- Brown, N. et al. *Combining Deep Reinforcement Learning and Search for Imperfect-Information Games*. arXiv:2007.13544.
- Ross, S., Gordon, G., and Bagnell, D. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS, 2011.
- Ng, A., Harada, D., and Russell, S. *Policy invariance under reward transformations: theory and application to reward shaping*. ICML, 1999.
- Assran, M. et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. arXiv:2301.08243.
- Assran, M. et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*. arXiv:2506.09985.
- Agarwal, R. et al. *Deep Reinforcement Learning at the Edge of the Statistical Precipice*. NeurIPS, 2021.
- Bebbington, P. et al. *Game-Generated Data: An Untapped Resource for Advanced AI Training*. arXiv:2504.16591.
- 17Lands public datasets. https://www.17lands.com/public_datasets
