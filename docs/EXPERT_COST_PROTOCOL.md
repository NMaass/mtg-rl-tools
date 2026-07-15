# Expert preference and causal-cost protocol

## Purpose

This protocol turns expert MTG judgment into a versioned, testable model without pretending that an expert can directly specify one universally correct dense reward.

The unit of supervision is a **comparison between two legal candidate lines in one information state**. Each line is summarized by declared consequence factors. Experts choose left, right, or tie; provide confidence, rationale, and assumptions; and may disagree. A monotone Bradley-Terry model learns non-negative weights over factors that have already been oriented so larger means better.

The output is an expert-preference utility model. Calling it a **causal cost** is justified only when the candidate factor values have causal provenance, such as:

- an XMage intervention/branch from the same state;
- an exact deterministic consequence produced by the rules engine;
- a rollout under an explicitly declared opponent/chance policy;
- an identified causal design with justified assumptions.

A factor estimated from an observational Arena trajectory, a language model, or an expert guess is a tactical/strategic prediction, not automatically a causal effect.

## Non-goals

This system does not:

- replace terminal win/loss reward;
- prove that the factor graph is causally identified;
- turn printed card text into a context-free card score;
- declare one expert's annotations optimal;
- resolve hidden information by reading the true opponent hand;
- certify a learned world model merely because its latent loss decreases.

## Core objects

### Factor ontology

A factor ontology defines stable names, orientation, scaling, bounds, defaults, units, and meaning. It is versioned independently of annotations and models.

```json
{
  "schemaVersion": 1,
  "kind": "expert-factor-ontology-v1",
  "name": "mtg-tactical-preference-factors-v1",
  "factors": [
    {
      "name": "opponent_threat_value_removed",
      "direction": "higher_better",
      "transform": "identity",
      "scale": 3.0,
      "minimum": 0.0,
      "maximum": 20.0,
      "default": 0.0,
      "group": "board",
      "unit": "expert points",
      "description": "Contextual value of opposing threats neutralized."
    }
  ]
}
```

`scale` is for numerical conditioning. It is not the strategic coefficient; strategic weights are learned. Bounds should catch annotation or prediction errors rather than compress every context to the same range.

### Candidate consequence vector

Each candidate represents one canonical legal action group or a declared multi-action line. The factor vector describes consequences over a stated horizon.

```json
{
  "candidateId": "bolt-creature",
  "label": "Lightning Bolt the opposing 3/3",
  "factors": {
    "opponent_threat_value_removed": 3.0,
    "tempo_delta": 1.0,
    "lethal_risk": 0.05
  },
  "metadata": {
    "actionCanonicalKey": "cast:lightning-bolt|target:creature-role-0",
    "factorSource": "xmage-branch-plus-expert-valuation",
    "horizon": "next-combat"
  }
}
```

Omitted factors use ontology defaults. Unknown factors fail closed.

### Preference annotation

One JSONL row records an exact pair. `contextId` should identify the information state **and this candidate pair**. Use the same `contextId` for redundant ratings of the same oriented pair; use a different ID for another pair in the same game state.

```json
{
  "contextId": "bolt-3-3-main-phase-v1",
  "expertId": "expert-01",
  "left": {"candidateId": "bolt-creature", "factors": {}},
  "right": {"candidateId": "bolt-face", "factors": {}},
  "preferred": "left",
  "confidence": 0.9,
  "weight": 1.0,
  "rationale": "The creature is the clock and face damage is not part of lethal.",
  "assumptions": ["No prevention effect is active"],
  "tags": ["removal", "threat-assessment"],
  "metadata": {"source": "expert-review-v1"}
}
```

A tie is a valid label. Low confidence is preferable to inventing certainty. Confidence zero retains a row for audit but excludes it from fitting.

## Factor design

### Separate measurement from preference

A good factor has two parts:

1. **measurement rule**: how the value is produced;
2. **preference orientation**: whether larger or smaller is generally better.

For example:

- `opponent_life_loss` can be exact over an immediate deterministic branch;
- `opponent_threat_value_removed` contains expert or learned contextual valuation;
- `expected_win_probability_delta` is a model estimate and must name the checkpoint;
- `lethal_risk` is a belief- and opponent-policy-dependent prediction;
- `rules_or_model_uncertainty` records that the preceding estimates are unreliable.

Do not merge uncertainty into the value itself. Preserve both value and provenance.

### Recommended factor families

#### Outcome

- expected game or match win-probability change;
- terminal win/loss indicator when reached;
- probability of losing before the next decision horizon.

#### Board and material

- threat value removed or added;
- own material delta;
- card advantage delta;
- board pressure/control change;
- permanent quality, resilience, or engine value.

#### Tempo

- mana efficiency;
- initiative/attack quality;
- time-to-stabilize;
- opponent mana denied;
- development delayed.

#### Survival and pressure

- own life buffer change;
- opponent life loss;
- clock length;
- lethal setup or prevention.

#### Optionality

- future legal-option value;
- flexible/specialized answer coverage retained;
- face reach retained;
- modal spell flexibility;
- irreversible commitment.

#### Information

- information gained by probing, revealing, or forcing a response;
- belief entropy reduced;
- information leaked to the opponent, if operationally modeled.

#### Risk and provenance

- outcome variance;
- lethal risk;
- sensitivity to one hidden card class;
- rules-engine/capture/model uncertainty;
- rollout-policy sensitivity.

### Avoid redundant aliases

`board_advantage`, `material_advantage`, `creature_advantage`, and `board_power` can encode the same observation four times and make learned weights uninterpretable. Start with the smallest ontology that experts can apply consistently. Add a factor only when:

- experts can define it independently;
- it changes preferences in contexts where existing factors do not;
- its measurement/provenance can be audited;
- held-out evidence supports the addition.

## Annotation workflow

### Step 1: select states without outcome leakage

Sample whole decision contexts from:

- held-out Arena games;
- fixed tactical scenarios;
- XMage self-play;
- states visited by a learned policy;
- disagreement between policy, expert, and search;
- uncertainty or poor calibration slices.

Do not show the annotator the actual later outcome unless the annotation task explicitly asks for retrospective review and records that fact. Otherwise hindsight leaks chance/opponent events into the target.

### Step 2: freeze perspective and legal choices

Record:

- perspective seat;
- public state;
- acting player's legal private information;
- event history made available to the model;
- canonical legal action groups;
- exact source version and state hash.

Never display the true opponent hand as ordinary context. An oracle review may use it only as a separately labeled analysis mode.

### Step 3: generate candidate consequences

Use a hierarchy of consequence sources:

1. exact deterministic XMage branch;
2. short exact branch plus declared opponent/chance responses;
3. multiple engine rollouts under a fixed policy/belief sampler;
4. learned world-model prediction with checkpoint and calibration metadata;
5. expert estimate.

Store the source for each factor. A single candidate may combine exact life loss, search-estimated win probability, and expert-valued threat removal.

### Step 4: choose pairs deliberately

All-pairs annotation is wasteful. Prefer:

- model top choice versus human action;
- model top two near a decision boundary;
- search top choice versus human action;
- two cards/lines with identical immediate effect but different optionality;
- a high-value factor conflict, such as tempo versus card advantage;
- active-learning pairs with maximum predictive entropy;
- pairs on which experts disagree.

Include easy controls—lethal, illegal-lookalike exclusions, and obvious equivalences—to detect annotation UI or schema errors.

### Step 5: collect independent judgments

Recommended proof-of-method design:

- 100–300 unique pairs across at least 50 states;
- at least 20% rated independently by two or more experts;
- randomized left/right presentation;
- confidence and rationale required for low-agreement/high-impact pairs;
- pair IDs stable across annotation rounds;
- expert identity pseudonymous but persistent;
- annotation time and ontology version recorded.

These are initial operational targets, not sample-complexity guarantees.

### Step 6: audit agreement before fitting

Agreement failures usually indicate one of:

- ambiguous horizon;
- missing hidden-information assumptions;
- factors with overlapping definitions;
- different beliefs about decklists/metagame;
- role-assignment disagreement;
- genuine strategically mixed choices.

Do not erase disagreement by majority vote. Keep individual labels and report:

- majority agreement by pair;
- pairwise expert agreement;
- performance by expert;
- disagreement by tag/factor family;
- rationales and assumptions;
- model uncertainty on disputed versus undisputed pairs.

### Step 7: split by context and expert

Never split annotations of one pair between train and holdout. Use at least:

- context holdout;
- expert holdout where multiple experts exist;
- matchup/archetype holdout;
- time-block holdout;
- card/set holdout for semantic-transfer claims.

A random row split can place near-identical annotations from the same state on both sides and overstate generalization.

### Step 8: fit simple models first

The supplied model is deliberately small:

```text
oriented_factor_i = transform(clamp(raw_i)) / scale_i
utility(candidate) = sum_i nonnegative_weight_i * oriented_factor_i
P(left preferred) = sigmoid((utility(left)-utility(right))/temperature)
cost(candidate) = -utility(candidate)
```

Non-negative weights enforce the ontology's monotonic directions. This improves auditability but does not capture every interaction. Compare against:

- equal weights;
- hand-set weights frozen before holdout evaluation;
- unconstrained logistic/Bradley-Terry model;
- monotone model with pairwise factor interactions;
- small context-conditioned network.

Only add capacity when held-out residuals show a specific missing interaction.

### Step 9: evaluate before deployment

Required preference metrics:

- pairwise accuracy excluding ties;
- weighted log loss;
- Brier score;
- preferred utility margin;
- calibration by probability bucket;
- metrics by expert, tag, matchup, horizon, and factor source;
- disagreement-conditioned metrics;
- coverage and parse failures.

For a cost model used in policy learning, also run:

- terminal-only versus shaped reward;
- equal-weight potential versus learned potential;
- auxiliary-only versus shaping;
- shuffled factor labels;
- matched-capacity scalar auxiliary control;
- reward-hacking/adversarial scenario tests;
- paired gameplay evaluation.

## Lightning Bolt, Abrade, and functional equivalence

### Lightning Bolt on a 3/3

The expert judgment should not be encoded as:

```text
Lightning Bolt prefers creature named X.
```

It should be encoded as a conflict between consequences:

| Factor | Bolt 3/3 | Bolt face |
| --- | ---: | ---: |
| threat value removed | high | zero |
| opponent life loss | zero | 3 |
| expected own life preserved | context-dependent | zero |
| tempo | context-dependent | usually lower without lethal pressure |
| face reach retained | zero after either cast | zero |
| lethal probability | state-dependent | state-dependent |

At opponent life three, face is terminal. At opponent life fourteen with the 3/3 as the only clock, removing it may dominate. The same card/action template yields different preference labels because the state and consequence factors differ.

### Abrade versus Lightning Strike

The immediate branch can be identical when both deal three to a creature. The strategic branch differs:

- using Abrade can preserve Lightning Strike's face/planeswalker reach;
- using Lightning Strike can preserve Abrade's artifact coverage;
- the correct choice depends on revealed cards, matchup priors, hand texture, life totals, and future mana;
- uncertainty about those beliefs should be recorded, not hidden in a card-name rule.

This is a direct test for whether the factor model represents functional equivalence and contextual exceptions.

## Causal interpretation

### Exact intervention

If XMage clones the same state and applies action A versus B while holding a deterministic response script/seed fixed, the immediate difference is an intervention under that simulator policy. Record:

- cloned state hash;
- action semantics;
- random seed/chance stream;
- response policy;
- horizon;
- variables measured;
- branch failures.

### Rollout-estimated effect

With stochastic draws/opponent policies, estimate a distribution rather than one scalar. Prefer common random numbers/paired seeds between A and B. Store:

- number of rollouts;
- mean, variance, and quantiles;
- belief model/checkpoint;
- opponent policy/checkpoint;
- sensitivity across policies and beliefs.

The effect is conditional on those choices.

### Observational association

A factor derived from “players who made this action won more often” is confounded by state quality and player skill. It may be a useful predictive feature, but it should be labeled observational. Do not call it a do-effect without an identification argument.

### Learned world-model counterfactual

A JEPA/RSSM/MuZero prediction is a model-based counterfactual. Validate it against held-out exact XMage branches before using it to generate preference factors. Report calibration by horizon and state slice; model uncertainty should increase when engine/card coverage is poor.

## Deployment roles

### Auxiliary head

Predict factor changes or expert utility from the shared state-action representation. This can shape representation learning while terminal value remains primary.

Advantages:

- no direct reward hacking path;
- easy calibration and ablation;
- explanations available at inference.

### Offline reranker

Use the cost model to annotate or rerank saved candidate lines. Keep raw policy/search scores alongside cost scores.

Advantages:

- safest first deployment;
- supports expert review and active learning;
- does not alter data collection policy.

### Search tie-breaker or prior

Use only after exact search values and factor provenance are available. Cap the contribution and test adversarial scenarios where the expert model conflicts with terminal search value.

### Potential-based shaping

Use:

```text
R_total = R_terminal + alpha * (gamma * Phi(next) - Phi(current))
```

where `Phi` is the expert utility over a perspective-consistent information state. Sweep `alpha`; include terminal-only and equal-weight controls. The validator requires the exact `gamma_phi_next_minus_phi` form for experiments labeled `potential_shaping`.

Partial observability and approximation matter: two histories with the same public snapshot can have different beliefs and value. `Phi` should therefore consume the same information state as the policy, not privileged state.

### Evaluation-only diagnostic

Use factor contributions to explain why two policies differ, even when no cost enters training. This is valuable when scalar win-rate intervals overlap.

## Active learning loop

After an initial model:

1. score held-out or newly collected legal pairs;
2. select high-entropy comparisons near 0.5 preference probability;
3. prioritize model-versus-human and model-versus-search disagreement;
4. stratify by underrepresented prompt types, matchups, and factor conflicts;
5. avoid repeatedly sampling near-duplicate states;
6. collect independent expert labels;
7. refit only after preserving a frozen benchmark set;
8. publish before/after agreement and calibration.

Keep a permanent audit set that is never selected adaptively.

## Versioning and provenance

Version separately:

- ontology schema and semantic version;
- annotation UI version;
- candidate generator/search checkpoint;
- state/decision schema;
- card database/rules engine commit;
- expert cohort;
- factor estimator checkpoints;
- fitted cost model hash;
- train/holdout split manifest.

Changing a factor definition without changing ontology version invalidates comparisons.

## CLI workflow

From `python/` after editable installation:

```bash
# Validate the broader experiment plan.
magic-cabt-research validate-plan \
  ../examples/research/experiment_matrix_v1.json --strict

# Fit a monotone preference model with a whole-context holdout.
magic-cabt-research fit-cost \
  --factors ../examples/research/causal_factors_v1.json \
  --preferences ../examples/research/expert_preferences.example.jsonl \
  --out runs/expert-cost-v1 \
  --holdout-fraction 0.25 \
  --seed 7

# Inspect one factor vector.
magic-cabt-research score-cost \
  --model runs/expert-cost-v1/model.json \
  --factors '{"opponent_threat_value_removed":3,"tempo_delta":1}'
```

`fit-cost` writes a model, train/holdout diagnostics, source hashes, and fit metadata. The example annotations are illustrative schema fixtures, not a validated strategic dataset.

## Acceptance criteria for proof of method

A first expert-cost result is credible only when:

- the ontology is frozen before final holdout scoring;
- all candidate actions are legal and perspective-correct;
- at least two simple weight baselines are reported;
- pair annotations stay grouped by context;
- duplicated experts/pairs do not cross partitions;
- disagreement and uncertainty are retained;
- held-out log loss/calibration improves, not only train accuracy;
- factor contributions survive expert review;
- terminal-only RL remains the primary control;
- any causal wording identifies the intervention source and assumptions;
- the model, data, ontology, and split hashes are published.
