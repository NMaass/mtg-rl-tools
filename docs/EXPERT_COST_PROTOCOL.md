# Expert preference and causal-cost protocol

## Purpose

This protocol turns expert MTG judgment into a versioned, testable signal without pretending that an expert can directly specify one universally correct dense reward.

The supervision unit is a **comparison between two legal candidate lines in one information state**. Each line is summarized by declared consequence factors. Experts choose left, right, or tie; provide confidence, rationale, and assumptions; and may disagree. The supplied fitter learns a small monotone Bradley–Terry utility model over those factors.

The output is an **expert-preference utility**. Calling it a **causal cost** is justified only when candidate factor values have causal provenance, for example:

- an XMage intervention from the same cloned state;
- an exact deterministic consequence produced by the rules engine;
- paired rollouts under a declared opponent, chance, and belief policy;
- an identified observational design with explicit assumptions.

A factor estimated from Arena trajectories, a language model, a value model, or an expert guess is still useful, but it is a prediction or judgment—not automatically a causal effect.

## Non-goals

This framework does not:

- replace terminal win/loss reward;
- prove that a factor graph is causally identified;
- turn printed card text into a context-free card score;
- declare one expert's annotations optimal;
- expose the true opponent hand to a deployable policy;
- certify a world model merely because its latent loss decreases.

## Core objects

### 1. Factor ontology

The ontology defines stable factor names, orientation, transform, scale, bounds, defaults, units, and meaning. It is versioned independently of annotations and fitted models.

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

`scale` is numerical conditioning, not strategic importance. Strategic weights are learned. Bounds should catch annotation or prediction errors rather than force every context into the same strategic range.

### 2. Candidate consequence vector

A candidate represents one canonical legal action group or a declared multi-action line. Its factors describe consequences over a stated horizon.

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

### 3. Preference annotation

One JSONL row records an exact pair. `contextId` identifies the information state **and the oriented candidate pair**. Repeated ratings of the same pair share the same ID; another pair from the same state receives another ID.

```json
{
  "contextId": "bolt-3-3-main-phase-v1",
  "expertId": "expert-01",
  "left": {
    "candidateId": "bolt-creature",
    "factors": {
      "opponent_threat_value_removed": 3.0,
      "tempo_delta": 1.0
    }
  },
  "right": {
    "candidateId": "bolt-face",
    "factors": {
      "opponent_life_loss": 3.0
    }
  },
  "preferred": "left",
  "confidence": 0.9,
  "weight": 1.0,
  "rationale": "The creature is the clock and face damage is not part of lethal.",
  "assumptions": ["No prevention effect is active"],
  "tags": ["removal", "threat-assessment"],
  "metadata": {"source": "expert-review-v1"}
}
```

A tie is valid. Low confidence is preferable to invented certainty. Confidence zero preserves a row for audit while excluding it from fitting.

## Factor design

### Separate measurement from preference

Every factor needs two definitions:

1. **measurement rule**: how its raw value is produced;
2. **preference orientation**: whether larger or smaller is generally better.

Examples:

- `opponent_life_loss` can be exact over an immediate deterministic branch;
- `opponent_threat_value_removed` contains contextual expert or learned valuation;
- `expected_win_probability_delta` is a model estimate and must name its checkpoint;
- `lethal_risk` depends on beliefs and an opponent policy;
- `rules_or_model_uncertainty` records unreliability in the preceding estimates.

Keep uncertainty and provenance separate from the estimate itself.

### Recommended factor families

| Family | Examples |
| --- | --- |
| Outcome | win-probability change, terminal result, loss-before-horizon probability |
| Board/material | threat value removed, material delta, card advantage, engine value |
| Tempo | mana efficiency, initiative, attack quality, development delay |
| Survival/pressure | life buffer, opponent life loss, clock length, lethal prevention |
| Optionality | future legal-option value, answer coverage retained, face reach retained |
| Information | information gained, belief entropy reduced, information leaked |
| Risk/provenance | variance, hidden-card sensitivity, rules/model uncertainty |

Avoid redundant aliases such as four slightly different versions of “board advantage.” Add a factor only when experts can define it independently, it changes preferences not represented by existing factors, its measurement can be audited, and held-out evidence supports it.

## Annotation workflow

### Step 1: sample complete contexts

Sample whole decision contexts from held-out Arena games, fixed tactical scenarios, XMage self-play, learned-policy trajectories, model/expert/search disagreement, and poor-calibration slices.

Do not reveal the actual later outcome unless the task is explicitly retrospective and records that fact. Otherwise hindsight leaks chance and opponent events into the target.

### Step 2: freeze perspective and legality

Record:

- perspective seat;
- public state;
- acting player's legal private information;
- event history available to the model;
- canonical legal action groups;
- source versions and state hash.

Never show the true opponent hand as ordinary context. Oracle review is a separately labeled analysis mode.

### Step 3: generate candidate consequences

Use this provenance hierarchy:

1. exact deterministic XMage branch;
2. short exact branch plus declared response script;
3. paired engine rollouts under fixed policy and belief samplers;
4. calibrated learned world-model prediction with checkpoint metadata;
5. expert estimate.

One vector may combine exact life loss, search-estimated win probability, and expert-valued threat removal. Store provenance per factor when the sources differ.

### Step 4: choose informative pairs

Prefer:

- model top choice versus the human action;
- model top two near a decision boundary;
- search top choice versus human action;
- lines with identical immediate effect but different optionality;
- high-value conflicts such as tempo versus card advantage;
- active-learning pairs near 0.5 predicted preference;
- pairs on which experts disagree.

Include easy controls—lethal, obvious equivalence, and illegal-lookalike exclusions—to detect UI and schema errors.

### Step 5: collect independent judgments

Initial proof-of-method target—not a sample-complexity guarantee:

- 100–300 unique pairs across at least 50 states;
- at least 20% rated by two or more experts;
- randomized left/right presentation;
- stable pair IDs and pseudonymous persistent expert IDs;
- rationale and confidence for disputed or high-impact pairs;
- annotation time, ontology version, and UI version.

### Step 6: audit disagreement before fitting

Disagreement often indicates ambiguous horizons, missing hidden-information assumptions, overlapping factors, different deck/metagame beliefs, role-assignment disagreement, or genuinely mixed strategic choices.

Do not erase disagreement by majority vote. Retain individual labels and report agreement by pair, expert, matchup, tag, and factor source.

### Step 7: split without leakage

Never split annotations of one pair between train and holdout. Use context holdout first, then add expert, matchup, time-block, card, or set holdouts for the corresponding generalization claim.

### Step 8: fit simple models first

The supplied model is deliberately small:

```text
oriented_factor_i = transform(clamp(raw_i)) / scale_i
utility(candidate) = sum_i nonnegative_weight_i * oriented_factor_i
P(left preferred) = sigmoid((utility(left)-utility(right))/temperature)
cost(candidate) = -utility(candidate)
```

Non-negative weights enforce ontology directions. Compare against equal weights, frozen hand-set weights, an unconstrained Bradley–Terry model, and only then interaction or context-conditioned models.

### Step 9: evaluate before deployment

Required preference metrics:

- pairwise accuracy excluding ties;
- weighted log loss;
- Brier score;
- preferred utility margin;
- calibration by probability bucket;
- metrics by expert, tag, matchup, horizon, and factor provenance;
- disagreement-conditioned metrics;
- coverage and parse failures.

For policy-learning use, also compare terminal-only reward, equal-weight potential, learned potential, auxiliary-only use, shuffled factor labels, a matched-capacity scalar auxiliary control, reward-hacking scenarios, and paired gameplay.

## Functional equivalence examples

### Lightning Bolt on a 3/3

Do not encode “Lightning Bolt prefers creature named X.” Encode the consequence conflict:

| Factor | Bolt 3/3 | Bolt face |
| --- | ---: | ---: |
| Threat value removed | high | zero |
| Opponent life loss | zero | 3 |
| Expected own life preserved | context-dependent | zero |
| Tempo | context-dependent | usually low without lethal pressure |
| Lethal probability | state-dependent | state-dependent |

At three opponent life, face is terminal. At fourteen life with the 3/3 as the only clock, removal may dominate. The action template is the same; the state and consequence vector change the label.

### Abrade versus Lightning Strike

The immediate branch can be identical when both deal three to a creature. The strategic branch differs:

- spending Abrade may preserve face or planeswalker reach;
- spending Lightning Strike may preserve artifact coverage;
- the choice depends on revealed cards, matchup priors, hand texture, life, and future mana;
- uncertainty about those beliefs is recorded rather than hidden in a card-name rule.

This is a direct test of functional equivalence plus contextual exceptions.

## Causal interpretation

### Exact intervention

Clone the same XMage state, apply A versus B, and hold the chance stream and response script fixed. Record state hash, semantic action, seed, response policy, horizon, measured variables, and branch failures.

### Rollout-estimated effect

Under stochastic draws or policies, estimate a distribution. Use common random numbers or paired seeds. Record rollout count, mean, variance, quantiles, belief checkpoint, opponent checkpoint, and sensitivity across policies.

### Observational association

“Players making action A won more often” is confounded by state quality and player skill. It may be predictive, but it is not a do-effect without an identification argument.

### Learned counterfactual

A JEPA, RSSM, or MuZero prediction is a model-based counterfactual. Validate it against held-out exact XMage branches before using it as factor provenance. Report calibration by horizon and state slice.

## Deployment roles

### Auxiliary head

Predict factor deltas or expert utility from a shared state-action representation while terminal value remains primary. This is the safest first training use because it limits direct reward-hacking paths and is easy to ablate.

### Offline reranker or diagnostic

Annotate saved candidate lines, retain raw policy/search scores, and use factor contributions for expert review and active learning. This does not alter the data-collection policy.

### Search prior or tie-breaker

Use only after exact search values and factor provenance exist. Cap its contribution and include scenarios where expert utility conflicts with terminal search value.

### Potential-based shaping

Use:

```text
R_total = R_terminal + alpha * (gamma * Phi(next) - Phi(current))
```

`Phi` must consume the same perspective-consistent information state as the policy, not privileged hidden state. Sweep `alpha` and retain terminal-only and equal-weight controls. The experiment validator requires the exact `gamma_phi_next_minus_phi` form for plans labeled `potential_shaping`.

## Active learning

After an initial model:

1. score held-out and newly collected pairs;
2. select high-entropy comparisons near 0.5;
3. prioritize model/human/search disagreement;
4. stratify underrepresented prompt types and factor conflicts;
5. avoid near-duplicate states;
6. collect independent labels;
7. preserve a permanent non-adaptive audit set;
8. refit and publish agreement/calibration changes.

## Versioning and provenance

Version separately:

- ontology and semantic version;
- annotation UI;
- candidate generator and search checkpoint;
- state/decision schema;
- card database and XMage commit;
- expert cohort;
- factor estimator checkpoints;
- fitted model hash;
- train/holdout manifest.

Changing a factor definition without changing ontology version invalidates comparisons.

## CLI workflow

From `python/` after editable installation:

```bash
# Validate the broader experiment plan.
magic-cabt-research validate-plan \
  ../examples/research/experiment_matrix_v1.json --strict

# Fit with a whole-context holdout.
magic-cabt-research fit-cost \
  --factors ../examples/research/causal_factors_v1.json \
  --preferences ../examples/research/expert_preferences.example.jsonl \
  --out runs/expert-cost-v1.json \
  --holdout-fraction 0.25 \
  --seed 7

# Inspect one factor vector.
magic-cabt-research score-cost \
  --model runs/expert-cost-v1.json \
  --factors-json '{"opponent_threat_value_removed":3,"tempo_delta":1}'
```

`fit-cost` writes model weights, train/holdout diagnostics, expert agreement, source hashes, and fit metadata. The example annotations are schema fixtures, not a validated strategic dataset.

## Acceptance criteria for proof of method

A first result is credible only when:

- the ontology is frozen before final holdout scoring;
- all candidates are legal and perspective-correct;
- at least two simple weight baselines are reported;
- pair annotations remain grouped by context;
- duplicated experts or pairs do not cross partitions;
- disagreement and uncertainty are retained;
- held-out log loss or calibration improves, not only train accuracy;
- factor contributions survive expert review;
- terminal-only RL remains the primary control;
- causal wording names the intervention source and assumptions;
- model, data, ontology, and split hashes are published.

## References

- Ng, Harada, and Russell. *Policy invariance under reward transformations: theory and application to reward shaping*. ICML, 1999.
- LeCun. *A Path Towards Autonomous Machine Intelligence*. 2022.
- Bebbington et al. *Game-Generated Data: An Untapped Resource for Advanced AI Training*. arXiv:2504.16591.
