# RSSM dynamics diagnostics

## Research claim under test

The structured RSSM tests whether an explicit stochastic latent transition model
predicts held-out Magic trajectories more usefully than deterministic latent
prediction. It is not treated as a planner or game-playing agent by default.

A checkpoint becomes a candidate for imagined search only after its held-out
open-loop behavior is characterized.

## Inputs

- horizon-one transitions only;
- complete game identity retained for splitting;
- perspective-safe state tensorization;
- semantic action vectors;
- exact engine-derived public causal deltas when available;
- terminal outcome labels when available.

Longer-horizon transition rows are excluded from recurrent training because
mixing horizon units would make the recurrence ambiguous.

## Model

The model contains:

- structured observation encoder;
- deterministic GRU state;
- diagonal-Gaussian prior conditioned on the previous latent and action;
- posterior conditioned on prior state and current observation;
- observation-latent decoder;
- public causal-delta head;
- terminal value head.

Training uses posterior reconstruction, one-step prior prediction, balanced KL
with free nats, causal-delta loss, and optional value loss.

## Mandatory held-out metrics

- total objective;
- posterior reconstruction MSE;
- one-step prior prediction MSE;
- prior/posterior KL;
- prior negative log likelihood of posterior means;
- standardized residual RMS;
- causal-delta error;
- value error where labeled;
- posterior latent per-dimension variance and effective rank;
- open-loop latent-decoder MSE at each horizon.

Open-loop error is computed by initializing from a posterior state and then
rolling only the prior with recorded actions. Later observations are used as
targets, not recurrent inputs.

## Interpretation gates

Do not describe the model as useful for planning when:

- open-loop error grows immediately or is no better than a constant predictor;
- standardized residual RMS is materially different from one;
- effective rank collapses;
- prior NLL improves while decoded trajectory accuracy degrades;
- performance depends on hidden-state fields removed by visibility tests;
- evaluation games overlap training games.

## Current boundaries

- The model does not expose a policy head or tree-search interface.
- It does not replace exact XMage legality or rule execution.
- Its Gaussian latent may be insufficient for highly multimodal opponent and
  chance responses.
- Prefix-bounded CLI collection preserves accepted game boundaries but is not a
  randomized dataset sampler; research runs should supply explicit manifests or
  preselected bundle lists.
