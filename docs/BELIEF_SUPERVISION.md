# Belief supervision and visibility contract

## Scope

The belief model estimates strategically defined opponent holdings from the
acting player's information state. It is an auxiliary supervised model, not a
source of hidden truth at inference time.

The implementation separates three objects:

1. **Model observation:** perspective state plus explicitly public history.
2. **Oracle training label:** engine-derived hidden-state truth used only in the
   loss function.
3. **Belief output:** calibrated probabilities produced from the model
   observation.

A record is invalid for belief training when these boundaries cannot be
verified.

## Required record shape

Belief labels are top-level training metadata:

```json
{
  "perspectiveSeat": 1,
  "observation": {
    "current": {},
    "publicHistory": []
  },
  "trainingLabels": {
    "visibility": "oracle-label-only",
    "source": "xmage-engine",
    "belief": {
      "spot_removal": 1,
      "counterspell": 0
    }
  }
}
```

Requirements:

- `trainingLabels.visibility` is exactly `oracle-label-only`;
- `trainingLabels.source` identifies the label producer;
- each present vocabulary label is binary;
- absent labels are masked rather than treated as negative;
- unlabeled decisions remain in their complete game sequence;
- oracle labels never appear inside `observation`;
- the vocabulary file is versioned and hashed in run metrics.

## Visibility-safe tensorization

`VisibilitySafeTensorizer` treats opponent-hand objects and unknown library
objects as anonymous hidden-card rows. Card name, text, mana value, power,
toughness, and other card-specific fields are not encoded.

Only `publicHistory` is accepted as temporal context. Generic `history` is
ignored because capture producers do not currently guarantee that it excludes
engine-only or private information.

The invariant tests change the identity and numeric characteristics of a hidden
opponent card and require identical tensor rows.

## Metrics

Report all of the following on complete held-out games:

- policy negative log likelihood, top-1, top-3, and MRR;
- aggregate and per-label binary log loss;
- aggregate and per-label Brier score;
- expected calibration error;
- reliability-bin confidence and empirical frequency;
- labeled records, labeled cells, unlabeled records, and label-source counts.

Accuracy alone is insufficient. A useful belief model must assign meaningful
probabilities, particularly when beliefs will later be sampled for search.

## Permitted uses

- auxiliary representation learning;
- replay diagnostics;
- calibrated policy/value features;
- belief-conditioned determinization baselines;
- active-learning selection for missing labels.

## Prohibited claims and uses

- Do not describe an oracle-label model as observing opponent cards.
- Do not feed `trainingLabels` into deployment or evaluation observations.
- Do not use true hidden state as a policy feature except in a separately named
  oracle upper-bound experiment.
- Do not infer planning strength from belief calibration alone.
- Do not merge results from different vocabularies or label producers without
  reporting the distinction.

## Current limitation

The checked-in vocabulary contains broad strategic categories. The repository
does not yet ship an engine label producer that maps every card and game state
to these categories. Until that producer exists, the trainer consumes only
records labeled externally under this contract.
