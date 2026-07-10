# Action-space research notes

This project has two action-space layers that should stay separate:

1. **Engine legality layer**: XMage/CABT enumerates the complete legal option
   list for the current prompt. This is the source of truth and should not be
   pruned by a learned prior.
2. **Model abstraction layer**: a policy can learn compressed action buckets,
   priors, entropy statistics, or top-k expansion rules over the legal options.
   These are training/search aids, not replacements for engine legality.

## What MTG-Causal-RL suggests

MTG-Causal-RL chooses a Gymnasium-style fixed masked action space: a
3,077-dimensional partial observation and a 478-action masked discrete action
space, plus archetype suites, reward profiles, causal factors, manifests, and
paired-seed evaluation. That design is useful because standard RL algorithms
like PPO consume fixed spaces and masks directly.

The same idea should be copied cautiously here. CABT has stronger rules fidelity
because XMage exposes real prompt callbacks and legal options. Therefore, our
closest equivalent is not a hard-coded 478-action enum; it is a set of **action
abstraction profiles** over the current legal option list:

- `small`: pass / play land / cast spell / activate / attack / block / target or
  choice / mana payment / mulligan / other.
- `full`: finer buckets for spell kinds, target side, mana payment, attacks,
  blockers, triggers, replacements, modes, numbers, and mulligans.

## Why not hard-prune rare actions?

Rare actions can still be strategically correct. Examples include self-targeting
or self-damage lines that are usually bad but occasionally enable cards like
Death's Shadow. The correct compromise is:

- keep the full legal action list;
- learn priors from logs;
- measure entropy/compressibility;
- use priors for ranking or top-k search expansion;
- keep strict legality validation at the bridge.

## What to measure

Run both profiles on any dataset:

```sh
magic-cabt-analyze-actions --input decisions.jsonl --profile small --out entropy.small.json
magic-cabt-analyze-actions --input decisions.jsonl --profile full --out entropy.full.json
```

Useful diagnostics:

- global selected-action entropy;
- prompt-conditional entropy;
- most common action bucket and rate per prompt type;
- legal action count distribution;
- rare bucket frequency.

Low conditional entropy means a small or hierarchical policy may work well.
High conditional entropy means the model needs richer state/option features,
search, or a finer prompt-specific decoder.

## Model sizes for ablation

The first ablation should compare:

- `small` config: coarse action profile, small option scorer, fast BC/PPO smoke;
- `full` config: finer action profile and causal auxiliary outputs.

The model configs live in `magic_cabt.models.configs`; they are framework-neutral
so a PyTorch trainer can consume them later without coupling the package to a
deep-learning framework today.
