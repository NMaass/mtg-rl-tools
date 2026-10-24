# Explainable heuristic, Jev escalation, and verified refinement

This is an **experimental XMage baseline**, not an expert Magic agent and not a
Forge integration. The engine supplies legal choices. Five independently
weighted modules explain their contribution to each candidate:

| Module | Implemented hypothesis | Important limit |
| --- | --- | --- |
| `role.py` | Visible clocks; develop, beatdown, control, stabilize | Not deck-level inevitability or belief modeling |
| `threats.py` | Visible stats and printed engine/keyword proxies; harmful versus beneficial targets | Not a general card-rules interpreter |
| `mana.py` | Land sequencing, missing colors, floating mana, preserve flexible sources | No multistep mana solver or full mana-pool observation |
| `tactics.py` | Bounded mulligans, apparent burn lethal, basic attacks/blocks/trades | Not exhaustive combat search; dynamic effects can invalidate estimates |
| `plans.py` | Role-conditioned objective and resource/development preferences | A plan label plus scoring, not a multiaction planner |

Every scored choice retains raw component scores, weights, reasons, the role,
plan, uncertainty margin, configuration hash, and coverage warnings. These are
heuristic scores, **not calibrated win probabilities**. No human logs or neural
policy were trained in this experiment.

## Setup

Install the Python package with `python -m pip install -e ./python`. Build the
existing XMage overlay and set `MAGIC_CABT_CLASSPATH` as described in the repository
README. Rebuild the Java overlay for the new target-source metadata. The CI
workflow provides an executable reference build against XMage revision
`fd40ad5c29a92cef824cf12ba6d0e4daa25db975`, using
`services/engine/prepare_reference.py` for compatibility. This project remains an
overlay; this repository alone is not a complete XMage checkout.

Run commands below from the repository root. Each run needs a new output
folder. Existing evidence is never silently overwritten.

## Fixed-pool complete-game comparison

```sh
python -m magic_cabt.heuristic compare \
  --pool examples/heuristic/deck_pool.json \
  --agents heuristic jev first random --seeds 11 23 \
  --out runs/heuristic-comparison
```

The fixed pool has three 60-card, four-copy kitchen-table lists: green creatures,
white flying creatures, and red creatures/burn. It is not a Standard/Modern
benchmark. All ordered deck pairs, agent pairs, seed blocks, and swapped seats
are scheduled in advance. P0 is explicitly chosen as starting player so the
paired game reverses play/draw as well as pilot/deck seats. The override is
recorded rather than scored as a policy decision.

A warm JVM is reused. Each game has explicit wall-clock, turn and decision
limits; errors restart the JVM. Crashes, limits, unresolved winners, skipped
provider games, and actual draws are distinct. Only resolved completed games
enter score summaries; **always inspect status counts alongside scores**.
Initial JVM/catalog setup is outside the per-game watchdog and timings.

The manifest pins the deck pool, Python and Java-overlay source hashes, configs,
schedule, seeds and limits. Replays retain each observation, selected indices,
explanations and terminal outcome. Random UUID ordering can still vary, so a
seed is not a promise of byte-identical replay across processes. Comparisons
here are descriptive; two seed blocks do not establish competitive strength.

### Engine integration corrections

XMage observation `playerIndex` follows a rotating turn-order list. The existing
runner previously used it as a persistent agent seat. Both the new experiment
runner and `eval/play.py` now map selecting `playerId` through exact configured
player names. Legacy fixtures without identity retain their old index path.

Target prompts now carry source ability text, source identity and `Outcome`,
so target ranking does not have to assume every target prompt is removal. The
new strategy projection interprets live `maxCount=0` literally, and rejects
invalid compound attacker/blocker selections. The legacy tournament's winner
aggregation is unchanged; use the **new experiment runner** for strict outcome
accounting rather than treating unknown winners as draws.

## Jev harness

The wrapper reuses the repository's existing typed Decisions API client and its
pinned `typesafe/jev-1.13` model. It scores the same `option_N` IDs as the
heuristic, with an allowlisted acting-player projection: no opponent hands,
libraries, raw payload copies, results or future frames. Static catalog data is
looked up only for visible named objects/actions. It does not yet include
recurrent public action history or belief state.

Default routing calls Jev only when the heuristic margin is small. The Python
API also supports `shadow`, `always`, and periodic auditing. Provider probabilities
are kept separate from heuristic scores. Thresholds (0.65 top probability and
0.10 probability gap) are initial **uncalibrated** acceptance rules. Every call
records disagreements, whether its choice was accepted, latency, billing,
request hash and safe request. Annotation via `.score()` never triggers payment.

Only exactly-one decisions with 2..255 options are escalated. Compound combat,
forced choices, oversized requests and unsupported envelopes stay with the
heuristic; nothing is truncated into a different legal-action set. Invalid
answers fall back explicitly. Successes and failures are cached; failures are
not silently retried. The key is never recorded in evidence.

Set `OPENROUTER_API_KEY` in your local environment to enable paid requests:

```sh
python -m magic_cabt.heuristic compare \
  --pool examples/heuristic/deck_pool.json --agents heuristic jev \
  --seeds 41 --jev-max-calls 100 --jev-cost-stop 1.00 \
  --out runs/jev-comparison
```

The call cap is hard. The dollar setting stops **after reported spending**; it
cannot guarantee an unknown next request will not cross the threshold. Unknown
billing stops further requests. The budget is shared across the run. Later Jev
games may be skipped when exhausted, and an already-running game falls back;
inspect per-game `jevActuallyCalled`, calls, fallbacks and skips. No credentials
are wired into CI. Missing credentials produce explicit skipped games, not a
purported Jev win rate. Fixture transports are labeled `fixture`.

## Watch decisions and play

```sh
python -m magic_cabt.heuristic review \
  --run runs/heuristic-comparison --out runs/heuristic-review.html
```

Open the generated HTML locally. It has previous/next, play/pause and a decision
slider, with legal actions, module explanations, role/plan, warnings and raw
frames. This is a text-based replay inspector, **not an animated XMage board**.
It changes acting-player perspective across decisions. Rendering every replay
embeds them all and can create a large HTML file; use a smaller review selection
for convenient browsing. Script labels are escaped and the JavaScript is syntax
tested; visual browser testing was unavailable in the implementation container.

The new agent is registered as `explainable` in the existing terminal interface:

```sh
python -m magic_cabt.play.human_vs_agent \
  --human-seat 0 --agent1 explainable \
  --deck0 examples/heuristic/green-curve.txt \
  --deck1 examples/heuristic/red-burn.txt \
  --seed 43 --out runs/my-game.jsonl
```

This is the existing legal-option terminal UI, not a new graphical play client.
Its runner now enriches visible card names with the same static engine catalog.

## Automated refinement

```sh
python -m magic_cabt.heuristic refine \
  --pool examples/heuristic/deck_pool.json \
  --suite examples/heuristic/regressions.json \
  --candidates 2 --tuning-seeds 301 --holdout-seeds 501 502 \
  --out runs/heuristic-refinement
```

This first implementation proposes one-module weight changes (+25%/-25%),
starting with tactics. It does **not** ask an LLM to rewrite executable policy
source. Each proposal has a hypothesis and immutable configuration. It runs the
implementation tests and 16 frozen, explicitly synthetic developer regression
cases, evaluates survivors on tuning games, selects once without holdout scores,
then evaluates that candidate on separate holdout seeds.

Promotion requires all planned games, identical policy/pool/code provenance, no
crashes, no duplicate/missing records, no reused tuning seeds, tactical passes,
at least 30 independent seed blocks, and a conservative one-sided Hoeffding
lower bound above 0.5. Games sharing a seed are grouped into one block, not
pretended independent. The bound includes the declared proposal family size.
The statistical interpretation assumes independent predeclared blocks; repeatedly
reusing a holdout does not make additional evidence. Future cycles need fresh
holdouts and an explicit repeated-testing policy.

The small command above exercises the loop but **cannot pass the sample-size
gate**. Promotion only writes the run's `champion.json`; it never edits main,
changes the default policy, or deploys automatically. Stronger opponent pools,
expert-reviewed roots, history/belief features, and complete tactical search are
still needed before trusting arbitrary-deck recommendations.

## Validation and measured evidence

```sh
cd python
python -m unittest discover -s tests -p 'test_heuristic*.py' -v
python -m unittest discover -s tests -v
```

See `docs/experiments/heuristic-jev-2026-10-24.json` for retained local run summaries.
Full comparison and refinement replays are separate downloadable artifacts, not
committed databases. `.github/workflows/heuristic-experiments.yml` rebuilds the
pinned engine and records fresh CI evidence. No paid Jev strength, expert strength,
or verified automatic improvement is claimed by this version.
