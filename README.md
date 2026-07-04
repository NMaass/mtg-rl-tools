# xmage-cabt-bridge

A CABT-style option-index bridge around [XMage](https://github.com/magefree/mage):
decisions the engine asks a player for are surfaced as prompts of indexed
options, selections come back as indices, and the bridge validates and applies
them through the engine's own APIs. Built as the Java half of a Python
training harness for Magic: The Gathering agents.

## Status: what works today, what doesn't

Implemented and tested against the real engine (`CabtRealGameSmokeTest` runs
actual `GameImpl` games through the bridge):

- **Priority with playable actions**: PASS_PRIORITY plus one option per
  ability from `Player.getPlayable` (play land, cast spell, activate ability,
  special action), dispatched through `PlayerImpl.activateAbility`. A smoke
  game plays a land from hand and casts a creature through the real mana
  payment and stack-resolution path.
- **Callback prompts**: targets, yes/no, generic choice, pile, modes,
  numbers/X/multi-amount, trigger ordering, replacement effects, mana
  payment, attacker/blocker declarations, mulligan — each with unit +
  callback-boundary tests; priority, targeting, mana payment and mulligan
  also covered by the full-engine smoke games.
- **Decision traces**: every surfaced decision is numbered and traced
  PENDING → SELECTED → APPLIED (or FAILED with the error), with the selected
  options resolvable per trace.
- **Fail-closed policy, enforced**: audited callbacks that are not surfaced
  throw `CabtUnhandledDecisionException` instead of letting the inherited
  `ComputerPlayer` AI decide; a reflection audit
  (`CabtBridgePlayerOverrideAuditTest`) fails the suite if a SURFACED or
  FAIL_CLOSED callback loses its override. Prompts reaching the bridge from
  simulation/playable-check games fail closed too.
- **Data layer**: static card metadata export and a JSONL transition dataset
  writer (Java), with `python/magic_cabt` parsers tested against
  Java-regenerated fixtures.

Not implemented yet — do not rely on these:

- **Subprocess protocol server** (`CabtProtocolServer`) and the Python
  live-game client: the Python package is a card-data/dataset **parser
  only**; there is no `game_start`/`game_select` interface yet. The
  verification script auto-enables protocol smoke tests once the server
  class exists.
- **Amount-distribution targeting** (`chooseTargetAmount`) fails closed by
  design until a distribution payload is built.
- **Player callbacks outside the audited surface** still fall back to
  `ComputerPlayer`; the audited surface and its enforcement are documented
  in `docs/cabt-decision-surface.md`.

## Layout

This repo is an **overlay**, not a fork: it contains only new files, at the
same paths they occupy inside an XMage checkout.

```
Mage.Server.Plugins/Mage.Player.AI/
  src/main/java/mage/player/cabt/   the bridge (player, prompts, appliers, data export)
  src/test/java/mage/player/cabt/   unit + callback-boundary + full-engine smoke tests
  docs/                             decision-surface audit, verification guide
python/                             magic_cabt package (card data + dataset parsers)
scripts/                            run-cabt-adapter-tests.sh
```

## Using it

Apply the overlay onto an XMage checkout and run the suite from there:

```sh
git clone https://github.com/magefree/mage.git
rsync -a --exclude .git --exclude README.md ./ mage/
cd mage && scripts/run-cabt-adapter-tests.sh
```

No XMage core file is modified — the bridge player extends `ComputerPlayer`
and everything lives in the new `mage.player.cabt` package. The code targets
Java 8 (XMage's build level).

Key docs:

- `Mage.Server.Plugins/Mage.Player.AI/docs/cabt-decision-surface.md` — the
  audited decision surface and fail-closed policy.
- `Mage.Server.Plugins/Mage.Player.AI/docs/cabt-verification.md` — test
  layers and failure-to-feature mapping.

## History note

The initial commits import the code feature by feature in the order the
features were built. Files are imported at their final state, so each commit
reads as one feature, but intermediate commits are not individually
buildable snapshots.
