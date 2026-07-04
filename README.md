# xmage-cabt-bridge

A CABT-style option-index bridge around [XMage](https://github.com/magefree/mage):
every decision the engine asks a player for (priority, targets, modes, mana
payment, combat, mulligan, ...) is surfaced as a prompt of indexed options,
selections come back as indices, and the bridge validates and applies them
through the engine's own APIs. Built as the Java half of a Python training
harness for Magic: The Gathering agents.

## Layout

This repo is an **overlay**, not a fork: it contains only new files, at the
same paths they occupy inside an XMage checkout.

```
Mage.Server.Plugins/Mage.Player.AI/
  src/main/java/mage/player/cabt/   the bridge (player, prompts, appliers, data export)
  src/test/java/mage/player/cabt/   unit + callback-boundary integration tests
  docs/                             decision-surface audit, verification guide
python/                             magic_cabt package (card data + dataset parsers)
scripts/                            run-cabt-adapter-tests.sh
```

## Using it

Apply the overlay onto an XMage checkout and run the suite from there:

```sh
git clone https://github.com/magefree/mage.git
rsync -a --exclude .git ./ mage/
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
