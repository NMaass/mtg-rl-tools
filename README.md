# mtg-rl-tools

[![python-mirror-tests](https://github.com/NMaass/mtg-rl-tools/actions/workflows/python-mirror-tests.yml/badge.svg)](https://github.com/NMaass/mtg-rl-tools/actions/workflows/python-mirror-tests.yml)

Tools for building reinforcement-learning and imitation-learning workflows around Magic: The Gathering.

This project connects three pieces that are usually separate:

1. **A real rules engine**: XMage owns legality, phases, priority, the stack, mana payment, combat, and hidden-information boundaries.
2. **A Python-friendly agent API**: game decisions are exposed as indexed legal options, and agents answer with option indices.
3. **Local gameplay data capture**: MTG Arena `Player.log` files can be followed, mirrored into XMage, and recorded as CABT-style decision datasets.

The long-term goal is to make Magic agent research more reproducible: legal action spaces should come from a rules engine, observations should be explicit and hidden-information-safe, and recorded games should use the same decision format as self-play.

---

## Project status

This is an active research/tooling project. The core loop is usable today for smoke games, local protocol experiments, and Arena-log dataset capture. It is not yet a complete competitive Magic training platform.

### Current capabilities

| Area | Status |
| --- | --- |
| XMage option-index bridge | Implemented. Real XMage player decisions are surfaced as `observation.select.option[]`, and selections are validated before touching the engine. |
| Python live-game client | Implemented. `magic_cabt.CabtBridge` speaks newline-delimited JSON to the Java protocol server. |
| Full-engine smoke games | Implemented. Tests drive real `GameImpl` games through priority, land play, spell casting, mana payment, stack resolution, mulligan, and hidden-hand checks. |
| Card/deck identity | Implemented. Repository-backed card resolution, deck validation, `resolve_card`, `validate_deck`, and by-name `repository_card_data` are available. |
| MTG Arena mirror/recorder | Implemented. A local Arena `Player.log` follower records CABT-format decisions and can mirror/replay the board in an XMage window. |
| Dataset artifacts | Implemented. Smoke games and Arena sessions write JSONL observations, decisions, transitions, summaries, and replayable state streams. |
| Search/lookahead API | Not implemented yet. A cloned-game `search_begin` / `search_step` API is on the roadmap. |

---

## How it works

### XMage bridge

The Java bridge lives in a new XMage package, `mage.player.cabt`, and does not modify XMage core files. It extends XMage's `ComputerPlayer` path, but replaces silent AI choices with a fail-closed bridge controller.

At each player decision:

1. XMage reaches a real engine callback such as priority, target selection, mana payment, combat declaration, mode selection, or mulligan.
2. The bridge builds a prompt with indexed legal options.
3. The Python side chooses option indices.
4. The bridge validates the selection and applies it through XMage's own APIs.
5. The game continues until the next required decision or game result.

The important invariant is that the bridge does not invent legality. Legal choices come from XMage state and callbacks.

### Python protocol

The protocol server is a newline-delimited JSON subprocess interface:

```json
{"command":"game_start","decks":[deck0,deck1],"options":{"seed":7,"maxTurns":20}}
{"command":"game_select","select":[0]}
{"command":"game_finish"}
```

The Python client wraps this as:

```python
from magic_cabt import CabtBridge, load_decklist

deck = load_decklist("examples/basic_deck.txt")

with CabtBridge() as bridge:
    response = bridge.game_start(deck, deck, seed=7, max_turns=20)
    while not bridge.finished:
        select = response["observation"]["select"]
        # Agent policy goes here. For example, choose the first legal option.
        response = bridge.game_select([0])
    print(bridge.result)
```

### Arena mirror and recorder

The Arena mirror is a separate local data-capture path:

```text
MTG Arena Player.log
        -> log follower
        -> GRE state tracker
        -> hidden-info-safe board snapshots
        -> CABT-format decision records
        -> optional XMage visual replay
```

It records the local player's real Arena decisions as indexed options paired with the pre-decision board state. Opponent hidden cards are redacted to face-down placeholders in the default output.

---

## Repository layout

This repository is an **overlay** for an XMage checkout, not a fork. Files are stored at the paths where they should be copied inside `magefree/mage`.

```text
Mage.Server.Plugins/Mage.Player.AI/
  src/main/java/mage/player/cabt/      XMage bridge, prompts, protocol server,
                                      card identity, dataset writers
  src/test/java/mage/player/cabt/      unit, callback-boundary, protocol, and
                                      full-engine smoke tests
  docs/                               decision-surface and verification docs

Mage.Client/
  src/main/java/mage/client/cabtmirror/ XMage client-side Arena mirror display

python/
  magic_cabt/                         Python client, parsers, Arena mirror,
                                      replay and dataset utilities
  tests/                              Python unit and protocol tests

examples/                             basic deck, random legal agent, self-play
scripts/                              setup, verification, and mirror launchers
docs/                                user-facing run guides
.github/workflows/                    Python mirror/package CI
```

---

## Quick start: run the XMage bridge tests

Clone this repo and a fresh XMage checkout, then apply the overlay and run the verification suite.

```sh
git clone https://github.com/NMaass/mtg-rl-tools.git
git clone https://github.com/magefree/mage.git
rsync -a --exclude .git --exclude README.md mtg-rl-tools/ mage/
cd mage
scripts/run-cabt-adapter-tests.sh
```

The verification script runs:

- Java unit tests for prompt builders, validators, serializers, card data, and dataset writing.
- Callback-boundary tests for real XMage player callbacks.
- Full-engine smoke games through `GameImpl`.
- Python tests against Java-regenerated fixtures.
- Python protocol smoke tests when the protocol server classpath is available.

More detail: [`Mage.Server.Plugins/Mage.Player.AI/docs/cabt-verification.md`](Mage.Server.Plugins/Mage.Player.AI/docs/cabt-verification.md)

---

## Quick start: run Python self-play

After the verification script has built the protocol server classpath:

```sh
export MAGIC_CABT_CLASSPATH="$(cat Mage.Server.Plugins/Mage.Player.AI/target/cabt-classpath.full.txt)"
python3 examples/run_selfplay.py --seed 42 --max-turns 15
```

This runs two simple legal-option agents through a real XMage game and writes:

```text
target/cabt-selfplay/replay.jsonl
```

The example agent is intentionally simple. It exists to prove the protocol, not to play well.

---

## Quick start: mirror MTG Arena games locally

The Arena mirror currently targets Windows because MTG Arena and the provided launcher scripts are Windows-oriented.

### 1. Prepare XMage and the mirror

You need:

- JDK 17
- Maven 3.x
- Python 3.9+
- An XMage checkout
- MTG Arena with **Detailed Logs (Plugin Support)** enabled

Run the setup script from this repo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-arena-mirror.ps1 `
  -XmageDir C:\path\to\mage
```

### 2. Start the GUI

```powershell
powershell -ExecutionPolicy Bypass -File scripts\arena-mirror-gui.ps1
```

Or double-click `Arena Mirror.bat` from the repository root.

The GUI can:

- locate the Arena `Player.log`,
- start/stop live following,
- open XMage when a live game appears,
- record decisions and states,
- list recorded bundles,
- replay a bundle back into XMage.

Step-by-step guide: [`docs/RUNNING.md`](docs/RUNNING.md)

### Recorded files

Each run writes a bundle such as:

```text
arena-mirror-runs/<timestamp>/
  decisions.jsonl
  mirror_states.jsonl
  game_history.jsonl
  summary.json
  card_cache.json
```

`decisions.jsonl` is the main training-data artifact: one pre-decision observation, one indexed legal-option list, and the indices chosen by the Arena player.

---

## Card and deck identity

The bridge includes a repository-backed card resolver so real decklists can be validated before a game starts.

Protocol commands:

| Command | Purpose |
| --- | --- |
| `resolve_card` | Resolve one requested card name and return diagnostics. |
| `validate_deck` | Validate a full name/count decklist without starting a game. |
| `repository_card_data` | Export static card metadata for a requested list of names. |
| `all_card_data` | Export the active game's deduped deck-pool metadata. Requires an active game. |

Resolution is fail-closed. Unknown cards return structured diagnostics rather than being omitted or replaced.

Python example:

```python
from magic_cabt import CabtBridge

with CabtBridge() as bridge:
    print(bridge.resolve_card("Boseiju, Who Endures"))
    validation = bridge.validate_deck("24 Forest\n4 Lightning Bolt")
    if not validation["valid"]:
        raise ValueError(validation["failures"])
    cards = bridge.repository_card_data(["Forest", "Lightning Bolt"])
```

---

## Roadmap

Near-term priorities:

- **Cloned-game search/lookahead**: expose `search_begin`, `search_step`, and `search_release` so agents can evaluate candidate lines without mutating the live game.
- **More real-card regression scenarios**: expand beyond the Forest/Bears smoke path into targeted spells, modal spells, triggers, replacement effects, X spells, and combat-heavy games.
- **Amount-distribution targeting**: surface `chooseTargetAmount` as an explicit prompt instead of failing closed.
- **Priority payload refinement**: improve option payloads for alternate costs, special casting paths, optional additional costs, activated abilities, and cast-from-non-hand zones.
- **Dataset tooling**: add utilities for filtering, validating, joining, and sampling self-play and Arena-recorded JSONL data.
- **Full overlay CI**: run the complete Java/XMage overlay suite in CI, not only the Python-only test gate.

Longer-term goals:

- trainable baseline agents,
- benchmark environments,
- stronger replay validation between Arena and XMage state,
- pluggable evaluation harnesses for RL and imitation-learning experiments.

---

## Contributing

Contributions should preserve the core safety and correctness invariants of the project.

### Design rules

- **XMage owns legality.** Do not replace engine legality with a static action enum.
- **Fail closed.** If a decision surface is not implemented, the bridge should throw a clear error instead of silently letting inherited AI choose.
- **Preserve hidden information.** Observations and recorded datasets must not leak opponent private zones.
- **Keep protocol shapes stable.** Python tests use Java-regenerated fixtures so format changes should be intentional and tested.
- **Add prompt surfaces completely.** New decision surfaces need an audit entry, builder/applier code, bridge override, and tests.

### Useful commands

```sh
# Full adapter verification from an overlaid XMage checkout
scripts/run-cabt-adapter-tests.sh

# Python-only tests
cd python && python3 -m unittest discover -s tests -v

# Run self-play once the classpath has been generated
python3 examples/run_selfplay.py --seed 42 --max-turns 15
```

### Where to start

Good first contributions are usually:

- adding a failing regression test for a specific card or prompt shape,
- improving README/docs clarity,
- adding Python dataset utilities,
- expanding Arena mirror parsing coverage with sanitized sample logs,
- adding a focused smoke scenario for one Magic mechanic.

For prompt-surface work, start with [`Mage.Server.Plugins/Mage.Player.AI/docs/cabt-decision-surface.md`](Mage.Server.Plugins/Mage.Player.AI/docs/cabt-decision-surface.md).

---

## License

No open-source license has been added yet. Until a license is chosen, treat the code as publicly visible but not formally licensed for reuse.
