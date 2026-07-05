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
  special action), dispatched through `PlayerImpl.activateAbility` — the
  engine's answer is recorded, not assumed (a declined activation traces as
  REJECTED). A smoke game plays a land from hand and casts a creature through
  the real mana payment and stack-resolution path. This is the current root
  priority implementation, not its final form: `getPlayableOptions` /
  `getPlayableObjects` remain future refinements for alternate-cost and
  casting-option payloads.
- **Callback prompts**: targets, yes/no, generic choice, pile, modes,
  numbers/X/multi-amount, trigger ordering, replacement effects, mana
  payment, attacker/blocker declarations, mulligan — each with unit +
  callback-boundary tests; priority, targeting, mana payment and mulligan
  also covered by the full-engine smoke games.
- **Decision traces**: every surfaced decision is numbered and traced
  PENDING → SELECTED → APPLIED — or REJECTED when the engine declines the
  selected action, or FAILED (with the error) when selecting, validating, or
  applying throws — with the selected options resolvable per trace.
- **Smoke-run artifact bundle**: the smoke test writes
  `target/cabt-smoke-run/` (manifest, decklists, `observations.jsonl`,
  `transitions.jsonl` with object-id zone-move deltas, `timeline.html`,
  `final-state.json`, `invariants.json`) and fails unless the files are
  mutually consistent — e.g. the selected PLAY_LAND option's card id is the
  id that moved HAND → BATTLEFIELD, and the cast spell's id resolves
  STACK → BATTLEFIELD.
- **Fail-closed policy, enforced**: audited callbacks that are not surfaced
  throw `CabtUnhandledDecisionException` instead of letting the inherited
  `ComputerPlayer` AI decide; a reflection audit
  (`CabtBridgePlayerOverrideAuditTest`) fails the suite if a SURFACED or
  FAIL_CLOSED callback loses its override. Prompts reaching the bridge from
  simulation/playable-check games fail closed too.
- **Data layer**: static card metadata export and a JSONL transition dataset
  writer (Java), with `python/magic_cabt` parsers tested against
  Java-regenerated fixtures. The Python package also includes a local
  MTG Arena `Player.log` normalizer that writes raw events, normalized events,
  game history (game states plus paired decision prompts and the client's
  chosen actions), per-match deck info, and summary JSON artifacts for later
  XMage validation — validated against real multi-match `Player.log` captures.
- **Subprocess protocol server + Python live-game client** (the CABT
  competition loop): `CabtProtocolServer` speaks newline-delimited JSON over
  stdin/stdout — `ping`, `capabilities`, `game_start`, `game_select`,
  `game_finish`, `all_card_data`, `visualize_data` — backed by
  `CabtGameSession` running a real `GameImpl` on a game thread. Selections
  are validated before the engine is touched: invalid ones return structured
  errors (`OPTION_INDEX_OUT_OF_RANGE`, `INVALID_SELECTION_COUNT`,
  `DUPLICATE_SELECTION`) and leave the pending decision answerable; unknown
  or malformed commands fail closed. Deck input is name+count entries
  resolved to real card classes (`CabtDeckFactory`), failing closed on
  unknown names; `game_start` takes optional `seed`, `maxTurns`, and player
  names. The Python client (`magic_cabt.CabtBridge`) is the
  `battle_start`/`battle_select`/`battle_finish`/`visualize_data`
  equivalent. Tested at three boundaries: `CabtGameSessionTest` (session
  API), `CabtProtocolServerTest` (raw request lines, full game from
  serialized observations only, hidden-hand checks per observation), and
  `python/tests/test_protocol_live.py` (real subprocess from Python).

Not implemented yet — do not rely on these:

- **Search/lookahead API** (the CABT `search_begin`/`search_step`
  equivalent): no cloned-game forward model is exposed to Python yet.
- **Global `all_card_data`** — the protocol's `all_card_data` is
  game-scoped (exports the active game's deduped deck pool only);
  `capabilities()` reports this as `"cardDataScope":
  "ACTIVE_GAME_DECK_POOL"`. A global static card-data export is future
  work.
- **Full card-name resolution** — `CabtDeckFactory` uses a class-name
  heuristic that works for simple names (Forest, Grizzly Bears) but
  fails closed for split cards, variant suffixes, and any card whose
  XMage class name differs from the naive transform. Full repository
  lookup is future work.
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
  src/main/java/mage/player/cabt/   the bridge (player, prompts, appliers, data export,
                                    game session + protocol server)
  src/test/java/mage/player/cabt/   unit + callback-boundary + full-engine smoke tests
  docs/                             decision-surface audit, verification guide
Mage.Client/
  src/main/java/mage/client/cabtmirror/   XMage-client display for the live Arena
                                          mirror (puppet game + GameView renderer)
python/                             magic_cabt package (card data + dataset parsers,
                                    live-game protocol client, arena_mirror live
                                    follower/tracker/recorder/replay)
examples/                           random legal agent, example deck, self-play runner
scripts/                            run-cabt-adapter-tests.sh,
                                    setup-arena-mirror.ps1, arena-mirror.ps1,
                                    arena-mirror-gui.ps1
.github/workflows/                  python-mirror-tests.yml (Python CI gate)
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

### Playing a live game from Python

After the suite has run once (it writes the launch classpath to
`Mage.Server.Plugins/Mage.Player.AI/target/cabt-classpath.full.txt`):

```sh
export MAGIC_CABT_CLASSPATH="$(cat Mage.Server.Plugins/Mage.Player.AI/target/cabt-classpath.full.txt)"
python3 examples/run_selfplay.py --seed 42 --max-turns 15
```

runs two random legal agents through a real engine game and writes a replay
to `target/cabt-selfplay/replay.jsonl`. The agent contract is the CABT one —
observation dict in, option-index list out:

```python
import random
from magic_cabt import CabtBridge, load_decklist

deck = load_decklist("examples/basic_deck.txt")   # "24 Forest" per line
with CabtBridge() as bridge:
    response = bridge.game_start(deck, deck, seed=7, max_turns=20)
    while not bridge.finished:
        select = response["observation"]["select"]
        count = random.randint(select["minCount"], select["maxCount"])
        picks = random.sample(range(len(select["option"])), count)
        response = bridge.game_select(picks)
    print(bridge.result["winner"])
```

Key docs:

- `Mage.Server.Plugins/Mage.Player.AI/docs/cabt-decision-surface.md` — the
  audited decision surface and fail-closed policy.
- `Mage.Server.Plugins/Mage.Player.AI/docs/cabt-verification.md` — test
  layers and failure-to-feature mapping.

### Live MTG Arena mirror (`magic_cabt.arena_mirror`)

Follow a live MTG Arena game from its `Player.log`, mirror the board into an
XMage window in real time, and record a CABT-format replay bundle — the same
`observation.select` option-index shape the engine bridge emits, so Arena
games and self-play games share one dataset schema.

```
Arena Player.log ──tail──▶ normalizer ──▶ GRE state tracker ──▶ snapshot
                                    │                              │
                            decision prompts + responses           ├─▶ XMage display (puppet game, GameView)
                            (indexed option lists)                 └─▶ CABT bundle (decisions.jsonl, mirror_states.jsonl)
```

One-time setup (needs JDK 17 + Maven; copies this overlay into an XMage
checkout, builds it, and prewarms XMage's card database):

```powershell
scripts\setup-arena-mirror.ps1 -XmageDir C:\path\to\xmage-checkout
```

Then, with MTG Arena set to log detailed data (Options → Account → "Detailed
Logs (Plugin Support)"), launch the GUI (or the CLI) and play a match:

```powershell
scripts\arena-mirror-gui.ps1               # window: Locate MTGA logs, live
                                           # log + actions panes, Start/Stop
scripts\arena-mirror.ps1 live              # CLI: follows the default Player.log
scripts\arena-mirror.ps1 live --from-start # also process the current log first
scripts\arena-mirror.ps1 replay <bundle>   # watch a recorded bundle back
```

The **GUI** has a "Locate MTGA logs" button, shows the log/status feed and the
recorded actions as they happen, and — with "Open XMage on live game" checked
— launches XMage automatically once a live game appears in the log, then
mirrors the current game while recording. XMage's own audio is muted for the
mirror window (the user's persisted XMage sound settings are left untouched).

Each live run writes `arena-mirror-runs/<timestamp>/`:

- `decisions.jsonl` — one CABT decision per line: the pre-decision board
  observation, the indexed legal-option list, and the indices the Arena
  player actually chose.
- `mirror_states.jsonl` — every board snapshot streamed to the display; also
  the replay playback stream.
- `game_history.jsonl` (raw game-state payloads redacted), `summary.json`,
  `card_cache.json`. `--raw-audit` (CLI) additionally writes an unredacted
  `raw_audit.jsonl` for debugging.

Hidden information is enforced, not assumed: any hand not owned by the log's
own seat (and every hand until the local seat is known) is redacted to
face-down placeholders — no grpId, name, or type line — **even if Arena
describes the card**. Enrichment refuses to resolve a name for a redacted
card, and raw Arena game-state payloads are kept out of the default bundle.
Card names for visible cards resolve from MTG Arena's own local card database.

Verified against real multi-match captures: decision prompt↔response pairing
is exact (respId == msgId) at 1145/1145 across four logs (17 games), a
simulated live tail reproduces the batch parse byte-for-byte, and replaying a
recorded bundle re-emits every state and decision in the original order
(`python -m unittest tests.test_arena_mirror tests.test_arena_mirror_e2e`).

## History note

The initial commits import the code feature by feature in the order the
features were built. Files are imported at their final state, so each commit
reads as one feature, but intermediate commits are not individually
buildable snapshots.
