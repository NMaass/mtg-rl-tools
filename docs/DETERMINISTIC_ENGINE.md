# Deterministic XMage reference engine

This directory exposes the pinned XMage rules implementation as a deterministic, agent-facing reference engine. It is the parity oracle for replay branching, offline labels, agent-vs-agent games, and any future faster Rust/Wasm implementation.

## Contract

A native root is identified by:

- pinned XMage revision `fd40ad5c29a92cef824cf12ba6d0e4daa25db975`;
- bridge/setup protocol version;
- two declared deck lists;
- integer seed;
- maximum-turn setting;
- the complete ordered prefix of legal bridge selections.

For that tuple, a fresh engine process must reproduce:

1. the same perspective-safe observation at every decision boundary;
2. the same legal semantic action list and stable option order;
3. the same selecting player seat;
4. the same private semantic verification digest; and
5. the same successor when the same semantic selection is applied.

A checkpoint is therefore a deterministic reconstruction recipe, not an in-memory JVM snapshot. Restore replays the prefix from the original seed and refuses the root if either the public/action fingerprint or the private digest differs.

## Stable identity

XMage runtime UUIDs are not reproducible across JVMs and are never the deterministic protocol identity.

CABT assigns:

- players: `P0`, `P1`;
- initial deck cards: `P<seat>:D<declared-slot>`.

Object views expose both the runtime `objectId` and the cross-process `semanticId`. Legal options similarly carry stable refs such as `sourceRef`, `targetRef`, `attackerRef`, and `blockerRef`. Runtime UUIDs remain available only for the live engine adapter.

The semantic observation fingerprint preserves those refs while scrubbing runtime UUIDs. This is important when two physical copies have the same card name.

## Randomness and concurrency

XMage's `RandomUtil` RNG is process-global. For deterministic games the bridge seeds it immediately before `game.start`, after cards and players have been constructed.

A seeded game therefore owns the RNG for the duration of the game. The bridge serializes seeded sessions inside one JVM. Scale deterministic actors with **one JVM process per worker**, not multiple concurrently running seeded games inside one JVM.

## Verification layers

The public fingerprint covers the perspective-safe game state, selecting seat, decision bounds, and semantic legal options. It intentionally excludes the recorded choice and hidden opponent information.

The private digest additionally covers hidden card identity/order, life/pass/in-game state, land-use state, player counters and floating mana, battlefield semantic state, stack, exile, and command state. It is a semantic verification digest, not a byte-for-byte serialization of every XMage field or RNG internals.

The strongest proof is behavioral: CI starts fresh JVM-backed sessions and compares every decision boundary, then reconstructs a checkpoint and takes the same alternative branch twice.

`services/engine/test_native.py` must prove:

- stale-root selections fail without advancing the game;
- a recorded prefix reconstructs the same public and private root;
- repeated same-seed runs produce identical traces;
- a different seed changes hidden state;
- creature/combat and targeted-spell rules paths are reproducible;
- two independent restores of one root produce the same alternative successor.

The pinned-engine Maven suite must also pass before this engine is considered usable.

## Agent interface

`services/engine/native.py` is newline-delimited JSON on stdin/stdout.

Start a game:

```json
{"command":"new","spec":{"decks":["24 Forest\n36 Grizzly Bears","24 Forest\n36 Grizzly Bears"],"seed":7,"maxTurns":20}}
```

The response contains the current observation, a public `fingerprint`, a private `engineFingerprint`, and the legal `observation.select.option` array.

Apply one legal choice using the fingerprint you observed:

```json
{"command":"step","selection":[0],"fingerprint":"<public fingerprint>"}
```

Save a deterministic root:

```json
{"command":"checkpoint"}
```

Restore it in a fresh process/session:

```json
{"command":"restore","checkpoint":{ "...":"checkpoint returned above" }}
```

Agents should choose only from the supplied option array. They do not need to implement Magic legality.

## Build and CI

The GitHub workflow `.github/workflows/engine-reference.yml` fetches the exact upstream XMage commit, overlays the CABT module, builds it, runs the bridge tests, then runs the native determinism/restore/branch harness. Its uploaded `pinned-xmage-evidence` artifact contains the Maven reports and native verification report.

Do not treat a skipped native step as a pass.

## What this proves—and what it does not

This gives the project a deterministic **XMage reference engine** through the declared CABT decision surface. Rules behavior is XMage's pinned implementation rather than a reimplementation inferred from logs.

It does not prove that XMage itself is bug-free, that every Magic card/mechanic is covered by our bridge, or that the private digest serializes every internal watcher/effect field. Unsupported decision callbacks fail closed. A future Rust/Wasm engine must be differential-tested against this reference and independently against official-rules fixtures where XMage behavior is disputed.

Imported Arena or MTGO view-only replays are not native roots unless enough information exists to reconstruct the authoritative hidden engine state.
