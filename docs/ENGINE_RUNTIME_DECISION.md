# Updated engine runtime decision — 19 September 2026

This updates the ordering in `BROWSER_REWRITE.md` section 2. Rust remains the preferred language **if** we write a replacement. The next experiment should instead test compiling the existing engine, because maintaining fewer independent rule implementations is the stronger correctness strategy.

## A relevant implementation already exists

Manabrew's 30 August 2026 article describes its production-but-experimental browser build of Java Forge using GraalVM Web Image. It reports byte-identical callback traces against its JVM reference for a bounded seed/turn experiment. It also explicitly says rollback is absent. This is evidence that compiling a mature Java Magic engine into a browser is feasible, not evidence that XMage compiles unchanged or that replay branching is solved.

Primary source: https://manabrew.app/blog/graalvm/

GraalVM's current documentation describes Web Image as an experimental Native Image backend producing Wasm plus a JavaScript runtime wrapper. It requires Oracle GraalVM 25.1+ and Binaryen 119+, uses newer Wasm GC/exception features, and is not a standalone WASI binary. Browser viability does not automatically imply Workers viability. Evaluate build/runtime licensing, browser features, memory and resource packaging before adoption.

Primary sources:
- https://www.graalvm.org/latest/reference-manual/web-image/
- https://www.graalvm.org/release-notes/25.1/
- https://github.com/oracle/graal

## Recommended order

1. Keep the pinned JVM implementation as the comparison reference, separate from the React UI.
2. Try GraalVM native-image for a server/container build. Keep the same decision protocol and run identical golden transcripts against the JVM.
3. Try Web Image for a browser-worker build. Replace only runtime boundaries such as controller waits, resource access, and thread assumptions; do not rewrite card rules merely to make the compiler accept them.
4. Use Cloudflare Containers for authoritative execution until the Wasm candidate passes compatibility and resource gates. Browser engines are suitable for local analysis, but cannot hide omniscient state from a person controlling that browser.
5. Consider a Rust native/Wasm port only when the retained reference and measured bottleneck justify its considerable semantic maintenance cost.

A target must match legal prompts, consequences, object relationships, reveals, terminal state and declared RNG continuation. A visually matching board or successful compilation is not enough. Do not substitute Forge parity for XMage parity. Do not change this repository's license by importing GPL/AGPL implementation code without a deliberate licensing decision. This branch has not copied Manabrew or Forge code.

## What our native experiment actually found

The first process-restart replay experiment failed at the opening-hand decision, before strategic play. Both runs used seed 7 and the same two declared decks, but one visible hand contained three Forests and four Grizzly Bears; the reconstructed run contained one Forest and six Grizzly Bears. `engine-native-divergence.json` preserves the expected and actual observations. This is not merely a transport-ID mismatch.

The pinned `Deck.getMaindeckCards()` filters a `LinkedHashSet` through `Collectors.toSet()`, discarding insertion order. `PlayerImpl.useDeck()` feeds that set into the library. A seeded shuffle is not reproducible when the starting sequence is not reproducible.

The CABT duel adapter now restores the declared main-deck insertion order after setup and before the engine's normal shuffle. It uses the existing library API and excludes extra-deck cards. This is a versioned experiment boundary for **new native sessions**, not a claim that old captures can recover their missing initial order. The original engine still supplies all rules, random shuffling, legal choices and resolution. A regression test checks exact initial ordering across fresh randomized object IDs; process-restart tests still compare the observations without relaxing their equality check.

Source paths at pinned XMage `fd40ad5c29a92cef824cf12ba6d0e4daa25db975`:
- `Mage/src/main/java/mage/cards/decks/Deck.java`, `getMaindeckCards`
- `Mage/src/main/java/mage/players/PlayerImpl.java`, `useDeck`
- `Mage/src/main/java/mage/players/Library.java`, `shuffle`

Future native roots need a complete setup manifest: engine/adapter build, card data, initial order, entropy/RNG contract, and full action prefix or a verified serialized snapshot. The current runner's semantic fingerprints are an initial divergence gate, not a proof of full hidden-state or RNG equality. Other unseeded randomness or unordered collections may remain; a successful bounded test is not permission to claim arbitrary-Magic determinism.

## Browser milestone is separate

The current browser app already views and analyzes captured replays without running a rules engine. Its Wasm module is **the reused Python parser runtime**, not a compiled Magic engine. It never accepts a view-only Arena/MTGO replay as a complete native execution snapshot. Complete private-event knowledge reconstruction and hosted agent matches remain separate work, rather than unimplemented buttons in the viewer.
