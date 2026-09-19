# Browser rewrite: evidence, boundaries, and engine strategy

This is a staged replacement, not a claim that all of Magic has been reimplemented. The browser workspace, private storage and parser adapter are independently testable. A native rules engine stays behind a separate interface. UI mock responses are not evidence of playing strength.

## 1. Exact compatibility comes before a new implementation language

The strongest way to preserve XMage behavior is to execute the same pinned engine, card definitions and RNG implementation. This branch pins upstream `fd40ad5c29a92cef824cf12ba6d0e4daa25db975`. It overlays the existing CABT bridge and adapts Maven's test-source layout; it does not translate card rules into a guessed universal action enum.

A new engine must specify what "same" means: equivalent legal decision prompts, consequences, trigger/replacement ordering, turn/priority transitions, random choices under an explicit RNG contract, terminal results, and information revealed to each observer. Engine bugs and official-rules disagreements require separate fixtures and a deliberate policy; silently fixing them is not exact behavioral cloning.

XMage's README describes roughly 9,000 tests and about 80% coverage. Those are upstream claims, not measurements made by this PR, and neither number proves complete rules/card correctness. The pinned workflow runs the bridge module tests and a bounded native continuation smoke test. It does NOT run every upstream card test or establish 80% coverage for our bridge.

Reference: https://github.com/magefree/mage

## 2. Language decision

**Recommendation for a replacement: Rust, built both natively and to WebAssembly.** This is a design judgment: explicit ownership, typed state/decisions, compact immutable snapshots, and shared native/Wasm tests are a good fit. Do not choose Rust merely to remove Java syntax, and do not infer throughput improvement without a measured baseline.

| Route | What it preserves | Principal cost | Decision |
| --- | --- | --- | --- |
| Pinned JVM in Cloudflare Containers | Existing card behavior and implementation | JVM resource use; no browser engine | Reference and practical first backend |
| Rust native + Wasm | Shared engine across actors/browser/Workers | A very large semantic port and validation burden | Preferred replacement after differential harness |
| C++ + Wasm | Mature native tooling; existing fast simulation approaches | Manual memory/ownership complexity; still a full port | Viable, not the default for this team |
| TypeScript engine | Shared UI/domain language | Runtime cost and another large rule/card implementation | Good adapter language, not a shortcut to full rules |
| TeaVM Java to WasmGC | Potential reuse of original Java code | Reflection/resource/classloading/threading dependencies; no drop-in proof | A bounded feasibility experiment, not the promised engine |

TeaVM 0.14 improved reflection support and removed its old non-GC/WASI backends. It would be misleading to say either "Java cannot run in Wasm" or "TeaVM compiles XMage unchanged". Test a reachable headless core and its resource/reflection surface before deciding.

References:
- https://teavm.org/docs/intro/overview.html
- https://teavm.org/docs/release-notes/0.14.0.html
- https://developers.cloudflare.com/containers/

## 3. Existing projects worth evaluating, not blindly adopting

Manabrew exposes a Rust/Wasm/browser route with a Forge-compatible layer and a parity harness against the Java Forge engine. That is a relevant methodological reference, but Forge parity is not XMage parity. Its stated Rust coverage remains in progress and its AGPL/GPL licensing cannot be silently mixed into this MIT repository.

Phase is another Rust/Wasm Magic project. A large number of parsed cards is not evidence that all their interactions implement the rules correctly. Both projects deserve a bounded common-scenario evaluation before adopting code or changing licenses.

References:
- https://github.com/witchesofthehill/manabrew
- https://github.com/phase-rs/phase

## 4. Differential port process

1. Pin engine revision, card catalogue, deck lists, seed, bridge schema and build hash. Persist these with each native execution root.
2. Record every engine prompt and response. The environment supplies valid choices; agents rank them. Compound targets, modes, payments and combat remain typed sequential decisions, not a combinatorial flat label.
3. Run candidate and reference against the same transcript. Compare actor-specific observations, legal options, events and terminal state at every decision boundary. Preserve object relationships while canonicalizing transport IDs. A signature that ignores all identity fields is insufficient to prove relationship equivalence.
4. Minimize the first divergent trace into a regression fixture. Separately maintain official-rules expectations so a shared engine bug cannot validate itself.
5. Add invariant tests: no future-information leakage; identical visible history implies identical agent input; hidden-card changes do not alter the opposing input; snapshot/load round-trip preserves legal options and RNG continuation; permutations of candidate order preserve semantic choices.
6. Port bounded mechanics/card families. Only mark them supported when the reference and candidate pass the declared corpus. Unsupported mechanics fail closed.
7. Benchmark warmed long-lived actors at 1/2/4/8/16 workers with the same workload and logging policy. Report decisions/second, games/second, RSS, GC, startup and failures separately. Only then decide whether simulation warrants a port.

The native runner in `services/engine/native.py` implements observe/step/checkpoint/restore/autoplay over the existing engine. A checkpoint stores original decks, seed and the entire action prefix; restore checks every recorded prompt fingerprint. This is deterministic reconstruction, not O(1) native state cloning. The current semantic signature is inherited from the existing replay-search module; it is a useful divergence gate, not a bit-for-bit full hidden-state proof. Imported Arena/MTGO view-only recordings are never accepted as native checkpoints.

## 5. Separate execution state from knowledge

An authoritative engine state includes hidden hands, ordered libraries, triggers, effects, RNG and pending decisions. It is not a model input. A perspective view contains only public state plus facts that player has learned.

A future information ledger must be event-derived with explicit recipients. For example: an omniscient import can retain both hands, but a private scry event grants its ordered-card knowledge only to the scrying player. Public reveal grants knowledge to both; shuffle invalidates known library positions; hidden zone transitions need knowledge invalidation rather than permanent identity tracking.

The browser contract already stores independent `views[seat]` and `decisions[seat]` per frame. Analysis selects one view and excludes recorded choice, later frames and other views. Face-down identities are redacted. A view absent from the recording is unavailable, not synthesized. The current Arena adapter reuses the existing snapshot tracker and does NOT yet reconstruct a complete private-event knowledge ledger. This limitation is surfaced in replay warnings.

## 6. Browser and Cloudflare shape

- React/TypeScript: actual board, timeline, one persistent right-side analysis panel, keyboard transport and minimal context-specific controls. No Tk or separate analysis window.
- Web Worker + self-hosted Pyodide: original Arena normalizer/tracker and MTGO text parser execute on the user's device. No OCR subprocess is executed. The adapter is the new part, and native Python/Wasm parity is tested on the same fixture.
- Cloudflare Worker: private API, JWT verification, encryption, upload validation and provider gateway.
- D1: replay index, encrypted per-user key, analysis cache, ratings and call-budget counter.
- R2: parsed replay objects. Raw logs remain local. R2 objects are never exposed through public bucket URLs.
- Cloudflare Access: initial private-beta sign-in. Verify issuer, audience, signature and subject; do not trust an email/header without JWT validation.
- Cloudflare Containers: appropriate future hosted location for the pinned native engine. It can keep the runtime 100% on Cloudflare without pretending a JVM is a Worker/Wasm module. Native engine hosting is not enabled by this browser PR's default deployment.

The key is AES-GCM encrypted with a Worker secret, unique nonce and owner-bound associated data. It is never returned to the browser, stored in localStorage, committed, or included in a replay export. This protects database contents, not against an attacker who controls both the Worker and its encryption secret. Rotation requires decrypt/re-encrypt with the prior secret; simply replacing the secret makes old records unreadable.

References:
- https://developers.cloudflare.com/workers/platform/limits/
- https://developers.cloudflare.com/d1/best-practices/local-development/
- https://developers.cloudflare.com/containers/reference/container-class/
- https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/authorization-cookie/validating-json/
- https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request

## 7. Import and UX decisions

A website cannot silently discover arbitrary MTGA/MTGO files. User-initiated file selection is the default; directory handles/polling can be a later Chromium enhancement, not a cross-browser promise. Import parses locally, previews the game count, and asks before uploading a parsed replay.

Arena setup follows the concise 17Lands sequence: Settings -> View Account -> Detailed Logs (Plugin Support), restart, then play. Windows Player.log lives under the user's AppData/LocalLow/Wizards Of The Coast/MTGA directory; macOS uses ~/Library/Logs/Wizards Of The Coast/MTGA. The file picker is the final authority because installs vary and logs can rotate.

MTGO support here is copied/exported text game-log input processed by the existing `mtgo_video.parse` and `state` modules. It is NOT a native .dat decoder or a full-fidelity private MTGO capture. Reconstruction warnings and missing legality are retained. We should inspect real user-provided native files before claiming to support their location/encoding.

The interaction model borrows the useful structure of chess analysis tools: stable board, adjacent evaluation, selectable move list, keyboard previous/next, and a scrubber. It does not copy another product's artwork. No engine-strength curve, future move tree, live coaching switch or multi-format dashboard is shown before the corresponding feature works. No API call on opening or autoplay; auto-analysis requires opt-in and occurs after manual navigation. Requests are bounded and coalesced; late responses stay attached to their original position.

References:
- https://www.17lands.com/getting_started
- https://developer.chrome.com/docs/capabilities/web-apis/file-system-access
- https://github.com/lichess-org/lila/tree/master/ui/analyse

## 8. Acceptance and remaining work

Required evidence: production build/typecheck; owner/key/origin tests; D1/R2 round-trip; no outcome/other-view leakage; browser keyboard/focus/geometry tests; same-fixture Python/Wasm parser equality; native engine smoke/restore results. A skip is not a pass. CI artifacts preserve screenshots, traces and the built app.

Not claimed complete: arbitrary full-fidelity MTGO imports, automatic watched directories, all-private-event knowledge tracking, exact branching of view-only replays, a Rust rules engine, whole-XMage differential equivalence, engine throughput scaling, hosted agent tournaments, or production deployment. These are not hidden behind buttons that imply they work.
