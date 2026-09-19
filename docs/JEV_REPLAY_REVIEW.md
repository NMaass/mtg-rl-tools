# Post-step Jev replay review

## Run

Use the existing `Arena Mirror.bat` / `scripts/arena-mirror-gui.ps1`, or:

```sh
cd python
python -m magic_cabt.arena_mirror gui
```

On the **Replays** tab, paste an OpenRouter key into the masked field, select a
match from the existing library, and click **Review with Jev**. The review window
shows the recorded position on the left and action rankings on the right. Existing
**Watch** playback in XMage is unchanged. Review itself needs no JVM or GPU.

**Next priority** / **Previous** (or arrow keys) select recorded priority decisions,
including recorded passes. The board updates first; analysis is then scheduled.
Opening a replay does not make a paid call. **Analyze / retry** analyzes the first
position or explicitly retries a failed request. Turn off **Analyze after stepping**
for manual-only calls. Empty-key mode allows offline inspection; reopen the review
after entering a key to enable calls.

Only one request runs at a time per review window. Fast stepping coalesces unsent
work to the newest position. Revisiting an identical request uses a session cache.
Failed requests are never automatically retried. Keep one review window open when
measuring total spending; usage totals and caching are per window.

The panel shows Jev's action probabilities, its selected action, the recorded
choice, latency, input/output tokens, and actual `usage.cost`. Missing costs stay
**unknown**, not zero. Choice probabilities are not win probabilities, and matching
the human move is not proof of correctness. Mark results Useful / Wrong / Unsure
and export JSON to retain inputs, model version, request hashes, call costs,
failures, human ratings, and agreement counts. Export is explicit and atomic;
there is no automatic analysis or key persistence.

## API and privacy

This uses `POST https://openrouter.ai/api/alpha/decisions`, **not chat completions**,
and pins `typesafe/jev-1.13`. The question has `type: choice` and criteria keyed by
captured option IDs. `state` contains separate `gameState` and `possibleActions`
objects. The provider must return probabilities for exactly those IDs. Invalid
answers are rejected while retaining any reported charge.

Reference: [OpenRouter Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request).
The alpha API may change. There is no silent model/endpoint fallback.

The key stays in process memory, is masked, is never included in GUI settings,
exported reports, requests' JSON bodies, or error messages. **Clear key** in the
library disconnects all review windows; closing clears their copies. A request
already sent cannot be recalled and may still incur a charge. The application
does not clear the operating-system clipboard or guarantee secure memory erasure.
HTTPS goes only to OpenRouter, with no redirects. OpenRouter forwards the submitted
game context to its provider; use this feature only for replays you permit it to
process. No raw account names, responses, match results, raw audit files, successor
states, or future replay frames are used as model input. The exported report does
contain game data and the local human/model comparison.

## Which recordings work?

- Existing Arena bundles: `decisions.jsonl` plus optional static `card_cache.json`.
- **Open decision file...** also accepts native XMage `game-NNNN.jsonl` streams from
  `examples/run_selfplay.py` / `magic_cabt.eval.play`.
- Supported prompt types: Arena `ACTIONSAVAILABLEREQ` and XMage `PRIORITY`, with
  exactly one indexed root choice. Targets, modes, combat assignments, and mana
  payment subprompts are intentionally out of scope for this first review slice.
- Missing options, mismatched responses, missing hero perspective, terminal
  snapshots, malformed options, or oversized inputs fail closed before any call.
- Board-only/video replays without captured action choices cannot be analyzed.
- Empty responses are not converted into an invented pass. A pass is ranked only
  when the captured prompt includes it. This cannot measure unrecorded auto-yields.
- The adapter only sends captured, perspective-safe fields; it does not invent
  missing mana, rules text, effects, or opponent knowledge. Missing card names
  weaken analysis. Static cache enrichment is limited to already-visible IDs;
  future replay states are never scanned to recover card identities.

The review panel is a read-only structured position view, not a replacement for
XMage's card-art display. Existing Watch remains available for visual playback.
These are retrospective recommendations computed from pre-action snapshots, not
proof of predictions made before the human acted.

## XMage findings: no rules-engine rewrite required

The existing `CabtBridgePlayer` obtains playable abilities from
`Player.getPlayable(Game, true)`. `CabtPriorityPromptBuilder` emits pass plus indexed
choices. `CabtPriorityOptionFactory` supplies source, ability, rule, and mana-cost
details for land plays, spells, activated abilities, and special actions. The
Python `CabtBridge` exposes this under `observation.select.option`.

This already supplies the requested root action space for native engine games.
Arena's recorded GRE action prompts provide the analogous captured option list.
A mirrored visual snapshot is not an independently reconstructed full engine
state: the reviewer therefore does not ask a partial mirror to invent legality.
Where choices are absent, improve capture or deterministically rebuild and verify
an engine replay root; changing implementation language cannot recover missing
information. No XMage performance benchmark or new engine-legality validation is
claimed by this PR, and no engine rewrite is included.

## Verification

```sh
cd python
python -m unittest discover -s tests -p 'test_jev_review.py' -v
xvfb-run -a python -m unittest discover -s tests -p 'test_jev_review*.py' -v
```

Tests cover request projection, hidden/future information exclusion, native and
Arena choices, invalid and missing options, exact typed-response validation,
unknown billing, endpoint/auth handling, duplicate suppression, coalescing,
explicit retries, offline mode, post-step timing, stale responses, focus/geometry,
feedback persistence, key-free export, and the existing library integration.
A dedicated CI workflow runs the GUI tests under Xvfb against the full checkout.
No real OpenRouter key is needed by the tests. A paid live Jev call and real-user
replay usefulness still require manual verification; synthetic provider fixtures
are not evidence of model strength or actual service latency.
