# Post-step Jev replay review

## Run

Use the existing `Arena Mirror.bat` / `scripts/arena-mirror-gui.ps1`, or:

```sh
cd python
python -m magic_cabt.arena_mirror gui
```

The **Replays** tab now has a persistent Jev panel on the right side of the
existing replay viewer. Select a replay and use **Watch** plus the normal transport
controls on the left. Paste an OpenRouter key directly into the masked field in
the Jev panel.

The panel follows the replay frame. A manual step, jump, or scrub seek updates the
replay first; when the resulting frame is a captured hero priority decision, Jev
is scheduled afterward. Normal autoplay does not issue Jev calls. Opening or
selecting a replay does not make a paid call. **Analyze / retry** explicitly
analyzes the currently displayed priority decision, and **Analyze after manual
step** can be disabled for fully manual requests.

Only one request runs at a time per replay panel. Fast stepping coalesces unsent
work to the newest position. Revisiting an identical request uses a session cache.
Failed requests are never automatically retried. Usage totals and caching are per
application session.

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
exported reports, requests' JSON bodies, or error messages. **Clear** in the
embedded panel disconnects the review session; closing the application clears the
in-memory key. A request
already sent cannot be recalled and may still incur a charge. The application
does not clear the operating-system clipboard or guarantee secure memory erasure.
HTTPS goes only to OpenRouter, with no redirects. OpenRouter forwards the submitted
game context to its provider; use this feature only for replays you permit it to
process. No raw account names, responses, match results, raw audit files, successor
states, or future replay frames are used as model input. The exported report does
contain game data and the local human/model comparison.

## Which recordings work?

- Existing Arena bundles: `decisions.jsonl` plus optional static `card_cache.json`.
- Native XMage `game-NNNN.jsonl` streams use the same readable projection layer,
  although the primary UX in this PR is the existing Arena replay library.
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

The Jev panel is intentionally beside the existing replay transport instead of
duplicating the board in another window. Existing Watch remains the visual
playback. Raw Arena/XMage transport identifiers are translated to readable
card/object names from the captured state and card cache before they are shown or
sent as semantic action labels; unresolved identities are called out rather than
presented as if an ID were a card name. These remain retrospective recommendations
computed from pre-action snapshots, not proof of predictions made before the human
acted.

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
