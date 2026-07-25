# Strategic macro-action records

XMage exposes the complete legal decision surface through independent player
callbacks. That is correct for execution, but one strategic play may span many
callbacks:

```text
PRIORITY: cast Lightning Bolt
CHOOSE_TARGET: opponent
PLAY_MANA: Mountain
```

Treating those rows as three unrelated strategic actions overweights interface
choreography and attaches the post-action state to the wrong semantic unit.
`magic-cabt-build-macro-actions` derives a conservative grouping view without
changing the canonical `DecisionRecord` format.

## Command

```bash
magic-cabt-build-macro-actions \
  --input arena-mirror-runs/run-001/decisions.jsonl \
  --out runs/run-001/macro_actions.jsonl \
  --transitions-out runs/run-001/transitions.jsonl
```

Inputs may use any format supported by `iter_decision_records`. Repeat
`--input` to process several files. Files are never grouped across their
boundaries.

Useful filters:

```text
--strategic-only   omit deterministic/non-strategic roots
--complete-only    omit groups without an observed boundary state
```

The unfiltered output is the audit artifact. Filters should be applied only to
training copies so incomplete and orphaned records remain measurable.

## Grouping rules

Grouping is deliberately conservative:

1. `PRIORITY`, mulligan, combat, starting-player, and special-action prompts
   begin roots.
2. Target, mode, amount, pile, and use prompts may attach to an open same-seat
   root.
3. Mana/payment/cost prompts may attach to an open same-seat root.
4. Trigger/replacement/ordering prompts may attach to an open same-seat root.
5. An unknown prompt always starts a new root. It is never silently attached to
   the previous play.
6. A game change, acting-player change, or new root closes the current group.
7. The following record's observation becomes the macro action's boundary
   state. An explicit `nextObservation` takes precedence.
8. EOF without either boundary is emitted as `complete: false`.

Parameter prompts encountered without a preceding root are retained as
`orphanedParameterGroup: true`. They are evidence of capture gaps or a prompt
family that needs a better classifier; they are not discarded.

## Schema

Each `MacroActionRecord-v1` includes:

```json
{
  "recordType": "MacroActionRecord",
  "groupId": "macro:[\"game-id\",1,9]:0:184",
  "playerIndex": 0,
  "rootSequenceNumber": 184,
  "lastSequenceNumber": 186,
  "rootObservation": {},
  "rootSelect": {},
  "rootSelectedIndices": [2],
  "steps": [
    {
      "sequenceNumber": 184,
      "promptType": "PRIORITY",
      "classification": {
        "role": "root",
        "confidence": "known-family"
      },
      "selectedOptions": []
    }
  ],
  "macroAction": {
    "promptType": "PRIORITY",
    "selectedOptions": [],
    "parameters": [],
    "strategic": true,
    "deterministic": false
  },
  "nextObservation": {},
  "complete": true,
  "completionReason": "new-root"
}
```

`selectedOptions` contains complete option semantics, including
`payload.canonicalKey` where the capture source supplied it. Concrete indices
remain present for replay provenance but are not treated as stable action
identity.

## Transition output

`--transitions-out` writes complete groups in the existing JEPA transition
shape:

```json
{
  "source": "macro_actions",
  "prev": {},
  "next": {},
  "action": {},
  "deltas": {},
  "outcome": {},
  "macroActionId": "..."
}
```

This file can be named `transitions.jsonl` and passed to the current structured
JEPA/RSSM training loaders. Its `horizon` is the number of CABT decisions
contained in the macro action, not a claim about turn distance.

## Trust boundary

This compiler does not replace the engine decision-surface audit and does not
claim that prompt-name heuristics are authoritative rules semantics. Every step
retains its classification reason and confidence. Unknown callbacks fail open
as separate roots, preventing accidental action merging.
