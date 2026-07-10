# Target fungibility and input masking

## The problem

If an opponent controls two identical tokens, choosing between them as
*targets* is strategically meaningless — but the two *bodies* both matter for
board presence. The pipeline must treat these two facts differently and
deliberately:

- **Action space**: identical targets should collapse to one canonical
  action. Otherwise a uniform policy double-counts them, an IL model splits
  its gradient across interchangeable labels, and per-game instance ids leak
  into features as noise.
- **State encoding**: the two bodies must stay visible as multiplicity
  (`Zombie 2/2 x2`), never collapsed away.

## Design decisions

1. **Fungibility is proven by the capture source, never guessed from
   labels.** Each option's `payload` may carry a `canonicalKey`: a fingerprint
   of everything strategically relevant about the object, *excluding* its
   instance id. Options without a key are never merged. There are no
   label-text heuristics.

2. **The recorded option list is never mutated.** `records.py` locks
   `option[i].index == i` and historical `selectedIndices` reference concrete
   positions, so dedup is a *derived view*
   (`magic_cabt.training.action_dedup.canonical_groups`) computed on top of
   the intact list. Board presence is unaffected either way — bodies live in
   the battlefield zone of the observation, not in the option list.

3. **Execution re-expands to a concrete instance.**
   `representative_index(group)` picks the group's lowest option index; its
   payload still carries the real `targetInstanceId` / `targetId`, so the
   engine paths (GRE response, `game_select`) need no changes.

4. **Feature text is identity-free.** `features.canonical_text` strips
   `instance=<n>` tokens and UUID literals from all option/state feature
   text; card identity (`name`, `grpId`) is preserved. Zone summaries
   aggregate identical objects into explicit counts with P/T and tapped
   state, so state features carry multiplicity while option features are
   invariant to instance relabeling.

## Fingerprint contents

A `canonicalKey` folds in (order-insensitive, serialized deterministically):

| Field | Why |
| --- | --- |
| card identity (`grpId` / name) | different cards are never fungible |
| controller and owner | your token ≠ the identical one they control |
| power / toughness | already reflect counters, anthems, auras |
| tapped, attacking/blocking | affects legality and combat value |
| counters (sorted) | +1/+1, loyalty, etc. |
| damage marked | a damaged body dies to less |
| card types / subtypes | changelings, animated lands |
| attachments (sorted card ids) | a Pacifism changes strategy without changing P/T |

Face-down or unknown objects get **no** key: hidden information can differ in
ways this perspective cannot see, so they are always treated as distinct.

## Where each source computes the key

- **Arena (implemented)**: `GameStateTracker.canonical_object_key` in
  `arena_mirror/tracker.py`, applied to target options at prompt time by
  `ArenaMatchTracker._fingerprint_target_options`. It must run in the tracker
  (not `options.build_prompt`) because only the tracker holds full object
  state.
- **XMage bridge (follow-up, Java)**: `CabtTargetOptionFactory` currently
  serializes only `targetId`/`targetName`/`zone` — not enough to prove
  fungibility (it would wrongly merge a tapped and an untapped copy). The
  fix mirrors what `MagicObjectViewFactory.permanentView` already extracts
  (P/T, tapped, counters, controller) plus `getAttachments()`, hashed into a
  `canonicalKey` payload field. Until that lands, XMage target options carry
  no key and are simply never merged — safe, just uncompressed.
- **Combat options (phase 2)**: the same fingerprint applies to
  attacker/blocker options in `CabtCombatOptionFactory` and the Arena combat
  builders; not yet wired.

## How training consumes it

- IL: fold a recorded concrete index through `group_index_of` so a human who
  clicked "the second identical token" trains the same canonical action as
  one who clicked the first.
- RL: `canonical_groups` gives the policy one action per group; the mask is
  over groups, and sampling weight no longer scales with how many
  interchangeable instances the engine happened to list.
