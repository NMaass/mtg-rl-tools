# Limited-play models: draft picks, deck construction, sideboarding

## Why one selection model, not JEPA layers

The gameplay JEPA (`docs/JEPA_PLAN.md`) is an action-conditioned world model:
its predictor, horizon embedding, and causal head exist to model board-state
transitions. Draft picks, limited deck construction, and Bo3 sideboarding
have no transitions to model — they are all one task shape:

> given a **context set of cards**, score **candidate cards**.

| mode        | context               | candidates                | positives          |
|-------------|-----------------------|---------------------------|--------------------|
| `draftPick` | pool drafted so far   | current pack              | cards picked       |
| `deckBuild` | deck built so far     | remaining pool            | cards still to add |
| `sideboard` | deck as last submitted| deck + sideboard          | deck actually kept |

So they share **one small model** (`models/draft_model.py`,
`CardSelectionModel`) with a mode feature, separate from the gameplay JEPA
but reusing its proven parts:

- the permutation-symmetric `StateEncoder` set transformer encodes the
  context set (a pool has no ordering identity, matching the fungibility
  invariant);
- cards are represented by **text embeddings** (`models/embeddings.py`, hash
  or sentence-transformer backend) plus a small numeric block (colors, types,
  mana value, P/T, pack/pick position), so unseen cards transfer through
  their rules text — no per-card id table to go stale between sets;
- grpIds resolve through the mirror's `CardDatabase` and the JEPA's
  `CardTextResolver` (`DraftCardResolver` merges both, degrades gracefully
  to bare grpId tokens).

Deck construction is unrolled into sequential inclusion decisions
(deck-so-far vs remaining pool). That same mode powers **mid-draft deck
outlook**: greedily auto-build from the current pool after any pick.

## Data flow

Arena `Player.log` already contains everything; `arena_log.py` now emits:

- `ARENA_DRAFT_PACK` — `Draft.Notify` pack contents (deduplicated);
- `ARENA_DRAFT_PICK` — `EventPlayerDraftMakePick` (`GrpIds` is a list:
  pick-two formats pick more than one card);
- `ARENA_DECK_SUBMIT` — `EventSetDeckV2` built-deck submission
  (quantities expanded);
- `ARENA_SIDEBOARD_PROMPT` / `ARENA_SIDEBOARD_SUBMIT` — Bo3
  `GREMessageType_SubmitDeckReq` and the client's `SubmitDeckResp`.

Known gap: Arena does not log a `Draft.Notify` for pick 1 of pack 1, so that
pick trains nothing (counted as `picks_missing_pack`, never fabricated).

## Commands

```bash
# 1. logs -> datasets (draft_picks / deck_builds / sideboards .jsonl)
magic-cabt-build-draft-dataset --log Player.log --log more.log --out data/

# 2. datasets -> checkpoint (all three modes by default)
magic-cabt-train-draft --input data/ --out runs/draft-model --epochs 30

# 3. pool -> deck outlook (greedy auto-build + colors/curve/cuts)
magic-cabt-draft-outlook --checkpoint runs/draft-model/checkpoint.pt \
    --picks data/draft_picks.jsonl        # or --pool "97967,97932,..."
```

Dataset notes: records are deduplicated across log copies by decision
content (not draft id — two seats of one pod share a draftId); a deck
submission logged in a different session than its draft recovers its pool
across sources; constructed-deck submissions keep `pool: null` and are
skipped by the trainer.

Fail-fast rules carry over from the gameplay pipeline: a pick that is not in
its joined pack raises; unalignable records are counted and skipped, never
patched to a different label.

## Status / next steps

- Pipeline validated end-to-end on 2026-07-11 logs: 40 pick decisions,
  4 deck builds, 10 sideboards (≈125 examples) → `local` preset checkpoint.
  At this volume the model memorizes; the value today is the pipeline and
  initial weights, which retrain in minutes as more logs accumulate.
- Scale options: keep feeding personal logs, and/or pretrain the
  `draftPick` mode on public 17Lands data (same example shape) before
  fine-tuning on personal picks.
- Live integration (draft mirror pane, outlook during a running draft via
  `StreamingNormalizer`) is a follow-up; the tracker already routes the new
  event types without breaking gameplay mirroring.
