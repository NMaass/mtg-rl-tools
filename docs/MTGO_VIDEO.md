# MTGO video ingestion

Turns MTGO gameplay footage into a structured game log, reconstructs the
board from it, and replays that board in XMage — then verifies the result
two independent ways.

```text
MTGO footage
  -> ffmpeg      crop the "Chat & Game Log" pane, one frame per second
  -> tesseract   OCR each frame
  -> reconstruct merge the overlapping scrolling windows into one log
  -> parse       MTGO's log grammar -> typed events
  -> catalog     resolve OCR'd card names offline against Scryfall data
  -> simulate    fold events into board snapshots (Arena mirror schema)
  -> XMage       replay the snapshots in the real client, and verify
```

The output is a bundle per game in the same `mirror_states.jsonl` format the
Arena mirror produces, so it replays through the existing
`magic_cabt.arena_mirror` machinery unchanged.

## Requirements

- `ffmpeg` and `tesseract` on `PATH`
- A built `Mage.Client` with the `cabtmirror` overlay, and
  `MAGIC_CABT_CLASSPATH` pointing at its classpath (see
  `scripts/setup-arena-mirror.ps1`; on macOS/Linux build with Maven and
  assemble the classpath by hand)
- Network access once, to build the card catalog

## Quick start

```sh
# One-time: build the local card catalog (~200MB download -> 0.8MB cache)
python3 -m magic_cabt.mtgo_video catalog

# Video -> per-game bundles
python3 -m magic_cabt.mtgo_video ingest match.mp4 --out bundle \
    --start 100 --end 640 --hero YourMTGOName --match-id league-1

# Timestamp inferred changes (combat damage) by locating them on screen
python3 -m magic_cabt.mtgo_video align bundle/game1 --video match.mp4

# Verify the decoded log against XMage's own rendering of it
python3 -m magic_cabt.mtgo_video verify bundle/game1

# Verify it against MTGO's on-screen life totals (independent ground truth)
python3 -m magic_cabt.mtgo_video crosscheck bundle/game1 --video match.mp4

# Replay in XMage and record it, beside the source footage
python3 -m magic_cabt.mtgo_video render bundle/game1 \
    --out xmage_replay.mp4 \
    --side-by-side comparison.mp4 --source-video match.mp4
```

`rebuild` re-runs parsing and simulation from a bundle's cached
`mtgo_log.json` without redoing OCR — use it while iterating on the parser
or the simulator.

## Bundle layout

```text
bundle/
  mtgo_log.json         reconstructed log entries
  mtgo_log_times.json   video timestamp each entry first appeared at
  mtgo_events.jsonl     parsed, name-resolved events for the whole match
  match.json            catalog provenance, cards in play, per-game summaries
  game1/
    mirror_states.jsonl board snapshots (Arena mirror schema)
    mtgo_events.jsonl   this game's events
    summary.json
    verification.json   log vs XMage
    hud_crosscheck.json log vs MTGO's on-screen life totals
```

Every snapshot carries `videoTime`, the moment its log line first appeared
on screen. That is what lets a decoded game be replayed in sync with the
footage it came from.

## The two verifications, and what each one proves

**`verify` — log vs XMage.** Feeds every snapshot through XMage's real
`MirrorStateApplier` and `GameView` (the same objects the client renders)
and diffs turn, life, library and hand counts, and the exact multiset of
battlefield and graveyard cards. This proves the mirror reproduces the
decoded game faithfully. It cannot tell you the decoded game is *right* —
XMage renders whatever the log claimed.

**`crosscheck` — log vs the footage's own HUD.** OCRs MTGO's on-screen life
totals and compares them against the life the simulator derived. This is
independent evidence from the same frames, and it catches what the first
check structurally cannot. It found two real bugs during development: combat
damage was missing entirely (MTGO logs no line for it), and a transformed
double-faced creature kept its front-face power, under-counting damage.

Only life is read from the HUD. The library and hand badges are ~10px tall
inside shaped icons and do not OCR reliably at 1080p, so they are
deliberately not used rather than reported at low confidence.

## Card names

Card names come from a local catalog condensed from Scryfall's
`oracle_cards` bulk file (201 MB in, 0.8 MB out), not from live lookups.
Live fuzzy lookups were tried first and were the wrong shape: rate limited,
network-dependent per run, non-deterministic across runs, and they 404 on
names they cannot match at all — losing a card the same log spells correctly
elsewhere.

Matching is:

1. exact hit on a folded key (casefold, strip diacritics and punctuation),
2. else fuzzy against the **run vocabulary** — the cards this log already
   named exactly — where a low threshold is safe because the candidate set
   is a few dozen names known to be in play,
3. else fuzzy against the whole catalog at a stricter threshold.

A match is accepted only if it also beats the runner-up by a margin. With
tens of thousands of names almost any string has a plausible neighbour, so
the margin — not the absolute score — is what separates "that card, misread"
from "not a card at all". It is why the garbled player name `Buz2Caldera` is
rejected rather than resolved to whatever it happens to resemble.

Every resolution is memoized, so one OCR string always decodes to the same
card. The catalog is restricted to cards that exist on MTGO: within that
pool the normalized keys are collision-free, and it excludes Alchemy cards,
which cannot appear in MTGO footage.

## Layout assumptions

`regions.py` targets a full-screen 1920x1080 MTGO client with the default
duel scene and the log docked top-right. The log crop deliberately stops
short of the scrollbar at x=1892: its arrows and thumb OCR as junk tokens
glued to the end of every wrapped line. Pass `--region WxH+X+Y` for other
layouts.

## Known limits

- Hand contents are never known: MTGO does not log them, so hands are
  rendered face-down at the right count.
- Library counts are derived from draws and mills starting at `--deck-size`,
  not read from the client.
- Combat damage is inferred from the attack, since MTGO logs no damage line.
  Blocks refund the blocked attacker's share. A creature whose power the
  catalog does not know is reported rather than guessed at.
- `align` needs the video; without it, an inferred combat-damage state
  carries the attack's timestamp, which runs a few seconds early.
