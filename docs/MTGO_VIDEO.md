# MTGO video ingestion

Turns MTGO gameplay footage into a replay file — board states in the same
schema the Arena mirror produces — then replays it in XMage and verifies it
three independent ways.

```text
MTGO footage
  -> layout       locate the UI in this capture, at whatever resolution
  -> extract      crop the game-log pane, normalized to a fixed OCR scale
  -> ocr          read each frame
  -> reconstruct  merge the scrolling windows; vote across repeated sightings
  -> parse        MTGO's log grammar, tolerant of how OCR mangles it
  -> catalog      resolve card names offline, or refuse to
  -> simulate     fold events into board snapshots
  -> verify       against XMage, against MTGO's own HUD, across captures
```

## Design

The pipeline is built around one observation: **every stage can be wrong,
and the only defence is evidence.** OCR misreads, the layout detector can
find the wrong rectangle, a card name can resolve to a plausible neighbour,
and the simulator can derive a life total the game never had. So each stage
either checks itself against something independent, or reports that it
could not.

Three properties follow from that, and they are what the architecture is
actually organised around:

**Redundancy across frames is the main resource.** A log line is on screen
for a dozen frames or more, and OCR does not fail identically on each,
because subpixel differences change the rasterisation. So the reconstructor
does not trust any single reading: it keeps every sighting of a line and
takes the majority. The same redundancy fixes the layout — the detected
pane drifts with the log's content (the scrollbar only exists once the log
overflows), so the bounds are a consensus across several frames rather than
one. And it settles questions that would otherwise be judgement calls: a
run of identical lines is capped at how many were ever on screen *at the
same time*, because MTGO really does log one sentence twice in a row, and
that is the difference between a real repeat and a misread.

**Nothing is guessed silently.** A card name that does not clear the
catalog's margin gate is reported unresolved, not resolved to whatever it
resembles. A layout that cannot be read back is a hard failure with a
message, not a confident wrong log. A repaired verb has to actually parse.
An attacker whose power is unknown is flagged rather than assumed.

**Derived facts are checked against the video.** Life totals are a
derivation — the log has no line for combat damage — so they are compared
against the life MTGO itself prints on screen, and the moment a derived
change actually happened is *located* in the footage rather than assumed.

### What changed, and why

The first version hardcoded 1080p pixel coordinates, trusted the longest
OCR reading of each line, and looked card names up live. Each of those
turned out to be a real defect rather than a simplification:

| Was | Failure it caused | Now |
| --- | --- | --- |
| Hardcoded 1080p crop | Nothing but 1080p worked at all | Detect the pane by brightness; consensus across frames; validate by reading it back |
| Fixed 3x upscale before OCR | Glyph size drifted with capture size, so results drifted too | Resample to a fixed target width, so tesseract sees the same thing at every resolution |
| Longest reading wins | One bad frame could define a line | Majority vote across every sighting of that line |
| Append on failed alignment | Noise produced duplicate entries (240 for a 75-line game at 576p) | Collapse duplicates, using frame co-occurrence to tell a misread apart from a genuine repeat |
| Alignment scored on similarity alone | A game log repeats every turn, so a window could match the wrong occurrence — duplicating or dropping a whole pane of lines | The pane scrolls one way, so a frame may not align behind the frame before it |
| Strict log grammar | A misread colon glued two lines into one unparsable entry | Tolerant clock shape, count from the noun not the article, self-validating verb repair |
| Live Scryfall fuzzy lookups | Rate-limited, non-deterministic, 404s on names the log spells correctly elsewhere | Offline catalog, run vocabulary, margin gate |

## Resolution independence

`layout.py` owns the entire dependency on capture size. It finds the
game-log pane by brightness — a near-white panel against a dark board,
separable at any resolution — trims the scrollbar band at its right edge,
and derives the HUD regions proportionally from the detected client area.
Letterboxing is removed first, so a padded or windowed capture works too.

Two things make this trustworthy rather than merely plausible:

- **Consensus.** The pane is detected on several frames spread through the
  clip and each edge is the median. A single frame is a poor witness: early
  in a game the log has not overflowed, so there is no scrollbar and the
  pane reads wider than it is.
- **Validation.** The located regions are read back before use. The log
  pane must contain timestamped lines; the life positions must contain
  small integers. If not, ingestion stops with a message instead of
  producing a confident wrong log.

`layout` also measures the log's glyph height *in source pixels* and warns
when it is below the point where OCR is dependable. Upscaling cannot add
detail the capture never had, so this is a property of the recording, not
of how hard the pipeline tries.

```sh
python3 -m magic_cabt.mtgo_video layout --video match.mp4
```

### Measured behaviour

The same game, decoded from five renderings of one recording:

| Capture | Detected pane | Glyph height | Verdict |
| --- | --- | --- | --- |
| 2560x1440 | 466px | 14px | works |
| 1920x1080 | 350px | 10px | works |
| 1600x900 | 292px | 8px | works |
| 1280x720 | 233px | 7px | works |
| 1024x576 | 186px | 4px | **below threshold, reported** |

At 576p the log text is four pixels tall. That is genuinely below what OCR
can read, and the pipeline says so rather than emitting a wrong game.

## Requirements

- `ffmpeg`, `ffprobe` and `tesseract` on `PATH`, and Python `Pillow`
- A built `Mage.Client` with the `cabtmirror` overlay, and
  `MAGIC_CABT_CLASSPATH` pointing at its classpath
- Network access once, to build the card catalog

## Usage

```sh
# One-time: build the local card catalog (~200MB download -> 0.8MB cache)
python3 -m magic_cabt.mtgo_video catalog

# Video -> per-game replay bundles
python3 -m magic_cabt.mtgo_video ingest match.mp4 --out bundle \
    --hero YourMTGOName --match-id league-1

# Locate inferred changes (combat damage) in the footage
python3 -m magic_cabt.mtgo_video align bundle/game1 --video match.mp4

# Verify: against XMage, and against MTGO's own on-screen life totals
python3 -m magic_cabt.mtgo_video verify bundle/game1
python3 -m magic_cabt.mtgo_video crosscheck bundle/game1 --video match.mp4

# Verify two captures of the same match decoded to the same game
python3 -m magic_cabt.mtgo_video compare bundle_1080p/game1 bundle_720p/game1

# Replay in XMage and record it, beside the source footage
python3 -m magic_cabt.mtgo_video render bundle/game1 --out replay.mp4 \
    --side-by-side comparison.mp4 --source-video match.mp4
```

`rebuild` re-runs parsing and simulation from a bundle's cached
`mtgo_log.json` without redoing OCR — use it while iterating.

## The three verifications

**`verify` — log vs XMage.** Feeds every snapshot through XMage's real
`MirrorStateApplier` and `GameView` and diffs turn, life, library and hand
counts, and the exact multiset of battlefield and graveyard cards. Proves
the mirror reproduces the decoded game. It cannot prove the decoding is
right: XMage renders whatever the log claimed.

**`crosscheck` — log vs the footage's own HUD.** OCRs MTGO's on-screen life
totals and compares them with the life the simulator derived. Independent
evidence from the same frames, and it catches what the first check
structurally cannot. It found two real bugs: combat damage missing entirely
(MTGO logs no line for it), and a transformed double-faced creature keeping
its front-face power.

**`compare` — capture vs capture.** Two recordings of the same match must
decode to the same game. This is what makes resolution independence a
checked claim rather than an assertion, and it catches OCR differences that
neither other check would notice, because both would happily agree with a
consistently-wrong log.

## Bundle layout

```text
bundle/
  mtgo_log.json         reconstructed log entries
  mtgo_log_times.json   video timestamp each entry first appeared at
  mtgo_events.jsonl     parsed, name-resolved events for the whole match
  match.json            layout, catalog provenance, cards in play, summaries
  game1/
    mirror_states.jsonl the replay file (Arena mirror schema)
    mtgo_events.jsonl   this game's events
    summary.json
    verification.json   log vs XMage
    hud_crosscheck.json log vs MTGO's on-screen life totals
```

Every snapshot carries `videoTime`, the moment its log line first appeared
on screen, which is what lets a decoded game be replayed in sync with the
footage it came from.

## Card names

Names resolve against a local catalog condensed from Scryfall's
`oracle_cards` bulk file (201MB in, 0.8MB out), restricted to cards that
exist on MTGO — within that pool normalized names are collision-free, and
it excludes Alchemy cards, which cannot appear in MTGO footage.

Matching is: exact hit on a folded key (casefold, strip diacritics and
punctuation); else fuzzy against the **run vocabulary**, the cards this log
already named exactly, where a low threshold is safe because the candidate
set is a few dozen names known to be in play; else fuzzy against the whole
catalog at a stricter threshold.

A match must also beat the runner-up by a margin. With tens of thousands of
names almost any string has a plausible neighbour, so the margin — not the
absolute score — is what separates "that card, misread" from "not a card at
all". It is why the garbled player name `Buz2Caldera` is rejected rather
than resolved to whatever it resembles.

## Known limits

- Hand contents are never known: MTGO does not log them, so hands are
  rendered face-down at the right count.
- Library counts are derived from draws and mills starting at `--deck-size`,
  not read from the client.
- Combat damage is inferred from the attack, since MTGO logs no damage line.
  Blocks refund the blocked attacker's share. An attacker whose power the
  catalog does not know is reported rather than guessed at.
- Below ~7px glyph height (roughly 720p full-screen) OCR stops being
  dependable. The pipeline measures this and warns.
- The HUD cross-check reads only life. The library and hand badges are
  ~10px tall inside shaped icons and do not OCR reliably even at 1080p, so
  they are deliberately unused rather than reported at low confidence.
