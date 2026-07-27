# MTGO video ingestion

Turns MTGO gameplay footage into a replay file — board states in the same
schema the Arena mirror produces — then replays it in XMage and verifies it
four independent ways.

```text
MTGO footage
  -> layout       locate the UI in this capture, however it is arranged
  -> extract      crop the game-log pane, normalized to a fixed OCR scale
  -> ocr          read each frame
  -> reconstruct  merge the scrolling windows; vote across repeated sightings
  -> parse        MTGO's log grammar, tolerant of how OCR mangles it
  -> catalog      resolve card names offline, or refuse to
  -> simulate     fold events into board snapshots
  -> verify       against XMage, against the board, against the HUD, across captures
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
takes the majority. The same redundancy fixes the layout — the text column
measured in any one frame is only as wide as that frame's longest line, so
the pane's bounds are the union across several frames rather than one
frame's guess. And it settles questions that would otherwise be judgement calls: a
run of identical lines is capped at how many were ever on screen *at the
same time*, because MTGO really does log one sentence twice in a row, and
that is the difference between a real repeat and a misread.

**Nothing is guessed silently.** A card name that does not clear the
catalog's margin gate is reported unresolved, not resolved to whatever it
resembles. A layout that cannot be read back is a hard failure with a
message, not a confident wrong log. A repaired verb has to actually parse.
An attacker whose power is unknown is flagged rather than assumed. And when
MTGO prints a player's name where a card's belongs — which it really does —
no permanent named after a player reaches the board.

**Derived facts are checked against the video.** Life totals are a
derivation — the log has no line for combat damage — so they are compared
against the life MTGO itself prints on screen, and the moment a derived
change actually happened is *located* in the footage rather than assumed.
Where the footage contradicts the derivation outright, the footage wins:
the observed life is adopted, and the fact that it was is recorded on the
state, so nothing downstream mistakes it for an independent confirmation.

**Each event is checked against the board it happens on.** The pipeline
derives the rules of the game by watching a log describe it, so a decoded
stream can be internally consistent and still describe an impossible game.
Running it through XMage one step ahead of itself — asking, before each
event, whether the board can support what it is about to claim — is what
catches a line that OCR damaged into something the game could not do.

### What changed, and why

The first version hardcoded 1080p pixel coordinates, trusted the longest
OCR reading of each line, and looked card names up live. Each of those
turned out to be a real defect rather than a simplification:

| Was | Failure it caused | Now |
| --- | --- | --- |
| Hardcoded 1080p crop | Nothing but 1080p worked at all | Detect the pane by brightness; consensus across frames; validate by reading it back |
| Regions as fractions of a default 1080p duel scene | Only one client arrangement worked; a streamed capture read a mana pip as a life total | Anchor on text MTGO draws — the phase bar, the names under each seat — and read every located region back |
| Fixed 3x upscale before OCR | Glyph size drifted with capture size, so results drifted too | Resample to a fixed target width, so tesseract sees the same thing at every resolution |
| Longest reading wins | One bad frame could define a line | Majority vote across every sighting of that line |
| Append on failed alignment | Noise produced duplicate entries (240 for a 75-line game at 576p) | Collapse duplicates, using frame co-occurrence to tell a misread apart from a genuine repeat |
| Alignment scored on similarity alone | A game log repeats every turn, so a window could match the wrong occurrence — duplicating or dropping a whole pane of lines | The pane scrolls one way, so a frame may not align behind the frame before it |
| Strict log grammar | A misread colon glued two lines into one unparsable entry | Tolerant clock shape, count from the noun not the article, self-validating verb repair |
| Live Scryfall fuzzy lookups | Rate-limited, non-deterministic, 404s on names the log spells correctly elsewhere | Offline catalog, run vocabulary, margin gate |
| Grammar fitted to one recording | A second capture used a dozen ordinary log forms it had never seen — including the blocker-first "X blocks Y", so blocks were invisible and blocked damage was still dealt | Those forms parse; parse coverage on that capture went from 77% to 97% |
| Combat resolved at the next log line | Ninjutsu bounces the attacker and swaps in a ninja *before* damage; the bounced creature's damage was dealt anyway and the ninja's was not | The lines that belong to combat do not resolve it, and a creature pulled out of combat withdraws its damage |

## Layout independence

`layout.py` owns the entire dependency on how the capture looks. That is a
larger job than resolution, because MTGO's duel scene is not a fixed
picture: panes are dockable and resizable, the whole UI can be scaled, the
client is often not full-screen, extra zone panes can be docked beside the
seats, and a streamed VOD wraps all of it in a facecam, a scoreboard and a
sponsor banner. Anything measured as "x percent across the frame" is a
guess about one recording rather than a fact about the client.

So nothing is measured that way. Every region is found from **text MTGO
itself draws**:

- **The phase bar.** A row of step labels — Untap, Upkeep, Draw, Main …
  Cleanup — appears in every duel and nowhere else. Finding that row is both
  the test for "is this a duel scene" and the anchor that separates the play
  area from the margins the seat panels sit in. It replaced the old
  duel-frame test, which read for both life totals — that is, it depended on
  already knowing where the seats were.
- **The seat panels.** Life is a large numeral and the player's name is
  ordinary small text centred under the avatar. Both are searched for, in
  both margins; the pair that sits one per half of the frame, in the same
  column, at the same size, wins. A candidate is only believed once its life
  box actually reads back as a number, which is what stops a scoreboard
  ("0 – 0") or a mana pip from being adopted as a seat.
- **The log pane.** Brightness finds candidates — but a facecam and a
  sponsor graphic are bright too, and "the rightmost bright panel" is
  whichever one the streamer put on the right. So every candidate is read,
  and the one whose text scans as timestamped log lines wins. Reading it is
  also how its text column is measured, which is what keeps the scrollbar
  out of the crop without assuming how wide a scrollbar is.

Three further things make this trustworthy rather than merely plausible:

- **Consensus.** Each region is detected on several frames spread through
  the clip and each edge is the median. A single frame is a poor witness:
  early in a game the log has not overflowed, so there is no scrollbar and
  the pane reads wider than it is, and a card dragged over a seat hides its
  life total for a second.
- **The right frames.** A league VOD spends much of its length in the deck
  editor and sideboarding, screens that carry a game log of their own in a
  different place. Samples are filtered to duel frames first.
- **Validation.** The located regions are read back before use. The log pane
  must contain timestamped lines, and *both* seats must yield a life total —
  one used to be enough, which let a layout pass while the other seat
  pointed at a mana pip. Numerals are read by majority vote across several
  thresholds and segmentation modes, because no single setting reads a white
  numeral over both a dark avatar and a bright one.

`layout` also measures the log's glyph height *in source pixels* and warns
when it is below the point where OCR is dependable. Upscaling cannot add
detail the capture never had, so this is a property of the recording, not
of how hard the pipeline tries — and when nothing at all can be located, it
distinguishes "the UI is somewhere unexpected" from "this capture is drawn
too small to read", which call for opposite responses.

```sh
python3 -m magic_cabt.mtgo_video layout --video match.mp4
```

### Measured against real captures

Four public league VODs, each a different arrangement, none of which the
previous detector handled:

| Capture | Arrangement | Log pane | Seats | Names |
| --- | --- | --- | --- | --- |
| Modern client, facecam right | client inset to 1620px, log docked right | found | 16 / 20 | Mafuhsa / MarshFlats |
| Modern client, scaled up | extra zone panes docked beside the seats, larger UI scale | found | 20 / 16 | marcomartinelli999 / impact36inc |
| Modern client, circular facecam over the log | facecam overlaps the log pane's lower half | found | 20 / 20 | Flexi98 / DB_ThatMillGuy |
| Windowed client in a 1920x810 stream frame | client ~1620x650 inside overlays and a banner | refused | — | — |

The fourth is the intended outcome, not a gap: its log text is about five
source pixels tall, below what OCR can read at any magnification, and the
pipeline says so — "recorded too small to read, whatever the frame size is"
— rather than producing a confident wrong log.

### End to end on a capture the old detector could not open

Ten minutes of the first of those VODs (a Pauper league match, client inset
to 1620px with a facecam beside it), ingested and put through all four
checks:

| | |
| --- | --- |
| Log entries reconstructed | 106 |
| Parse coverage | 97% (3 unparsed, all glued by an unreadable timestamp) |
| `verify` — log vs XMage | 78/78 states |
| `rules` — events vs the board | 67 checked, 2 flagged (both real: MTGO printed a player's name where a card's belonged) |
| `crosscheck` — vs the on-screen life | 152/156 readings agree (97%), 2 of which came from the HUD and are counted apart |
| `align` — derived combat damage | 9 of 9 located in the footage: 8 confirmed, 1 corrected from the screen |

Getting there turned up four defects worth naming, each found by a check
rather than by inspection: the log crop was a few pixels narrow and clipped
the last character of full-width lines; a dozen ordinary MTGO log forms had
no grammar; ninjutsu's attacker swap was not modelled; and an attacker
killed by a spell mid-combat still dealt its damage.

### Measured behaviour

The same game — a complete six-turn Pauper league game, 75 log lines —
decoded from five renderings of one recording, and verified all three ways:

| Capture | Detected pane | Log vs XMage | Log vs HUD life | Same game as 1080p? |
| --- | --- | --- | --- | --- |
| 2560x1440 | 466x629 | 61/61 | 122/122 | yes |
| 1920x1080 | 350x449 | 61/61 | 118/118 | reference |
| 1600x900 | 292x374 | 61/61 | 122/122 | yes |
| 1280x720 | 233x299 | 68/68 | 136/136 | **no — 84 events, not 75** |
| 1024x576 | 186x238 | 71/189 | — | **no — 249 events** |

1440p, 1080p and 900p decode to the identical game and verify completely.

720p is the interesting case, and the reason the third check exists. Its
decode passes both of the other verifications — every state it claims is
rendered faithfully by XMage, and every life total it derives matches
MTGO's own display — while still containing nine events that never
happened, a stretch of the log recorded twice. Neither of the first two
checks can see that: XMage renders what it is told, and duplicating a land
drop does not change anyone's life total. Only comparing against another
capture of the same match catches it.

576p is below the floor: the log text is four pixels tall, which is less
than OCR can read at any magnification. The pipeline measures that and
warns before spending the time.

## Requirements

- `ffmpeg`, `ffprobe` and `tesseract` on `PATH`, and Python `Pillow`
- A built `Mage.Client` with the `cabtmirror` overlay, and
  `MAGIC_CABT_CLASSPATH` pointing at its classpath. XMage opens its card
  database relative to the working directory, so the JVM has to start where
  `db/cards.h2.mv.db` lives; the verifications find that directory from the
  classpath themselves, and say so plainly if it is missing rather than
  reporting every library as empty
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

# Verify: against XMage, against the board each event happens on, and
# against MTGO's own on-screen life totals
python3 -m magic_cabt.mtgo_video verify bundle/game1
python3 -m magic_cabt.mtgo_video rules bundle/game1
python3 -m magic_cabt.mtgo_video crosscheck bundle/game1 --video match.mp4

# Verify two captures of the same match decoded to the same game
python3 -m magic_cabt.mtgo_video compare bundle_1080p/game1 bundle_720p/game1

# Replay in XMage and record it, beside the source footage
python3 -m magic_cabt.mtgo_video render bundle/game1 --out replay.mp4 \
    --side-by-side comparison.mp4 --source-video match.mp4
```

`rebuild` re-runs parsing and simulation from a bundle's cached
`mtgo_log.json` without redoing OCR — use it while iterating.

## The four verifications

**`verify` — log vs XMage.** Feeds every snapshot through XMage's real
`MirrorStateApplier` and `GameView` and diffs turn, life, library and hand
counts, and the exact multiset of battlefield and graveyard cards. Proves
the mirror reproduces the decoded game. It cannot prove the decoding is
right: XMage renders whatever the log claimed.

**`rules` — each event vs the board it happens on.** MTGO's log is a stream
of assertions, and each one presupposes something about the board: a
creature cannot be destroyed unless it is on the battlefield, a player
cannot discard from an empty hand, a card cannot attack from a graveyard.
This walks the game one step ahead of itself, checking each decoded event
against the board XMage holds the moment before it.

That is what catches damage the other checks structurally cannot. A missed
"casts X" line leaves the later "X is destroyed" with nothing to destroy —
and `verify` still passes (XMage renders what it is told) and `crosscheck`
still passes (destroying a creature changes nobody's life). Run against the
reference bundle it found four real decode defects that both other checks
had passed: three log lines whose player was read as `Buz2Caldera`, and a
spell whose target was read as `BuzzCaldera. 2` with a scrollbar glyph stuck
to it.

It reports which event types it *cannot* check as well as which it can, so
the coverage is visible rather than assumed. What it is not: XMage is not
being asked to play the game, so this does not prove the original match was
legal — it proves the decoded event stream is coherent against the board the
mirror actually built, which is where OCR and parse damage shows up.

**`crosscheck` — log vs the footage's own HUD.** OCRs MTGO's on-screen life
totals and compares them with the life the simulator derived. Independent
evidence from the same frames, and it catches what the first check
structurally cannot. It found two real bugs: combat damage missing entirely
(MTGO logs no line for it), and a transformed double-faced creature keeping
its front-face power.

**`crosscheck` also corrects.** Combat damage is the one quantity the log
does not state, and MTGO's silence goes further than that: it prints nothing
when an attacker is killed by a spell mid-combat, so an attack it describes
in full can still deal less than the attackers' total power. Where `align`
cannot find the derived life anywhere in the footage, the life the screen
settles on — before anything else that could change it — is adopted instead,
the difference carried into every later state, and the correction recorded.
The state is marked `lifeSource: "hud"`, and `crosscheck` counts readings
that came from the screen apart from readings that agreed with it on their
own, so a corrected state cannot inflate the agreement rate it fed.

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
    rules_check.json    each event vs the board it happens on
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
- The stack is not modelled, so `rules` cannot check what a spell targeted:
  a counterspell legally targets something that is on the stack rather than
  the battlefield, and "not on the battlefield" would fire on ordinary play.
  It checks the targets it *can* judge — a target that is a garbled version
  of a player's name — and leaves the rest alone.
- MTGO sometimes prints a player's name where a card's belongs. Nothing can
  recover which card that was, so the cast is recorded, reported, and left
  off the board rather than becoming a permanent named after a player.
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
