# MTGO video ingestion

Turns MTGO gameplay footage into a replay file — board states in the same
schema the Arena mirror produces — then replays it in XMage and verifies it
five independent ways.

```text
MTGO footage
  -> layout       locate the UI in this capture, however it is arranged
  -> extract      crop the game-log pane, normalized to a fixed OCR scale
  -> ocr          read each frame
  -> reconstruct  merge the scrolling windows; vote across repeated sightings
  -> parse        MTGO's log grammar, tolerant of how OCR mangles it
  -> catalog      resolve card names offline, or refuse to
  -> simulate     fold events into board snapshots
  -> verify       against XMage, the rules, the picture, the HUD, other captures
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
| A seek per reading in the HUD checks | A ten-minute game's crosscheck was a few hundred ffmpeg launches to read a few hundred numerals | One decode for the whole game; the same 156 readings, 216s to 82s |
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
  ("0 – 0") or a mana pip from being adopted as a seat. The numeral's
  position is expressed in units of the name under it — the panel is one
  widget drawn at whatever scale the client is set to, so its own text is
  the only ruler that survives being rescaled.
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

### One thing that does not work: averaging frames

The obvious way to lower that floor is the standard video-OCR trick: several
consecutive frames of the same still text are the same picture plus
independent noise, so averaging N of them cuts the noise by about √N. It was
built, measured, and removed.

On this kind of footage there is no noise to average. A modern codec spends
no bits re-encoding a region that has not changed, and the log pane between
scrolls has not: measured on a 720p re-encode, **eleven of eleven consecutive
frame pairs were bit-identical** in the pane. Averaging identical frames
returns the same frame. The measured effect was slightly negative — 191 log
entries became 186 — because the frames consumed by each composite would
otherwise have been separate sightings for the reconstruction's vote.

The redundancy that *does* exist is across scroll positions, not across
frames within one, and the reconstruction already takes it at the text level.
Pixel-level compositing would pay on a source with genuine per-frame noise —
an analog capture, a camera pointed at a screen — and not on a screen
recording of a deterministic renderer.

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

Five public VODs, each a different arrangement, none of which the previous
detector handled. These are the rows of `python/tests/captures.json`, and
`scripts/check-mtgo-captures.py` re-checks every one of them:

| Capture | Arrangement | Log pane | Seats | Names |
| --- | --- | --- | --- | --- |
| Modern client, facecam right | client inset to 1620px, log docked right | found | 16 / 20 | Mafuhsa / MarshFlats |
| Modern client, scaled up | extra zone panes docked beside the seats, larger UI scale | found | 20 / 16 | marcomartinelli999 / impact36inc |
| Modern client, circular facecam over the log | facecam overlaps the log pane's lower half | found | 20 / 20 | Flexi98 / DB_ThatMillGuy |
| 2016 championship broadcast, old client | different skin, no game-log pane at all | refused | 20 / 20 | leearson / beena |
| Windowed client in a 1920x810 stream frame | client ~1620x650 inside overlays and a banner | refused | — | — |

The last two are the intended outcomes rather than gaps. The broadcast is
refused because it shows no game log to read — but its phase bar and both
seat panels are still located, which is what says *why* it was refused. The
windowed capture's log text is about five source pixels tall, below what OCR
can read at any magnification, and the pipeline reports that rather than
producing a confident wrong log.

The broadcast is also what the inverted word pass is for: its seat plates
are dark enough that the white names on them are invisible to the ordinary
passes and read cleanly inverted. That pass only runs when the seats were
not found otherwise.

### End to end on a capture the old detector could not open

Ten minutes of the first of those VODs (a Pauper league match, client inset
to 1620px with a facecam beside it), ingested and put through every check:

| | |
| --- | --- |
| Log entries reconstructed | 106 |
| Parse coverage | 97% (3 unparsed, all glued by an unreadable timestamp) |
| `verify` — log vs XMage | 78/78 states |
| `rules` — events vs the board | 67 checked, 2 flagged (both real: MTGO printed a player's name where a card's belonged) |
| `crosscheck` — vs the on-screen life | 153/156 readings agree (98%), 2 of which came from the HUD and are counted apart |
| `board` — the screen vs the derived board | 96 cards read, 2 unlogged permanents (both real, both the same glued log line) |
| `align` — derived combat damage | 9 of 9 located in the footage: 8 confirmed, 1 corrected from the screen |

Getting there turned up five defects worth naming, each found by a check
rather than by inspection: the log crop was a few pixels narrow and clipped
the last character of full-width lines; a dozen ordinary MTGO log forms had
no grammar; ninjutsu's attacker swap was not modelled; an attacker killed by
a spell mid-combat still dealt its damage; and one land reached the board
under a garbage name because its log line had another glued onto it.

Those numbers are reproducible rather than remembered. The captures are other
people's videos and cannot live in the repository, so
`python/tests/captures.json` records each one's YouTube id, the timestamp of
the frame, and what the detector should find in it, and
`scripts/check-mtgo-captures.py` fetches whatever is missing and checks the
lot:

```sh
scripts/check-mtgo-captures.py            # fetch what is missing, check all
scripts/check-mtgo-captures.py --offline  # only what is already cached
```

### Measured behaviour

The same ten-minute Pauper league match, re-encoded at three sizes and put
through the whole pipeline. Every number here came out of the current code;
nothing is carried over from an earlier measurement.

| Capture | Log text | Log entries | Parse coverage | vs XMage | vs the rules | vs the HUD | Same game as 1080p? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1920x1080 | 13px | 106 | 97% | 78/78 | 2 findings, both real | 153/156 (98%) | reference |
| 1280x720 | 7px | 191 | 81% | 63/123 | 17 findings, 4 naming a lost line | seats unreadable | **no — 170 events, not 90** |
| 1024x576 | 5px | 779 | 51% | — | — | seats unreadable | **no — split into five games that never happened** |

The two degraded captures were both warned about before any of that work
started: the pipeline measures the log's glyph height in source pixels and
says when it is below what OCR can read at any magnification.

720p is the case worth reading. It decodes into a game that is nearly twice
as long as the one that was played, because at seven pixels the
reconstruction cannot always tell a re-read line from a new one. Three of the
five checks catch it, and they catch different parts: XMage rejects half its
states outright, the rules check finds four permanents attacking from nowhere
and matches four of them to the unparsed lines they were mangled out of, and
`compare` simply reports that this is not the same game as the 1080p decode.
The HUD check cannot run at all, because at that size the life numerals are
not readable either — and the layout detector says so rather than pretending.

576p is well below the floor: 779 log entries for a 106-entry game, split
into five "games" whose boundaries are OCR noise.

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

# Run every check this bundle and this machine allow, in one go. Anything
# that cannot run is reported as skipped with the reason -- and a bundle
# nothing could be run against does not come back "ok".
python3 -m magic_cabt.mtgo_video check bundle/game1 --video match.mp4

# Or one at a time: against XMage, against the board each event happens on,
# against the permanents on screen, against MTGO's own life totals
python3 -m magic_cabt.mtgo_video verify bundle/game1
python3 -m magic_cabt.mtgo_video rules bundle/game1
python3 -m magic_cabt.mtgo_video board bundle/game1 --video match.mp4
python3 -m magic_cabt.mtgo_video crosscheck bundle/game1 --video match.mp4

# Verify two captures of the same match decoded to the same game
python3 -m magic_cabt.mtgo_video compare bundle_1080p/game1 bundle_720p/game1

# Replay in XMage and record it, beside the source footage
python3 -m magic_cabt.mtgo_video render bundle/game1 --out replay.mp4 \
    --side-by-side comparison.mp4 --source-video match.mp4
```

`rebuild` re-runs parsing and simulation from a bundle's cached
`mtgo_log.json` without redoing OCR — use it while iterating.

## The five verifications

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
still passes (destroying a creature changes nobody's life).

On the reference capture it reports two findings, and both are real: MTGO
itself printed a player's name where a card's belonged, twice, in plain black
rather than the blue it uses for cards. Nothing can recover which card that
was, so the cast is recorded, reported, and kept off the board rather than
becoming a permanent named after a player. On the degraded 720p decode of the
same match it reports seventeen, and names the lost line behind four of
them.

It does more than flag the impossible line, because the impossible line is
almost never the broken one. "Faerie Seer is destroyed, but it is not on the
battlefield" is half an answer; the other half is that an *earlier* line
never arrived. So each violation is turned into the smallest edit that would
have made it possible — the shape process mining calls an alignment, where a
*log move* is an event the model cannot perform and a *model move* is an
event the model needed that the log lacks. Those are exactly this pipeline's
two failure modes: OCR inventing a line and OCR dropping one.

And a dropped line usually did not vanish — it was mangled into something the
grammar could not parse and is still sitting in the log as an unparsed entry.
So each proposed insertion is matched against the unparsed entries mentioning
the same card, and the likely culprit is reported with it. Deleting one
"casts Tolarian Terror" line from a decoded game and re-running produces three
violations, all naming the same missing event: *a line putting Tolarian Terror
onto the battlefield is missing.*

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

**`board` — the board on screen vs the board the log produced.** Every other
check reasons about the decoded log. This one looks at the picture: it finds
the two battlefield panels from the phase bar, slides a card-sized window
across the permanents on them, and identifies each by a perceptual signature
of its art against Scryfall's, restricted to the cards this match has named.

That is the only check that can see a *dropped* line. A missed "casts X"
leaves the event stream internally consistent, XMage renders the board it was
given, and nobody's life changes — but the permanent is still sitting there
on screen. Run against the ten-minute capture it found one: MTGO's log line
"Mafuhsa plays Volatile Fjord" had another line glued onto it by an
unreadable timestamp, so the land went onto the board under a garbage name.
The screen shows Volatile Fjord in play; the derived board does not.

Colour, not gradient. The usual perceptual hashes for matching card scans
are built on gradient structure, and at a hundred pixels wide that structure
is mostly compression noise: a 16x16 dHash put the right card outside the top
two for three cards in four. A 6x6 grid of average colour identified all
four, each beating the runner-up by about a factor of two.

Three things keep it from inventing permanents. An art window with no
contrast is not a card — an empty stretch of panel is flat grey, and flat
grey matches whichever card has the flattest art. A tapped permanent is drawn
rotated, so its art is a column of its box rather than a band across it, and
reading it upright lands on rules text. And nothing is claimed from one
frame: a permanent does not flicker, so a card the log never mentioned must
be seen at the same place in states next to each other before it is
reported. Twice at some point in the game with nothing in between is what a
matcher finding a plausible neighbour twice looks like, and was a real false
positive until the rule demanded adjacency.

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
    board_check.json    the board on screen vs the board derived
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
