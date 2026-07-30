# Dataset feasibility reanalysis — July 2026

## Conclusion

The working target — on the order of 100k full game logs — is reachable, but not
by scaling any single approach tried so far. The reanalysis changes the picture
in four ways:

1. **The 17lands verdict was rendered against the wrong file.** The public
   datasets include a third type beyond draft and game data: **replay data**,
   per-turn card-level event columns for both players across millions of limited
   games since early 2021. It is not a full log, but it is close enough that the
   rules-repair machinery this repository already built for OCR-damaged MTGO
   logs can plausibly *reconstruct* full trajectories from it, with a confidence
   gate. This is the largest accessible corpus by two orders of magnitude.
2. **Full-text MTGO game logs already exist in bulk.** The Penny Dreadful
   community's PDBot has recorded league and tournament games since 2016 and
   publishes complete textual game logs with linked decklists
   (logs.pennydreadfulmagic.com). Text in the exact grammar `mtgo_video/parse.py`
   already speaks, with no OCR stage and no resolution floor.
3. **The binding constraint is not games, it is hero-perspective decisions.**
   Sources split sharply on whether the acting player's hand is known. A
   public-only game (MTGO video today, PD logs) is worth much less for behavior
   cloning than a hero-perspective game (Arena logs, 17lands reconstruction,
   MTGO video with hand reading added). The target should be restated in
   hero-perspective decisions, and the actual number should come from the
   scaling curves the scope audit already ordered, not from a guess.
4. **For MTGO video, compute was never the bottleneck.** At the measured 1 fps
   extraction and per-game check costs, ingesting tens of thousands of VODs is
   hundreds of dollars of CPU. The real constraints are the ≥1080p floor, VOD
   acquisition bandwidth, and rights — all three of which creator partnerships
   (ask grinders for their raw local recordings) solve at once.

Recommended portfolio: 17lands replay reconstruction for limited-format scale,
PD logs for constructed public-state scale, MTGO video with hero-hand reading
for constructed hero-perspective quality, Arena capture and XMage-native play
for options-complete gold data — plus four cheap letters (Daybreak, 17lands,
PD, XMage server operators) any one of which could moot the rest.

## What "quantity and quality" means for this repository

The supported path trains on DecisionRecords: hero-perspective state, the legal
options XMage enumerates, and the chosen action. Raw sources differ on every
axis that matters, so "a game" is not a unit. The axes:

| Axis | Why it matters |
| --- | --- |
| Hero hand known | Without it, BC imitates decisions while blind to the information that caused them. Public-only games still feed world-model, belief, value, and opponent-model lines — but not clean policy imitation. |
| Decklists known | Bounds the hidden zones; enables belief supervision and library inference. |
| Exact action order | Within-turn ordering and targets; required for exact replay and options enumeration. |
| Legal options recoverable | Native in XMage games; derived via mirror for everything else. |
| Skill measurable | Lets the corpus be filtered or weighted (league records, ranks). |
| Format | Limited vs constructed changes card pool and decision distribution, not the schema. |

Scale intuition: at roughly 25–40 strategic hero decisions per game after macro
grouping, 100k games is ~3M decisions. Whether the baselines need 3M, or
saturate at 300k, is precisely what the ordered-but-unrun scaling curves
(scope audit, next-work item 6) will say. Run them on the corpus that exists
before committing a quarter to acquisition — the result calibrates every
estimate below.

## Reassessment of the approaches already tried

| Approach | Prior verdict | Reassessment |
| --- | --- | --- |
| 17lands public data | "Too nerfed, no full logs" | True of `game_data`/`draft_data`. **False of `replay_data`**, which was easy to miss: per-turn, per-player card-level events plus end-of-turn life. See approach A. |
| Own logs | "Too few preserved" | Arena's `Player.log` rotates, but if the 17lands client (or Untapped) was running, the vendor retains full personal history — 17lands offers complete-history export to Mythic patrons, and a support request for one's own data is reasonable regardless. Partial recovery is plausible. |
| Ask 17lands | "No relationship; unsure they store it" | They store it — the site renders per-game replays, and replay-level public files exist. The ask has precedent (academic use of 17lands data is common and they invite citation). Reframe the ask around *research access or an expanded replay export*, referencing the public replay schema. Cheap letter, moderate odds. |
| Opt-in product | "Too much overhead, not ready" | Agree for a public product. But the expensive parts are already built and parked (`arena_mirror` GUI, `upload/` consent-envelope + redaction). A *private* donation pilot — a dozen recruited grinders, not a launch — reuses them with near-zero new surface. See approach G. |
| MTGO video parsing | "Most viable; fix scaling/overlays; estimate throughput" | Pipeline is real and verified, and the scaling work is done: the ≥1080p floor is measured and enforced, refusals are safe. Compute math below says CPU is trivial; supply, bandwidth, and rights dominate. Two upgrades change its value class: read the hero's hand strip (streamer's hand is on screen; battlefield art-matching already exists) and prefer creator-provided raw recordings over YouTube re-encodes. See approach D. |
| MTGA video parsing | "Maybe as easy; worth trying" | Deprioritize. MTGA shows no persistent textual log pane, so the log-OCR architecture does not transfer — it becomes a much harder general vision problem, to recover data the client will hand over perfectly in `Player.log`. The only unique asset MTGA VODs hold is *historical streamer archives*; revisit only if a specific corpus is identified and the owner won't run the recorder. |

## New or adjusted approaches, ranked

### A. Reconstruct full games from 17lands replay data (new; highest expected value, research risk)

`replay_data_public.{SET}.{EVENT}.csv.gz` files on 17lands' public S3 contain,
per game, per turn, per player: cards drawn, discarded, tutored (by name),
creatures / non-creatures / instants+sorceries cast, attackers, blockers,
blocked/unblocked, lands played, mana spent, and end-of-turn life and creature
counts ([schema helper][17l-dtypes], [datasets page][17l-data]). Coverage runs
back to early 2021 across every premier set — order 10^7 limited games.

What is missing is within-turn ordering, targets, and resolution detail. That
gap is exactly the shape of problem `mtgo_video/rules.py` already solves for
OCR damage: find the event stream a rules engine accepts that satisfies the
observed constraints, propose the smallest edits where none does, and report
confidence instead of guessing. Here the constraints are *clean* — merely
underdetermined — and limited-format turns are short and low-interaction, so a
large fraction of games should admit a unique (or unique-up-to-irrelevance)
legal ordering. End-of-turn life for both players every turn is a strong
disambiguator for combat and burn. Games that don't reconstruct uniquely get
flagged or dropped; capture-confidence plumbing for exactly this already exists
in the manifests.

Decisive quality property: the hero's opening hand and every draw are named, so
**reconstructed games are hero-perspective** — hand known at every decision.
Ranks and win rates are attached, so the corpus can be filtered by skill.

- Ceiling: 10^6+ hero-perspective limited games — two orders beyond the target.
- Cost: a reconstruction spike (~1–2 weeks) reusing rules/replay-search code;
  then batch compute.
- Risk: ambiguity rate is unknown until measured; instants held at instant
  speed and combat tricks are the hard cases. The spike's deliverable is a
  measured unique-reconstruction rate on ~100 games; that number decides.
- Terms: the datasets are published for public analysis (17lands asks for
  citation); confirm their data-use terms cover model training, which is also a
  good pretext for the research-access letter.

### B. Penny Dreadful game-log corpus (new; largest clean-text MTGO source)

PDBot spectates Penny Dreadful league and tournament matches on MTGO and has
published complete textual game logs since 2016, linked to registered
decklists and standings ([logsite][pd-logs], [tools repo][pd-tools]). This is
the MTGO-video end product — the textual game log — with no video, no OCR, no
resolution floor, and a decklist attached. `parse.py`'s grammar (built
tolerant of OCR damage) should hit ~100% parse on clean text; the
reconstruct/simulate/verify stages apply unchanged.

- Ceiling: plausibly 10^5+ matches (24/7 league + six weekly tournaments for
  ten years); exact count is one friendly ask away — the whole toolchain is
  open source and the community is famously approachable. Ask before bulk
  scraping.
- Quality: public-state only (spectator view — hands face-down), constructed,
  one budget format with quarterly rotations, mixed but *measurable* skill
  (standings). Ideal for world-model / belief / value / opponent lines and for
  belief supervision against known decklists; discounted value for BC.
- Cost: days — a log fetcher plus grammar touch-ups for PDBot markup and chat
  lines.

### C. Creator raw-archive partnerships (adjusted from "parse MTGO videos")

Instead of (or before) mass-crawling YouTube: ask the prolific MTGO grinders
directly for their raw local recordings, with permission to train on them.
A handful of daily-upload creators each sit on thousands of matches; raw OBS
output is 1080p60+ (well above the measured OCR floor), has a stable personal
layout (amortize layout consensus per creator), avoids YouTube's re-encode,
download throttling, and rights ambiguity — and the hero's hand is on screen.
Five to ten cooperative creators plausibly yields 10k–50k matches of archive
plus an ongoing feed, at pristine quality, for the cost of outreach and disk.

### D. MTGO VOD crawl and batch ingest (the current plan, quantified)

The user asked "how, and how long." Anchors from the measured pipeline: extract
samples at 1 fps; a 40-minute league VOD is ~2.4k small-crop tesseract calls
(~4–12 core-min) plus ~2–4 min decode; the HUD crosscheck is a measured 82 s
per 10-minute game; parse/simulate are negligible. Call it **15–30 core-minutes
per VOD** for ingest plus the full check suite, embarrassingly parallel.

For 100k games at ~2.4 games per match VOD ≈ 42k VODs:

| Resource | Estimate | Verdict |
| --- | --- | --- |
| CPU | 10k–21k core-hours (~2–4 weeks on one 32-core box; ~$200–600 spot) | Non-issue |
| Bandwidth/storage | 0.7–2 GB per 1080p VOD → 30–80 TB transfer; keep bundles+crops, discard video | Real logistics; weeks-to-months at polite rates |
| Supply at ≥1080p | Unknown; guess 30k–100k usable VODs across 20–50 archive channels | **The open question** |
| Accuracy | 97% parse and five-way verification on the reference; sub-1080p and odd layouts refuse safely | Yield haircut, not correctness risk — expect 40–70% of downloaded VODs usable; measure on the first 200 |

So the plan's missing number is supply, and it is cheap to get: a metadata-only
crawler (channel lists, durations, resolutions, upload dates — no video
download) turns the guess into a count in a weekend. Do that before bulk
downloading anything, and pair bulk use with creator permission (C) — which
also sidesteps the bandwidth throttle.

**Quality upgrade that changes the value class:** read the hero's hand. MTGO
renders the streamer's hand at ~100 px card width at 1080p — the scale at
which the battlefield art-matcher already identifies cards (6×6 color grid,
run-vocabulary restricted). Hand contents turn MTGO-video output from
public-only state streams into hero-perspective games; combined with
XMage-side option enumeration at hero priority, they become true
DecisionRecords. Without this, 100k MTGO-video games count at the public-only
discount.

### E. MTGOSDK structured capture (new; replaces pixels wherever an account is present)

[MTGOSDK][mtgosdk] (Videre Project; active, ~770 commits) attaches to a running
MTGO client and exposes games, zones, events, history, and replays as objects —
no OCR. Three uses, in ascending ambition: (1) perfect capture of one's own
play; (2) a spike to check whether client-side match history/replays are
readable in bulk for an account; (3) a PDBot-style watcher that records
watchable open-play games as structured logs — effectively approach B
generalized beyond Penny Dreadful (note league games are not spectatable, so
volume is capped; and PDBot's decade of open, community-positive operation is
the model to imitate). EULA reverse-engineering tension exists; the SDK's
DMCA-interoperability posture and the tracker precedent (Videre, PDBot) are the
mitigations. A [docker image][mtgo-docker] runs MTGO on Linux for headless use.

### F. XMage-native human games (new; the only options-complete human source)

Every other source needs mirroring to recover legal options. Games played *on
XMage* natively emit them through the same engine the CabtBridge already
instruments — plus both hands, server-side. Public XMage servers host hundreds
of concurrent players ([xmage.today][xmage]); games/day is unknown publicly.
Two moves: ask operators (one email) whether they retain or would record
anonymized game histories; and/or contribute an opt-in, consent-bannered
recording feature upstream — the community-goodwill version of "produce a
useful product," on a product that already has users. Even a few thousand
full-information, options-complete human games is a uniquely valuable
evaluation and calibration set, whatever the bulk corpus ends up being.

### G. Private Arena log-donation pilot (descoped from "product with opt-in")

The public-product overhead was the reason to park this; a private pilot has
almost none. The recorder GUI, redaction, consent envelope, and upload bundle
code exist (`arena_mirror/`, `upload/` — parked, not missing). Recruit 10–30
known grinders/streamers personally; 20 donors × 6 games/day ≈ 3.5k
hero-perspective games/month of the highest-fidelity constructed data
available, with consent by construction. This also keeps faith with the scope
audit: no new first-class surface, no hosting product — a script and a shared
drive suffice for a pilot.

### H. Four letters (near-zero cost, any hit is decisive)

1. **Daybreak Games (MTGO):** server-side logs exist for every game ever
   played (IDs run to nine digits). A research proposal with privacy terms is
   a long shot with a decisive payoff, and Daybreak has been notably
   community-open (trackers tolerated, data licensed to deck sites).
2. **17lands:** research access / expanded replay export, framed by the public
   replay schema (approach A) and their academic-use precedent. Also request
   personal full-history export.
3. **Penny Dreadful maintainers:** blessing plus a bulk export for B, rather
   than scraping.
4. **XMage server operators:** retained histories or an opt-in recorder (F).

### Deprioritized

- **MTGA VOD parsing** — see reassessment table; the client's own log
  dominates it for any cooperating player.
- **Cockatrice/Untap replays** — structured event streams exist, but no rules
  engine stands behind them (players enforce rules by hand), so every log
  needs full inference *and* the player base tolerates illegal states. Weakest
  quality-per-effort here.
- **Synthetic-only self-play** — already in hand via `run_selfplay.py`; it is
  not human data, but it sets the demand side: the scaling curve over
  self-play + existing Arena captures is what converts "100k" from a guess
  into a requirement.

## Portfolio and sequence

Expected hero-perspective yield is the number to watch (public-only games
counted separately):

| Horizon | Action | Converts |
| --- | --- | --- |
| 2 weeks | Send the four letters; run the YouTube metadata crawler; run scaling curves on the existing corpus; spike 17lands reconstruction on ~100 games; prototype hero-hand reading on one 1080p VOD | Guesses → numbers |
| 1–2 months | Whichever of A/B/C/D the numbers favor, plus the G pilot | First 10k–50k games banked |
| Quarter | Scale the winner; keep F as the gold evaluation set | 100k-equivalent, measured against the curve |

Realistic composition of a 100k-equivalent corpus: majority limited-format
hero-perspective games from A; tens of thousands of constructed public-state
games from B (and D before hand-reading); thousands to tens of thousands of
constructed hero-perspective games from C/D-with-hands and G; a small
options-complete XMage set from F for evaluation. If the A spike's
unique-reconstruction rate comes back strong, quantity stops being the
project's risk at all — the risk moves back to where the scope audit already
points: benchmarks and scaling evidence, not acquisition.

## Rights, consent, and privacy (summary)

- **17lands public files:** published for analysis; confirm training use in
  the letter; cite them.
- **PD logs:** public site, but ask before bulk use; logs contain player
  handles — apply the existing redaction pass before corpus inclusion.
- **VODs:** prefer explicit creator permission (C); a metadata crawl is
  harmless, mass download without asking is gray in both ToS and copyright.
- **MTGOSDK/watchers:** EULA tension acknowledged; follow the PDBot model —
  open operation, community benefit, watchable games only.
- **Donations (G):** consent envelope and redaction already implemented;
  keep the pilot invite-only to stay inside the scope audit's parking
  decision.

[17l-data]: https://www.17lands.com/public_datasets
[17l-dtypes]: https://17lands-public.s3.amazonaws.com/analysis_data/helper_files/replay_dtypes.py
[pd-logs]: https://logs.pennydreadfulmagic.com/
[pd-tools]: https://github.com/PennyDreadfulMTG/Penny-Dreadful-Tools
[mtgosdk]: https://github.com/videre-project/MTGOSDK
[mtgo-docker]: https://github.com/videre-project/mtgo-docker
[xmage]: https://xmage.today/
