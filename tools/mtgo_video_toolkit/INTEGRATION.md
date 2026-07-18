# mtgo_video_toolkit — integration notes

Six vendored Python packages that turn **MTGO gameplay video** (and native MTGO
logs) into **canonical symbolic game states**, then replay those states through
the existing XMage CABT bridge for verification. Vendored here without editing
any existing repository source, per the toolkit's own `assemble` contract
(destination `tools/mtgo_video_toolkit`, `integrationMode:
vendored-no-existing-files-edited`).

## Packages (dependency order)

| Package | Console script | Role |
|---|---|---|
| `mtg_state_contract` | — | Source-neutral canonical state schema + confidence-aware comparator. Foundation for the rest. |
| `mtgo_video_acquisition` | `mtgo-video-source` | Reproducible local video corpus manifest; thin, opt-in yt-dlp wrapper (no auth/DRM/cookies). |
| `mtgo_video_parser` | `mtgo-video` | Headless OCR/VLM screen reader → `canonical_states.jsonl` + `observed_actions.jsonl`. |
| `mtgo_native_logs` | `mtgo-native-log` | Parse native MTGO GameLog/DraftLog text into the same `ObservedAction` shape. |
| `xmage_state_follower` | `xmage-follow` | Replay observed actions through `CabtProtocolServer`, compare states, emit mismatch report. |
| `mtgo_pipeline_coordinator` | `mtgo-pipeline`, `mtgo-toolkit-assemble` | Local orchestrator: extract → follow → bundle. Leaves GitHub out of the execution path. |

## Install (isolated venv)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install \
  -e tools/mtgo_video_toolkit/mtg_state_contract \
  -e tools/mtgo_video_toolkit/mtgo_video_acquisition \
  -e tools/mtgo_video_toolkit/mtgo_video_parser \
  -e tools/mtgo_video_toolkit/mtgo_native_logs \
  -e tools/mtgo_video_toolkit/xmage_state_follower \
  -e tools/mtgo_video_toolkit/mtgo_pipeline_coordinator
```

Optional backends: PaddleOCR GPU — `pip install paddleocr` plus a matching
`paddlepaddle-gpu` wheel from the Paddle CUDA index (this repo was validated with
`paddlepaddle-gpu==3.2.0` from the `cu126` index on an RTX 3070 Ti). yt-dlp needs
`ffmpeg` on `PATH` to merge 1080p adaptive streams. `mtgo-pipeline doctor`
reports what is present.

All 27 vendored unit tests pass (`pytest` in each package dir), 1 skipped
(tesseract-binary dependent).

## First validated run — WotC MOCS 2016 clip

Tested on `youtube FeNbwi7-lT0` — *Magic Online Championship Series 2016 February
Playoff Quarterfinal 3* (official Magic: The Gathering channel, 3:39, 1080p,
video sha256 `6a1dac65…`), downloaded with the acquisition package's explicit
`--acknowledge-rights-and-terms` gate.

```bash
# metadata-only manifest
mtgo-video-source discover --url https://www.youtube.com/watch?v=FeNbwi7-lT0 --out mtgo-runs/sources.json
# download (requires rights/terms acknowledgement)
mtgo-video-source download --manifest mtgo-runs/sources.json --out mtgo-runs/corpus --acknowledge-rights-and-terms
# extract canonical states (paddle GPU + the calibrated profile below)
mtgo-pipeline extract --video "<downloaded .mp4>" --bundle mtgo-runs/match-001 \
  --layout tools/mtgo_video_toolkit/mtgo_video_parser/src/mtgo_video_parser/profiles/mocs_2016_oldclient.yaml \
  --ocr paddle
```

Result: **262 sampled frames → 262 canonical states, 0 extraction errors.**
Scalar fields extract at high confidence — life totals, turn number, and the
game-log text read at 0.92–0.98 (paddle read "20" at 1.00 on the life crop). The
log-action parser recovered 2 `MULLIGAN` events. The life/turn timeline is
internally coherent (life decrements within a segment; the broadcast's repeated
scrubbing back to 20/20 between game segments is faithfully mirrored, not a
parser fault).

### The `mocs_2016_oldclient.yaml` layout profile

The bundled `mtgo-1080p-standard` profile targets the **modern** MTGO client
(life totals on the right, log on the right). This 2016 MOCS broadcast uses the
**old** client — player panels (avatar, life, timer, library/hand pills) in the
**left sidebar**, game log as the left text box, turn indicator on the bottom
bar. `mocs_2016_oldclient.yaml` is a new profile calibrated to that layout; it is
the concrete instance of the "calibrate per channel before bulk processing" step
the parser README documents. Calibration coordinates were read directly off
sampled frames and verified against known frames (turn-1 mulligan and turn-10
board).

### Phase detection (`phase_bar` region kind)

The old client marks the active step with an orange *colour highlight* on a fixed
`Untap…Cleanup` bar, not distinct text, so a `phase_bar` region kind
(`perception.py`) locates the highlight centroid and maps it to the nearest step
(the two Main phases disambiguate by x-position). Step centres were derived by
clustering the highlight centroid across the clip. Result on this video:

| | without phase | with `phase_bar` |
|---|---|---|
| states with phase | 0 / 262 | **242 / 262** (0.82 mean conf) |
| quarantined states | 262 | **55** |
| training candidates | 0 | **207** |
| quality mean | 0.567 | **0.652** |

### Card-name recognition (`--card-db`)

With the Scryfall `oracle_cards` bulk supplied via `--card-db`, the recognizer
OCRs each detected card's title bar and validates against the 39.6k-card
dictionary (fail-closed: unresolved text stays unknown). On the turn-10 board it
identifies the real cards — Deathmist Raptor, Canopy Vista, Lumbering Falls,
Silkwrap, Prairie Stream, Forest/Plains, Yavimaya Coast (leearson); Knight Ally,
Jace Telepath Unbound, Shambling Vent, Battlefield Forge, Sunken Hollow,
Island/Mountain (beena); "Morph" for face-down 2/2s. A minority of fuzzy matches
are wrong (raise `minimum_resolver_score` to trade recall for precision). Full
per-frame card OCR is slow (~an hour for the clip on this GPU); use it on
targeted segments rather than whole broadcasts.

### Remaining honest limitation

- **Hand-count** is reliable mid-game (0.96) but low-confidence on the turn-1
  mulligan pill (~0.38) — correctly flagged rather than trusted.

## The XMage follower stage — wired and proven

`mtgo-pipeline follow` / `xmage-follow` replays actions through the CABT
`CabtProtocolServer` and needs **Java + a compiled CABT classpath**
(`MAGIC_CABT_CLASSPATH`). Built here from the `xmage-goldflush` checkout
(JDK17 + Maven, `mvn -pl Mage.Server.Plugins/Mage.Player.AI test-compile`;
classpath = `target/classes` + `target/cabt-deps.txt`). The bridge boots and
speaks the protocol end-to-end: `game_start` → decision options, `game_select`,
and `visualize_data` snapshots return full board state.

Running the follower on this clip's bundle **correctly fails closed** — it launches
XMage, ranks the observed MULLIGAN action against the engine's options, and
reports "no hypothesis survived" because *exact* replay needs both real decklists,
the seed, and every intervening decision (per the follower README), none of which
a 3.6-min broadcast provides. The stack is verified; a clip with recoverable
decklists is what turns "runs" into "verified".
