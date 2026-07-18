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

### Known limitations on this clip (all honest / by-design)

- **Phase is not modelled.** In the old client the active step is a *colour
  highlight* on a fixed `Untap…Cleanup` bar, not distinct on-screen text, so text
  OCR cannot isolate it. A colour-highlight detector is a future enhancement. Its
  absence means every state fails the training-quality gate (`/phase` is a
  critical field) → all 262 states land in `quarantine.jsonl`, `trainingCandidates
  = 0`. This is the quality gate working conservatively, not a failure.
- **Card names are unknown** (`identifiedCards = 0`). Battlefield *rectangles* are
  detected (giving zone counts, noisily over-counted) but names need a card-DB
  resolver (`--card-db`), which was not supplied. The parser never guesses a name.
- **Hand-count** is reliable mid-game (0.96) but low-confidence on the turn-1
  mulligan pill (~0.38) — correctly flagged rather than trusted.

## Not yet run here: the XMage follower stage

`mtgo-pipeline follow` / `xmage-follow` replays the extracted actions through the
CABT `CabtProtocolServer` and needs **Java + a compiled CABT classpath**
(`MAGIC_CABT_CLASSPATH`). Neither `java` nor the classpath is present in this
environment (`doctor` → `readyForXmage: false`), so verification against XMage is
the remaining step once the CABT module is built.
