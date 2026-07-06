# Running the Arena → XMage mirror

Follow a live MTG Arena game from its log, watch it mirrored on an XMage board
in real time, and record a replay you can watch back — all locally.

---

## 1. One-time setup

You need, once:

- **JDK 17** and **Maven** (any recent 3.x). The provided scripts default to a
  portable JDK at `C:\Users\nicho\tools\jdk-17.0.19+10` and Maven at
  `C:\Users\nicho\tools\apache-maven-3.9.9`; pass `-JavaHome` / `-MavenHome` to
  the setup script if yours live elsewhere.
- **Python 3.9+** on `PATH` (with `tkinter`, which ships with the standard
  Windows installer).
- An **XMage checkout** to overlay onto — a clone of
  [magefree/mage](https://github.com/magefree/mage). The scripts default to
  `C:\Users\nicho\Code\xmage-goldflush`.

Then build everything once (this copies the overlay into the XMage checkout,
compiles it, and prewarms XMage's card database — allow 20–30 minutes the
first time):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-arena-mirror.ps1
```

Optional: put a launcher icon on your Desktop:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create-desktop-shortcut.ps1
```

## 2. Turn on MTG Arena logging

In MTG Arena: **Options → Account → check "Detailed Logs (Plugin Support)"**,
then restart Arena. Without this, the log contains no gameplay to follow.

The log lives at:

```
%USERPROFILE%\AppData\LocalLow\Wizards Of The Coast\MTGA\Player.log
```

## 3. Launch the mirror

Double-click **`Arena Mirror.bat`** in the repo root (or the **Arena Mirror**
Desktop icon). No terminal needed. From a terminal you can instead run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\arena-mirror-gui.ps1
```

The window has two tabs.

### Follow tab

1. The **MTGA Player.log** path is pre-filled with the standard location; use
   **Locate MTGA logs** if yours differs.
2. Pick an **Output folder** for the recording (default:
   `arena-mirror-runs\session`).
3. Leave **Open XMage on live game** checked.
4. Click **Start following** and play a match in MTG Arena.

When a game begins, XMage opens on its own and mirrors the board — your hand
face-up, your opponent's hand face-down. The left pane shows a status feed; the
right pane lists each decision as it's recorded. The XMage window stays open
across Start/Stop until you close it (or close the GUI). XMage's audio is muted.

> First XMage launch takes ~15–30 s while it loads its card database — give it
> a moment after the first game starts.

### Replays tab

Every recording appears here with its game / decision / state counts. Select
one and click **Watch replay** (or double-click) to play it back into XMage at
the chosen **Speed**.

## 4. What gets recorded

Each run writes a bundle under your output folder:

| File | Contents |
| --- | --- |
| `decisions.jsonl` | one decision per line: the pre-decision board observation, the indexed legal-option list, and the option indices you actually chose (the CABT training format) |
| `mirror_states.jsonl` | every board snapshot streamed to XMage; also the replay stream |
| `game_history.jsonl` | decision prompts/responses + lifecycle events (raw game-state payloads redacted) |
| `summary.json` | counts of games, decisions, and states |
| `card_cache.json` | grpId → card info, so replays resolve cards on other machines |

Hidden information is never written: your opponent's hand and any face-down
cards are recorded as face-down placeholders with no card identity.

---

## Command line (no GUI)

```powershell
# set the classpath once per shell (setup wrote it here):
$env:MAGIC_CABT_CLASSPATH = Get-Content C:\...\xmage-goldflush\Mage.Client\target\mirror-classpath.txt -Raw

python -m magic_cabt.arena_mirror live               # follow the default log
python -m magic_cabt.arena_mirror live --no-display  # record only, no XMage
python -m magic_cabt.arena_mirror replay <bundle>    # watch a bundle back
python -m magic_cabt.arena_mirror gui                # the GUI
```

## Troubleshooting

- **"Player.log not found"** — Arena isn't logging. Enable Detailed Logs (step
  2) and start a game, or click **Locate MTGA logs** and point at the file.
- **GUI opens but XMage never does** — XMage isn't built yet; run
  `scripts\setup-arena-mirror.ps1`. The GUI still records in the meantime.
- **The board stops updating mid-game** — recording is unaffected (check
  `summary.json`); watch the replay to review the full game. If you see a
  "cannot render" note in the log for a specific card, report it.
