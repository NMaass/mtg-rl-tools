# mtgo-native-logs

Native MTGO GameLog/DraftLog files are an opportunistic, high-precision
supplement to the video corpus. This package discovers candidate files and
extracts play-by-play semantic events without pretending that text logs contain
a complete board state.

```powershell
mtgo-native-log discover --out log-files.json
mtgo-native-log parse --input GameLog.txt --out runs\game-001
```

The parser writes the same `ObservedAction` shape used by the video parser and
canonical public-history events from `mtg-state-contract`, so native-log events
can align with video and XMage without a separate model format.

Player names are pseudonymized by default. Use `--keep-player-names` only for a
private local analysis where retaining names is deliberate.
