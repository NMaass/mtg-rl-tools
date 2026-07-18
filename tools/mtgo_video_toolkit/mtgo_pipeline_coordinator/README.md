# mtgo-pipeline-coordinator

A local coordinator that leaves GitHub out of the execution path. It runs the
video parser, invokes the XMage follower, writes a reproducible bundle, and can
vendor all modules into a local checkout without editing any existing source
file.

```powershell
mtgo-pipeline doctor

mtgo-pipeline extract `
  --video D:\videos\match.mp4 `
  --bundle D:\mtgo-runs\match-001 `
  --layout D:\profiles\channel-a.yaml `
  --ocr paddle

mtgo-pipeline follow `
  --bundle D:\mtgo-runs\match-001 `
  --manifest D:\manifests\match-001.json `
  --classpath $env:MAGIC_CABT_CLASSPATH
```

To copy the entire toolkit into a local `mtg-rl-tools` checkout:

```powershell
mtgo-toolkit-assemble `
  --toolkit-root D:\downloads\mtgo_video_toolkit `
  --repo D:\src\mtg-rl-tools
```

The assembler writes under `tools/mtgo_video_toolkit` only. It does not modify
existing repository files, create branches, invoke GitHub, or push anything.
