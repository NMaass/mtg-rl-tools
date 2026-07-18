# mtgo-video-parser

A headless, plugin-oriented MTGO screen reader that writes canonical symbolic
states and semantic actions. The intended large-scale path is:

1. adaptive frame sampling;
2. fixed-layout region extraction;
3. fast local OCR on every selected frame;
4. temporal consensus and event-log parsing;
5. optional VLM second pass for uncertain crops;
6. canonical-state output for XMage comparison and model tensorization.

## Why fixed layout first

MTGO lets users resize and pop out zones. A generic UI model is useful for
calibration, but a channel-specific normalized profile is substantially cheaper
and more stable for millions of frames. Use `print-layout`, adjust the YAML once
for a channel/layout, then process headlessly.

## Commands

```powershell
mtgo-video doctor
mtgo-video print-layout --out profiles\my-channel.yaml
mtgo-video sample --video match.mp4 --out sample-frames
mtgo-video extract --video match.mp4 --layout profiles\my-channel.yaml `
  --ocr paddle --out runs\match-001 --fps 2
```

OpenRouter or a local OpenAI-compatible VLM server can be used as the OCR
backend:

```powershell
$env:OPENROUTER_API_KEY = "..."
mtgo-video extract --video match.mp4 --out runs\match-001 `
  --ocr openrouter --model qwen/qwen3-vl-8b-instruct
```

For a local server:

```powershell
mtgo-video extract --video match.mp4 --out runs\match-001 `
  --ocr local-vlm --endpoint http://127.0.0.1:8000/v1/chat/completions `
  --model Qwen/Qwen3-VL-2B-Instruct
```

The base install uses Tesseract. PaddleOCR is an optional high-throughput local
backend. Install the PaddlePaddle GPU wheel matching the machine's CUDA runtime,
then `pip install -e ".[paddle]"`.

## Output

```text
manifest.json
perceived_frames.jsonl
canonical_states.jsonl
observed_actions.jsonl
frames/                  # optional
```

The parser never labels an unreadable opponent card as known. Low-confidence
values remain explicit unknowns for the comparator and training quarantine.

Additional extraction files include `training_candidates.jsonl`,
`quarantine.jsonl`, `extraction_errors.jsonl`, and `quality_report.json`.
Player names are pseudonymized by default; `--keep-player-names` is an explicit
private-local opt-out.
