# Cloudflare upload ingest

This directory contains Cloudflare Worker scaffolding for opt-in Magic CABT training bundle uploads.

The worker is intentionally conservative: it accepts bounded JSON envelopes built by `magic-cabt-upload`, validates consent and record shape, rejects raw Player.log-style uploads, deduplicates by bundle hash, stores accepted envelopes in R2, and writes dedupe metadata to KV when bindings are configured.

## Local client flow

```sh
magic-cabt-upload \
  --bundle runs/arena/2026-07-09 \
  --dry-run \
  --yes

magic-cabt-upload \
  --bundle runs/arena/2026-07-09 \
  --endpoint https://<worker-host>/bundles \
  --api-key "$UPLOAD_TOKEN" \
  --yes
```

The uploader builds a redacted envelope from `decisions.jsonl` and `summary.json`; it does not upload raw MTGA logs by default.

## Worker deployment sketch

```sh
cd cloudflare/ingest-worker
wrangler r2 bucket create magic-cabt-uploads
wrangler kv namespace create UPLOAD_KV
wrangler secret put UPLOAD_TOKEN
wrangler deploy
```

Update `wrangler.toml` with the created KV namespace id before deployment.

## Server-side junk filters included

- bearer token support through `UPLOAD_TOKEN`;
- 5 MB body limit;
- required `kind = magic_cabt_upload_bundle`;
- required `consent.allowTraining = true`;
- required manifest;
- required decision array of 1..20,000 records;
- required decision prompt plus selected indices;
- raw-log uploads rejected via `redactionReport.rawPlayerLogUploaded === false`;
- dedupe by stable bundle hash.

These checks are not proof of honest play. They are intake hygiene. Training ingestion should still perform deeper schema validation, duplicate detection, and statistical outlier filtering before records enter a trusted dataset.
