# Cloudflare ingest

This directory contains Cloudflare Worker scaffolding for opt-in Magic CABT training bundle uploads.

The worker is intentionally conservative: it accepts bounded JSON envelopes built by `magic-cabt-upload`, validates the consent and record shape, deduplicates by hash, and stores accepted envelopes in R2 with metadata in KV.
