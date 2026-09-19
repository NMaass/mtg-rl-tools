# Priority: browser replay workspace

A desktop-only React replay viewer with adjacent Jev analysis. Raw logs are parsed locally by the original Python parsers running in a same-origin Web Worker. D1 stores a private index, encrypted keys, ratings and analysis records; R2 stores parsed replays.

This is the browser foundation, not a completed replacement Magic rules engine. See [engine strategy and research](../docs/BROWSER_REWRITE.md).

## Local UI

Requirements: Node 22.12+ and Python 3.9+.

```sh
cd web
npm install
python3 scripts/sync_cards.py
npm run build
npm run preview
```

Open the printed local address and choose **Explore example** or import a real log. Preview works without Cloudflare credentials. Saving a replay or analyzing it requires the local or hosted API below. The example is labeled synthetic and never sends paid requests.

Card metadata comes from Scryfall's bulk export. `sync_cards.py` accepts legacy JSON and gzipped JSONL descriptors and can use `--bulk /path/to/default-cards.jsonl.gz` offline. A missing catalogue does not magically resolve card IDs. Do not evaluate model strength on unresolved recordings.

## Local full-stack API

```sh
cd web
cp .dev.vars.example .dev.vars
# Replace KEY_ENCRYPTION_KEY with the output of: openssl rand -base64 32
npm run db:local
npm run build
npx wrangler dev
```

`DEV_AUTH=local-test` only works on localhost/127.0.0.1. It is refused on a workers.dev or custom hostname. Never deploy it. Local Wrangler D1/R2 data remains separate from production.

## Deploy to your Worker address

Use an authenticated Cloudflare account. No account IDs, keys or deployment credentials are committed.

```sh
cd web
npx wrangler login
npx wrangler d1 create priority
npx wrangler r2 bucket create priority-replays
```

Put the returned D1 ID in `wrangler.jsonc`. In Cloudflare Zero Trust, create an Access application protecting the planned hostname `priority-replay.<your-subdomain>.workers.dev`, or protect this Worker through its Access tab. Restrict the policy to the intended beta users. Put the Access issuer (`https://<team>.cloudflareaccess.com`) and application audience in `ACCESS_ISSUER` and `ACCESS_AUD`.

The Worker validates the JWT itself; an unvalidated email/header never becomes an owner ID. Until this is configured, API requests fail closed. The UI can still be a local preview.

```sh
openssl rand -base64 32 | npx wrangler secret put KEY_ENCRYPTION_KEY
npm run db:remote
python3 scripts/sync_cards.py
node scripts/preflight.mjs
npm run deploy
```

Keep the encryption secret in your own secret manager. The database contains AES-GCM ciphertext, not raw OpenRouter keys. Rotating the master key requires a migration using the old key; replacing it blindly makes saved keys unreadable. The Worker itself must be trusted: encryption is not protection from someone controlling the application and its master key.

The deployment is one Worker plus its static assets, D1 and R2. It does not deploy a native engine or provision tournaments. No OpenRouter key is required for deployment; each user saves their own in Settings. OpenRouter/TypeSafe remains the requested external inference provider; application hosting and storage are on Cloudflare.

## Import and review

**Arena:** Settings → View Account → Detailed Logs (Plugin Support), restart, then play. Choose Player.log. The import dialog shows the conventional Windows/macOS paths. Browsers cannot silently search a user's disk.

**MTGO:** choose a copied/exported text game log. Binary .dat replay decoding is not implemented. The existing text parser's reconstructed public state is clearly marked incomplete; it does not infer hands, legal action sets, or exact engine roots.

Import first previews the parsed games. **Save to library** uploads parsed replays only, never the original log. Replays are private to the validated account. The library shows the latest 200 entries. There is no public bucket URL.

Arrow keys step positions; Shift + arrows jump captured decisions; Home/End seek boundaries; Space toggles playback. Keyboard shortcuts do not fire while editing or inside a dialog. Autoplay never calls Jev. **After stepping** is opt-in; unsent work is cancelled on another navigation, perspective change, dialog, or playback action.

Jev receives only the selected pre-action perspective and captured legal options. It does not receive the actual move, later frames, the result, or the other perspective. Results are cached. An invalid or failed paid response retains reported cost. Unknown costs stay unknown. Separate attempt records preserve previous charges after a retry. The private `/api/replays/<id>/report` endpoint exports ratings and attempt records without credentials.

## Verification

```sh
cd python
python3 -m unittest discover -s tests -p test_browser_export.py -v
cd ../web
python3 scripts/make_fixture.py
npm run build
npm test
npx playwright install chromium
npm run test:browser
```

CI uploads the tested source, built assets, parser runtime, screenshots, dependency lock, and failing browser traces. Provider fixtures test behavior without making paid calls. Local D1/R2 tests use Wrangler's actual local bindings. Native Python/Wasm tests use exactly the same raw Arena fixture.

The pinned XMage workflow separately builds the real Java engine and overlay, runs its bridge tests, and exercises the native observe/step/checkpoint/restore boundary. This is not every upstream card test or a complete equivalence proof. Recorded view-only imports cannot be resumed as full native games.

## Current limits

The Arena importer is only as complete as the existing tracker. It does not yet derive a full private-knowledge ledger from scry/reveal/shuffle events. Independent perspective views are represented and tested, but missing perspectives are never invented. There is no Rust rules port, hosted agent arena, automatic watched-directory integration, or proven arbitrary-point continuation for partial logs in this branch.
