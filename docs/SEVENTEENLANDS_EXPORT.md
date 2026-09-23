# Getting your 17Lands data out

Mythic patronage backfills your 17Lands account with your **entire** history
(draft and sealed only — 17Lands computes no statistics for constructed), and
allow up to 48 hours after upgrading for it to finish populating.

**17Lands does not offer data exports.** Their FAQ's promise that Mythic
patrons "get personal data for their entire history" means their site computes
and displays your personal statistics over your whole history — lower tiers
are backfilled only about 30 days. It is a backfill of what the site shows
you, not an archive you can download, and their developers have said plainly
that they do not provide exports. Do not buy a tier expecting a download
button; there isn't one at any price. So there are two paths, and they are complementary rather than
alternatives:

1. **Harvest it yourself** through the same JSON API the site's own pages
   call. `magic-cabt-17lands-export` does this. It gets your complete *draft*
   history.
2. **Ask 17Lands for the rest.** The game endpoint is capped at roughly your
   last hundred games, so per-game data does not go back the way draft data
   does. Only they can lift that. A ready-to-send letter is at the bottom of
   this page.

## 1. Harvest it yourself

### Get a session cookie

The exporter authenticates as you, the way your browser already does. It never
asks for a password.

1. Log in at <https://www.17lands.com> in your browser.
2. Open the developer tools (F12) and select the **Network** tab.
3. Load any page that shows your own data — your user page will do.
4. Click any request to a `/data/...` URL, find **Request Headers**, and copy
   the entire value of the `Cookie:` header.
5. Save it to a file, readable only by you:

```sh
umask 077
cat > ~/.17lands-cookie   # paste the cookie, then press Ctrl-D
chmod 600 ~/.17lands-cookie
```

Pass it with `--cookie-file`. Do not pass a cookie as a command-line
argument — it would be recorded in your shell history and in the process
table. `SEVENTEENLANDS_COOKIE` works too if you prefer an environment
variable.

The cookie is a live credential for your 17Lands account: treat it like a
password, do not commit it, and expect it to expire (the exporter says so
plainly when it does, rather than writing empty files).

### Run it — slowly, in four steps

**1. Find out what it would cost.** `--plan` spends exactly one request (the
draft list) and then tells you what the real run needs:

```sh
magic-cabt-17lands-export --cookie-file ~/.17lands-cookie \
    --out 17lands-export --plan
```

> `account has 2148 draft(s); 0 already saved`
> `this run would make 6445 request(s) over about 7h 10m`

**2. Prove it works on five drafts** before pointing it at thousands:

```sh
magic-cabt-17lands-export --cookie-file ~/.17lands-cookie \
    --out 17lands-export --limit 5
```

Open one of the files in `17lands-export/drafts/` and check it holds what you
expect. If the API has moved, you have found out at a cost of sixteen
requests.

**3. Spread the real harvest over several evenings.** There is no prize for
finishing today. Time-box a session and stop:

```sh
magic-cabt-17lands-export --cookie-file ~/.17lands-cookie \
    --out 17lands-export --max-duration 45m
```

Forty-five minutes at the default pace is a few hundred requests — a rounding
error in anyone's traffic. Run it again tomorrow; finished drafts are skipped,
so each session picks up exactly where the last stopped, and each run tells
you how far along you are and how many sessions remain:

```text
account has 2148 draft(s); 1305 already saved (61%)
remaining: 2530 request(s), about 4h 12m at this pace
at 45m per session that is about 6 more session(s)
```

`--max-requests N` is the same idea measured in requests instead of minutes.
Either way, stopping costs nothing, so stop often.

**4. Better still, stop remembering to run it.** `--schedule` prints a
scheduler entry that reruns the exact command you just typed, nightly, until
the history is complete — after which each run costs two requests and does
nothing:

```sh
magic-cabt-17lands-export --cookie-file ~/.17lands-cookie \
    --out 17lands-export --max-duration 45m --schedule cron
```

`--schedule systemd` and `--schedule launchd` emit a timer unit and a
LaunchAgent plist respectively. Printing a snippet makes no request at all.

### Choosing a pace

`--pace` sets the delay, jitter, and breather in one word. The default is
`slow`, not the fastest option, because none of this is urgent:

| Pace | Roughly | Good for |
| --- | --- | --- |
| `glacial` | 230 requests/hour | Leave it running for weeks; barely visible in anyone's logs |
| `slow` (default) | 510 requests/hour | A few thousand drafts across some evenings |
| `steady` | 1300 requests/hour | A short catch-up run, not a full history |

Any individual flag overrides the preset, so `--pace glacial --delay 20` is
glacial's breathers with an even longer gap.

### What it does to stay a good guest

| Behaviour | Why |
| --- | --- |
| One request at a time, `--delay` seconds apart, plus jitter | Never a burst, never a metronome |
| Only the `--parts` you ask for | Requests per draft is the multiplier on everything; three parts instead of four is a quarter less load |
| `Retry-After` obeyed exactly when given | The service knows better than our backoff formula |
| Any 429 slows the **rest of the run** permanently | If it says it is busy once, we do not go back to the old pace |
| A run of consecutive failures trips a breaker and stops | A struggling server should not also have to carry us |
| `--max-requests` / `--max-duration` / `--pause-every` | Spread the cost over days |
| `--pace` defaults to `slow`, not the fastest option | The gentle path is the one you get without thinking |
| Stops instantly on 401/403 | An auth problem is not something to retry |
| Resume on re-run | Stopping is free, so stopping is easy |

Measured on a stub server that pushed back mid-run: request gaps of 1.8s,
1.9s, 1.8s, 1.6s, then exactly 3.0s when the server sent `Retry-After: 3`,
then 2.3–2.7s for every subsequent request — the slowdown persisted rather
than decaying back.

The one lever that reduces load more than any pacing flag is asking for less.
`--parts draft` fetches only the pick-by-pick data — one request per draft
instead of three — and that alone is the training signal:

```sh
magic-cabt-17lands-export --cookie-file ~/.17lands-cookie \
    --out 17lands-export --parts draft
```

Please do not raise the pace or disable the breaker. If you want the data
faster, the letter below is the right lever — not more parallelism.

### What lands on disk

```text
17lands-export/
  user.json               the raw /data/user response
  game_list.json          your recent games (bounded, see below)
  export_manifest.json    counts, failures, notes, request total
  drafts/
    <draft_id>.json       summary + picks + metadata + deck + pool
```

`export_manifest.json` is the honest record: how many drafts were listed
versus written, which parts failed and why, and whether the game list came
back at its ceiling. Read it before assuming a run was complete.

### What it can and cannot reach

| Data | Reachable | Notes |
| --- | --- | --- |
| Every draft you have played | yes | `/data/user` is not paginated or capped |
| Pick-by-pick for each draft | yes | pack contents and your pick, per pick |
| Registered deck and drafted pool | yes | per draft |
| Event metadata, record | yes | format, wins/losses |
| Recent games | ~last 100 | hard service ceiling, not a patron tier limit |
| Full per-game action logs | **no** | see the letter below |

That table is the fidelity answer the training plan was waiting on
([IL_TRAINING_RUN_2026-08.md](IL_TRAINING_RUN_2026-08.md), "Step zero"): the
self-serve harvest yields a complete **draft** corpus and only a recent
**game** sample. So it fully supplies the draft, deck-building, and (with the
game list's opening-hand and result fields) mulligan models, and it does not
supply the gameplay policy corpus. For that, keep the Arena mirror recorder
running on your own play — it captures full-fidelity DecisionRecords with
legal options, which even a perfect 17Lands export would not contain.

## 2. Ask 17Lands for the full export

Worth sending regardless: it is the only route to your complete *game*
history, and 17Lands have been receptive to research use of their data
before. Contact them via the address on their site, their Discord, or
[@17Lands](https://x.com/17lands).

> Subject: Mythic patron — request for a full personal data export
>
> Hi,
>
> I'm a Mythic patron (account: `<your 17Lands username>`, `<your email>`).
> Thank you for backfilling my full history — it's exactly what I was hoping
> for.
>
> I'm doing personal research on imitation learning from Magic gameplay, using
> only my own play data, and I'd like to ask about two things.
>
> First: is there any way to get a bulk export of my own account's data? I can
> reconstruct my draft history through the site, but I'd rather pull a single
> archive once than make thousands of requests against your API, which seems
> better for both of us.
>
> Second, and the one I actually care about: `/data/user_game_list` returns
> roughly my last hundred games. Is the per-game data behind that (the same
> data your replay viewer renders) retrievable for my whole history? If it's
> event-level — the actions in sequence rather than the per-turn aggregates in
> the public `replay_data` files — that would be enormously useful, and I'm
> happy to receive it in whatever form is least work for you: a dump, an
> S3 link, a raised limit on my account, anything.
>
> Entirely my own data, no redistribution, and I'll cite 17Lands in anything
> that comes of it.
>
> On that last point — I'm currently pulling my own draft history through the
> site's API at one request every four seconds, in sessions of a few hundred,
> spread over several days, and I stop entirely if I see a 429. If you'd
> rather I went slower, stayed inside a particular window, or didn't do it at
> all, say the word and I'll follow it.
>
> Thanks for building and running this — it's a remarkable resource.
>
> `<your name>`

The second question is the one to press. The public `replay_data` files are
per-turn aggregates that lose within-turn ordering and targets; whether the
data behind their replay viewer is event-level decides whether your own
history can serve as a gameplay policy corpus or only as value, belief, and
mulligan supervision. See
[DATASET_FEASIBILITY_2026-07.md](DATASET_FEASIBILITY_2026-07.md) for why that
distinction drives the whole plan.

## 3. Feeding it into the pipeline

The exporter writes raw 17Lands JSON, deliberately unconverted — it is your
archive, and the conversion is a separate, re-runnable step. Nothing in the
gameplay training path consumes it yet, because what it yields is draft-level
rather than decision-level. The converter to write next depends on 17Lands'
answer above:

- **event-level game data** → an ingest converter to DecisionRecords, and the
  corpus becomes the gameplay policy corpus the plan assumes;
- **per-turn aggregates only** → the draft/mulligan/deck models, plus
  win-probability and opponent-holding priors, exactly as the addendum in the
  feasibility doc scoped them.

Until then the full-fidelity gameplay corpus comes from the Arena mirror
recorder, and `magic-cabt-training-run` is what turns those captures into a
trained-and-evaluated run:

```sh
magic-cabt-training-run --log captures/Player.log --out runs/first-run
```
