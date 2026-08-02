"""One-command training run: saved MTGA ``Player.log`` files in, report out.

``magic-cabt-training-run`` chains the supported pipeline —

    ingest -> collect -> validate -> audit -> manifest -> split -> macro
           -> compile -> baselines -> train-bc -> train-torch -> compare
           -> report

— into a single resumable run directory, so "point it at the logs" is the
whole workflow. It orchestrates the existing supported commands rather than
reimplementing them; every stage's artifact is the same file the standalone
CLI would have produced.

Contracts the orchestrator adds on top of the stages it wraps:

- **Fail closed.** A stage that cannot produce its artifact stops the run
  with an actionable message; nothing downstream trains on data that did not
  pass ``validate``. An untrusted ``audit`` verdict also stops the run unless
  ``--allow-untrusted`` is given (the report records the verdict either way).
- **Whole-game splits.** Train/val/test assignment hashes ``gameId`` with the
  run seed, so adjacent decisions from one game can never cross a split
  boundary, and the same corpus + seed always yields the same split.
- **Resumable.** Each stage records an input fingerprint in ``state.json``;
  re-running the same command skips stages whose inputs did not change.
  ``--force`` redoes everything.
- **Degrades honestly.** Torch model stages are skipped with a recorded
  reason when torch is not installed; the report says "skipped", never "ok".
- **Reproducible.** ``config.json`` snapshots arguments, resolved inputs,
  environment, and the combined-dataset hash.

No XMage build is required: batch ingest replays saved logs through the same
normalizer/tracker/recorder the live Arena mirror uses, with no display.

``--toy N`` generates a small synthetic corpus with a learnable rule instead
of reading logs — an end-to-end smoke test of the whole pipeline (and of a
fresh install) that finishes in seconds.
"""

import argparse
import contextlib
import datetime
import hashlib
import io
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback

from magic_cabt.arena_log import iter_log_entries
from magic_cabt.training.io import iter_decision_records

__all__ = [
    "STAGE_NAMES",
    "StageError",
    "build_parser",
    "generate_toy_records",
    "main",
    "run",
    "split_by_game",
]

STAGE_NAMES = (
    "ingest",
    "collect",
    "validate",
    "audit",
    "manifest",
    "split",
    "macro",
    "compile",
    "baselines",
    "train-bc",
    "train-torch",
    "compare",
    "report",
)

# Bytes of head/tail hashed to fingerprint a log file cheaply. Player.log
# files are append-only, so size + head + tail identifies a snapshot without
# reading a multi-hundred-MB file twice per run.
_FINGERPRINT_WINDOW = 1 << 20

_STATE_FILE = "state.json"
_CONFIG_FILE = "config.json"


class StageError(Exception):
    """A stage failed in a way the user has to resolve; message is the fix."""


# ---------------------------------------------------------------------------
# Run context


class RunContext(object):
    """Paths, config, and the persistent stage state for one run directory."""

    def __init__(self, args):
        self.args = args
        self.out = os.path.abspath(args.out)
        self.logs_dir = os.path.join(self.out, "logs")
        self.ingest_dir = os.path.join(self.out, "ingest")
        self.dataset_dir = os.path.join(self.out, "dataset")
        self.il_dir = os.path.join(self.out, "il")
        self.models_dir = os.path.join(self.out, "models")
        self.eval_dir = os.path.join(self.out, "eval")
        self.combined = os.path.join(self.dataset_dir, "all_decisions.jsonl")
        self.state = {"schemaVersion": 1, "stages": {}}
        self.stage_meta = {}
        self.console = _Console(quiet=args.quiet)
        self._torch_available = None

    @property
    def torch_available(self):
        """Lazy: importing torch costs seconds, so pay only when a torch
        stage (or the dry-run plan) actually asks."""
        if self._torch_available is None:
            self._torch_available = _torch_available()
        return self._torch_available

    # -- state -------------------------------------------------------------

    def load_state(self):
        path = os.path.join(self.out, _STATE_FILE)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict) and isinstance(loaded.get("stages"), dict):
                self.state = loaded

    def save_state(self):
        _atomic_json(os.path.join(self.out, _STATE_FILE), self.state)

    def stage_state(self, name):
        return self.state["stages"].get(name) or {}

    def record_stage(self, name, status, fingerprint=None, seconds=None,
                     detail=None):
        entry = {
            "status": status,
            "finishedAt": _utc_now(),
        }
        if fingerprint is not None:
            entry["fingerprint"] = fingerprint
        if seconds is not None:
            entry["seconds"] = round(seconds, 3)
        if detail:
            entry["detail"] = detail
        self.state["stages"][name] = entry
        self.save_state()

    # -- console -----------------------------------------------------------

    def say(self, text):
        self.console.info(text)

    def stage_log_path(self, name):
        return os.path.join(self.logs_dir, "%s.log" % name)


class _Console(object):
    """Terminal presenter: stable plain lines when piped, live lines on a TTY.

    Each stage renders as one aligned line. On a TTY the line appears as soon
    as the stage starts and is rewritten in place when it finishes (and while
    it reports progress), so the display stays put instead of scrolling
    mid-stage. When piped — CI, log files — no control codes are emitted and
    the start and finish are separate plain lines, so a hung run still shows
    where it stopped. Color follows the stream (a TTY, ``TERM`` not ``dumb``)
    and the ``NO_COLOR`` convention.
    """

    _STATUS_COLORS = {"ok": "32", "cached": "36", "skipped": "33",
                      "failed": "31", "interrupted": "31"}

    def __init__(self, stream=None, quiet=False):
        self.stream = stream if stream is not None else sys.stderr
        self.quiet = quiet
        isatty = getattr(self.stream, "isatty", None)
        self.live = bool(isatty and isatty()) \
            and os.environ.get("TERM") != "dumb"
        self.color = self.live and not os.environ.get("NO_COLOR")
        self._open = None

    # -- painting ----------------------------------------------------------

    def _paint(self, text, code):
        if self.color and code:
            return "\x1b[%sm%s\x1b[0m" % (code, text)
        return text

    def _stage_prefix(self, index, total, name):
        return "[%2d/%d] %-11s" % (index, total, name)

    # -- stage lines -------------------------------------------------------

    def stage_begin(self, index, total, name):
        if self.quiet:
            return
        prefix = self._stage_prefix(index, total, name)
        if self.live:
            self._open = (index, total, name)
            self.stream.write("\r\x1b[K%s %s" % (
                prefix, self._paint("running", "2")))
        else:
            self.stream.write("%s running\n" % prefix)
        self.stream.flush()

    def progress(self, text):
        """Refresh the open stage line with a short progress note (TTY only)."""
        if self.quiet or not self.live or self._open is None:
            return
        index, total, name = self._open
        self.stream.write("\r\x1b[K%s %s  %s" % (
            self._stage_prefix(index, total, name),
            self._paint("running", "2"), text))
        self.stream.flush()

    def stage_end(self, index, total, name, status, seconds=None, note=None):
        if self.quiet:
            self._open = None
            return
        parts = [self._stage_prefix(index, total, name),
                 self._paint("%-11s" % status,
                             self._STATUS_COLORS.get(status))]
        if seconds is not None:
            parts.append("%6.1fs" % seconds)
        if note:
            note = str(note)
            if len(note) > 60:
                note = note[:59] + "…"
            parts.append(" %s" % note)
        line = " ".join(parts).rstrip()
        if self.live:
            self.stream.write("\r\x1b[K%s\n" % line)
        else:
            self.stream.write("%s\n" % line)
        self._open = None
        self.stream.flush()

    # -- prose lines -------------------------------------------------------

    def info(self, text):
        if self.quiet:
            return
        code = "33" if text.startswith("warning:") else None
        self._write_line("[training-run] %s" % self._paint(text, code))

    def error(self, text):
        self._write_line(self._paint("error: %s" % text, "31"))

    def _write_line(self, line):
        """Print a prose line without destroying an open live stage line."""
        if self.live and self._open is not None:
            index, total, name = self._open
            self.stream.write("\r\x1b[K%s\n" % line)
            self.stream.write("%s %s" % (
                self._stage_prefix(index, total, name),
                self._paint("running", "2")))
        else:
            self.stream.write("%s\n" % line)
        self.stream.flush()


# ---------------------------------------------------------------------------
# Small utilities


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _atomic_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_log(path):
    """Cheap content identity for a (possibly large) append-only log file."""
    size = os.path.getsize(path)
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with open(path, "rb") as handle:
        digest.update(handle.read(_FINGERPRINT_WINDOW))
        if size > _FINGERPRINT_WINDOW:
            handle.seek(max(0, size - _FINGERPRINT_WINDOW))
            digest.update(handle.read(_FINGERPRINT_WINDOW))
    return digest.hexdigest()


def _fingerprint_value(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _torch_available():
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@contextlib.contextmanager
def _stage_output(ctx, name):
    """Send a wrapped CLI's stdout/stderr to this stage's log file."""
    os.makedirs(ctx.logs_dir, exist_ok=True)
    with open(ctx.stage_log_path(name), "a", encoding="utf-8") as handle:
        handle.write("---- %s %s\n" % (name, _utc_now()))
        with contextlib.redirect_stdout(handle), \
                contextlib.redirect_stderr(handle):
            yield handle


def _run_wrapped_main(ctx, stage, module_main, argv, allowed_codes=(0,)):
    """Call another CLI's ``main(argv)`` in-process, logging its output.

    Returns the exit code. Codes outside ``allowed_codes`` raise StageError
    pointing at the captured log.
    """
    with _stage_output(ctx, stage) as handle:
        handle.write("$ %s\n" % " ".join(argv))
        try:
            code = module_main(argv)
        except SystemExit as exc:  # argparse errors and explicit exits
            code = exc.code if isinstance(exc.code, int) else 1
        except StageError:
            raise
        except Exception:
            handle.write(traceback.format_exc())
            raise StageError(
                "%s crashed; full traceback in %s"
                % (stage, ctx.stage_log_path(stage)))
    code = 0 if code is None else code
    if code not in allowed_codes:
        raise StageError(
            "%s exited with code %s; details in %s"
            % (stage, code, ctx.stage_log_path(stage)))
    return code


# ---------------------------------------------------------------------------
# Toy corpus

_TOY_GOOD = "Toy Champion"
_TOY_BAD = "Toy Blunder"
_TOY_LANDS = ("Toy Plains", "Toy Island", "Toy Swamp")


def generate_toy_records(games, seed=0):
    """Yield synthetic DecisionRecords with a learnable preference rule.

    The latent rule is simple enough for the bag-of-words baseline to find,
    but invisible to the nonlearned controls: when an option casts
    ``Toy Champion`` the hero takes it 95% of the time; otherwise the hero
    acts uniformly at random, so ``first-legal`` gains nothing from the
    hero's habits. A trained policy therefore has to beat both baselines on
    held-out games before the toy run reports success. Records follow the
    canonical DecisionRecord v1 shape used by the engine self-play fixtures,
    so they pass ``validate`` and ``audit`` unchanged.
    """
    rng = random.Random(seed)
    for game_number in range(games):
        game_id = "toy-%d-%d" % (seed, game_number)
        decisions = rng.randint(14, 26)
        life = [20, 20]
        for sequence in range(decisions):
            terminal = sequence == decisions - 1
            turn = 1 + sequence // 2
            options = [{"index": 0, "type": "PASS_PRIORITY",
                        "label": "Pass priority"}]
            cast_indices = {}
            for name in _toy_hand(rng):
                option = {
                    "index": len(options),
                    "type": "CAST_SPELL",
                    "label": "Cast %s" % name,
                    "payload": {"card": {"name": name}},
                }
                cast_indices[name] = option["index"]
                options.append(option)
            if _TOY_GOOD in cast_indices and rng.random() < 0.95:
                chosen = cast_indices[_TOY_GOOD]
            else:
                chosen = rng.randrange(len(options))
            if terminal:
                life[1] = 0
            yield {
                "schemaVersion": 1,
                "source": "engine_selfplay",
                "gameId": game_id,
                "sequenceNumber": sequence,
                "playerIndex": 0,
                "observation": {
                    "current": {
                        "turnNumber": turn,
                        "activePlayerId": "p0",
                        "priorityPlayerId": "p0",
                        "phase": "PRECOMBAT_MAIN",
                        "players": [
                            {"playerIndex": 0, "playerId": "p0",
                             "name": "Hero", "life": life[0],
                             "handCount": max(0, 7 - turn // 2),
                             "libraryCount": 53 - sequence},
                            {"playerIndex": 1, "playerId": "p1",
                             "name": "Rival", "life": life[1],
                             "handCount": 6, "libraryCount": 52},
                        ],
                        "stack": [],
                        "battlefield": [{"name": name}
                                        for name in _TOY_LANDS[:turn % 4]],
                    },
                    "select": {"type": "PRIORITY", "minCount": 1,
                               "maxCount": 1, "option": options},
                },
                "select": {"type": "PRIORITY", "playerIndex": 0,
                           "minCount": 1, "maxCount": 1, "option": options},
                "selectedIndices": [chosen],
                "terminal": terminal,
                "result": {"winnerPlayerIndex": 0} if terminal else None,
                "metadata": {"toy": True, "toySeed": seed},
            }


def _toy_hand(rng):
    hand = []
    if rng.random() < 0.6:
        hand.append(_TOY_GOOD)
    if rng.random() < 0.5:
        hand.append(_TOY_BAD)
    for name in _TOY_LANDS:
        if rng.random() < 0.25:
            hand.append(name)
    rng.shuffle(hand)
    return hand


# ---------------------------------------------------------------------------
# Whole-game splitting


def split_by_game(game_ids, seed, val_fraction, test_fraction):
    """Deterministically assign whole games to train/val/test.

    The assignment hashes ``(seed, gameId)`` so it is stable across runs and
    independent of input order, then fills test and val to their target game
    counts from the hash-ordered list. Returns ``{gameId: split}``.
    """
    unique = sorted(set(game_ids))
    if not unique:
        return {}
    ordered = sorted(
        unique,
        key=lambda gid: hashlib.sha256(
            ("%d:%s" % (seed, gid)).encode("utf-8")).hexdigest())
    total = len(ordered)
    test_count = int(round(total * test_fraction))
    val_count = int(round(total * val_fraction))
    if test_fraction > 0 and test_count == 0:
        test_count = 1
    if val_fraction > 0 and val_count == 0:
        val_count = 1
    if test_count + val_count >= total:
        raise StageError(
            "cannot hold out %d val + %d test games from only %d games; "
            "add data or lower --val-fraction/--test-fraction"
            % (val_count, test_count, total))
    assignment = {}
    for position, gid in enumerate(ordered):
        if position < test_count:
            assignment[gid] = "test"
        elif position < test_count + val_count:
            assignment[gid] = "val"
        else:
            assignment[gid] = "train"
    return assignment


# ---------------------------------------------------------------------------
# Stages


def stage_ingest(ctx):
    """Turn each saved Player.log into an Arena-mirror bundle, no display."""
    logs = _discover_logs(ctx.args.log)
    produced = []
    if ctx.args.toy:
        os.makedirs(ctx.ingest_dir, exist_ok=True)
        toy_path = os.path.join(ctx.ingest_dir, "toy_decisions.jsonl")
        count = 0
        with open(toy_path, "w", encoding="utf-8") as handle:
            for record in generate_toy_records(ctx.args.toy, seed=ctx.args.seed):
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                count += 1
        produced.append({"source": "toy", "path": toy_path,
                         "decisions": count})
    for log_path in logs:
        produced.append(_ingest_one_log(ctx, log_path))
    ctx.stage_meta["ingest"] = {"bundles": produced}
    if produced:
        os.makedirs(ctx.ingest_dir, exist_ok=True)
        _atomic_json(os.path.join(ctx.ingest_dir, "ingest_report.json"),
                     {"bundles": produced})
    return {"bundles": produced}


def _discover_logs(entries):
    """Expand ``--log`` values: files as-is, directories scanned for logs."""
    logs = []
    for entry in entries or []:
        path = os.path.abspath(os.path.expanduser(entry))
        if os.path.isfile(path):
            logs.append(path)
        elif os.path.isdir(path):
            found = []
            for name in sorted(os.listdir(path)):
                if name.lower().endswith(".log"):
                    found.append(os.path.join(path, name))
            if not found:
                raise StageError(
                    "no .log files found in directory %s" % path)
            logs.extend(found)
        else:
            raise StageError(
                "log path does not exist: %s (pass a Player.log file or a "
                "directory containing .log files)" % path)
    return logs


def _ingest_one_log(ctx, log_path):
    from magic_cabt.arena_mirror.recorder import MirrorRecorder
    from magic_cabt.arena_mirror.session import MirrorSession

    fingerprint = _fingerprint_log(log_path)
    bundle_dir = os.path.join(ctx.ingest_dir, fingerprint[:16])
    marker = os.path.join(bundle_dir, "ingest_source.json")
    if os.path.exists(marker):
        try:
            existing = _read_json(marker)
        except ValueError:
            existing = {}
        if existing.get("fingerprint") == fingerprint:
            ctx.say("ingest cached: %s" % os.path.basename(log_path))
            existing["cached"] = True
            return existing
    if os.path.isdir(bundle_dir):
        # No marker (a previous ingest died mid-way) or a marker for other
        # content. The recorder appends, so ingesting into the leftovers
        # would silently duplicate records; the bundle is derived data, so
        # rebuild it from the log instead.
        ctx.say("rebuilding incomplete ingest bundle for %s"
                % os.path.basename(log_path))
        shutil.rmtree(bundle_dir)

    card_db = _open_card_db(ctx)
    os.makedirs(bundle_dir, exist_ok=True)
    with _stage_output(ctx, "ingest") as log_stream:
        log_stream.write("ingesting %s -> %s\n" % (log_path, bundle_dir))
        recorder = MirrorRecorder(bundle_dir, card_db=card_db)
        session = MirrorSession(recorder=recorder, display=None,
                                card_db=card_db, verbose=True,
                                log_stream=log_stream)
        try:
            size = os.path.getsize(log_path) or 1
            label = os.path.basename(log_path)
            with open(log_path, "rb") as fh:
                session.feed_entries(iter_log_entries(
                    _decoded_lines_with_progress(ctx, fh, size, label)))
        finally:
            recorder.close()

    summary = {}
    summary_path = os.path.join(bundle_dir, "summary.json")
    if os.path.exists(summary_path):
        summary = _read_json(summary_path)
    counts = summary.get("counts") or summary
    decisions = counts.get("decisions", 0)
    info = {
        "source": log_path,
        "path": os.path.join(bundle_dir, "decisions.jsonl"),
        "bundle": bundle_dir,
        "fingerprint": fingerprint,
        "decisions": decisions,
        "summary": counts,
    }
    _atomic_json(marker, info)
    ctx.say("ingested %s: %s decisions"
            % (os.path.basename(log_path), decisions))
    if not decisions:
        ctx.say("warning: %s produced no decisions -- is detailed logging "
                "enabled in Arena, and is this a gameplay session?"
                % os.path.basename(log_path))
    return info


def _decoded_lines_with_progress(ctx, handle, size, label):
    """Decode a raw log stream, reporting percent-through on the live line.

    The byte count is tracked directly (text-mode iteration forbids
    ``tell``), so the meter is exact and free.
    """
    seen = 0
    last = 0.0
    for raw in handle:
        seen += len(raw)
        now = time.monotonic()
        if now - last >= 0.5:
            last = now
            ctx.console.progress("%s %d%%" % (label,
                                              min(100, 100 * seen // size)))
        yield raw.decode("utf-8", "replace")


def _open_card_db(ctx):
    from magic_cabt.arena_mirror.cards import CardDatabase
    try:
        return CardDatabase(db_path=ctx.args.card_db)
    except IOError as error:
        ctx.say("warning: card database unavailable (%s); ingest continues "
                "with log-provided names only" % error)
        return None


def stage_collect(ctx):
    """Normalize every source into one canonical JSONL with collision checks."""
    sources = [info["path"] for info in
               (ctx.stage_meta.get("ingest") or {}).get("bundles", [])]
    for entry in ctx.args.bundle or []:
        path = os.path.abspath(os.path.expanduser(entry))
        if os.path.isdir(path):
            candidate = os.path.join(path, "decisions.jsonl")
            if not os.path.exists(candidate):
                raise StageError("bundle %s has no decisions.jsonl" % path)
            sources.append(candidate)
        elif os.path.isfile(path):
            sources.append(path)
        else:
            raise StageError("bundle path does not exist: %s" % path)
    if not sources:
        raise StageError(
            "no inputs: pass --log Player.log, --bundle <dir-or-jsonl>, "
            "or --toy N")

    os.makedirs(ctx.dataset_dir, exist_ok=True)
    seen = {}
    games = set()
    written = 0
    duplicates = 0
    with open(ctx.combined, "w", encoding="utf-8") as out_handle:
        for source in sources:
            try:
                records = iter_decision_records(source)
                for record in records:
                    key = (record.get("gameId"), record.get("sequenceNumber"))
                    if key in seen:
                        duplicates += 1
                        if seen[key] != source and not ctx.args.allow_duplicate_games:
                            raise StageError(
                                "gameId %r sequence %r appears in both %s and "
                                "%s; the same session was passed twice (or two "
                                "logs overlap). Drop one input, or pass "
                                "--allow-duplicate-games to keep first "
                                "occurrences." % (key[0], key[1],
                                                  seen[key], source))
                        continue
                    seen[key] = source
                    if record.get("gameId") is not None:
                        games.add(record["gameId"])
                    out_handle.write(json.dumps(record, sort_keys=True) + "\n")
                    written += 1
            except ValueError as error:
                raise StageError("could not read %s: %s" % (source, error))

    if not written:
        raise StageError(
            "collected 0 decisions from %d source(s); nothing to train on. "
            "Check the ingest warnings in %s"
            % (len(sources), ctx.stage_log_path("ingest")))
    if len(games) < ctx.args.min_games:
        raise StageError(
            "corpus has %d whole games; --min-games is %d. Add more logs or "
            "lower the gate deliberately." % (len(games), ctx.args.min_games))

    meta = {
        "sources": sources,
        "decisions": written,
        "games": len(games),
        "duplicatesDropped": duplicates,
        "sha256": _sha256_file(ctx.combined),
    }
    ctx.stage_meta["collect"] = meta
    _atomic_json(os.path.join(ctx.dataset_dir, "collect_report.json"), meta)
    return meta


def stage_validate(ctx):
    from magic_cabt.training import validate_dataset
    code = _run_wrapped_main(ctx, "validate", validate_dataset.main,
                             [ctx.combined], allowed_codes=(0, 1, 2))
    if code == 2:
        raise StageError("dataset unreadable; see %s"
                         % ctx.stage_log_path("validate"))
    if code == 1:
        raise StageError(
            "dataset failed validation -- training on it would violate the "
            "fail-closed contract. Error summary: %s"
            % ctx.stage_log_path("validate"))
    return {"valid": True}


def stage_audit(ctx):
    from magic_cabt.research import trust_audit
    audit_path = os.path.join(ctx.dataset_dir, "trust_audit.json")
    argv = ["--input", ctx.combined, "--out", audit_path]
    code = _run_wrapped_main(ctx, "audit", trust_audit.main, argv,
                             allowed_codes=(0, 1))
    trusted = code == 0
    if not trusted and not ctx.args.allow_untrusted:
        raise StageError(
            "trust audit verdict: NOT trusted (report: %s). Review it, then "
            "either fix the data or re-run with --allow-untrusted to proceed "
            "with the verdict recorded." % audit_path)
    if not trusted:
        ctx.say("warning: proceeding on an untrusted corpus "
                "(--allow-untrusted); see %s" % audit_path)
    return {"trusted": trusted, "report": audit_path}


def stage_manifest(ctx):
    from magic_cabt.training import build_manifest
    manifest_path = os.path.join(ctx.dataset_dir, "manifest.json")
    _run_wrapped_main(ctx, "manifest", build_manifest.main, [
        "--input", ctx.combined,
        "--out", manifest_path,
        "--name", ctx.args.name or os.path.basename(ctx.out),
    ])
    return {"manifest": manifest_path}


def stage_split(ctx):
    game_ids = []
    for record in iter_decision_records(ctx.combined):
        if record.get("gameId") is not None:
            game_ids.append(record["gameId"])
    assignment = split_by_game(game_ids, ctx.args.seed,
                               ctx.args.val_fraction, ctx.args.test_fraction)
    paths = {name: os.path.join(ctx.dataset_dir, "%s_decisions.jsonl" % name)
             for name in ("train", "val", "test")}
    handles = {name: open(path, "w", encoding="utf-8")
               for name, path in paths.items()}
    counts = {name: {"games": set(), "decisions": 0} for name in paths}
    try:
        for record in iter_decision_records(ctx.combined):
            split = assignment.get(record.get("gameId"))
            if split is None:
                raise StageError(
                    "decision without a gameId cannot be split whole-game; "
                    "sequence %r" % record.get("sequenceNumber"))
            handles[split].write(json.dumps(record, sort_keys=True) + "\n")
            counts[split]["decisions"] += 1
            counts[split]["games"].add(record["gameId"])
    finally:
        for handle in handles.values():
            handle.close()

    report = {name: {"games": len(counts[name]["games"]),
                     "decisions": counts[name]["decisions"],
                     "path": paths[name]}
              for name in paths}
    report["seed"] = ctx.args.seed
    report["valFraction"] = ctx.args.val_fraction
    report["testFraction"] = ctx.args.test_fraction
    _atomic_json(os.path.join(ctx.dataset_dir, "splits.json"), report)
    ctx.stage_meta["split"] = report

    test_bundle = os.path.join(ctx.dataset_dir, "test_bundle")
    os.makedirs(test_bundle, exist_ok=True)
    shutil.copyfile(paths["test"],
                    os.path.join(test_bundle, "decisions.jsonl"))
    return report


def stage_macro(ctx):
    from magic_cabt.training import build_macro_actions
    macro_path = os.path.join(ctx.dataset_dir, "macro_actions.jsonl")
    transitions_path = os.path.join(ctx.dataset_dir, "macro_transitions.jsonl")
    _run_wrapped_main(ctx, "macro", build_macro_actions.main, [
        "--input", ctx.combined,
        "--out", macro_path,
        "--transitions-out", transitions_path,
    ])
    return {"macroActions": macro_path, "transitions": transitions_path}


def stage_compile(ctx):
    from magic_cabt.training import compile_il
    os.makedirs(ctx.il_dir, exist_ok=True)
    outputs = {}
    for split in ("train", "val", "test"):
        source = os.path.join(ctx.dataset_dir, "%s_decisions.jsonl" % split)
        out_path = os.path.join(ctx.il_dir, "%s.jsonl" % split)
        _run_wrapped_main(ctx, "compile", compile_il.main, [
            "--input", source, "--out", out_path,
        ])
        with open(out_path, "r", encoding="utf-8") as handle:
            rows = sum(1 for _ in handle)
        outputs[split] = {"path": out_path, "examples": rows}
        if rows == 0:
            if split == "train":
                raise StageError(
                    "0 single-choice IL examples compiled from the train "
                    "split; every decision was multi-select or invalid. "
                    "See %s" % ctx.stage_log_path("compile"))
            ctx.say("warning: the %s split compiled to 0 single-choice IL "
                    "examples; its evaluation metrics will be empty" % split)
    ctx.stage_meta["compile"] = outputs
    return outputs


def stage_baselines(ctx):
    from magic_cabt.dataset import read_dataset
    from magic_cabt.training.eval_bc import evaluate
    os.makedirs(ctx.eval_dir, exist_ok=True)
    results = {}
    test_il = os.path.join(ctx.il_dir, "test.jsonl")
    for policy in ("first", "random"):
        metrics = evaluate(read_dataset(test_il), policy=policy,
                           seed=ctx.args.seed)
        metrics["modelType"] = "baseline:%s" % policy
        metrics["split"] = "test"
        out_path = os.path.join(ctx.eval_dir, "baseline_%s_test.json" % policy)
        _atomic_json(out_path, metrics)
        results[policy] = {"path": out_path,
                           "top1Accuracy": metrics.get("top1Accuracy")}
    ctx.stage_meta["baselines"] = results
    return results


def stage_train_bc(ctx):
    from magic_cabt.dataset import read_dataset
    from magic_cabt.models import BagOfWordsBCPolicy
    from magic_cabt.training import train_bc

    out_dir = os.path.join(ctx.models_dir, "bc")
    _run_wrapped_main(ctx, "train-bc", train_bc.main, [
        "--input", os.path.join(ctx.il_dir, "train.jsonl"),
        "--out", out_dir,
    ])
    checkpoint = os.path.join(out_dir, "checkpoint.json")
    policy = BagOfWordsBCPolicy.load(checkpoint)
    results = {"checkpoint": checkpoint}
    for split in ("val", "test"):
        examples = list(read_dataset(os.path.join(ctx.il_dir,
                                                  "%s.jsonl" % split)))
        metrics = train_bc.evaluate_policy(policy, examples)
        metrics["modelType"] = "bag_of_words_bc"
        metrics["split"] = split
        out_path = os.path.join(ctx.eval_dir, "bc_%s.json" % split)
        _atomic_json(out_path, metrics)
        results[split] = {"path": out_path,
                          "top1Accuracy": metrics.get("top1Accuracy")}
    ctx.stage_meta["train-bc"] = results
    return results


def stage_train_torch(ctx):
    if ctx.args.skip_torch:
        return _skip("torch stages disabled with --skip-torch")
    if not ctx.torch_available:
        return _skip("torch is not installed; "
                     "pip install -e '.[torch]' to enable the ranker and "
                     "structured BC")
    from magic_cabt.training import train_ranker, train_structured_bc
    results = {}
    train_split = os.path.join(ctx.dataset_dir, "train_decisions.jsonl")

    ranker_dir = os.path.join(ctx.models_dir, "ranker")
    _run_wrapped_main(ctx, "train-torch", train_ranker.main, [
        "--input", train_split,
        "--out", ranker_dir,
        "--epochs", str(ctx.args.epochs),
        "--batch-size", str(ctx.args.batch_size),
        "--lr", str(ctx.args.lr),
        "--seed", str(ctx.args.seed),
    ])
    results["ranker"] = {"checkpoint": os.path.join(ranker_dir,
                                                    "checkpoint.pt")}

    structured_dir = os.path.join(ctx.models_dir, "structured-bc")
    _run_wrapped_main(ctx, "train-torch", train_structured_bc.main, [
        "--input", train_split,
        "--out", structured_dir,
        "--preset", "local",
        "--epochs", str(ctx.args.epochs),
        "--batch-size", str(ctx.args.batch_size),
        "--lr", str(ctx.args.lr),
    ])
    results["structuredBc"] = {"dir": structured_dir}
    ctx.stage_meta["train-torch"] = results
    return results


def stage_compare(ctx):
    from magic_cabt.analysis import suite
    bundle = os.path.join(ctx.dataset_dir, "test_bundle")
    out_html = os.path.join(ctx.eval_dir, "comparison.html")
    argv = ["--bundle", bundle, "--out", out_html,
            "--model", "first=baseline:first-legal",
            "--model", "random=baseline:random"]
    torch_meta = ctx.stage_meta.get("train-torch") or {}
    ranker = (torch_meta.get("ranker") or {}).get("checkpoint")
    if ranker and os.path.exists(ranker):
        argv += ["--model", "ranker=%s" % ranker]
    structured = (torch_meta.get("structuredBc") or {}).get("dir")
    if structured:
        best = os.path.join(structured, "best.pt")
        if os.path.exists(best):
            argv += ["--model", "structured-bc=%s" % best]
    _run_wrapped_main(ctx, "compare", suite.main, argv)
    return {"html": out_html,
            "json": os.path.splitext(out_html)[0] + ".json"}


def _skip(reason):
    return {"skipped": True, "reason": reason}


# ---------------------------------------------------------------------------
# Report


def stage_report(ctx):
    report = build_report(ctx)
    json_path = os.path.join(ctx.out, "report.json")
    md_path = os.path.join(ctx.out, "report.md")
    _atomic_json(json_path, report)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_report_markdown(report))
    return {"json": json_path, "markdown": md_path}


def build_report(ctx):
    collect = ctx.stage_meta.get("collect") or _maybe_json(
        os.path.join(ctx.dataset_dir, "collect_report.json"))
    splits = ctx.stage_meta.get("split") or _maybe_json(
        os.path.join(ctx.dataset_dir, "splits.json"))
    models = {}
    for stage, name, path in (
            ("baselines", "baseline:first",
             os.path.join(ctx.eval_dir, "baseline_first_test.json")),
            ("baselines", "baseline:random",
             os.path.join(ctx.eval_dir, "baseline_random_test.json")),
            ("train-bc", "bag-of-words-bc",
             os.path.join(ctx.eval_dir, "bc_test.json"))):
        # A metrics file is only this run's result when its producing stage
        # ran (or was cache-valid) this run; a skipped stage must not
        # republish a previous run's artifact as current.
        metrics = _maybe_json(path) if _stage_ok(ctx, stage) else None
        if metrics:
            models[name] = {
                "top1Accuracy": metrics.get("top1Accuracy"),
                "top3Accuracy": metrics.get("top3Accuracy"),
                "meanReciprocalRank": metrics.get("meanReciprocalRank"),
                "examples": metrics.get("examples"),
            }
    bc_val = _maybe_json(os.path.join(ctx.eval_dir, "bc_val.json")) \
        if _stage_ok(ctx, "train-bc") else None
    warnings = []
    if collect and collect.get("duplicatesDropped"):
        warnings.append("%d duplicate decisions dropped during collect"
                        % collect["duplicatesDropped"])
    audit_state = ctx.state["stages"].get("audit") or {}
    if (audit_state.get("detail") or {}).get("trusted") is False:
        warnings.append("trust audit verdict was NOT trusted "
                        "(run continued via --allow-untrusted)")
    torch_state = ctx.state["stages"].get("train-torch") or {}
    if (torch_state.get("detail") or {}).get("skipped"):
        warnings.append("torch stages skipped: %s"
                        % torch_state["detail"].get("reason"))
    comparison = _maybe_json(os.path.join(ctx.eval_dir, "comparison.json")) \
        if _stage_ok(ctx, "compare") else None
    comparison_models = ((comparison or {}).get("metrics") or {}).get("models")
    return {
        "schemaVersion": 1,
        "generatedAt": _utc_now(),
        "runDir": ctx.out,
        "corpus": collect,
        "splits": splits,
        "testMetrics": models,
        "comparison": comparison_models,
        "bcValTop1": (bc_val or {}).get("top1Accuracy"),
        "stages": ctx.state["stages"],
        "warnings": warnings,
    }


def _stage_ok(ctx, name):
    return (ctx.state["stages"].get(name) or {}).get("status") == "ok"


def _maybe_json(path):
    if path and os.path.exists(path):
        try:
            return _read_json(path)
        except ValueError:
            return None
    return None


def render_report_markdown(report):
    lines = ["# Training run report", ""]
    corpus = report.get("corpus") or {}
    lines.append("Generated %s in `%s`."
                 % (report.get("generatedAt"), report.get("runDir")))
    lines.append("")
    lines.append("## Corpus")
    lines.append("")
    lines.append("| Games | Decisions | Sources | Duplicates dropped |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| %s | %s | %s | %s |" % (
        corpus.get("games", "?"), corpus.get("decisions", "?"),
        len(corpus.get("sources") or []),
        corpus.get("duplicatesDropped", 0)))
    splits = report.get("splits") or {}
    if splits:
        lines += ["", "## Whole-game splits", "",
                  "| Split | Games | Decisions |", "| --- | --- | --- |"]
        for name in ("train", "val", "test"):
            entry = splits.get(name) or {}
            lines.append("| %s | %s | %s |" % (
                name, entry.get("games", "?"), entry.get("decisions", "?")))
    metrics = report.get("testMetrics") or {}
    if metrics:
        lines += ["", "## Held-out test metrics", "",
                  "| Model | Top-1 | Top-3 | MRR |", "| --- | --- | --- | --- |"]
        for name, row in sorted(metrics.items()):
            lines.append("| %s | %s | %s | %s |" % (
                name, _fmt(row.get("top1Accuracy")),
                _fmt(row.get("top3Accuracy")),
                _fmt(row.get("meanReciprocalRank"))))
    comparison = report.get("comparison") or []
    if comparison:
        lines += ["", "## Head-to-head on the test bundle "
                      "(`eval/comparison.html`)", "",
                  "| Model | Coverage | Top-1 | MRR |",
                  "| --- | --- | --- | --- |"]
        for row in comparison:
            lines.append("| %s | %s | %s | %s |" % (
                row.get("name"), _fmt(row.get("coverage")),
                _fmt(row.get("playedTop1")), _fmt(row.get("playedMRR"))))
    lines += ["", "## Stages", "",
              "| Stage | Status | Seconds |", "| --- | --- | --- |"]
    for name in STAGE_NAMES:
        entry = (report.get("stages") or {}).get(name) or {}
        lines.append("| %s | %s | %s |" % (
            name, entry.get("status", "-"), entry.get("seconds", "-")))
    warnings = report.get("warnings") or []
    if warnings:
        lines += ["", "## Warnings", ""]
        lines += ["- %s" % warning for warning in warnings]
    lines.append("")
    return "\n".join(lines)


def _fmt(value):
    if value is None:
        return "-"
    return "%.3f" % value


# ---------------------------------------------------------------------------
# Orchestration


def _stage_fingerprint(ctx, name):
    """What a stage's result depends on; changes invalidate the cache."""
    args = ctx.args
    base = {
        "seed": args.seed,
        "valFraction": args.val_fraction,
        "testFraction": args.test_fraction,
    }
    if name == "ingest":
        base["logs"] = [
            _fingerprint_log(path) for path in _discover_logs(args.log)]
        base["toy"] = args.toy
    elif name == "collect":
        base["bundles"] = sorted(args.bundle or [])
        base["allowDuplicates"] = bool(args.allow_duplicate_games)
        base["ingest"] = (ctx.stage_state("ingest") or {}).get("fingerprint")
    else:
        base["dataset"] = (ctx.stage_meta.get("collect") or {}).get("sha256") \
            or ((ctx.stage_state("collect") or {}).get("detail") or {}).get("sha256")
        if name == "train-torch":
            base["epochs"] = args.epochs
            base["batchSize"] = args.batch_size
            base["lr"] = args.lr
            base["torchAvailable"] = ctx.torch_available
            base["skipTorch"] = bool(args.skip_torch)
        if name == "compare":
            # The comparison scores whatever checkpoints train-torch produced,
            # so a retrained model must invalidate the cached comparison even
            # though the dataset hash is unchanged.
            base["trainTorch"] = (ctx.stage_state("train-torch") or {}).get(
                "fingerprint")
    return _fingerprint_value(base)


_STAGE_FUNCTIONS = {
    "ingest": stage_ingest,
    "collect": stage_collect,
    "validate": stage_validate,
    "audit": stage_audit,
    "manifest": stage_manifest,
    "split": stage_split,
    "macro": stage_macro,
    "compile": stage_compile,
    "baselines": stage_baselines,
    "train-bc": stage_train_bc,
    "train-torch": stage_train_torch,
    "compare": stage_compare,
    "report": stage_report,
}

# Stages whose cached results later stages read back from stage_meta; on a
# cache hit their recorded detail is restored so downstream stages still see
# their outputs.
_META_STAGES = ("ingest", "collect", "split", "compile", "baselines",
                "train-bc", "train-torch")

# The data-integrity chain. Skipping any of these on a reused run directory
# would let later stages consume a previous corpus's files as if they were
# current, so --skip refuses them; caching makes rerunning them free anyway.
_UNSKIPPABLE_STAGES = frozenset(
    ("collect", "validate", "audit", "split", "compile", "report"))


def run(args):
    ctx = RunContext(args)
    skips = set(args.skip or [])
    unknown = skips.difference(STAGE_NAMES)
    if unknown:
        raise StageError("unknown --skip stage(s): %s (choose from %s)"
                         % (", ".join(sorted(unknown)),
                            ", ".join(STAGE_NAMES)))
    integrity = skips.intersection(_UNSKIPPABLE_STAGES)
    if integrity:
        raise StageError(
            "%s cannot be skipped: the data-integrity chain must run on the "
            "current corpus, or later stages would silently reuse a previous "
            "run's files" % ", ".join(sorted(integrity)))

    if args.dry_run:
        plan = _dry_run_plan(ctx, skips)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    os.makedirs(ctx.out, exist_ok=True)
    os.makedirs(ctx.logs_dir, exist_ok=True)
    if args.force:
        ctx.state = {"schemaVersion": 1, "stages": {}}
        for path in (ctx.ingest_dir, ctx.dataset_dir, ctx.il_dir,
                     ctx.models_dir, ctx.eval_dir):
            if os.path.isdir(path):
                shutil.rmtree(path)
    else:
        ctx.load_state()

    _write_config(ctx)

    run_started = time.monotonic()
    total = len(STAGE_NAMES)
    for position, name in enumerate(STAGE_NAMES, start=1):
        if name in skips:
            ctx.record_stage(name, "skipped",
                             detail=_skip("skipped via --skip"))
            ctx.console.stage_end(position, total, name, "skipped",
                                  note="--skip")
            continue
        fingerprint = _stage_fingerprint(ctx, name)
        previous = ctx.stage_state(name)
        if (not args.force and previous.get("status") in ("ok", "skipped")
                and previous.get("fingerprint") == fingerprint
                and name != "report"):
            if name in _META_STAGES and previous.get("detail"):
                ctx.stage_meta[name] = previous["detail"]
            ctx.console.stage_end(position, total, name, "cached")
            continue
        ctx.console.stage_begin(position, total, name)
        started = time.monotonic()
        try:
            detail = _STAGE_FUNCTIONS[name](ctx)
        except StageError as error:
            seconds = time.monotonic() - started
            ctx.record_stage(name, "failed", fingerprint=fingerprint,
                             seconds=seconds, detail={"error": str(error)})
            ctx.console.stage_end(position, total, name, "failed", seconds)
            ctx.console.error(str(error))
            return 1
        except KeyboardInterrupt:
            seconds = time.monotonic() - started
            ctx.record_stage(name, "interrupted", fingerprint=fingerprint,
                             seconds=seconds,
                             detail={"error": "interrupted by user"})
            ctx.console.stage_end(position, total, name, "interrupted",
                                  seconds)
            ctx.console.error("interrupted during %s -- re-run the same "
                              "command to resume from there" % name)
            return 130
        seconds = time.monotonic() - started
        status = "skipped" if (isinstance(detail, dict)
                               and detail.get("skipped")) else "ok"
        note = detail.get("reason") if status == "skipped" \
            else _stage_note(name, detail)
        ctx.record_stage(name, status, fingerprint=fingerprint,
                         seconds=seconds, detail=_json_safe(detail))
        ctx.console.stage_end(position, total, name, status, seconds, note)
    _print_summary(ctx, time.monotonic() - run_started)
    return 0


def _stage_note(name, detail):
    """The one fact worth showing beside a finished stage's status."""
    if not isinstance(detail, dict):
        return None
    if name == "collect":
        return "%s games, %s decisions" % (detail.get("games"),
                                           detail.get("decisions"))
    if name == "split":
        return "train/val/test %s/%s/%s games" % (
            (detail.get("train") or {}).get("games"),
            (detail.get("val") or {}).get("games"),
            (detail.get("test") or {}).get("games"))
    if name == "audit":
        return "trusted" if detail.get("trusted") else "NOT trusted"
    if name == "compile":
        train = (detail.get("train") or {}).get("examples")
        return "%s train examples" % train if train is not None else None
    if name == "ingest":
        bundles = detail.get("bundles")
        if bundles:
            return "%d source%s" % (len(bundles),
                                    "" if len(bundles) == 1 else "s")
    return None


def _dry_run_plan(ctx, skips):
    logs = _discover_logs(ctx.args.log)
    plan = {
        "out": ctx.out,
        "logs": logs,
        "bundles": ctx.args.bundle or [],
        "toyGames": ctx.args.toy,
        "torchAvailable": ctx.torch_available,
        "stages": [
            {"stage": name,
             "planned": name not in skips}
            for name in STAGE_NAMES],
    }
    if not logs and not (ctx.args.bundle or []) and not ctx.args.toy:
        plan["warning"] = ("no inputs; the run would stop at collect. Pass "
                          "--log, --bundle, or --toy N.")
    return plan


def _write_config(ctx):
    config = {
        "argv": {key: value for key, value in sorted(vars(ctx.args).items())},
        "createdAt": _utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packageVersion": _package_version(),
    }
    _atomic_json(os.path.join(ctx.out, _CONFIG_FILE), config)


def _package_version():
    try:
        from importlib import metadata
        return metadata.version("magic-cabt")
    except Exception:
        return None


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return {"repr": repr(value)}


def _print_summary(ctx, elapsed):
    report = _maybe_json(os.path.join(ctx.out, "report.json")) or {}
    metrics = report.get("testMetrics") or {}
    if metrics:
        ctx.say("test top-1: %s" % " · ".join(
            "%s %s" % (name, _fmt(row.get("top1Accuracy")))
            for name, row in sorted(metrics.items())))
    for warning in report.get("warnings") or []:
        ctx.say("warning: %s" % warning)
    artifacts = ["report: %s" % os.path.join(ctx.out, "report.md")]
    if _stage_ok(ctx, "compare"):
        artifacts.append("comparison: %s"
                         % os.path.join(ctx.eval_dir, "comparison.html"))
    ctx.say("done in %.1fs -- %s" % (elapsed, " · ".join(artifacts)))


# ---------------------------------------------------------------------------
# CLI


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-training-run",
        description="Run the full supported training pipeline from saved "
                    "MTGA Player.log files (or existing decision bundles) "
                    "to a trained-and-evaluated report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  # smoke-test the whole pipeline on a synthetic corpus\n"
               "  magic-cabt-training-run --toy 30 --out runs/toy\n"
               "\n"
               "  # real run over saved Arena logs, resumable in place\n"
               "  magic-cabt-training-run --log captures/Player.log \\\n"
               "      --log captures/old-logs --out runs/first-run\n")

    inputs = parser.add_argument_group(
        "inputs", "at least one of --log, --bundle, or --toy")
    inputs.add_argument("--log", action="append", default=[],
                        help="saved Player.log file, or a directory of .log "
                             "files (repeatable)")
    inputs.add_argument("--bundle", action="append", default=[],
                        help="existing bundle dir or DecisionRecord JSONL "
                             "(repeatable)")
    inputs.add_argument("--toy", type=int, default=0, metavar="GAMES",
                        help="generate a synthetic corpus of GAMES games "
                             "instead of reading logs (pipeline smoke test)")

    corpus = parser.add_argument_group("corpus and splits")
    corpus.add_argument("--name", default=None,
                        help="corpus name recorded in the manifest")
    corpus.add_argument("--card-db", default=None,
                        help="Arena card database path for name resolution")
    corpus.add_argument("--seed", type=int, default=0,
                        help="seed for splits, baselines, and trainers")
    corpus.add_argument("--val-fraction", type=float, default=0.1)
    corpus.add_argument("--test-fraction", type=float, default=0.1)
    corpus.add_argument("--min-games", type=int, default=2,
                        help="fail if the corpus has fewer whole games")
    corpus.add_argument("--allow-duplicate-games", action="store_true",
                        help="keep first occurrence when the same "
                             "game/sequence appears in multiple inputs")
    corpus.add_argument("--allow-untrusted", action="store_true",
                        help="continue past a NOT-trusted audit verdict "
                             "(recorded in the report)")

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=10,
                          help="epochs for torch trainers")
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--skip-torch", action="store_true",
                          help="skip torch model stages even if torch is "
                               "installed")

    control = parser.add_argument_group("run control")
    control.add_argument("--out", required=True,
                         help="run directory for datasets, models, logs, and "
                              "the final report")
    control.add_argument("--skip", action="append", default=[],
                         metavar="STAGE",
                         help="skip an optional stage (repeatable): ingest, "
                              "macro, baselines, train-bc, train-torch, or "
                              "compare; the data-integrity chain cannot be "
                              "skipped")
    control.add_argument("--force", action="store_true",
                         help="ignore cached stages and rebuild everything")
    control.add_argument("--dry-run", action="store_true",
                         help="print the resolved plan and exit")
    control.add_argument("--quiet", action="store_true",
                         help="suppress progress output (errors still print)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.val_fraction < 0 or args.test_fraction < 0 \
            or args.val_fraction + args.test_fraction >= 1:
        sys.stderr.write("error: --val-fraction and --test-fraction must be "
                         ">= 0 and sum to < 1\n")
        return 2
    if args.toy < 0:
        sys.stderr.write("error: --toy takes a positive game count\n")
        return 2
    if args.epochs < 1 or args.batch_size < 1:
        sys.stderr.write("error: --epochs and --batch-size must be >= 1\n")
        return 2
    if args.lr <= 0:
        sys.stderr.write("error: --lr must be > 0\n")
        return 2
    if args.min_games < 1:
        sys.stderr.write("error: --min-games must be >= 1\n")
        return 2
    try:
        return run(args)
    except StageError as error:
        sys.stderr.write("error: %s\n" % error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
