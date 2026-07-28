"""CLI for the MTGO video ingestion pipeline.

    python -m magic_cabt.mtgo_video ingest VIDEO --out BUNDLE_DIR \
        [--start S] [--end S] [--fps 1.0] [--hero NAME] [--deck-size 60]

Produces a match directory containing one replayable bundle per game:

    BUNDLE_DIR/
      mtgo_log.json        reconstructed log entries (OCR windows merged)
      mtgo_events.jsonl    parsed + name-corrected events, whole match
      card_cache.json      Scryfall metadata cache
      match.json           match-level summary
      game1/
        mirror_states.jsonl  board snapshots (Arena mirror schema)
        mtgo_events.jsonl    this game's events
        summary.json
      game2/ ...
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
from typing import Optional

from .catalog import CardCatalog, DEFAULT_CATALOG, build_catalog
from .extract import extract_log_frames, grab_frame, probe_duration
from .layout import (
    Layout,
    detect_layout,
    detect_layout_from_frames,
    select_duel_frames,
    validate_layout,
)
from .games import split_games, game_is_complete, discover_players
from .ocr import ocr_image, lines_to_entries
from .parse import parse_log
from .reconstruct import LogReconstructor
from .regions import Region
from .state import GameSimulator

_CARD_FIELDS = ("card", "into", "counter", "source", "attacker", "blocker")
_CARD_LIST_FIELDS = ("cards", "targets")


def _canonical_player(name, players, cutoff=0.7):
    """Map an OCR'd player name onto the match's known players."""
    import difflib

    best, score = name, cutoff
    for known in players:
        ratio = difflib.SequenceMatcher(None, name.lower(), known.lower()).ratio()
        if ratio >= score:
            best, score = known, ratio
    return best


def correct_events(events, catalog, players=()):
    """Resolve OCR'd card names to canonical names against the local catalog.

    A spell's target may be a player rather than a card ("casts Thought
    Scour targeting BuzzCaldera"), so known player names are left alone
    instead of being matched against the catalog.
    """
    import difflib

    def is_player(text):
        return any(
            difflib.SequenceMatcher(None, text.lower(), name.lower()).ratio() >= 0.8
            for name in players
        )

    def fields(event):
        for field in _CARD_FIELDS:
            if field in event:
                yield field, None
        for field in _CARD_LIST_FIELDS:
            for i in range(len(event.get(field, ()))):
                yield field, i

    def get(event, field, i):
        return event[field] if i is None else event[field][i]

    def put(event, field, i, value):
        if i is None:
            event[field] = value
        else:
            event[field][i] = value

    # Lock the run vocabulary to the cards this log spells correctly, before
    # resolving anything fuzzily: a garbled sighting then snaps onto the card
    # the log already named exactly, whichever came first.
    catalog.seed_from_log(
        get(event, field, i)
        for event in events
        for field, i in fields(event)
        if not is_player(get(event, field, i))
    )
    for event in events:
        for field, i in fields(event):
            name = get(event, field, i)
            if not is_player(name):
                put(event, field, i, catalog.canonical_name(name))
        # Player names are OCR'd too ("golubtsoy", "BurzCaldera"). Resolve
        # them here rather than only inside the simulator, so the event log
        # itself is canonical and two captures of one match can be compared.
        for field in ("player", "target", "owner", "whose"):
            if field in event:
                event[field] = _canonical_player(event[field], players)
    return events


def _ocr_one(path):
    # Non-strict: an ingest spans thousands of frames and must survive a few
    # unreadable ones. run_ingest reports the empty-frame count.
    return lines_to_entries(ocr_image(path, strict=False))


def write_game_bundle(out_dir, events, args, catalog, index):
    os.makedirs(out_dir, exist_ok=True)
    sim = GameSimulator(
        hero=args.hero,
        match_id=args.match_id,
        game_number=index,
        deck_size=args.deck_size,
        card_info=catalog.lookup,
    )
    sim.seed_players(discover_players(events))
    states = []
    for event in events:
        for snap in sim.apply(event):
            # A snapshot the simulator already attributed (combat damage)
            # keeps its own event and timestamp.
            snap.setdefault("sourceEvent",
                            {k: v for k, v in event.items() if k != "clock"})
            if "videoTime" in event:
                snap.setdefault("videoTime", event["videoTime"])
            states.append(snap)

    with open(os.path.join(out_dir, "mirror_states.jsonl"), "w") as f:
        for state in states:
            f.write(json.dumps(state) + "\n")
    with open(os.path.join(out_dir, "mtgo_events.jsonl"), "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    players = [p.name for p in sorted(sim.players.values(), key=lambda q: q.seat)]
    summary = {
        "source": "mtgo_video",
        "matchId": args.match_id,
        "gameNumber": index,
        "players": players,
        "hero": args.hero,
        "events": len(events),
        "unparsedEvents": sum(1 for e in events if e["type"] == "UNPARSED"),
        "states": len(states),
        "complete": game_is_complete(events),
        "winner": sim.winner,
        "turns": sim.turn_number,
        "simulatorWarnings": sim.warnings,
        "title": "MTGO %s game %d: %s" % (args.match_id, index, " vs ".join(players)),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def sample_layout_frames(video, out_dir, start, end, count=5, at=None):
    """Frames spread through the clip, for layout consensus."""
    out_dir = os.path.realpath(out_dir)
    if at is not None:
        return [grab_frame(video, at, os.path.join(out_dir, "layout_frame.png"))]
    begin = start or 0.0
    # Sample across everything that will be ingested, and only that. Taking
    # the layout from the first few minutes of a three-hour VOD and then
    # cropping the whole thing to it assumes the client never moves -- and
    # the first few minutes of a league VOD are usually the deck editor.
    finish = end
    if finish is None:
        duration = probe_duration(video)
        finish = duration if duration else begin + 300.0
    span = max(1.0, finish - begin)
    paths = []
    for index in range(count):
        when = begin + span * (index + 1) / (count + 1)
        paths.append(grab_frame(
            video, when, os.path.join(out_dir, "layout_frame_%d.png" % index)))
    return paths


def resolve_layout(args, out_dir):
    """Locate the MTGO UI in this capture, however it was recorded."""
    samples = sample_layout_frames(args.video, out_dir, args.start, args.end,
                                   count=args.layout_samples,
                                   at=args.layout_frame)
    # Agree the layout over duel frames only; menus have a game log too, in
    # a different place, and would pull the consensus off the board.
    samples = select_duel_frames(samples)
    layout = detect_layout_from_frames(samples, detect=not args.no_detect)
    if args.region:
        layout.log_pane = Region.parse(args.region)
        layout.detected = False
    print("      layout: %s" % layout.describe(), file=sys.stderr)

    # Validate against every sample, not just until one passes. A VOD spends
    # time on deck-building and sideboarding screens, so one failure means
    # nothing -- but *which* samples fail is the only evidence available that
    # the client was rearranged partway through the recording, and one crop
    # cannot serve a video whose UI moved in the middle of it.
    checks = [validate_layout(layout, sample) for sample in samples]
    passed = [c for c in checks if c["ok"]]
    if not passed:
        raise SystemExit(
            "layout validation failed on all %d sampled frames (%s). The UI "
            "could not be located in this capture; pass --region WxH+X+Y, or "
            "--start/--end covering actual gameplay."
            % (len(samples), "; ".join(checks[-1]["problems"])))
    # Prefer a sample whose seats were readable too: the recorded layout is
    # what the HUD checks will use later, and one that could read them is a
    # better record of this capture than one that happened not to.
    check = dict(next((c for c in passed if c.get("seatsReadable")), passed[0]),
                 samplesValidated=len(passed), samplesChecked=len(checks))
    print("      layout validated on %d/%d sampled frames: %d timestamped "
          "lines, life %s"
          % (len(passed), len(checks), check["logTimestamps"], check["life"]),
          file=sys.stderr)
    if len(passed) < len(checks):
        # Not fatal on its own: the failing sample may be a menu screen. But
        # if the client really did move, every frame after the move is being
        # cropped to the wrong rectangle, and this is the only warning of it.
        print("      WARNING: %d of %d sampled frames did not validate "
              "against this layout. If the client was resized or rearranged "
              "partway through, ingest each part separately with "
              "--start/--end." % (len(checks) - len(passed), len(checks)),
              file=sys.stderr)
    for warning in check.get("warnings", ()):
        print("      WARNING: %s" % warning, file=sys.stderr)
    return layout, check


def run_ingest(args):
    os.makedirs(args.out, exist_ok=True)
    frames_dir = args.frames_dir or tempfile.mkdtemp(prefix="mtgo_frames_")

    print("[1/6] locating the MTGO UI and extracting frames (fps=%g)..."
          % args.fps, file=sys.stderr)
    layout, layout_check = resolve_layout(args, args.out)
    frames = extract_log_frames(
        args.video, frames_dir, start=args.start, end=args.end,
        fps=args.fps, region=layout.log_pane,
    )
    print("      %d frames" % len(frames), file=sys.stderr)

    print("[2/6] OCR (%d workers)..." % args.workers, file=sys.stderr)
    paths = [path for _, path in frames]
    # Threads, not processes: each task is a tesseract subprocess, so the
    # GIL is released for the whole of the actual work.
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        per_frame = list(pool.map(_ocr_one, paths))
    readable = sum(1 for entries in per_frame if entries)
    print("      %d/%d frames yielded log entries" % (readable, len(per_frame)),
          file=sys.stderr)

    if args.dump_ocr:
        # The raw per-frame readings, so a reconstruction difference between
        # two captures can be replayed offline instead of re-OCR'd.
        with open(os.path.join(args.out, "ocr_frames.json"), "w") as f:
            json.dump([{"time": t, "entries": e}
                       for (t, _), e in zip(frames, per_frame)], f)

    print("[3/6] merging log windows...", file=sys.stderr)
    rec = LogReconstructor()
    for (timestamp, _), entries in zip(frames, per_frame):
        rec.feed(entries, timestamp)
    with open(os.path.join(args.out, "mtgo_log.json"), "w") as f:
        json.dump(rec.entries, f, indent=1)
    with open(os.path.join(args.out, "mtgo_log_times.json"), "w") as f:
        json.dump(rec.times, f)
    print("      %d log entries" % len(rec.entries), file=sys.stderr)

    print("[4/6] parsing events...", file=sys.stderr)
    print("[5/6] resolving card names against the local catalog...",
          file=sys.stderr)
    print("[6/6] splitting games and simulating board states...", file=sys.stderr)
    match = dict(build_games(args, rec.entries, args.out, open_catalog(args),
                             rec.times), **{
        "video": os.path.abspath(args.video),
        "startSeconds": args.start,
        "endSeconds": args.end,
        "fps": args.fps,
        "layout": layout.to_dict(),
        "layoutCheck": layout_check,
        "framesRead": readable,
        "framesTotal": len(per_frame),
    })
    with open(os.path.join(args.out, "match.json"), "w") as f:
        json.dump(match, f, indent=2)

    if not args.frames_dir:
        shutil.rmtree(frames_dir, ignore_errors=True)
    print(json.dumps(match, indent=2))


def build_games(args, rec_entries, out_dir, catalog, times=None):
    """Parse, correct, split, and simulate; returns the match summary."""
    events = parse_log(rec_entries, times)
    correct_events(events, catalog, players=discover_players(events))
    with open(os.path.join(out_dir, "mtgo_events.jsonl"), "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    games = split_games(events)
    game_summaries = []
    for i, game_events in enumerate(games, start=1):
        if args.game and i != args.game:
            continue
        game_summaries.append(
            write_game_bundle(os.path.join(out_dir, "game%d" % i),
                              game_events, args, catalog, i))

    if catalog.unresolved:
        print("WARNING: %d card names did not match the catalog: %s"
              % (len(catalog.unresolved), ", ".join(sorted(catalog.unresolved))),
              file=sys.stderr)

    unparsed = sum(1 for e in events if e["type"] == "UNPARSED")
    return {
        "catalogSource": catalog.source,
        "catalogBuiltAt": catalog.built_at,
        "cardsInPlay": sorted(catalog.vocabulary),
        "cardsUnresolved": catalog.unresolved,
        "source": "mtgo_video",
        "matchId": args.match_id,
        "hero": args.hero,
        "logEntries": len(rec_entries),
        "events": len(events),
        "unparsedEvents": unparsed,
        "parseCoverage": round(1.0 - unparsed / max(1, len(events)), 4),
        "games": game_summaries,
    }


def open_catalog(args):
    """Load the local card catalog, building it once if it is missing."""
    path = args.catalog or DEFAULT_CATALOG
    if not os.path.exists(path):
        print("building card catalog (one-time Scryfall bulk download)...",
              file=sys.stderr)
        build_catalog(path)
    return CardCatalog(path)


def run_catalog(args):
    path = build_catalog(args.catalog or DEFAULT_CATALOG)
    catalog = CardCatalog(path)
    print(json.dumps({"path": path, "cards": len(catalog.cards),
                      "source": catalog.source, "builtAt": catalog.built_at},
                     indent=2))


def run_rebuild(args):
    """Re-run parsing and simulation from an already-OCR'd log."""
    with open(os.path.join(args.bundle, "mtgo_log.json")) as f:
        entries = json.load(f)
    times_path = os.path.join(args.bundle, "mtgo_log_times.json")
    times = None
    if os.path.exists(times_path):
        with open(times_path) as f:
            times = json.load(f)
    match = build_games(args, entries, args.bundle, open_catalog(args), times)
    match_path = os.path.join(args.bundle, "match.json")
    if os.path.exists(match_path):
        with open(match_path) as f:
            match = dict(json.load(f), **match)
    with open(match_path, "w") as f:
        json.dump(match, f, indent=2)
    print(json.dumps(match, indent=2))


def run_verify(args):
    from .verify import verify_bundle

    classpath = args.classpath or os.environ.get("MAGIC_CABT_CLASSPATH")
    if not classpath:
        raise SystemExit("pass --classpath or set $MAGIC_CABT_CLASSPATH")
    report = verify_bundle(args.bundle, classpath, java=args.java, cwd=args.cwd)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


def run_rules(args):
    """Check each decoded event against XMage's board at that moment."""
    from .rules import check_bundle

    classpath = args.classpath or os.environ.get("MAGIC_CABT_CLASSPATH")
    if not classpath:
        raise SystemExit("pass --classpath or set $MAGIC_CABT_CLASSPATH")
    report = check_bundle(args.bundle, classpath, java=args.java, cwd=args.cwd)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


def run_check(args):
    """Run every verification that this bundle and this machine allow.

    The five checks have different prerequisites -- two need XMage on the
    classpath, two need the capture and a readable HUD, one needs a second
    recording of the same match -- and running them one at a time means
    finding that out five times. Anything that cannot run is reported as
    skipped, with the reason, rather than quietly left out: a check that did
    not run is not a check that passed.
    """
    classpath = args.classpath or os.environ.get("MAGIC_CABT_CLASSPATH")
    results = []

    def record(name, runner, prerequisite=None):
        if prerequisite:
            results.append({"check": name, "status": "skipped",
                            "detail": prerequisite})
            return
        try:
            report = runner()
        except Exception as error:  # a check that cannot run is not a pass
            results.append({"check": name, "status": "error",
                            "detail": str(error).splitlines()[0][:200]})
            return
        results.append({"check": name,
                        "status": "ok" if report.get("ok") else "failed",
                        "detail": _summarize(name, report)})

    from .verify import verify_bundle

    record("verify", lambda: verify_bundle(args.bundle, classpath,
                                           java=args.java, cwd=args.cwd),
           None if classpath else "no --classpath or $MAGIC_CABT_CLASSPATH")

    from .rules import check_bundle

    record("rules", lambda: check_bundle(args.bundle, classpath,
                                         java=args.java, cwd=args.cwd),
           None if classpath else "no --classpath or $MAGIC_CABT_CLASSPATH")

    record("board", lambda: _board_report(args),
           None if args.video else "no --video")
    record("crosscheck", lambda: _crosscheck_report(args),
           None if args.video else "no --video")

    from .compare import compare_bundles

    def compared():
        report = compare_bundles([args.bundle] + list(args.against))
        return dict(report, ok=report["allIdentical"])

    record("compare", compared,
           None if args.against else "no --against BUNDLE (needs a second "
                                     "recording of the same match)")

    ran = [r for r in results if r["status"] != "skipped"]
    report = {
        "bundle": os.path.abspath(args.bundle),
        "checks": results,
        "checksRun": len(ran),
        # A bundle nothing could be run against has not been verified, and
        # reporting that as a pass is exactly the failure this command exists
        # to prevent.
        "ok": bool(ran) and all(r["status"] == "ok" for r in ran),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    for result in results:
        print("%-11s %-8s %s" % (result["check"], result["status"],
                                 result["detail"]), file=sys.stderr)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


def _summarize(name, report):
    if name == "verify":
        return "%s/%s states" % (report.get("statesVerified"),
                                 report.get("states"))
    if name == "rules":
        alignment = report.get("alignment") or {}
        return ("%d events checked, %d findings (%d naming a lost line)"
                % (report.get("eventsChecked", 0),
                   len(report.get("violations") or []),
                   alignment.get("attributed", 0)))
    if name == "board":
        return ("%d cards read, %d unlogged permanents"
                % (report.get("cardsIdentified", 0),
                   len(report.get("unloggedPermanents") or [])))
    if name == "crosscheck":
        return ("%s/%s life readings agree"
                % (report.get("lifeReadingsAgreed"),
                   report.get("lifeReadingsChecked")))
    if name == "compare":
        return "identical" if report.get("ok") else "decodes differ"
    return ""


def _board_report(args):
    from .art import ArtIndex
    from .board import check_states
    from .render import load_states

    layout = bundle_layout(args.bundle)
    if layout is None or layout.phase_bar is None:
        raise RuntimeError("bundle has no recorded phase bar; re-ingest it")
    vocabulary = _bundle_vocabulary(args.bundle)
    index = ArtIndex(args.art_cache)
    index.ensure(vocabulary)
    return check_states(args.video, load_states(args.bundle), layout, index,
                        sample=args.sample)


def _crosscheck_report(args):
    from .hud import crosscheck_life
    from .render import load_states

    return crosscheck_life(args.video, load_states(args.bundle),
                           sample=args.sample,
                           layout=bundle_layout(args.bundle))


def run_board(args):
    """Check the board on screen against the board the log produced."""
    from .art import ArtIndex
    from .board import check_states
    from .render import load_states

    states = load_states(args.bundle)
    layout = bundle_layout(args.bundle)
    if layout is None or layout.phase_bar is None:
        raise SystemExit(
            "this bundle has no recorded phase bar, so the board cannot be "
            "located; re-ingest it with a build that detects one")
    vocabulary = _bundle_vocabulary(args.bundle)
    index = ArtIndex(args.art_cache)
    missing = [name for name in vocabulary
               if name not in index.signatures and name not in index.missing]
    if missing:
        print("fetching art for %d cards..." % len(missing), file=sys.stderr)
        index.ensure(vocabulary)
    report = check_states(args.video, states, layout, index,
                          sample=args.sample, settle=args.settle)
    report["bundle"] = os.path.abspath(args.bundle)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


def _bundle_vocabulary(bundle: str):
    """The cards this match named, which is what the board can contain."""
    for candidate in (os.path.join(bundle, "match.json"),
                      os.path.join(os.path.dirname(bundle.rstrip("/")),
                                   "match.json")):
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f).get("cardsInPlay") or []
    return []


def bundle_layout(bundle: str) -> Optional[Layout]:
    """The layout recorded when this bundle was ingested, if any.

    Reusing it means the HUD is read from the same places the log was, so a
    720p bundle is cross-checked against 720p coordinates rather than
    silently falling back to 1080p ones.
    """
    for candidate in (os.path.join(bundle, "match.json"),
                      os.path.join(os.path.dirname(bundle.rstrip("/")),
                                   "match.json")):
        if not os.path.exists(candidate):
            continue
        with open(candidate) as f:
            data = json.load(f)
        raw = data.get("layout")
        if not raw:
            continue
        return Layout.from_dict(raw)
    return None


def run_layout(args):
    """Detect and validate the UI layout of a capture, without ingesting."""
    work = tempfile.mkdtemp(prefix="mtgo_layout_")
    samples = sample_layout_frames(args.video, work, args.start, args.end,
                                   count=args.layout_samples, at=args.at)
    samples = select_duel_frames(samples)
    layout = detect_layout_from_frames(samples, detect=not args.no_detect)
    checks = [validate_layout(layout, sample) for sample in samples]
    passed = [c for c in checks if c["ok"]]
    check = dict(passed[0] if passed else checks[-1],
                 samplesValidated=len(passed), samplesChecked=len(checks))
    print(json.dumps({"layout": layout.to_dict(), "check": check,
                      "summary": layout.describe()}, indent=2))
    if not passed:
        raise SystemExit(1)


def run_compare(args):
    """Check that captures of the same match decoded to the same game."""
    from .compare import compare_bundles

    report = compare_bundles(args.bundles)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not report["allIdentical"]:
        raise SystemExit(1)


def run_align(args):
    """Timestamp inferred state changes by locating them in the footage."""
    from .hud import align_derived_states
    from .render import load_states

    states = load_states(args.bundle)
    report = align_derived_states(args.video, states,
                                  layout=bundle_layout(args.bundle))
    path = os.path.join(args.bundle, "mirror_states.jsonl")
    with open(path, "w") as f:
        for state in states:
            f.write(json.dumps(state) + "\n")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


def run_crosscheck(args):
    """Check derived life totals against MTGO's on-screen HUD."""
    from .hud import crosscheck_life
    from .render import load_states

    states = load_states(args.bundle)
    report = crosscheck_life(args.video, states, sample=args.sample,
                             settle=args.settle,
                             layout=bundle_layout(args.bundle))
    report["bundle"] = os.path.abspath(args.bundle)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


def run_render(args):
    from .render import capture_states, render_replay, render_side_by_side

    classpath = args.classpath or os.environ.get("MAGIC_CABT_CLASSPATH")
    if not classpath:
        raise SystemExit("pass --classpath or set $MAGIC_CABT_CLASSPATH")
    # Capture once, then encode every requested cut from the same shots.
    shots = capture_states(args.bundle, classpath=classpath, java=args.java,
                           cwd=args.cwd, reuse=args.reuse_frames)
    outputs = [render_replay(args.bundle, args.out, seconds_per_state=
                             args.seconds_per_state, shots=shots)]
    if args.side_by_side:
        if not args.source_video:
            raise SystemExit("--side-by-side needs --source-video")
        outputs.append(render_side_by_side(
            args.bundle, args.source_video, args.side_by_side,
            speed=args.speed, shots=shots,
        ))
    for path in outputs:
        print(path)


def build_parser():
    parser = argparse.ArgumentParser(prog="magic_cabt.mtgo_video")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="video -> log -> per-game mirror bundles")
    ingest.add_argument("video")
    ingest.add_argument("--out", required=True)
    ingest.add_argument("--start", type=float, default=None)
    ingest.add_argument("--end", type=float, default=None)
    ingest.add_argument("--fps", type=float, default=1.0)
    ingest.add_argument("--region", default=None,
                        help="override the detected log-pane crop, as WxH+X+Y")
    ingest.add_argument("--no-detect", action="store_true",
                        help="skip detection; scale the reference layout instead")
    ingest.add_argument("--layout-frame", type=float, default=None,
                        help="video seconds to sample for layout detection")
    ingest.add_argument("--dump-ocr", action="store_true",
                        help="also write the raw per-frame OCR readings")
    ingest.add_argument("--layout-samples", type=int, default=5,
                        help="frames to sample for layout consensus")
    ingest.add_argument("--hero", default=None,
                        help="local player name (perspective for the mirror)")
    ingest.add_argument("--match-id", default="mtgo-video")
    ingest.add_argument("--game", type=int, default=None,
                        help="only build this game number (1-based)")
    ingest.add_argument("--deck-size", type=int, default=60)
    ingest.add_argument("--catalog", default=None,
                        help="card catalog path (default: %s)" % DEFAULT_CATALOG)
    ingest.add_argument("--frames-dir", default=None,
                        help="keep extracted frames here (default: temp dir)")
    ingest.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    ingest.set_defaults(func=run_ingest)

    rebuild = sub.add_parser(
        "rebuild",
        help="re-parse and re-simulate from a bundle's cached mtgo_log.json")
    rebuild.add_argument("bundle")
    rebuild.add_argument("--hero", default=None)
    rebuild.add_argument("--match-id", default="mtgo-video")
    rebuild.add_argument("--game", type=int, default=None)
    rebuild.add_argument("--deck-size", type=int, default=60)
    rebuild.add_argument("--catalog", default=None)
    rebuild.set_defaults(func=run_rebuild)

    catalog = sub.add_parser(
        "catalog", help="download/refresh the local card-name catalog")
    catalog.add_argument("--catalog", default=None,
                         help="output path (default: %s)" % DEFAULT_CATALOG)
    catalog.set_defaults(func=run_catalog)

    verify = sub.add_parser(
        "verify", help="check a game bundle against XMage's mirrored behavior")
    verify.add_argument("bundle")
    verify.add_argument("--classpath", default=None)
    verify.add_argument("--java", default="java")
    verify.add_argument("--cwd", default=None,
                        help="working directory for the JVM (the Mage.Client dir)")
    verify.add_argument("--out", default=None, help="write the report here too")
    verify.set_defaults(func=run_verify)

    rules = sub.add_parser(
        "rules",
        help="check each decoded event against XMage's board at that moment")
    rules.add_argument("bundle")
    rules.add_argument("--classpath", default=None)
    rules.add_argument("--java", default="java")
    rules.add_argument("--cwd", default=None,
                       help="working directory for the JVM (the Mage.Client dir)")
    rules.add_argument("--out", default=None, help="write the report here too")
    rules.set_defaults(func=run_rules)

    check_cmd = sub.add_parser(
        "check", help="run every verification this bundle allows")
    check_cmd.add_argument("bundle")
    check_cmd.add_argument("--video", default=None,
                           help="the capture, for the checks that read it")
    check_cmd.add_argument("--against", nargs="*", default=[],
                           help="other recordings' bundles of the same match")
    check_cmd.add_argument("--classpath", default=None)
    check_cmd.add_argument("--java", default="java")
    check_cmd.add_argument("--cwd", default=None)
    check_cmd.add_argument("--sample", type=int, default=1)
    check_cmd.add_argument("--art-cache", default=None)
    check_cmd.add_argument("--out", default=None)
    check_cmd.set_defaults(func=run_check)

    board_cmd = sub.add_parser(
        "board",
        help="check the board on screen against the board the log produced")
    board_cmd.add_argument("bundle")
    board_cmd.add_argument("--video", required=True)
    board_cmd.add_argument("--sample", type=int, default=1,
                           help="check every Nth state (default: all)")
    board_cmd.add_argument("--settle", type=float, default=0.5,
                           help="seconds after the log line to read the board")
    board_cmd.add_argument("--art-cache", default=None,
                           help="card-art signature cache path")
    board_cmd.add_argument("--out", default=None)
    board_cmd.set_defaults(func=run_board)

    compare_cmd = sub.add_parser(
        "compare",
        help="check that bundles of the same match decoded identically")
    compare_cmd.add_argument("bundles", nargs="+",
                             help="game bundle dirs; the first is the reference")
    compare_cmd.add_argument("--out", default=None)
    compare_cmd.set_defaults(func=run_compare)

    layout_cmd = sub.add_parser(
        "layout", help="detect and validate a capture's UI layout")
    layout_cmd.add_argument("--video", required=True)
    layout_cmd.add_argument("--at", type=float, default=None,
                            help="sample one frame here instead of several")
    layout_cmd.add_argument("--start", type=float, default=None)
    layout_cmd.add_argument("--end", type=float, default=None)
    layout_cmd.add_argument("--no-detect", action="store_true")
    layout_cmd.add_argument("--layout-samples", type=int, default=5)
    layout_cmd.set_defaults(func=run_layout)

    align = sub.add_parser(
        "align",
        help="timestamp inferred changes (combat damage) from the footage")
    align.add_argument("bundle")
    align.add_argument("--video", required=True)
    align.set_defaults(func=run_align)

    crosscheck = sub.add_parser(
        "crosscheck",
        help="check derived life totals against MTGO's on-screen HUD")
    crosscheck.add_argument("bundle")
    crosscheck.add_argument("--video", required=True)
    crosscheck.add_argument("--sample", type=int, default=1,
                            help="check every Nth state (default: all)")
    crosscheck.add_argument("--settle", type=float, default=0.5,
                            help="seconds after the log line to read the HUD")
    crosscheck.add_argument("--out", default=None)
    crosscheck.set_defaults(func=run_crosscheck)

    render = sub.add_parser(
        "render", help="replay a game bundle in XMage and record it to video")
    render.add_argument("bundle")
    render.add_argument("--out", required=True, help="output mp4 path")
    render.add_argument("--classpath", default=None)
    render.add_argument("--java", default="java")
    render.add_argument("--cwd", default=None)
    render.add_argument("--seconds-per-state", type=float, default=1.0)
    render.add_argument("--side-by-side", default=None,
                        help="also write footage-beside-XMage mp4 here")
    render.add_argument("--source-video", default=None,
                        help="the MTGO footage, for --side-by-side")
    render.add_argument("--speed", type=float, default=1.0,
                        help="playback speed multiplier for --side-by-side")
    render.add_argument("--reuse-frames", action="store_true",
                        help="re-encode from existing screenshots, no XMage")
    render.set_defaults(func=run_render)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
