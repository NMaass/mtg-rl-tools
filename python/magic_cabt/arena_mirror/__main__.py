"""CLI: live-follow MTGA and mirror into XMage, or replay a recording.

    python -m magic_cabt.arena_mirror live [--log PATH] [--out DIR] ...
    python -m magic_cabt.arena_mirror replay BUNDLE_DIR [--speed X] ...

Defaults point at the standard Windows MTGA install; the display needs
$MAGIC_CABT_CLASSPATH (or --classpath) with the built XMage + bridge classes.
"""

import argparse
import datetime
import os
import sys

from .cards import CardDatabase
from .mirror import MirrorDisplay
from .recorder import MirrorRecorder
from .replay import ReplayPlayer
from .session import MirrorSession

DEFAULT_LOG_PATH = os.path.expanduser(
    r"~\AppData\LocalLow\Wizards Of The Coast\MTGA\Player.log")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m magic_cabt.arena_mirror",
        description="Mirror live MTG Arena gameplay into an XMage board "
                    "display and record a CABT-format replay.")
    commands = parser.add_subparsers(dest="mode", required=True)

    live = commands.add_parser("live", help="follow a live Player.log")
    live.add_argument("--log", default=DEFAULT_LOG_PATH,
                      help="Player.log path (default: standard MTGA location)")
    live.add_argument("--out", default=None,
                      help="output bundle directory (default: "
                           "arena-mirror-runs/<timestamp>)")
    live.add_argument("--from-start", action="store_true",
                      help="process the log's existing content before tailing")
    live.add_argument("--no-display", action="store_true",
                      help="record only; do not launch the XMage window")
    live.add_argument("--classpath", default=None,
                      help="Java classpath for the display "
                           "(default: $MAGIC_CABT_CLASSPATH)")
    live.add_argument("--java", default="java", help="java executable")
    live.add_argument("--card-db", default=None,
                      help="MTGA Raw_CardDatabase path (default: auto-detect)")
    live.add_argument("--raw-audit", action="store_true",
                      help="also write raw_audit.jsonl (unredacted normalized "
                           "events; may contain raw Arena payloads)")
    live.add_argument("--quiet", action="store_true")

    replay = commands.add_parser("replay", help="replay a recorded bundle")
    replay.add_argument("bundle", help="bundle directory from a live run")
    replay.add_argument("--speed", type=float, default=None,
                        help="wall-clock multiplier over recorded timing")
    replay.add_argument("--interval", type=float, default=0.5,
                        help="fixed seconds per state (default 0.5; ignored "
                             "when --speed or --step is given)")
    replay.add_argument("--step", action="store_true",
                        help="advance one state per Enter keypress")
    replay.add_argument("--no-display", action="store_true",
                        help="narrate to the console only")
    replay.add_argument("--classpath", default=None)
    replay.add_argument("--java", default="java")

    gui = commands.add_parser("gui", help="launcher GUI (logs, actions, "
                                          "locate-logs, auto-open XMage)")
    gui.add_argument("--classpath", default=None)
    gui.add_argument("--java", default="java")

    args = parser.parse_args(argv)
    if args.mode == "live":
        return run_live(args)
    if args.mode == "gui":
        from . import gui as gui_mod
        return gui_mod.main(
            ["--java", args.java]
            + (["--classpath", args.classpath] if args.classpath else []))
    return run_replay(args)


def run_live(args):
    out_dir = args.out
    if out_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join("arena-mirror-runs", stamp)

    try:
        card_db = CardDatabase(db_path=args.card_db)
    except IOError as error:
        sys.stderr.write("warning: %s\n         cards will render face-down\n"
                         % (error,))
        card_db = None

    display = None
    if not args.no_display:
        display = MirrorDisplay(classpath=args.classpath, java=args.java)
        display.ping()

    recorder = MirrorRecorder(out_dir, card_db=card_db,
                              raw_audit=args.raw_audit)
    session = MirrorSession(recorder=recorder, display=display,
                            card_db=card_db, verbose=not args.quiet)
    sys.stderr.write("[arena-mirror] recording to %s\n" % out_dir)
    try:
        session.follow(args.log, from_start=args.from_start)
    except KeyboardInterrupt:
        sys.stderr.write("\n[arena-mirror] stopped\n")
    finally:
        recorder.close()
        if display is not None:
            display.close()
    return 0


def run_replay(args):
    display = None
    if not args.no_display:
        display = MirrorDisplay(classpath=args.classpath, java=args.java)
        display.ping()
    interval = None if (args.speed or args.step) else args.interval
    player = ReplayPlayer(display=display, speed=args.speed,
                          interval=interval, step=args.step)
    try:
        player.play(args.bundle)
        if display is not None and display.alive:
            input("replay done - press Enter to close the display... ")
    except KeyboardInterrupt:
        pass
    finally:
        if display is not None:
            display.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
