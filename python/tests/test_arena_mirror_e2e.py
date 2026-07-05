"""End-to-end mirror verification against the real captured Player.logs.

Runs only when the captures are present (they are large and live outside the
repo). Set MTGA_CAPTURE_DIR to point elsewhere; defaults to the known local
capture directory. Skips cleanly when absent so the suite stays portable.
"""

import json
import os
import shutil
import tempfile
import threading
import time
import unittest

from magic_cabt.arena_log import iter_log_entries
from magic_cabt.arena_mirror.follower import LogFollower
from magic_cabt.arena_mirror.recorder import MirrorRecorder
from magic_cabt.arena_mirror.replay import ReplayPlayer, load_bundle
from magic_cabt.arena_mirror.session import MirrorSession
from magic_cabt.arena_mirror.tracker import ArenaMatchTracker, StreamingNormalizer

CAPTURE_DIR = os.environ.get(
    "MTGA_CAPTURE_DIR", r"C:\Users\nicho\Code\mtg-ai\mtga_logs")
CAPTURE_LOGS = ("nick1.log", "nick2.log", "levi1.log", "levi2.log")

# Ground truth from the batch-validated normalizer (see the mtga-real-log
# memory): 17 games, ~1155 prompts. These are the aggregate expectations.
EXPECTED_GAMES = 17


def _available_logs():
    return [os.path.join(CAPTURE_DIR, name)
            for name in CAPTURE_LOGS
            if os.path.exists(os.path.join(CAPTURE_DIR, name))]


def _run_pipeline(log_path, out_dir):
    recorder = MirrorRecorder(out_dir)
    session = MirrorSession(recorder=recorder, display=None, card_db=None,
                            verbose=False)
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        session.feed_entries(iter_log_entries(handle))
    recorder.close()
    return recorder.counts


@unittest.skipUnless(_available_logs(), "no MTGA captures present")
class MirrorEndToEndTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_all_decisions_match_and_counts_reconcile(self):
        total_games = 0
        total_decisions = 0
        total_matched = 0
        for log_path in _available_logs():
            out_dir = os.path.join(self.dir, os.path.basename(log_path))
            counts = _run_pipeline(log_path, out_dir)
            total_games += len(counts["games"])
            total_decisions += counts["decisions"]
            total_matched += counts["decisionsMatched"]
        # every decision the client made resolves to option indices
        self.assertEqual(total_decisions, total_matched)
        self.assertGreater(total_decisions, 1000)
        if len(_available_logs()) == len(CAPTURE_LOGS):
            self.assertEqual(EXPECTED_GAMES, total_games)

    def test_replay_reproduces_recorded_decisions(self):
        log_path = _available_logs()[0]
        out_dir = os.path.join(self.dir, "bundle")
        _run_pipeline(log_path, out_dir)

        states, decisions, summary = load_bundle(out_dir)
        self.assertTrue(states)
        self.assertTrue(decisions)

        # Play the bundle back through the real ReplayPlayer with a fake
        # display, capturing the states it renders and the decisions it
        # narrates. This is exactly the "watch the replay back" path.
        rendered = []

        class FakeDisplay(object):
            def start_game(self, players, **kwargs):
                pass

            def send_state(self, state):
                rendered.append((state.get("matchId"), state.get("seq")))

            def send_message(self, text):
                pass

        devnull = open(os.devnull, "w")
        try:
            player = ReplayPlayer(display=FakeDisplay(), interval=0,
                                  log_stream=devnull)
            player.play(out_dir)
        finally:
            devnull.close()

        # every recorded state is rendered, in order
        self.assertEqual([(s.get("matchId"), s.get("seq")) for s in states],
                         rendered)
        # every recorded decision is narrated exactly once, in recorded order
        self.assertEqual([d["sequence"] for d in decisions], player.narrated)
        self.assertEqual(summary["decisions"], len(decisions))

    def test_live_follow_matches_batch(self):
        """A simulated live tail produces the same events as a batch parse."""
        log_path = _available_logs()[0]
        with open(log_path, "rb") as handle:
            data = handle.read()

        tmp = os.path.join(self.dir, "Player.log")
        open(tmp, "wb").close()

        def writer():
            with open(tmp, "ab") as out:
                position = 0
                chunk = 65536
                while position < len(data):
                    out.write(data[position:position + chunk])
                    out.flush()
                    position += chunk
                    time.sleep(0.002)

        follower = LogFollower(tmp, poll_seconds=0.02, from_start=True)
        live = {"decisions": 0, "snapshots": 0}
        normalizer = StreamingNormalizer()
        tracker = ArenaMatchTracker(
            on_snapshot=lambda snap, event: live.__setitem__(
                "snapshots", live["snapshots"] + 1),
            on_decision=lambda record: live.__setitem__(
                "decisions", live["decisions"] + 1))

        writer_thread = threading.Thread(target=writer)
        writer_thread.start()

        def stopper():
            writer_thread.join()
            time.sleep(1.0)
            follower.stop()

        threading.Thread(target=stopper).start()
        for entry in follower.follow():
            for event in normalizer.feed(entry)[1]:
                tracker.handle_event(event)

        batch = {"decisions": 0, "snapshots": 0}
        batch_normalizer = StreamingNormalizer()
        batch_tracker = ArenaMatchTracker(
            on_snapshot=lambda snap, event: batch.__setitem__(
                "snapshots", batch["snapshots"] + 1),
            on_decision=lambda record: batch.__setitem__(
                "decisions", batch["decisions"] + 1))
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            for entry in iter_log_entries(handle):
                for event in batch_normalizer.feed(entry)[1]:
                    batch_tracker.handle_event(event)

        self.assertEqual(batch, live)


if __name__ == "__main__":
    unittest.main()
