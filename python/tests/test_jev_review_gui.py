import json
import os
import tempfile
import threading
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

from magic_cabt.replay_review import client
from magic_cabt.replay_review.data import make_point
from magic_cabt.replay_review.session import ReviewSession
from magic_cabt.replay_review.ui import ReviewWindow
from test_jev_review import record, response


class GuiTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError:
            self.skipTest("A display is required (run with xvfb-run).")
        self.root.withdraw()
        self.window = None

    def tearDown(self):
        if self.window is not None and not self.window._closed:
            self.window.close()
        self.root.destroy()

    def pump(self, predicate=lambda: False, duration=.2):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.root.update()
            if predicate():
                return
            time.sleep(.005)

    def open(self, analyzer=None, key="fixture-key"):
        session = ReviewSession(analyzer or (lambda p, k: client.analyze(p, k, lambda p, k: response())))
        self.window = ReviewWindow(self.root, "fixture", api_key=key, session=session,
                                   points=[make_point(record(t), t) for t in (3, 4, 5)])
        self.pump(lambda: bool(self.window.points))
        return self.window

    def test_no_call_on_open_and_cached_revisit(self):
        window = self.open()
        self.assertEqual(window.session.summary()["calls"], 0)
        window.step(1)
        self.pump(lambda: window.session.summary()["calls"] == 1)
        window.step(-1)
        self.pump(lambda: window.session.summary()["calls"] == 2)
        window.step(1)
        self.pump()
        self.assertEqual(window.session.summary()["calls"], 2)

    def test_no_key_offline_and_manual_mode(self):
        window = self.open(key="")
        window.step(1)
        self.pump()
        self.assertEqual(window.session.summary()["calls"], 0)
        window.session.connect("fixture-key")
        window.auto.set(False)
        window.step(1)
        self.pump()
        self.assertEqual(window.session.summary()["calls"], 0)
        window.analyze()
        self.pump(lambda: window.session.summary()["calls"] == 1)

    def test_delayed_result_preserves_position_focus_and_geometry(self):
        started, release = threading.Event(), threading.Event()
        def delayed(payload, key):
            started.set()
            release.wait(2)
            return client.analyze(payload, key, lambda p, k: response())
        window = self.open(delayed)
        window.step(1)
        self.pump(started.is_set)
        window.next.focus_force()
        before = (window.ratings.winfo_rootx(), window.ratings.winfo_rooty(),
                  window.ratings.winfo_width(), window.ratings.winfo_height())
        window.step(1)
        release.set()
        self.pump(lambda: window.session.summary()["calls"] == 2, duration=2)
        self.pump()
        self.assertEqual(window.index, 2)
        self.assertEqual(window.caption.get(), window.points[2].caption)
        self.assertIn("Lightning Bolt", window.recommendation.get())
        self.assertEqual(before, (window.ratings.winfo_rootx(), window.ratings.winfo_rooty(),
                                 window.ratings.winfo_width(), window.ratings.winfo_height()))
        self.assertEqual(window.focus_get(), window.next)

    def test_feedback_and_export_never_contain_key(self):
        window = self.open()
        window.feedback_value.set("Useful")
        window._rate()
        window.auto.set(False)
        window.step(1)
        window.step(-1)
        self.assertEqual(window.feedback_value.get(), "Useful")
        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / "review.json")
            with patch("magic_cabt.replay_review.ui.filedialog.asksaveasfilename", return_value=output):
                window.export()
            raw = Path(output).read_text()
            self.assertNotIn("fixture-key", raw)
            self.assertEqual(json.loads(raw)["decisions"][0]["humanRating"], "Useful")

    def test_existing_library_integration_and_settings(self):
        try:
            from magic_cabt.arena_mirror.gui import ArenaMirrorApp
        except ModuleNotFoundError:
            if os.environ.get("CI"):
                raise
            self.skipTest("Full repository required for launcher integration.")
        from magic_cabt.replay_review.launcher import ReviewLibraryMixin
        class Integrated(ReviewLibraryMixin, ArenaMirrorApp):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            settings = str(Path(tmp) / "settings.json")
            with patch("magic_cabt.arena_mirror.gui.SETTINGS_PATH", settings):
                app = Integrated(self.root)
                app._review_key.set("fixture-key")
                app._save_settings()
                self.assertNotIn("fixture-key", Path(settings).read_text())
                app._clear_review_key()
                self.assertEqual(app._review_key.get(), "")
                self.assertTrue(app.replay_table.winfo_exists())


if __name__ == "__main__":
    unittest.main()
