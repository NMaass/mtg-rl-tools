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
from magic_cabt.replay_review.ui import ReplayReviewPanel
from test_jev_review import record, response


class GuiTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError:
            self.skipTest("A display is required (run with xvfb-run).")
        self.root.withdraw()
        self.panel = None

    def tearDown(self):
        if self.panel is not None and not self.panel._closed:
            self.panel.close()
        try:
            for callback in self.root.tk.call("after", "info"):
                self.root.after_cancel(callback)
            self.root.destroy()
        except tk.TclError:
            pass

    def pump(self, predicate=lambda: False, duration=.2):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.root.update()
            if predicate():
                return
            time.sleep(.005)

    def open_panel(self, analyzer=None, key="fixture-key"):
        session = ReviewSession(
            analyzer or
            (lambda p, k: client.analyze(
                p, k, lambda payload, key: response())))
        self.panel = ReplayReviewPanel(self.root, session=session)
        self.panel.pack(fill=tk.BOTH, expand=True)
        if key:
            self.panel.key_var.set(key)
        self.panel.points = [
            make_point(record(3), 0, frame_index=2),
            make_point(record(4), 1, frame_index=5),
        ]
        self.pump()
        return self.panel

    def test_call_happens_after_manual_step_not_on_load(self):
        panel = self.open_panel()
        panel.sync_frame(2, analyze_after_step=False)
        self.pump()
        self.assertEqual(panel.session.summary()["calls"], 0)
        panel.sync_frame(5, analyze_after_step=True)
        self.pump(
            lambda: "Lightning Bolt" in panel.recommendation.get(),
            duration=2)
        self.assertEqual(panel.session.summary()["calls"], 1)
        self.assertIn("Lightning Bolt", panel.recorded.get())
        self.assertIn("Lightning Bolt", panel.recommendation.get())

    def test_autoplay_style_progress_does_not_spend(self):
        panel = self.open_panel()
        panel.sync_frame(2, analyze_after_step=False)
        panel.sync_frame(5, analyze_after_step=False)
        self.pump()
        self.assertEqual(panel.session.summary()["calls"], 0)

    def test_delayed_result_does_not_move_panel_or_focus(self):
        started, release = threading.Event(), threading.Event()

        def delayed(payload, key):
            started.set()
            release.wait(2)
            return client.analyze(
                payload, key, lambda p, k: response())

        panel = self.open_panel(delayed)
        panel.sync_frame(2, analyze_after_step=True)
        self.pump(
            lambda: started.is_set() and
            panel.recommendation.get().startswith("Analyzing"),
            duration=2)
        panel.key_entry.focus_force()
        before = (
            panel.ratings.winfo_rootx(), panel.ratings.winfo_rooty(),
            panel.ratings.winfo_width(), panel.ratings.winfo_height())
        release.set()
        self.pump(
            lambda: panel.session.summary()["calls"] == 1, duration=2)
        self.pump()
        self.assertEqual(
            before,
            (panel.ratings.winfo_rootx(), panel.ratings.winfo_rooty(),
             panel.ratings.winfo_width(), panel.ratings.winfo_height()))
        self.assertEqual(panel.focus_get(), panel.key_entry)

    def test_feedback_and_export_never_contain_key(self):
        panel = self.open_panel()
        panel.sync_frame(2, analyze_after_step=False)
        panel.feedback_value.set("Useful")
        panel._rate()
        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / "review.json")
            with patch(
                    "magic_cabt.replay_review.ui.filedialog.asksaveasfilename",
                    return_value=output):
                panel.export()
            raw = Path(output).read_text()
            self.assertNotIn("fixture-key", raw)
            self.assertEqual(
                json.loads(raw)["decisions"][0]["humanRating"], "Useful")

    def test_existing_replay_library_contains_embedded_panel_and_does_not_save_key(self):
        try:
            from magic_cabt.arena_mirror.gui import ArenaMirrorApp
        except ModuleNotFoundError:
            if os.environ.get("CI"):
                raise
            self.skipTest("Full repository required for launcher integration.")
        from magic_cabt.replay_review.launcher import ReviewLibraryMixin

        class Integrated(ReviewLibraryMixin, ArenaMirrorApp):
            pass

        self.panel = None
        self.root.deiconify()
        with tempfile.TemporaryDirectory() as tmp:
            settings = str(Path(tmp) / "settings.json")
            with patch(
                    "magic_cabt.arena_mirror.gui.SETTINGS_PATH", settings):
                app = Integrated(self.root)
                self.panel = app._review_panel
                app._notebook.select(app._replays_tab)
                self.root.update()
                self.assertIs(
                    app._review_panel.winfo_toplevel(), self.root)
                self.assertTrue(app.replay_table.winfo_exists())
                self.assertTrue(app._review_panel.winfo_exists())
                self.assertGreater(
                    app._review_panel.winfo_rootx(),
                    app.replay_table.winfo_rootx())
                app._review_panel.key_var.set("fixture-key")
                app._save_settings()
                self.assertNotIn(
                    "fixture-key", Path(settings).read_text())
                app._review_panel.clear_key()
                self.assertEqual(app._review_panel.key_var.get(), "")


if __name__ == "__main__":
    unittest.main()
