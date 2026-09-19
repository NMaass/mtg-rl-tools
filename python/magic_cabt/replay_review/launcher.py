"""Embed Jev review beside the existing replay transport."""

import argparse
import tkinter as tk
from tkinter import ttk

from .ui import ReplayReviewPanel


class ReviewLibraryMixin:
    def __init__(self, *args, **kwargs):
        self._review_manual_navigation = False
        self._review_active_bundle = None
        super().__init__(*args, **kwargs)
        self.root.geometry("1440x820")
        self.root.minsize(1180, 700)

    def _build_replays_tab(self, frame):
        layout = tk.PanedWindow(
            frame, orient=tk.HORIZONTAL, sashwidth=8, bd=0,
            bg="#1b1d24", opaqueresize=True)
        layout.pack(fill=tk.BOTH, expand=True)

        replay_side = ttk.Frame(layout)
        review_side = ttk.Frame(layout, width=410, style="Card.TFrame")
        layout.add(replay_side, stretch="always", minsize=700)
        layout.add(review_side, stretch="never", minsize=360, width=410)

        super()._build_replays_tab(replay_side)

        self._review_panel = ReplayReviewPanel(review_side)
        self._review_panel.pack(fill=tk.BOTH, expand=True)
        selection = self.replay_table.selection()
        if selection:
            bundle = self._replay_paths.get(selection[0])
            if bundle:
                self._review_panel.load_bundle(bundle)

    def _on_replay_selected(self, event=None):
        super()._on_replay_selected(event)
        panel = getattr(self, "_review_panel", None)
        if panel is None or self._replay_active():
            return
        selection = self.replay_table.selection()
        bundle = self._replay_paths.get(selection[0]) if selection else None
        if bundle and bundle != panel.bundle:
            panel.load_bundle(bundle)

    def watch_replay(self):
        was_active = self._replay_active()
        selection = self.replay_table.selection()
        panel = getattr(self, "_review_panel", None)
        if not was_active and selection and panel is not None:
            bundle = self._replay_paths.get(selection[0])
            if bundle:
                self._review_active_bundle = bundle
                if bundle != panel.bundle:
                    panel.load_bundle(bundle)
        result = super().watch_replay()
        if was_active:
            self._review_active_bundle = None
        return result

    def _transport(self, action, arg=None):
        if (self._replay_controller is not None and
                action in ("step", "jump", "seek")):
            self._review_manual_navigation = True
        return super()._transport(action, arg)

    def _on_scrub_seek(self, index):
        self._review_manual_navigation = True
        return super()._on_scrub_seek(index)

    def _handle_event(self, kind, payload):
        super()._handle_event(kind, payload)
        panel = getattr(self, "_review_panel", None)
        if kind == "replay_reset":
            self._review_active_bundle = None
            if panel is not None:
                selection = self.replay_table.selection()
                bundle = self._replay_paths.get(selection[0]) if selection else None
                if bundle and bundle != panel.bundle:
                    panel.load_bundle(bundle)
            return
        if panel is None or kind != "replay_progress":
            return
        generation, info = payload
        if (generation != self._replay_generation or
                self._replay_controller is None):
            return
        manual_step = bool(
            self._review_manual_navigation and not info.get("playing"))
        panel.sync_frame(info.get("index"), analyze_after_step=manual_step)
        if manual_step:
            self._review_manual_navigation = False

    def _on_close(self):
        panel = getattr(self, "_review_panel", None)
        if panel is not None:
            panel.close()
        super()._on_close()


def main(argv=None):
    from ..arena_mirror.gui import ArenaMirrorApp

    class JevArenaMirrorApp(ReviewLibraryMixin, ArenaMirrorApp):
        pass

    parser = argparse.ArgumentParser(
        description="Arena replay viewer with embedded post-step Jev review.")
    parser.add_argument("--classpath", default=None)
    parser.add_argument("--java", default="java")
    args = parser.parse_args(argv)
    root = tk.Tk()
    JevArenaMirrorApp(root, classpath=args.classpath, java=args.java)
    root.mainloop()
    return 0


if __name__ == "__main__":
    main()
