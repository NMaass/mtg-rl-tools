"""Add Jev review to the existing replay library without replacing XMage playback."""

import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .ui import ReviewWindow


class ReviewLibraryMixin:
    def _build_replays_tab(self, frame):
        self._review_windows = []
        self._review_key = tk.StringVar()
        toolbar = ttk.Frame(frame)
        toolbar.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(toolbar, text="OpenRouter key").pack(side=tk.LEFT)
        self._review_entry = ttk.Entry(toolbar, textvariable=self._review_key,
                                       show="*", width=28)
        self._review_entry.pack(side=tk.LEFT, padx=8)
        ttk.Button(toolbar, text="Clear key", command=self._clear_review_key).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Review with Jev", style="Accent.TButton",
                   command=self._open_review).pack(side=tk.RIGHT)
        ttk.Button(toolbar, text="Open decision file...", command=self._open_review_file).pack(side=tk.RIGHT, padx=6)
        ttk.Label(frame, text="Session-only key. Review sends the visible pre-action state and captured choices to OpenRouter / TypeSafe. No calls on opening.",
                  wraplength=940, style="Muted.TLabel").pack(anchor="w", pady=(0, 12))
        super()._build_replays_tab(frame)

    def _open_review(self):
        selected = self.replay_table.selection()
        bundle = self._replay_paths.get(selected[0]) if selected else None
        if not bundle:
            messagebox.showinfo("Select a replay", "Select a recorded match from the replay library.", parent=self.root)
            return
        self._review(bundle)

    def _open_review_file(self):
        path = filedialog.askopenfilename(parent=self.root, title="Open a captured decision stream",
                                         filetypes=[("JSONL decisions / XMage game", "*.jsonl")])
        if path:
            self._review(path)

    def _review(self, bundle):
        self._review_windows = [w for w in self._review_windows if not w._closed]
        try:
            window = ReviewWindow(self.root, bundle, api_key=self._review_key.get())
        except ValueError:
            messagebox.showerror("Invalid key", "Paste an OpenRouter key without spaces, or leave it blank to inspect offline.", parent=self.root)
            return
        self._review_windows.append(window)

    def _clear_review_key(self):
        self._review_key.set("")
        for window in self._review_windows:
            if not window._closed:
                window.disconnect()

    def _on_close(self):
        self._clear_review_key()
        for window in self._review_windows:
            if not window._closed:
                window.close()
        super()._on_close()


def main(argv=None):
    from ..arena_mirror.gui import ArenaMirrorApp

    class JevArenaMirrorApp(ReviewLibraryMixin, ArenaMirrorApp):
        pass

    parser = argparse.ArgumentParser(description="Arena mirror with post-step Jev replay review.")
    parser.add_argument("--classpath", default=None)
    parser.add_argument("--java", default="java")
    args = parser.parse_args(argv)
    root = tk.Tk()
    JevArenaMirrorApp(root, classpath=args.classpath, java=args.java)
    root.mainloop()
    return 0


if __name__ == "__main__":
    main()
