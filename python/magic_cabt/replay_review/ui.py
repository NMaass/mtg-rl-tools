"""Embedded, read-only Jev priority review UI."""

import json
import os
import queue
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .data import MODEL, load_points
from .session import ReviewSession


def _label(parent, variable, **kwargs):
    return ttk.Label(parent, textvariable=variable, **kwargs)


def _table(parent, columns, widths, height):
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.BOTH, expand=True)
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=height)
    for name, width in zip(columns, widths):
        tree.heading(name, text=name)
        tree.column(name, width=width, minwidth=48,
                    stretch=name == columns[0])
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scroll.set)
    return tree


class ReplayReviewPanel(ttk.Frame):
    """Persistent Jev panel driven by the existing replay transport."""

    def __init__(self, parent, session=None):
        super().__init__(parent, padding=12, style="Card.TFrame")
        self.session = session or ReviewSession()
        self.points = []
        self.feedback = {}
        self.current_point = None
        self.current_frame = None
        self.bundle = None
        self._closed = False
        self._render_signature = None
        self._load_generation = 0
        self._loaded = queue.Queue()
        self._build()
        self._timer = self.after(75, self._poll)

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(6, weight=1)

        ttk.Label(self, text="Jev priority review",
                  style="H2.TLabel").grid(row=0, column=0, sticky="w")

        key_row = ttk.Frame(self, style="Card.TFrame")
        key_row.grid(row=1, column=0, sticky="ew", pady=(10, 8))
        key_row.columnconfigure(0, weight=1)
        self.key_var = tk.StringVar()
        self.key_entry = ttk.Entry(
            key_row, textvariable=self.key_var, show="*", width=24)
        self.key_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(key_row, text="Clear",
                   command=self.clear_key).grid(row=0, column=1)
        ttk.Label(
            self,
            text="Paste an OpenRouter key. It stays in memory only. "
                 "A call is made after a manual replay step that lands on "
                 "a captured priority decision.",
            wraplength=390, style="CardMuted.TLabel",
        ).grid(row=2, column=0, sticky="ew", pady=(0, 10))

        self.position = tk.StringVar(value="Select a replay.")
        _label(self, self.position, style="Card.TLabel",
               wraplength=390).grid(row=3, column=0, sticky="ew")

        self.recorded = tk.StringVar(value="Recorded action: —")
        _label(self, self.recorded, style="CardMuted.TLabel",
               wraplength=390).grid(row=4, column=0, sticky="ew", pady=(5, 3))

        self.recommendation = tk.StringVar(
            value="Step the replay to a priority decision to analyze it.")
        _label(self, self.recommendation, style="Card.TLabel",
               wraplength=390).grid(row=5, column=0, sticky="ew", pady=(4, 8))

        table_parent = ttk.Frame(self, style="Card.TFrame")
        table_parent.grid(row=6, column=0, sticky="nsew")
        self.ratings = _table(
            table_parent, ("Action", "Jev %", "Played"), (250, 64, 58), 9)
        self.ratings.tag_configure("best", foreground="#46d19e")
        self.ratings.bind("<<TreeviewSelect>>", self._show_action)

        self.detail = tk.StringVar(
            value="Captured legal choices will appear here with card names.")
        _label(self, self.detail, style="CardMuted.TLabel",
               wraplength=390).grid(row=7, column=0, sticky="ew", pady=(8, 4))

        self.metrics = tk.StringVar(
            value="Latency — | Cost unknown\nInput — | Output —")
        _label(self, self.metrics, style="Card.TLabel").grid(
            row=8, column=0, sticky="w", pady=(4, 8))

        controls = ttk.Frame(self, style="Card.TFrame")
        controls.grid(row=9, column=0, sticky="ew")
        self.auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls, text="Analyze after manual step", variable=self.auto,
            command=self._toggle_auto).pack(side=tk.LEFT)
        self.analyze_button = ttk.Button(
            controls, text="Analyze / retry", command=self.analyze,
            state=tk.DISABLED)
        self.analyze_button.pack(side=tk.RIGHT)

        feedback = ttk.Frame(self, style="Card.TFrame")
        feedback.grid(row=10, column=0, sticky="ew", pady=(10, 4))
        ttk.Label(feedback, text="Your review:",
                  style="Card.TLabel").pack(side=tk.LEFT)
        self.feedback_value = tk.StringVar(value="Unrated")
        self.feedback_input = ttk.Combobox(
            feedback, textvariable=self.feedback_value, state="readonly",
            values=("Unrated", "Useful", "Wrong", "Unsure"), width=10)
        self.feedback_input.pack(side=tk.RIGHT)
        self.feedback_input.bind("<<ComboboxSelected>>", self._rate)

        self.totals = tk.StringVar(
            value="No calls | Jev probabilities are not win probabilities.")
        _label(self, self.totals, style="CardMuted.TLabel",
               wraplength=390).grid(row=11, column=0, sticky="ew", pady=(6, 4))

        ttk.Button(self, text="Export review…",
                   command=self.export).grid(row=12, column=0, sticky="e")

    def load_bundle(self, bundle):
        self._load_generation += 1
        generation = self._load_generation
        self.bundle = bundle
        self.points = []
        self.current_point = None
        self.current_frame = None
        self._render_signature = None
        self.position.set("Loading captured priority decisions…")
        self.recorded.set("Recorded action: —")
        self.recommendation.set("No request sent.")
        self.ratings.delete(*self.ratings.get_children())
        threading.Thread(
            target=self._load, args=(bundle, generation), daemon=True).start()

    def _load(self, bundle, generation):
        try:
            points, ignored = load_points(bundle)
            self._loaded.put((generation, points, ignored, None))
        except (OSError, ValueError) as exc:
            self._loaded.put((generation, [], 0, str(exc)))

    def sync_frame(self, frame_index, analyze_after_step=False):
        """Show the priority decision at this replay frame, if one exists."""
        if not isinstance(frame_index, int):
            return
        self.current_frame = frame_index
        point = next(
            (p for p in self.points if p.frame_index == frame_index), None)
        if point is None:
            self.current_point = None
            self._render_signature = None
            self.position.set("Frame %d · no captured priority decision" %
                              (frame_index + 1))
            self.recorded.set("Recorded action: —")
            self.recommendation.set(
                "Jev is called only on captured hero priority decisions.")
            self.ratings.delete(*self.ratings.get_children())
            self.analyze_button.configure(state=tk.DISABLED)
            return
        changed = self.current_point is None or             self.current_point.key != point.key
        self.current_point = point
        self._render_signature = None
        self._show_point()
        if changed and analyze_after_step and self.auto.get():
            self.analyze(retry=False)

    def connect_key(self):
        key = self.key_var.get().strip()
        if not key:
            return False
        try:
            self.session.connect(key)
        except ValueError:
            self.recommendation.set(
                "Invalid OpenRouter key. Paste it without whitespace.")
            return False
        return True

    def clear_key(self):
        self.key_var.set("")
        self.session.disconnect()
        self._render_signature = None
        self._render_result()

    def analyze(self, retry=True):
        point = self.current_point
        if point is None or point.request is None:
            return
        if not self.session.connected and not self.connect_key():
            self.recommendation.set(
                "Paste an OpenRouter key to analyze this priority decision.")
            return
        self.session.request(point, retry=retry)
        self._render_signature = None

    def _show_point(self):
        point = self.current_point
        if point is None:
            return
        self.position.set(point.caption)
        self.feedback_value.set(self.feedback.get(point.key, "Unrated"))
        played = next(
            (o["label"] for o in point.options
             if o["id"] == point.recorded_choice),
            "unscored / response not matched")
        self.recorded.set("Recorded action: " + played)
        self.detail.set(
            "; ".join(point.warnings) or
            "Card/object transport IDs are resolved to readable names.")
        self._render_result()

    def _render_result(self):
        point = self.current_point
        if point is None:
            return
        status, result = self.session.status(point)
        can_analyze = point.request is not None and status in ("idle", "error")
        self.analyze_button.configure(
            state=tk.NORMAL if can_analyze else tk.DISABLED)
        signature = (
            point.key, status, self.session.connected,
            (result or {}).get("choice"),
            tuple(sorted(((result or {}).get("probabilities") or {}).items())),
        )
        if signature == self._render_signature:
            return
        self._render_signature = signature

        status_text = {
            "idle": "Ready. Step again or select Analyze.",
            "queued": "Queued behind the current Jev request.",
            "loading": "Analyzing this captured pre-action state…",
            "complete": "Jev recommendation complete.",
            "error": (result or {}).get("error") or
                     "Request failed; retry explicitly.",
            "unsupported": point.issue or "Unsupported recording.",
        }.get(status, "No recommendation yet.")
        self.recommendation.set(status_text)
        self.metrics.set("Latency — | Cost unknown\nInput — | Output —")

        scroll = self.ratings.yview()
        self.ratings.delete(*self.ratings.get_children())
        probabilities = (result or {}).get("probabilities", {})
        for option in sorted(
                point.options,
                key=lambda item: -probabilities.get(item["id"], 0)):
            probability = probabilities.get(option["id"])
            best = (result or {}).get("choice") == option["id"]
            self.ratings.insert(
                "", "end", iid=option["id"],
                values=(
                    option["label"],
                    "%.1f%%" % (probability * 100)
                    if probability is not None else "—",
                    "Yes" if option["id"] == point.recorded_choice else "",
                ),
                tags=("best",) if best else (),
            )
        if scroll:
            self.ratings.yview_moveto(scroll[0])

        if result:
            cost = result.get("costUsd")
            self.metrics.set(
                "Latency %s ms | %s\nInput %s | Output %s" % (
                    result.get("latencyMs", "—"),
                    "$%.8f" % cost if cost is not None else "Cost unknown",
                    result.get("inputTokens", "—"),
                    result.get("outputTokens", "—"),
                ))
            if status == "complete":
                choice = next(
                    (o for o in point.options
                     if o["id"] == result.get("choice")), None)
                if choice is not None:
                    comparison = (
                        "Matches your recorded action"
                        if point.recorded_choice == result.get("choice")
                        else "Different from your recorded action"
                        if point.recorded_choice else
                        "Recorded action is unscored")
                    self.recommendation.set(
                        "%s\n%s" % (choice["label"], comparison))

    def _show_action(self, event=None):
        selected = self.ratings.selection()
        point = self.current_point
        if selected and point is not None:
            option = next(
                (o for o in point.options if o["id"] == selected[0]), None)
            if option is not None:
                detail = option["label"]
                details = option.get("details") or {}
                readable = [
                    "%s: %s" % (key, value)
                    for key, value in details.items()
                    if value not in (None, "", [], {})
                ]
                if readable:
                    detail += " · " + " · ".join(readable)
                self.detail.set(detail)

    def _toggle_auto(self):
        if not self.auto.get():
            self.session.cancel_pending()

    def _rate(self, event=None):
        if self.current_point is not None:
            self.feedback[self.current_point.key] = self.feedback_value.get()

    def _poll(self):
        if self._closed:
            return
        try:
            while True:
                generation, points, ignored, error = self._loaded.get_nowait()
                if generation != self._load_generation:
                    continue
                self.points = points
                if error:
                    self.position.set("Replay unavailable")
                    self.recommendation.set(error)
                elif points:
                    self.position.set(
                        "%d priority decisions loaded · %d other decisions skipped"
                        % (len(points), ignored))
                    if self.current_frame is not None:
                        self.sync_frame(self.current_frame, False)
        except queue.Empty:
            pass

        if self.current_point is not None:
            self._render_result()
        usage = self.session.summary()
        self.totals.set(
            "%d calls | Known $%.8f | %d unknown-cost | %d failed" % (
                usage["calls"], usage["knownCostUsd"],
                usage["unknownCostCalls"], usage["failedCalls"]))
        self._timer = self.after(75, self._poll)

    def export(self):
        if not self.points:
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Export replay review",
            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8",
                    dir=str(Path(path).parent), delete=False) as handle:
                temporary = handle.name
                json.dump(
                    self.session.report(self.points, self.feedback),
                    handle, indent=2, allow_nan=False)
            os.replace(temporary, path)
        except (OSError, ValueError):
            messagebox.showerror(
                "Export failed",
                "The review could not be saved. Choose a writable location.",
                parent=self)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.key_var.set("")
        self.session.close()
        try:
            self.after_cancel(self._timer)
        except tk.TclError:
            pass


class ReviewWindow(tk.Toplevel):
    """Standalone wrapper retained for opening an arbitrary decision JSONL."""

    def __init__(self, parent, bundle, api_key="", session=None):
        super().__init__(parent)
        self.title("Jev priority replay review")
        self.geometry("470x760")
        self.minsize(420, 620)
        self.panel = ReplayReviewPanel(self, session=session)
        self.panel.pack(fill=tk.BOTH, expand=True)
        if api_key:
            self.panel.key_var.set(api_key)
            self.panel.connect_key()
        self.panel.load_bundle(bundle)
        self.protocol("WM_DELETE_WINDOW", self.close)

    @property
    def _closed(self):
        return self.panel._closed

    def close(self):
        self.panel.close()
        self.destroy()
