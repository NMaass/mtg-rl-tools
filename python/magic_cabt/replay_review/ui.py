"""A native, read-only priority review workspace; network work never touches Tk."""

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
        tree.column(name, width=width, minwidth=50, stretch=name == columns[0])
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll = ttk.Scrollbar(frame, command=tree.yview)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scroll.set)
    return tree


def _name(obj):
    return str(obj.get("name") or obj.get("cardName") or
               "Unknown card #%s" % obj.get("grpId", obj.get("instanceId", "?")))


def board_rows(state):
    hero = str(state["hero"])
    zones = state["zones"]
    rows = []
    for obj in zones["battlefield"]:
        owner = str(obj.get("controllerSeat", obj.get("controllerId", "?")))
        stats = "%s/%s" % (obj["power"], obj["toughness"]) if obj.get("power") is not None and obj.get("toughness") is not None else ""
        flags = [stats, "Tapped" if obj.get("tapped") else "Untapped"]
        if obj.get("damage"):
            flags.append("%s damage" % obj["damage"])
        if obj.get("counters"):
            flags.append("Counters: %s" % obj["counters"])
        rows.append((_name(obj), "Your field" if owner == hero else "Opponent field", " | ".join(filter(None, flags))))
    for zone in ("stack", "exile", "command"):
        for obj in zones[zone]:
            rows.append((_name(obj), zone.capitalize(), obj.get("rule", "")))
    for obj in zones["hands"].get(hero, []):
        rows.append((_name(obj), "Your hand", obj.get("manaCost", "")))
    for owner, cards in zones["graveyards"].items():
        for obj in cards:
            rows.append((_name(obj), "Your graveyard" if owner == hero else "Opponent graveyard", ""))
    return rows


class ReviewWindow(tk.Toplevel):
    def __init__(self, parent, bundle, api_key="", session=None, points=None):
        if api_key and (len(api_key.strip()) > 2048 or any(c.isspace() for c in api_key.strip())):
            raise ValueError("Invalid key")
        super().__init__(parent)
        self.title("Jev | Priority replay review")
        self.geometry("1240x790")
        self.minsize(1080, 720)
        self.session = session or ReviewSession()
        if api_key:
            self.session.connect(api_key)
        self.points = []
        self.index = 0
        self.feedback = {}
        self._closed = False
        self._render_signature = None
        self._generation = 0
        self._step_timer = None
        self._loaded = queue.Queue()
        self._build(Path(bundle).name)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Left>", lambda event: self._key_step(event, -1))
        self.bind("<Right>", lambda event: self._key_step(event, 1))
        if points is None:
            threading.Thread(target=self._load, args=(bundle,), daemon=True).start()
        else:
            self._loaded.put((points, 0, None))
        self._timer = self.after(50, self._poll)

    def _build(self, name):
        shell = ttk.Frame(self, padding=18)
        shell.pack(fill=tk.BOTH, expand=True)
        head = ttk.Frame(shell)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(3, weight=1)
        head.grid(row=0, column=0, sticky="ew")
        ttk.Label(head, text="Priority review", style="H1.TLabel").pack(side=tk.LEFT)
        ttk.Label(head, text=name, style="Muted.TLabel").pack(side=tk.RIGHT)
        ttk.Label(shell, text="Recorded pre-action state | Jev ranks captured choices after you step. No moves are executed.",
                  style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 12))
        controls = ttk.Frame(shell)
        controls.grid(row=2, column=0, sticky="ew")
        self.previous = ttk.Button(controls, text="Previous", width=10, command=lambda: self.step(-1), state="disabled")
        self.previous.pack(side=tk.LEFT)
        self.next = ttk.Button(controls, text="Next priority", width=14, command=lambda: self.step(1), state="disabled")
        self.next.pack(side=tk.LEFT, padx=6)
        self.position = tk.StringVar(value="Loading replay...")
        _label(controls, self.position, width=24).pack(side=tk.LEFT, padx=10)
        self.auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Analyze after stepping", variable=self.auto,
                        command=self._toggle_auto).pack(side=tk.RIGHT)
        body = ttk.Frame(shell)
        body.grid(row=3, column=0, sticky="nsew", pady=12)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0, minsize=440)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body, padding=12, style="Card.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right = ttk.Frame(body, width=440, padding=12, style="Card.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_propagate(False)
        self.caption = tk.StringVar(value="Waiting for recorded decisions")
        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)
        _label(left, self.caption, style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self.state_info = tk.StringVar(value="")
        _label(left, self.state_info, style="CardMuted.TLabel", wraplength=530,
               width=58).grid(row=1, column=0, sticky="ew", pady=(8, 12))
        board_parent = ttk.Frame(left)
        board_parent.grid(row=2, column=0, sticky="nsew")
        self.board = _table(board_parent, ("Card / object", "Zone", "State"), (210, 135, 180), 13)
        self.played = tk.StringVar(value="Recorded action: --")
        _label(left, self.played, style="Card.TLabel", wraplength=530,
               width=58).grid(row=3, column=0, sticky="ew", pady=(12, 6))
        ttk.Button(left, text="Inspect exact request", command=self.inspect).grid(row=4, column=0, sticky="w")
        right.rowconfigure(1, minsize=48)
        right.rowconfigure(3, weight=1)
        right.columnconfigure(0, weight=1)
        ttk.Label(right, text="Jev's recommendation", style="H2.TLabel").grid(row=0, column=0, sticky="w")
        self.status_text = tk.StringVar(value="No request sent")
        _label(right, self.status_text, wraplength=412, width=45,
               style="CardMuted.TLabel").grid(row=1, column=0, sticky="ew", pady=(8, 6))
        status_space = ttk.Frame(right, height=64, style="Card.TFrame")
        status_space.grid(row=2, column=0, sticky="ew")
        status_space.pack_propagate(False)
        self.recommendation = tk.StringVar(value="Step or select Analyze to request a recommendation.")
        _label(status_space, self.recommendation, wraplength=405,
               style="Card.TLabel").pack(anchor="w", fill=tk.X)
        table_parent = ttk.Frame(right)
        table_parent.grid(row=3, column=0, sticky="nsew", pady=(6, 10))
        self.ratings = _table(table_parent, ("Action", "Jev %", "Played"), (260, 72, 70), 8)
        self.ratings.tag_configure("best", foreground="#46d19e")
        self.detail = tk.StringVar(value="Select an action to read its full label.")
        detail_space = ttk.Frame(right, height=48, style="Card.TFrame")
        detail_space.grid(row=4, column=0, sticky="ew")
        detail_space.pack_propagate(False)
        _label(detail_space, self.detail, wraplength=405, style="CardMuted.TLabel").pack(anchor="w")
        self.ratings.bind("<<TreeviewSelect>>", self._show_action)
        self.metrics = tk.StringVar(value="Latency -- | Cost unknown\nInput -- | Output --")
        _label(right, self.metrics, style="Card.TLabel").grid(row=5, column=0, sticky="w", pady=10)
        buttons = ttk.Frame(right)
        buttons.grid(row=6, column=0, sticky="ew")
        self.analyze_button = ttk.Button(buttons, text="Analyze / retry", width=18, command=self.analyze, state="disabled")
        self.analyze_button.pack(side=tk.LEFT)
        self.disconnect_button = ttk.Button(buttons, text="Disconnect key", command=self.disconnect)
        self.disconnect_button.pack(side=tk.RIGHT)
        feedback = ttk.Frame(right)
        feedback.grid(row=7, column=0, sticky="ew", pady=(12, 8))
        ttk.Label(feedback, text="Your review:").pack(side=tk.LEFT)
        self.feedback_value = tk.StringVar(value="Unrated")
        self.feedback_input = ttk.Combobox(feedback, textvariable=self.feedback_value, state="readonly",
                                           values=("Unrated", "Useful", "Wrong", "Unsure"), width=11)
        self.feedback_input.pack(side=tk.RIGHT)
        self.feedback_input.bind("<<ComboboxSelected>>", self._rate)
        ttk.Label(right, text="Choice probabilities are not win rates. Agreement with your move is not correctness.",
                  wraplength=405, style="CardMuted.TLabel").grid(row=8, column=0, sticky="ew")
        foot = ttk.Frame(shell)
        foot.grid(row=4, column=0, sticky="ew")
        self.totals = tk.StringVar(value="No calls | Key stays in memory; closing clears it.")
        _label(foot, self.totals, width=90).pack(side=tk.LEFT)
        ttk.Button(foot, text="Export review...", command=self.export).pack(side=tk.RIGHT)
        self.notice = tk.StringVar(value="")
        _label(shell, self.notice, wraplength=1170, width=110,
               style="Muted.TLabel").grid(row=5, column=0, sticky="w", pady=(6, 0))

    def _load(self, bundle):
        try:
            points, ignored = load_points(bundle)
            self._loaded.put((points, ignored, None))
        except (OSError, ValueError) as exc:
            self._loaded.put(([], 0, str(exc)))

    def _poll(self):
        if self._closed:
            return
        try:
            points, ignored, error = self._loaded.get_nowait()
            self.points = points
            if error:
                self.position.set("Replay unavailable")
                self.notice.set(error)
            elif points:
                self.notice.set("%d non-priority decisions excluded. Only captured options are ranked." % ignored)
                self._show_point()
        except queue.Empty:
            pass
        if self.points:
            self._render_result()
        usage = self.session.summary()
        self.totals.set("%d calls | Known $%.8f | %d unknown-cost calls | %d failed | Key %s" % (
            usage["calls"], usage["knownCostUsd"], usage["unknownCostCalls"], usage["failedCalls"],
            "connected" if self.session.connected else "disconnected"))
        self._timer = self.after(75, self._poll)

    def _key_step(self, event, direction):
        if isinstance(self.focus_get(), (tk.Entry, ttk.Entry, tk.Text, ttk.Combobox)):
            return
        self.step(direction)
        return "break"

    def _toggle_auto(self):
        if not self.auto.get():
            self.session.cancel_pending()

    def step(self, direction):
        target = self.index + direction
        if not self.points or not 0 <= target < len(self.points):
            return
        self.index = target
        self._generation += 1
        generation = self._generation
        self.session.cancel_pending()
        self._show_point()
        if self._step_timer is not None:
            self.after_cancel(self._step_timer)
        self._step_timer = self.after_idle(lambda: self._analyze_step(generation))

    def _analyze_step(self, generation):
        self._step_timer = None
        if not self._closed and generation == self._generation and self.auto.get():
            self.session.request(self.points[self.index])

    def _show_point(self):
        point = self.points[self.index]
        self.position.set("%d / %d priority decisions" % (self.index + 1, len(self.points)))
        self.caption.set(point.caption)
        self.previous.configure(state="normal" if self.index else "disabled")
        self.next.configure(state="normal" if self.index + 1 < len(self.points) else "disabled")
        self.feedback_value.set(self.feedback.get(point.key, "Unrated"))
        self.board.delete(*self.board.get_children())
        self.played.set("Recorded action: " + next((o["label"] for o in point.options if o["id"] == point.recorded_choice), "unscored / not matched"))
        if point.request:
            state = point.request["state"]["gameState"]
            players = ["%s: %s life, %s cards" % (
                "You" if str(p.get("seat", p.get("playerId"))) == str(state["hero"]) else "Opponent",
                p.get("life", "?"), p.get("handCount", "?")) for p in state["players"]]
            self.state_info.set("Turn %s | %s / %s\n%s" % (state.get("turnNumber", "?"), state.get("phase", "?"), state.get("step", "?"), " | ".join(players)))
            for row in board_rows(state):
                self.board.insert("", "end", values=row)
        else:
            self.state_info.set("Unsupported capture. No request will be sent.")
        self._render_signature = None
        self._render_result()

    def analyze(self):
        if self.points:
            self.session.request(self.points[self.index], retry=True)

    def _render_result(self):
        point = self.points[self.index]
        status, result = self.session.status(point)
        self.analyze_button.configure(state="normal" if self.session.connected and status in ("idle", "error") else "disabled")
        signature = (point.key, status, self.session.connected)
        if signature == self._render_signature:
            return
        self._render_signature = signature
        self.status_text.set({"idle": "Ready | " + MODEL, "queued": "Queued; waiting for the current call", "loading": "Analyzing captured pre-action state...", "complete": "Complete | session-cached", "error": "Call failed | explicit retry only", "unsupported": "Unsupported recording"}[status])
        self.recommendation.set(point.issue or (result or {}).get("error") or "No recommendation yet.")
        self.metrics.set("Latency -- | Cost unknown\nInput -- | Output --")
        self.detail.set("; ".join(point.warnings) or "Select an action to read its full label.")
        scroll = self.ratings.yview()
        self.ratings.delete(*self.ratings.get_children())
        probabilities = (result or {}).get("probabilities", {})
        for option in sorted(point.options, key=lambda o: -probabilities.get(o["id"], 0)):
            prob = probabilities.get(option["id"])
            best = (result or {}).get("choice") == option["id"]
            self.ratings.insert("", "end", iid=option["id"], values=(option["label"], "%.1f%%" % (prob * 100) if prob is not None else "--", "Yes" if option["id"] == point.recorded_choice else ""), tags=("best",) if best else ())
        if scroll:
            self.ratings.yview_moveto(scroll[0])
        if result:
            cost = result.get("costUsd")
            self.metrics.set("Latency %s ms | %s\nInput %s | Output %s" % (result.get("latencyMs", "--"), "$%.8f" % cost if cost is not None else "Cost unknown", result.get("inputTokens", "--"), result.get("outputTokens", "--")))
            if status == "complete":
                choice = next(o for o in point.options if o["id"] == result["choice"])
                self.recommendation.set(choice["label"] + "\n" + ("Matches recorded choice" if point.recorded_choice == result["choice"] else "Different from recorded choice" if point.recorded_choice else "Recorded choice unscored"))

    def _show_action(self, event=None):
        selected = self.ratings.selection()
        if selected and self.points:
            option = next(o for o in self.points[self.index].options if o["id"] == selected[0])
            self.detail.set(option["label"])

    def _rate(self, event=None):
        if self.points:
            self.feedback[self.points[self.index].key] = self.feedback_value.get()

    def inspect(self):
        if not self.points:
            return
        window = tk.Toplevel(self)
        window.title("Exact request | no key or recorded action included")
        text = tk.Text(window, wrap="word", width=100, height=35)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert("1.0", json.dumps(self.points[self.index].request, indent=2))
        text.configure(state="disabled")

    def export(self):
        path = filedialog.asksaveasfilename(parent=self, title="Export local review (contains game data, never the API key)",
                                           defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(Path(path).parent), delete=False) as handle:
                temporary = handle.name
                json.dump(self.session.report(self.points, self.feedback), handle, indent=2, allow_nan=False)
            os.replace(temporary, path)
        except (OSError, ValueError):
            messagebox.showerror("Export failed", "The review could not be saved. Choose a writable location.", parent=self)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def disconnect(self):
        self.session.disconnect()
        self._render_signature = None

    def close(self):
        self._closed = True
        self.session.close()
        self.after_cancel(self._timer)
        if self._step_timer is not None:
            self.after_cancel(self._step_timer)
        self.destroy()
