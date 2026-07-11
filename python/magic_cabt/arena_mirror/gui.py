"""Desktop launcher + replay viewer for the Arena -> XMage mirror.

Two tabs, one window:

* **Follow** — point it at MTGA's ``Player.log``, press Start, and every live
  game is mirrored into XMage and recorded to a CABT bundle.
* **Replays** — a browsable library of recorded matches (named the way a
  player thinks of them: "Dimir vs Boros — Win (2-1) — Ladder"), with a full
  transport bar: play/pause, single-step, jump to the next action or the next
  *non-pass* action, and a live speed control and scrubber.

The look is a flat dark theme built entirely on ttk (no third-party deps), so
it reads like a modern desktop app rather than raw Tk. All log/session work
runs off the Tk thread; UI updates are marshalled back through a thread-safe
queue drained on the event loop.
"""

import glob
import json
import os
import queue
import shutil
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, scrolledtext, ttk

from .cards import CardDatabase
from .draft_advisor import DraftAdvisor
from .mirror import CLASSPATH_ENV_VAR, MirrorDisplay
from .recorder import MirrorRecorder
from .replay import ReplayController
from .session import MirrorSession

DEFAULT_LOG_PATH = os.path.expanduser(
    r"~\AppData\LocalLow\Wizards Of The Coast\MTGA\Player.log")
DEFAULT_RUNS_DIR = "arena-mirror-runs"
# Where setup-arena-mirror.ps1 builds XMage; used to auto-find the classpath
# and the java runtime when the GUI wasn't launched through that script.
DEFAULT_XMAGE_DIR = r"C:\Users\nicho\Code\xmage-goldflush"


# --- flat dark palette -------------------------------------------------------
class Palette(object):
    BG = "#1b1d24"          # window background
    PANEL = "#232630"       # cards / panes
    PANEL_ALT = "#2b2f3a"   # zebra / hover
    BORDER = "#333743"
    TEXT = "#e7e9ee"
    MUTED = "#9aa1b0"
    ACCENT = "#6c8cff"      # primary action / indigo
    ACCENT_DK = "#5a76e0"
    SUCCESS = "#46d19e"
    DANGER = "#ff6b6b"
    WARN = "#f0b429"
    SELECT = "#38415a"


class ArenaMirrorApp(object):
    """The launcher + replay-viewer window."""

    def __init__(self, root, classpath=None, java="java"):
        self.root = root
        self.classpath = classpath
        self.java = java
        root.title("Arena → XMage Mirror")
        root.geometry("1040x680")
        root.minsize(880, 560)
        root.configure(bg=Palette.BG)

        self._init_fonts()
        self._init_style()

        self.log_var = tk.StringVar(value=DEFAULT_LOG_PATH)
        self.out_var = tk.StringVar(value=os.path.join(
            DEFAULT_RUNS_DIR, "session"))
        self.display_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Idle")

        self._queue = queue.Queue()
        self._session = None
        self._thread = None
        self._import_thread = None
        self._stop_requested = False
        # The XMage window is GUI-owned: it opens on first live activity (or a
        # replay) and stays up across Start/Stop until the user closes it.
        self._display = None
        self._display_lock = threading.Lock()
        self._replay_controller = None
        self._replay_at_end = False
        self._replay_total = 0

        self._build_layout()
        self.root.after(100, self._drain_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------- theming
    def _init_fonts(self):
        self.font_ui = tkfont.nametofont("TkDefaultFont")
        self.font_ui.configure(family="Segoe UI", size=10)
        self.font_h1 = tkfont.Font(family="Segoe UI Semibold", size=15)
        self.font_h2 = tkfont.Font(family="Segoe UI Semibold", size=11)
        self.font_mono = tkfont.Font(family="Consolas", size=10)
        self.font_chip = tkfont.Font(family="Segoe UI Semibold", size=9)

    def _init_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        p = Palette
        style.configure(".", background=p.BG, foreground=p.TEXT,
                        fieldbackground=p.PANEL, bordercolor=p.BORDER,
                        font=self.font_ui)
        style.configure("TFrame", background=p.BG)
        style.configure("Card.TFrame", background=p.PANEL)
        style.configure("TLabel", background=p.BG, foreground=p.TEXT)
        style.configure("Card.TLabel", background=p.PANEL, foreground=p.TEXT)
        style.configure("H1.TLabel", background=p.BG, foreground=p.TEXT,
                        font=self.font_h1)
        style.configure("H2.TLabel", background=p.BG, foreground=p.TEXT,
                        font=self.font_h2)
        style.configure("Muted.TLabel", background=p.BG, foreground=p.MUTED)
        style.configure("CardMuted.TLabel", background=p.PANEL,
                        foreground=p.MUTED)

        # buttons
        style.configure("TButton", background=p.PANEL_ALT, foreground=p.TEXT,
                        borderwidth=0, focuscolor=p.PANEL_ALT, padding=(12, 6))
        style.map("TButton",
                  background=[("active", p.BORDER), ("disabled", p.PANEL)],
                  foreground=[("disabled", p.MUTED)])
        style.configure("Accent.TButton", background=p.ACCENT,
                        foreground="#ffffff", padding=(14, 7))
        style.map("Accent.TButton",
                  background=[("active", p.ACCENT_DK), ("disabled", p.PANEL)],
                  foreground=[("disabled", p.MUTED)])
        style.configure("Transport.TButton", padding=(10, 6),
                        font=self.font_h2)

        # entries / checkbuttons
        style.configure("TEntry", fieldbackground=p.PANEL, foreground=p.TEXT,
                        insertcolor=p.TEXT, borderwidth=1, padding=5)
        style.configure("TCheckbutton", background=p.BG, foreground=p.TEXT)
        style.map("TCheckbutton", background=[("active", p.BG)],
                  foreground=[("active", p.TEXT)])

        # notebook — flat pill tabs, identical size in every state (clam
        # otherwise pads/expands the selected tab differently, so the inactive
        # tab looks bigger). Fixed padding + zero expand keeps them uniform.
        style.configure("TNotebook", background=p.BG, borderwidth=0,
                        tabmargins=(0, 4, 0, 0))
        style.configure("TNotebook.Tab", background=p.BG, foreground=p.MUTED,
                        padding=(22, 10), borderwidth=0, font=self.font_h2,
                        focuscolor=p.BG)
        style.map("TNotebook.Tab",
                  background=[("selected", p.PANEL), ("active", p.PANEL_ALT)],
                  foreground=[("selected", p.ACCENT), ("active", p.TEXT)],
                  padding=[("selected", (22, 10)), ("active", (22, 10))],
                  expand=[("selected", (0, 0, 0, 0)),
                          ("!selected", (0, 0, 0, 0))])

        # treeview
        style.configure("Treeview", background=p.PANEL, fieldbackground=p.PANEL,
                        foreground=p.TEXT, borderwidth=0, rowheight=28)
        style.map("Treeview", background=[("selected", p.SELECT)],
                  foreground=[("selected", p.TEXT)])
        style.configure("Treeview.Heading", background=p.PANEL_ALT,
                        foreground=p.MUTED, borderwidth=0, relief="flat",
                        font=self.font_chip, padding=(8, 6))
        style.map("Treeview.Heading", background=[("active", p.BORDER)])

        # scale + progressbar
        style.configure("Horizontal.TScale", background=p.BG,
                        troughcolor=p.PANEL_ALT, borderwidth=0)

    # ------------------------------------------------------------- layout
    def _build_layout(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 12))

        follow_tab = ttk.Frame(notebook, padding=16)
        draft_tab = ttk.Frame(notebook, padding=16)
        replays_tab = ttk.Frame(notebook, padding=16)
        notebook.add(follow_tab, text="Follow")
        notebook.add(draft_tab, text="Draft")
        notebook.add(replays_tab, text="Replays")
        self._notebook = notebook
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._build_follow_tab(follow_tab)
        self._build_draft_tab(draft_tab)
        self._build_replays_tab(replays_tab)

    def _build_follow_tab(self, frm):
        header = ttk.Frame(frm)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Follow live game", style="H1.TLabel").pack(
            side=tk.LEFT)
        self._status_chip = tk.Label(
            header, textvariable=self.status_var, font=self.font_chip,
            bg=Palette.PANEL_ALT, fg=Palette.MUTED, padx=12, pady=4)
        self._status_chip.pack(side=tk.RIGHT)
        ttk.Label(frm, text="Mirror MTG Arena into XMage and record a replay.",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 14))

        self._path_row(frm, "MTGA Player.log", self.log_var,
                       "Locate…", self._pick_log)
        self._path_row(frm, "Output folder", self.out_var,
                       "Choose…", self._pick_out)

        opts = ttk.Frame(frm)
        opts.pack(fill=tk.X, pady=(8, 4))
        ttk.Checkbutton(opts, text="Open XMage on live game",
                        variable=self.display_var).pack(side=tk.LEFT)

        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=(10, 6))
        self.start_btn = ttk.Button(btns, text="Start following",
                                    style="Accent.TButton", command=self.start)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=8)
        # one-shot: batch-import every game already in the log, headless and at
        # full speed (no XMage, no pacing) — a fast way to record past games
        self.import_btn = ttk.Button(btns, text="Process existing log",
                                     command=self.import_existing)
        self.import_btn.pack(side=tk.LEFT)
        ttk.Button(btns, text="Open output folder",
                   command=self._open_output).pack(side=tk.LEFT, padx=8)

        panes = tk.PanedWindow(frm, orient=tk.HORIZONTAL, sashwidth=8,
                               bg=Palette.BG, bd=0)
        panes.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.log_view = self._log_pane(panes, "Log / status")
        self.action_view = self._log_pane(panes, "Actions (recorded decisions)")

    def _log_pane(self, panes, title):
        wrap = ttk.Frame(panes, style="Card.TFrame")
        ttk.Label(wrap, text=title, style="Card.TLabel",
                  font=self.font_h2).pack(fill=tk.X, padx=10, pady=(8, 4))
        view = scrolledtext.ScrolledText(
            wrap, height=16, wrap=tk.WORD, bd=0, relief=tk.FLAT,
            bg=Palette.PANEL, fg=Palette.TEXT, insertbackground=Palette.TEXT,
            font=self.font_mono, padx=10, pady=6)
        view.configure(highlightthickness=0)
        view.pack(fill=tk.BOTH, expand=True, padx=2, pady=(0, 4))
        panes.add(wrap, stretch="always", minsize=260)
        return view

    def _build_draft_tab(self, frm):
        header = ttk.Frame(frm)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Live draft", style="H1.TLabel").pack(
            side=tk.LEFT)
        self.draft_status_var = tk.StringVar(
            value="No draft seen yet — start following on the Follow tab.")
        ttk.Label(frm, textvariable=self.draft_status_var,
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 14))

        panes = tk.PanedWindow(frm, orient=tk.HORIZONTAL, sashwidth=8,
                               bg=Palette.BG, bd=0)
        panes.pack(fill=tk.BOTH, expand=True)

        pack_wrap = ttk.Frame(panes, style="Card.TFrame")
        ttk.Label(pack_wrap, text="Current pack", style="Card.TLabel",
                  font=self.font_h2).pack(fill=tk.X, padx=10, pady=(8, 4))
        self.draft_pack_table = ttk.Treeview(
            pack_wrap, columns=("rank", "card", "score"), show="headings",
            height=15)
        for column, heading, width, anchor, stretch in (
                ("rank", "#", 36, tk.CENTER, False),
                ("card", "Card", 260, tk.W, True),
                ("score", "Score", 70, tk.CENTER, False)):
            self.draft_pack_table.heading(column, text=heading)
            self.draft_pack_table.column(column, width=width, anchor=anchor,
                                         stretch=stretch)
        self.draft_pack_table.tag_configure("odd", background=Palette.PANEL)
        self.draft_pack_table.tag_configure("even",
                                            background=Palette.PANEL_ALT)
        self.draft_pack_table.tag_configure("best",
                                            foreground=Palette.SUCCESS)
        self.draft_pack_table.pack(fill=tk.BOTH, expand=True, padx=2,
                                   pady=(0, 4))
        panes.add(pack_wrap, stretch="always", minsize=280)

        self.draft_outlook_view = self._log_pane(panes, "Deck outlook")

    def _update_draft_pane(self, payload):
        kind = payload.get("kind")
        draft = payload.get("draft") or {}
        pool_size = len(draft.get("pool") or [])
        if kind == "deck_submit":
            self.draft_status_var.set(
                "Deck submitted for %s." % (payload.get("eventName"),))
        else:
            self.draft_status_var.set(
                "Pack %s, pick %s — pool %d cards%s" % (
                    draft.get("packNumber"), draft.get("pickNumber"),
                    pool_size,
                    "" if payload.get("scored") else
                    " (no draft model — showing names only)"))
        self.draft_pack_table.delete(
            *self.draft_pack_table.get_children())
        for index, entry in enumerate(payload.get("pack") or []):
            score = entry.get("score")
            tags = ["odd" if index % 2 else "even"]
            if index == 0 and payload.get("scored"):
                tags.append("best")
            self.draft_pack_table.insert(
                "", tk.END, tags=tags,
                values=(index + 1, entry.get("name"),
                        "%.2f" % score if score is not None else "—"))
        if payload.get("error"):
            self.draft_status_var.set(
                self.draft_status_var.get() +
                " — advisor error: %s" % payload["error"])
        outlook = payload.get("outlook")
        if outlook is not None:
            self.draft_outlook_view.delete("1.0", tk.END)
            self._append(self.draft_outlook_view, _outlook_text(outlook))

    def _build_replays_tab(self, frm):
        header = ttk.Frame(frm)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Replay library", style="H1.TLabel").pack(
            side=tk.LEFT)
        ttk.Button(header, text="Refresh",
                   command=self._refresh_replays).pack(side=tk.RIGHT)
        ttk.Button(header, text="Browse folder…",
                   command=self._pick_runs_dir).pack(side=tk.RIGHT, padx=8)

        columns = ("title", "result", "format", "games", "decisions", "date")
        headings = (("title", "Match", 360), ("result", "Result", 90),
                    ("format", "Event", 130), ("games", "Games", 64),
                    ("decisions", "Decisions", 90), ("date", "Played", 130))
        table_wrap = ttk.Frame(frm, style="Card.TFrame")
        table_wrap.pack(fill=tk.BOTH, expand=True, pady=(12, 8))
        self.replay_table = ttk.Treeview(table_wrap, columns=columns,
                                         show="headings", height=10)
        for column, heading, width in headings:
            anchor = tk.W if column in ("title",) else tk.CENTER
            self.replay_table.heading(column, text=heading)
            self.replay_table.column(column, width=width, anchor=anchor,
                                     stretch=(column == "title"))
        self.replay_table.tag_configure("odd", background=Palette.PANEL)
        self.replay_table.tag_configure("even", background=Palette.PANEL_ALT)
        self.replay_table.tag_configure(
            "win", foreground=Palette.SUCCESS)
        self.replay_table.tag_configure(
            "loss", foreground=Palette.DANGER)
        self.replay_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                               padx=(2, 0), pady=2)
        scroll = ttk.Scrollbar(table_wrap, orient=tk.VERTICAL,
                               command=self.replay_table.yview)
        self.replay_table.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.replay_table.bind("<Double-1>", lambda e: self.watch_replay())
        self.replay_table.bind("<<TreeviewSelect>>", self._on_replay_selected)

        self._build_matchup_strip(frm)
        self._build_transport(frm)

        self._runs_dir = DEFAULT_RUNS_DIR
        self._replay_paths = {}
        self._replay_meta = {}
        self._refresh_replays()

    def _build_matchup_strip(self, frm):
        # 17Lands-style: the selected match's decks shown as colored mana pips
        strip = ttk.Frame(frm, style="Card.TFrame")
        strip.pack(fill=tk.X, pady=(0, 8))
        row = ttk.Frame(strip, style="Card.TFrame")
        row.pack(fill=tk.X, padx=12, pady=9)

        self.you_label = ttk.Label(row, text="You", style="CardMuted.TLabel")
        self.you_label.pack(side=tk.LEFT)
        self.you_pips = ManaPips(row, bg=Palette.PANEL)
        self.you_pips.pack(side=tk.LEFT, padx=(8, 6))
        self.you_arch = ttk.Label(row, text="—", style="Card.TLabel",
                                  font=self.font_h2)
        self.you_arch.pack(side=tk.LEFT)

        ttk.Label(row, text="vs", style="CardMuted.TLabel").pack(
            side=tk.LEFT, padx=14)

        self.opp_pips = ManaPips(row, bg=Palette.PANEL)
        self.opp_pips.pack(side=tk.LEFT, padx=(0, 6))
        self.opp_arch = ttk.Label(row, text="—", style="Card.TLabel",
                                  font=self.font_h2)
        self.opp_arch.pack(side=tk.LEFT)

        self.matchup_result = ttk.Label(row, text="", style="Card.TLabel")
        self.matchup_result.pack(side=tk.RIGHT)

    def _build_transport(self, frm):
        bar = ttk.Frame(frm, style="Card.TFrame")
        bar.pack(fill=tk.X)
        inner = ttk.Frame(bar, style="Card.TFrame")
        inner.pack(fill=tk.X, padx=10, pady=10)

        # start / stop the whole replay session
        self.watch_btn = ttk.Button(inner, text="▶  Watch",
                                    style="Accent.TButton",
                                    command=self.watch_replay)
        self.watch_btn.pack(side=tk.LEFT, padx=(0, 12))

        # the transport controls, enabled only while a replay is loaded
        self._transport_buttons = []
        self.playpause_btn = self._transport_button(
            inner, "⏸  Pause", "play / pause", self._toggle_play, width=9)
        for text, tip, command in (
                ("⏮", "step back one state", lambda: self._transport("step", -1)),
                ("⏭", "step forward one state", lambda: self._transport("step", 1)),
                ("↷ Next", "jump to the next action",
                 lambda: self._transport("jump", False)),
                ("⇥ Next play", "skip passes to the next real action",
                 lambda: self._transport("jump", True)),
        ):
            self._transport_button(inner, text, tip, command)

        # readout (far right) + speed presets
        self.replay_readout_var = tk.StringVar(value="")
        ttk.Label(inner, textvariable=self.replay_readout_var,
                  style="Card.TLabel").pack(side=tk.RIGHT)

        ttk.Label(inner, text="Speed", style="CardMuted.TLabel").pack(
            side=tk.LEFT, padx=(16, 6))
        self.replay_speed = 4.0
        self._speed_buttons = {}
        for value in (1, 2, 4, 8, 16):
            button = ttk.Button(inner, text="%d×" % value, width=3,
                                style="Transport.TButton",
                                command=lambda v=value: self._set_speed(v))
            button.pack(side=tk.LEFT, padx=(0, 4))
            self._speed_buttons[value] = button

        # scrubber (with turn ticks) + status line
        scrub_row = ttk.Frame(bar, style="Card.TFrame")
        scrub_row.pack(fill=tk.X, padx=10, pady=(0, 8))
        self.scrubber = ReplayScrubber(
            scrub_row, bg=Palette.PANEL, on_seek=self._on_scrub_seek,
            on_preview=self._on_scrub_preview)
        self.scrubber.pack(fill=tk.X)
        self.replay_status_var = tk.StringVar(value="Select a replay to watch.")
        ttk.Label(scrub_row, textvariable=self.replay_status_var,
                  style="CardMuted.TLabel").pack(anchor="w", pady=(4, 0))
        self._highlight_speed(self.replay_speed)

    def _transport_button(self, parent, text, tip, command, width=None):
        button = ttk.Button(parent, text=text, style="Transport.TButton",
                            command=command, state=tk.DISABLED)
        if width is not None:
            button.configure(width=width)
        button.pack(side=tk.LEFT, padx=(0, 8))
        _Tooltip(button, tip)
        self._transport_buttons.append(button)
        return button

    def _path_row(self, parent, label, var, button_text, command):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(row, text=button_text, command=command).pack(side=tk.LEFT)

    # ------------------------------------------------------------- replays
    def _on_tab_changed(self, event):
        try:
            selected = self._notebook.tab(self._notebook.select(), "text")
        except tk.TclError:
            return
        if selected.strip() == "Replays":
            self._refresh_replays()

    def _pick_runs_dir(self):
        path = filedialog.askdirectory(
            title="Folder containing recorded replays")
        if path:
            self._runs_dir = path
            self._refresh_replays()

    def _refresh_replays(self):
        self.replay_table.delete(*self.replay_table.get_children())
        self._replay_paths = {}
        self._replay_meta = {}
        rows = []
        for path in self._discover_bundles(self._runs_dir):
            summary = _load_summary(path)
            rows.append((path, summary))
        # newest first, by recorded start time when available
        rows.sort(key=lambda item: _sort_key(item[1], item[0]), reverse=True)
        for i, (path, summary) in enumerate(rows):
            values, tags = self._row_for(path, summary, i)
            item = self.replay_table.insert("", tk.END, values=values,
                                            tags=tags)
            self._replay_paths[item] = path
            self._replay_meta[item] = summary
        self.replay_status_var.set(
            "%d replay(s) in %s" % (len(rows), self._runs_dir))

    def _row_for(self, path, summary, index):
        title = summary.get("title")
        if not title:
            title = os.path.relpath(path, self._runs_dir)
            if title == ".":
                title = os.path.basename(os.path.abspath(path))
        result = (summary.get("result") or "").capitalize() or "—"
        record = summary.get("gameRecord")
        if record and result != "—":
            result = "%s %s" % (result, record)
        event = summary.get("eventName") or "—"
        games = len(summary.get("games", [])) or "—"
        decisions = summary.get("decisions", "—")
        date = _format_date(summary.get("startTime"))
        tags = ["even" if index % 2 else "odd"]
        outcome = summary.get("result")
        if outcome in ("win", "loss"):
            tags.append(outcome)
        return (title, result, event, games, decisions, date), tuple(tags)

    def _on_replay_selected(self, event=None):
        selection = self.replay_table.selection()
        if not selection:
            return
        summary = self._replay_meta.get(selection[0]) or {}
        you = summary.get("you") or {}
        opp = summary.get("opponent") or {}
        # colored mana pips + archetype for each side (17Lands style)
        self.you_pips.set_colors(you.get("colors"))
        self.opp_pips.set_colors(opp.get("colors"))
        self.you_arch.config(text=_side_text(you))
        self.opp_arch.config(text=_side_text(opp))
        result = (summary.get("result") or "").capitalize()
        record = summary.get("gameRecord")
        if result and record:
            result = "%s %s" % (result, record)
        color = {"Win": Palette.SUCCESS, "Loss": Palette.DANGER}.get(
            result.split(" ")[0] if result else "", Palette.MUTED)
        self.matchup_result.config(text=result or "", foreground=color)
        if not self._replay_active():
            self.replay_status_var.set("Ready to watch.")

    @staticmethod
    def _discover_bundles(runs_dir):
        """A bundle is any directory containing mirror_states.jsonl. Look at
        the runs dir itself and one level down (the usual <runs>/<name> layout)."""
        found = []
        if not os.path.isdir(runs_dir):
            return found
        if os.path.isfile(os.path.join(runs_dir, "mirror_states.jsonl")):
            found.append(runs_dir)
        for entry in sorted(os.listdir(runs_dir)):
            path = os.path.join(runs_dir, entry)
            if os.path.isfile(os.path.join(path, "mirror_states.jsonl")):
                found.append(path)
        return found

    # ------------------------------------------------------------- transport
    def watch_replay(self):
        if self._replay_active():
            self._transport("stop")
            return
        selection = self.replay_table.selection()
        if not selection:
            self.replay_status_var.set("Select a replay first.")
            return
        bundle = self._replay_paths.get(selection[0])
        if not bundle:
            return
        self.watch_btn.config(state=tk.DISABLED)
        self.replay_status_var.set("Opening XMage for replay…")
        threading.Thread(target=self._launch_replay, args=(bundle,),
                         daemon=True).start()

    def _launch_replay(self, bundle):
        try:
            display = self._get_display()
        except Exception as error:
            self._post(("replay_status", "Cannot open XMage: %s" % error))
            self._post(("replay_reset", None))
            return
        try:
            controller = ReplayController(
                bundle, display=display, speed=self.replay_speed,
                on_progress=lambda info: self._post(("replay_progress", info)),
                on_message=lambda text: self._post(("log", text)))
        except Exception as error:
            self._post(("replay_status", "Replay error: %s" % error))
            self._post(("replay_reset", None))
            return
        self._replay_controller = controller
        self._replay_total = max(1, controller.total - 1)
        self._post(("replay_ready", os.path.basename(bundle)))
        controller.start()
        controller.play()

    def _toggle_play(self):
        controller = self._replay_controller
        if controller is None:
            return
        if self._replay_at_end:
            controller.seek(0)
            controller.play()
        else:
            controller.toggle()

    def _transport(self, action, arg=None):
        controller = self._replay_controller
        if controller is None:
            return
        if action == "toggle":
            controller.toggle()
        elif action == "step":
            controller.step(arg)
        elif action == "jump":
            controller.next_decision(meaningful=arg)
        elif action == "stop":
            controller.stop()
            self._replay_controller = None
            self._post(("replay_reset", None))

    def _set_speed(self, value):
        self.replay_speed = float(value)
        self._highlight_speed(value)
        if self._replay_controller is not None:
            self._replay_controller.set_speed(float(value))

    def _highlight_speed(self, value):
        for preset, button in self._speed_buttons.items():
            button.configure(style="Accent.TButton" if preset == value
                             else "Transport.TButton")

    def _on_scrub_preview(self, index):
        # live position while dragging; the actual seek happens on release
        self.replay_readout_var.set(
            "%d / %d   ·   release to seek" % (index + 1, self._replay_total + 1))

    def _on_scrub_seek(self, index):
        if self._replay_controller is not None:
            self._replay_controller.seek(index)

    def _replay_active(self):
        return self._replay_controller is not None

    # ------------------------------------------------------------- actions
    def _pick_log(self):
        initial = os.path.dirname(self.log_var.get()) or "/"
        path = filedialog.askopenfilename(
            title="Locate MTGA Player.log",
            initialdir=initial if os.path.isdir(initial) else "/",
            filetypes=[("Player log", "Player*.log"), ("Log files", "*.log"),
                       ("All files", "*.*")])
        if path:
            self.log_var.set(path)

    def _pick_out(self):
        path = filedialog.askdirectory(title="Output folder")
        if path:
            self.out_var.set(path)

    def _open_output(self):
        path = self.out_var.get()
        if os.path.isdir(path):
            try:
                os.startfile(path)  # noqa: E1101 (Windows only)
            except Exception as error:
                self._log("Could not open folder: %s" % error)
        else:
            self._log("Output folder does not exist yet.")

    def start(self):
        if self._thread and self._thread.is_alive():
            self._log("Already following.")
            return
        log_path = self.log_var.get()
        if not os.path.exists(log_path):
            self._log("Player.log not found at: %s" % log_path)
            self._log("Set MTGA to Detailed Logs (Plugin Support) and start a "
                      "game, then click Locate.")
            return
        self._stop_requested = False
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._set_status("Following", Palette.ACCENT)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_requested = True
        self._set_status("Stopping…", Palette.WARN)
        if self._session is not None:
            self._session.stop()

    def import_existing(self):
        """Batch-record every game already in the log — headless, full speed."""
        if self._import_thread and self._import_thread.is_alive():
            return
        if self._thread and self._thread.is_alive():
            self._log("Stop live following before importing the existing log.")
            return
        log_path = self.log_var.get()
        if not os.path.exists(log_path):
            self._log("Player.log not found at: %s" % log_path)
            return
        self.import_btn.config(state=tk.DISABLED, text="Processing…")
        self.start_btn.config(state=tk.DISABLED)
        self._set_status("Importing…", Palette.WARN)
        self._import_thread = threading.Thread(
            target=self._run_import, args=(log_path,), daemon=True)
        self._import_thread.start()

    def _run_import(self, log_path):
        import datetime
        from ..arena_log import iter_log_entries

        runs_root = os.path.dirname(self.out_var.get()) or DEFAULT_RUNS_DIR
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(runs_root, "imported-%s" % stamp)
        try:
            card_db = CardDatabase()
        except Exception as error:
            self._post(("log", "No card DB (%s); cards render face-down."
                        % error))
            card_db = None

        recorder = MirrorRecorder(out_dir, card_db=card_db)
        # headless: no display_factory, so XMage never opens; batch feed has no
        # pacing, so it runs as fast as the log parses
        session = MirrorSession(
            recorder=recorder, card_db=card_db, verbose=False,
            on_game=lambda kind, match_id, game: self._post(
                ("log", "• %s (match=%s game=%s)" % (kind, match_id, game))))
        self._post(("log", "Processing existing log → %s"
                    % os.path.abspath(out_dir)))
        try:
            with open(log_path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            session.feed_entries(iter_log_entries(text.splitlines(True)))
            self._post(("log", "Done: %d states, %d decisions across %d game(s)."
                        % (recorder.counts["mirrorStates"],
                           recorder.counts["decisions"],
                           len(recorder.counts["games"]))))
        except Exception as error:
            self._post(("log", "Import error: %s" % error))
        finally:
            recorder.close()
            self._post(("import_done", out_dir))

    # ------------------------------------------------------------- worker
    def _run(self):
        out_dir = self.out_var.get()
        try:
            card_db = CardDatabase()
        except Exception as error:
            self._post(("log", "No card DB (%s); cards render face-down."
                        % error))
            card_db = None

        recorder = MirrorRecorder(out_dir, card_db=card_db)
        display_factory = self._get_display if self.display_var.get() else None
        advisor = DraftAdvisor.maybe_load(
            card_db, log=lambda text: self._post(("log", text)))

        session = MirrorSession(
            recorder=recorder, display_factory=display_factory,
            card_db=card_db, verbose=False,
            on_status=lambda text: self._post(("log", text)),
            on_action=lambda line, record: self._post(("action", line)),
            on_game=lambda kind, match_id, game: self._post(
                ("log", "• %s (match=%s game=%s)" % (kind, match_id, game))),
            on_draft=lambda kind, draft, event: self._post(
                ("draft", self._draft_payload(advisor, card_db, kind,
                                              draft, event))))
        self._session = session
        self._post(("log", "Recording to %s" % os.path.abspath(out_dir)))
        try:
            session.follow(self.log_var.get(), from_start=False)
        except Exception as error:
            self._post(("log", "Follow error: %s" % error))
        finally:
            recorder.close()
            session.display = None
            self._post(("done", None))

    def _get_display(self):
        """Return the shared, GUI-owned XMage display, launching it (or
        relaunching it if the user closed the window) on demand.

        java, the classpath, and the working directory are resolved here so the
        board opens even when the GUI wasn't started through the launch script
        (which is otherwise what wires them up)."""
        with self._display_lock:
            if self._display is not None and self._display.alive:
                return self._display
            java = self._resolve_java()
            if java is None:
                raise RuntimeError(
                    "Java 17 wasn't found. Install a JDK 17 and set JAVA_HOME, "
                    "or start the app with scripts\\arena-mirror-gui.ps1.")
            classpath = self._resolve_classpath()
            if not classpath:
                raise RuntimeError(
                    "XMage isn't built yet — run scripts\\setup-arena-mirror.ps1 "
                    "once to enable the replay board.")
            self._post(("log", "Launching XMage window…"))
            self._display = MirrorDisplay(classpath=classpath, java=java,
                                          cwd=self._display_cwd(classpath))
            self._display.ping()
            return self._display

    def _resolve_java(self):
        """A launchable java executable, or None. Tries the configured value,
        JAVA_HOME, a portable JDK under ~\\tools, then PATH."""
        candidates = [self.java]
        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            candidates.append(os.path.join(java_home, "bin", "java.exe"))
            candidates.append(os.path.join(java_home, "bin", "java"))
        candidates.extend(sorted(glob.glob(os.path.expanduser(
            r"~\tools\jdk-*\bin\java.exe")), reverse=True))
        candidates.append("java")
        for candidate in candidates:
            if not candidate:
                continue
            if os.path.isfile(candidate):
                return candidate
            found = shutil.which(candidate)
            if found:
                return found
        return None

    def _resolve_classpath(self):
        """The XMage display classpath: the configured value, the env var, or
        the file setup writes under the default XMage checkout."""
        if self.classpath:
            return self.classpath
        env = os.environ.get(CLASSPATH_ENV_VAR)
        if env:
            return env
        cp_file = os.path.join(DEFAULT_XMAGE_DIR, "Mage.Client", "target",
                               "mirror-classpath.txt")
        if os.path.isfile(cp_file):
            with open(cp_file, encoding="utf-8") as handle:
                return handle.read().strip()
        return None

    @staticmethod
    def _display_cwd(classpath):
        """XMage must run from its Mage.Client dir to find its card db/images;
        derive it from the classpath (or the default checkout)."""
        for part in (classpath or "").split(os.pathsep):
            index = part.lower().find("mage.client")
            if index >= 0:
                base = part[:index + len("mage.client")]
                if os.path.isdir(base):
                    return base
        base = os.path.join(DEFAULT_XMAGE_DIR, "Mage.Client")
        return base if os.path.isdir(base) else None

    @staticmethod
    def _draft_payload(advisor, card_db, kind, draft, event):
        """Build the draft-pane update on the session thread, so model
        scoring never runs on (or blocks) the Tk event loop."""
        payload = {"kind": kind, "draft": draft, "scored": False}
        if kind == "deck_submit":
            payload["eventName"] = event.get("eventName")

        def name_for(grp_id):
            if advisor is not None:
                return advisor.name_for(grp_id)
            if card_db is not None:
                try:
                    return card_db.name_for(grp_id) or str(grp_id)
                except Exception:
                    return str(grp_id)
            return str(grp_id)

        pack = draft.get("packCards") or []
        payload["pack"] = [{"grpId": grp_id, "name": name_for(grp_id),
                            "score": None} for grp_id in pack]
        if advisor is not None:
            try:
                if pack:
                    payload["pack"] = advisor.score_pack(draft)
                    payload["scored"] = True
                if kind in ("pick", "deck_submit") and draft.get("pool"):
                    payload["outlook"] = advisor.outlook(draft["pool"])
            except Exception as error:
                payload["error"] = str(error)
        return payload

    # ------------------------------------------------------------- Tk queue
    def _post(self, item):
        self._queue.put(item)

    def _drain_queue(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                self._handle_event(kind, payload)
        except queue.Empty:
            pass
        self.root.after(80, self._drain_queue)

    def _handle_event(self, kind, payload):
        if kind == "log":
            self._log(payload)
        elif kind == "action":
            self._append(self.action_view, payload)
        elif kind == "draft":
            self._update_draft_pane(payload)
        elif kind == "done":
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self._set_status("Stopped", Palette.MUTED)
            if self._display is not None and self._display.alive:
                self._log("Stopped following. XMage stays open — close its "
                          "window or this app to end.")
        elif kind == "import_done":
            self.import_btn.config(state=tk.NORMAL, text="Process existing log")
            self.start_btn.config(state=tk.NORMAL)
            self._set_status("Idle", Palette.MUTED)
            if os.path.isdir(payload):
                self._runs_dir = os.path.dirname(payload) or self._runs_dir
                self._refresh_replays()
            self._log("Imported replay is in the Replays tab.")
        elif kind == "replay_status":
            self.replay_status_var.set(payload)
        elif kind == "replay_ready":
            self._set_transport_enabled(True)
            self.watch_btn.config(text="⏹  Stop", state=tk.NORMAL)
            controller = self._replay_controller
            if controller is not None:
                self.scrubber.configure_marks(controller.total,
                                              controller.turn_marks)
            self.scrubber.set_enabled(True)
            self.replay_status_var.set("Playing %s" % payload)
        elif kind == "replay_reset":
            self._set_transport_enabled(False)
            self.watch_btn.config(text="▶  Watch", state=tk.NORMAL)
            self.scrubber.set_enabled(False)
            self._replay_controller = None
        elif kind == "replay_progress":
            self._update_progress(payload)

    def _update_progress(self, info):
        self.scrubber.set_position(info.get("index", 0))
        playing = info.get("playing")
        at_end = info.get("atEnd")
        self._replay_at_end = bool(at_end) and not playing
        self.playpause_btn.config(
            text="⏸  Pause" if playing else (
                "↺  Replay" if at_end else "▶  Play"))
        bits = ["%d / %d" % (info.get("index", 0) + 1, info.get("total", 0))]
        if info.get("turn"):
            active = info.get("activeName")
            turn = "Turn %s" % info["turn"]
            bits.append("%s (%s)" % (turn, active) if active else turn)
        if info.get("phase"):
            bits.append(info["phase"])
        self.replay_readout_var.set("   ·   ".join(bits))
        state = "playing" if info.get("playing") else (
            "finished" if info.get("atEnd") else "paused")
        line = "Replay %s" % state
        if info.get("decision"):
            line = "%s   —   %s" % (line, info["decision"])
        self.replay_status_var.set(line)

    def _set_transport_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in self._transport_buttons:
            button.config(state=state)

    def _log(self, text):
        self._append(self.log_view, text)

    @staticmethod
    def _append(view, text):
        view.insert(tk.END, text + "\n")
        view.see(tk.END)

    def _set_status(self, text, color):
        self.status_var.set(text)
        if hasattr(self, "_status_chip"):
            self._status_chip.config(fg=color)

    def _on_close(self):
        # closing the GUI ends everything: stop following, stop any replay,
        # and close the XMage window the user left open.
        self._stop_requested = True
        if self._session is not None:
            self._session.stop()
        controller = getattr(self, "_replay_controller", None)
        if controller is not None:
            try:
                controller.stop()
            except Exception:
                pass
        with self._display_lock:
            if self._display is not None:
                try:
                    self._display.close()
                except Exception:
                    pass
                self._display = None
        self.root.after(200, self.root.destroy)


class ManaPips(tk.Canvas):
    """A row of colored WUBRG pips, the way 17Lands shows a deck's colors."""

    # (fill, outline) per color — tuned to read on the dark panel
    _COLORS = {
        "W": ("#f3efd8", "#c9c3a2"),
        "U": ("#3f86c9", "#2b6197"),
        "B": ("#3d3d46", "#7f7f8b"),
        "R": ("#d24d43", "#9c332c"),
        "G": ("#53a463", "#3a7746"),
    }

    def __init__(self, parent, bg, diameter=15, gap=3):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0,
                         height=diameter + 3, width=1)
        self._d = diameter
        self._gap = gap

    def set_colors(self, colors):
        self.delete("all")
        letters = [c for c in "WUBRG" if c in (colors or "")]
        d, gap = self._d, self._gap
        self.configure(width=max(1, len(letters) * (d + gap) - gap) if letters
                       else 1)
        x = 0
        for letter in letters:
            fill, outline = self._COLORS[letter]
            self.create_oval(x + 1, 2, x + d, d + 1, fill=fill,
                             outline=outline, width=1)
            x += d + gap


class ReplayScrubber(tk.Canvas):
    """A seek bar that draws a tick at every turn boundary.

    Seeks on release (not on every drag pixel, which would flood the engine).
    ``on_preview(index)`` fires live while dragging so the UI can show the
    position; ``on_seek(index)`` fires once, on release.
    """

    _PAD = 10

    def __init__(self, parent, bg, on_seek=None, on_preview=None, height=34):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0,
                         height=height)
        self._on_seek = on_seek
        self._on_preview = on_preview
        self._total = 1
        self._marks = []
        self._pos = 0.0
        self._enabled = False
        self._dragging = False
        self.bind("<Configure>", lambda e: self._redraw())
        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)

    def set_enabled(self, enabled):
        self._enabled = enabled
        self._redraw()

    def configure_marks(self, total, marks):
        self._total = max(1, (total or 1) - 1)
        self._marks = marks or []
        self._redraw()

    def set_position(self, index):
        if self._dragging:
            return
        self._pos = max(0.0, min(1.0, index / self._total))
        self._redraw()

    # --- interaction ---
    def _press(self, event):
        if not self._enabled:
            return
        self._dragging = True
        self._update_from_x(event.x, preview=True)

    def _drag(self, event):
        if self._dragging:
            self._update_from_x(event.x, preview=True)

    def _release(self, event):
        if not self._dragging:
            return
        self._dragging = False
        self._update_from_x(event.x, preview=False)
        if self._on_seek is not None:
            self._on_seek(int(round(self._pos * self._total)))

    def _update_from_x(self, x, preview):
        width = self.winfo_width()
        span = max(1, width - 2 * self._PAD)
        self._pos = max(0.0, min(1.0, (x - self._PAD) / span))
        self._redraw()
        if preview and self._on_preview is not None:
            self._on_preview(int(round(self._pos * self._total)))

    # --- drawing ---
    def _redraw(self):
        self.delete("all")
        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 1:
            return
        pad = self._PAD
        mid = height // 2
        right = width - pad
        track = Palette.PANEL_ALT if self._enabled else Palette.BORDER
        # base track
        self.create_line(pad, mid, right, mid, fill=track, width=5,
                         capstyle=tk.ROUND)
        # turn ticks
        tick = Palette.MUTED if self._enabled else Palette.BORDER
        for mark in self._marks:
            tx = pad + (mark["index"] / self._total) * (right - pad)
            self.create_line(tx, mid - 8, tx, mid + 8, fill=tick, width=1)
        # progress fill
        x = pad + self._pos * (right - pad)
        if self._enabled:
            self.create_line(pad, mid, x, mid, fill=Palette.ACCENT, width=5,
                             capstyle=tk.ROUND)
            self.create_oval(x - 7, mid - 7, x + 7, mid + 7,
                             fill=Palette.ACCENT, outline="#ffffff", width=1)


class _Tooltip(object):
    """A tiny hover tooltip for the icon-only transport buttons."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self._tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        if self._tip is not None:
            return
        x = self.widget.winfo_rootx() + 8
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry("+%d+%d" % (x, y))
        tk.Label(self._tip, text=self.text, bg=Palette.PANEL_ALT,
                 fg=Palette.TEXT, padx=8, pady=3,
                 font=("Segoe UI", 9)).pack()

    def _hide(self, _event=None):
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


# --- module helpers ----------------------------------------------------------
def _load_summary(path):
    summary_path = os.path.join(path, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            pass
    return {}


def _sort_key(summary, path):
    start = summary.get("startTime")
    if start:
        return start
    try:
        return "mtime:%f" % os.path.getmtime(path)
    except OSError:
        return ""


def _format_date(iso):
    if not iso:
        return "—"
    import datetime
    try:
        dt = datetime.datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return "—"
    return dt.strftime("%b %d, %H:%M")


def _side_text(side):
    archetype = side.get("archetype")
    name = side.get("name")
    if archetype and name:
        return "%s (%s)" % (archetype, name)
    return archetype or name or "?"


def _outlook_text(outlook):
    """Render a deck-outlook dict (analysis.draft_outlook) as pane text."""
    lines = ["Auto-built deck (%d of %d spells, pool %d):" % (
        len(outlook.get("deck") or []), outlook.get("spellTarget") or 0,
        outlook.get("poolSize") or 0)]
    for index, entry in enumerate(outlook.get("deck") or []):
        lines.append("%2d. %s" % (index + 1,
                                  entry.get("name") or entry.get("grpId")))
    colors = outlook.get("colors") or {}
    if colors:
        lines.append("")
        lines.append("Colors: " + "  ".join(
            "%s×%d" % (letter, count) for letter, count in colors.items()))
    curve = outlook.get("curve") or {}
    if curve:
        lines.append("Curve:  " + "  ".join(
            "%s:%d" % (cost, count) for cost, count in curve.items()))
    cuts = outlook.get("cuts") or []
    if cuts:
        lines.append("")
        lines.append("Cuts: " + ", ".join(
            str(entry.get("name") or entry.get("grpId"))
            for entry in cuts[:12]))
    return "\n".join(lines)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m magic_cabt.arena_mirror gui",
        description="Launcher GUI for the Arena -> XMage mirror.")
    parser.add_argument("--classpath", default=None,
                        help="Java classpath for the XMage display "
                             "(default: $MAGIC_CABT_CLASSPATH)")
    parser.add_argument("--java", default="java", help="java executable")
    args = parser.parse_args(argv)

    root = tk.Tk()
    ArenaMirrorApp(root, classpath=args.classpath, java=args.java)
    root.mainloop()
    return 0


if __name__ == "__main__":
    main()
