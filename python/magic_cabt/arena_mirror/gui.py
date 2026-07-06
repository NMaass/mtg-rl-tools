"""Tkinter launcher GUI for the Arena -> XMage mirror.

A slight adaptation of the rconroy293/mtga-log-client tracker UI: a path row
for the MTGA ``Player.log`` with a "Locate MTGA logs" browse button, an
output-directory row, start/stop controls, and two live panes — a status/log
feed and an actions (decisions) feed. When following starts and live gameplay
appears in the log, it launches the XMage display and follows the current
game; recording to a CABT bundle happens the whole time.
"""

import json
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

from .cards import CardDatabase
from .mirror import MirrorDisplay
from .recorder import MirrorRecorder
from .replay import ReplayPlayer, load_bundle
from .session import MirrorSession

DEFAULT_LOG_PATH = os.path.expanduser(
    r"~\AppData\LocalLow\Wizards Of The Coast\MTGA\Player.log")
DEFAULT_RUNS_DIR = "arena-mirror-runs"


class ArenaMirrorApp(object):
    """The follower GUI. All log/session work runs off the Tk thread; UI
    updates are marshalled back through a thread-safe queue drained on the Tk
    event loop."""

    def __init__(self, root, classpath=None, java="java"):
        self.root = root
        self.classpath = classpath
        self.java = java
        root.title("Arena -> XMage Mirror")
        root.geometry("900x620")

        self.log_var = tk.StringVar(value=DEFAULT_LOG_PATH)
        self.out_var = tk.StringVar(value=os.path.join(
            "arena-mirror-runs", "session"))
        self.display_var = tk.BooleanVar(value=True)
        self.from_start_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="idle")

        self._queue = queue.Queue()
        self._session = None
        self._thread = None
        self._stop_requested = False
        # The XMage window is owned by the GUI, not by a single follow run:
        # it opens on first live activity and stays up across Start/Stop until
        # the user closes the XMage window itself or closes this GUI.
        self._display = None
        self._display_lock = threading.Lock()
        self._replay_thread = None

        self._build_layout()
        self.root.after(100, self._drain_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------- layout
    def _build_layout(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        follow_tab = tk.Frame(notebook, padx=10, pady=10)
        replays_tab = tk.Frame(notebook, padx=10, pady=10)
        notebook.add(follow_tab, text="Follow")
        notebook.add(replays_tab, text="Replays")
        self._notebook = notebook
        # refresh the replay list whenever the Replays tab is opened
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._build_follow_tab(follow_tab)
        self._build_replays_tab(replays_tab)

    def _build_follow_tab(self, frm):
        self._path_row(frm, "MTGA Player.log", self.log_var,
                       "Locate MTGA logs", self._pick_log)
        self._path_row(frm, "Output folder", self.out_var,
                       "Choose...", self._pick_out)

        opts = tk.Frame(frm, pady=4)
        opts.pack(fill=tk.X)
        tk.Checkbutton(opts, text="Open XMage on live game",
                       variable=self.display_var).pack(side=tk.LEFT)
        tk.Checkbutton(opts, text="Process existing log first",
                       variable=self.from_start_var).pack(side=tk.LEFT, padx=12)

        btns = tk.Frame(frm, pady=6)
        btns.pack(fill=tk.X)
        self.start_btn = tk.Button(btns, text="Start following",
                                   command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn = tk.Button(btns, text="Stop", command=self.stop,
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Open output folder",
                  command=self._open_output).pack(side=tk.LEFT, padx=4)
        tk.Label(btns, textvariable=self.status_var, fg="#0a6").pack(
            side=tk.RIGHT)

        panes = tk.PanedWindow(frm, orient=tk.HORIZONTAL, sashwidth=6)
        panes.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        left = tk.Frame(panes)
        tk.Label(left, text="Log / status", anchor="w").pack(fill=tk.X)
        self.log_view = scrolledtext.ScrolledText(left, height=20, wrap=tk.WORD)
        self.log_view.pack(fill=tk.BOTH, expand=True)
        panes.add(left, stretch="always")

        right = tk.Frame(panes)
        tk.Label(right, text="Actions (recorded decisions)",
                 anchor="w").pack(fill=tk.X)
        self.action_view = scrolledtext.ScrolledText(right, height=20,
                                                     wrap=tk.WORD)
        self.action_view.pack(fill=tk.BOTH, expand=True)
        panes.add(right, stretch="always")

    def _build_replays_tab(self, frm):
        top = tk.Frame(frm)
        top.pack(fill=tk.X)
        tk.Label(top, text="Recorded replays", anchor="w").pack(side=tk.LEFT)
        tk.Button(top, text="Refresh",
                  command=self._refresh_replays).pack(side=tk.RIGHT)
        tk.Button(top, text="Browse folder...",
                  command=self._pick_runs_dir).pack(side=tk.RIGHT, padx=4)

        # a table of bundles with their summary counts
        columns = ("bundle", "games", "decisions", "states")
        self.replay_table = ttk.Treeview(frm, columns=columns,
                                         show="headings", height=10)
        for column, heading, width in (
                ("bundle", "Replay", 320), ("games", "Games", 70),
                ("decisions", "Decisions", 90), ("states", "States", 80)):
            self.replay_table.heading(column, text=heading)
            self.replay_table.column(column, width=width,
                                     anchor=tk.W if column == "bundle" else tk.CENTER)
        self.replay_table.pack(fill=tk.BOTH, expand=True, pady=(4, 4))
        self.replay_table.bind("<Double-1>", lambda e: self.watch_replay())

        controls = tk.Frame(frm)
        controls.pack(fill=tk.X)
        self.watch_btn = tk.Button(controls, text="Watch replay",
                                   command=self.watch_replay)
        self.watch_btn.pack(side=tk.LEFT, padx=4)
        tk.Label(controls, text="Speed").pack(side=tk.LEFT, padx=(12, 2))
        self.replay_speed_var = tk.StringVar(value="4")
        tk.Spinbox(controls, from_=1, to=50, width=4,
                   textvariable=self.replay_speed_var).pack(side=tk.LEFT)
        self.replay_status_var = tk.StringVar(value="")
        tk.Label(controls, textvariable=self.replay_status_var,
                 fg="#06a").pack(side=tk.RIGHT)

        self._runs_dir = DEFAULT_RUNS_DIR
        self._replay_paths = {}
        self._refresh_replays()

    def _path_row(self, parent, label, var, button_text, command):
        row = tk.Frame(parent, pady=2)
        row.pack(fill=tk.X)
        tk.Label(row, text=label, width=16, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(row, text=button_text, command=command).pack(
            side=tk.LEFT, padx=4)

    # ------------------------------------------------------------- replays
    def _on_tab_changed(self, event):
        try:
            selected = self._notebook.tab(self._notebook.select(), "text")
        except Exception:
            return
        if selected == "Replays":
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
        bundles = self._discover_bundles(self._runs_dir)
        for path in bundles:
            summary = {}
            summary_path = os.path.join(path, "summary.json")
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, encoding="utf-8") as handle:
                        summary = json.load(handle)
                except Exception:
                    pass
            label = os.path.relpath(path, self._runs_dir)
            if label == ".":
                label = os.path.basename(os.path.abspath(path))
            item = self.replay_table.insert("", tk.END, values=(
                label,
                len(summary.get("games", [])) or "?",
                summary.get("decisions", "?"),
                summary.get("mirrorStates", "?")))
            self._replay_paths[item] = path
        self.replay_status_var.set("%d replay(s) in %s"
                                   % (len(bundles), self._runs_dir))

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

    def watch_replay(self):
        if self._replay_thread and self._replay_thread.is_alive():
            self.replay_status_var.set("A replay is already playing.")
            return
        selection = self.replay_table.selection()
        if not selection:
            self.replay_status_var.set("Select a replay first.")
            return
        bundle = self._replay_paths.get(selection[0])
        if not bundle:
            return
        try:
            speed = max(1.0, float(self.replay_speed_var.get()))
        except ValueError:
            speed = 4.0
        self.watch_btn.config(state=tk.DISABLED)
        self._replay_thread = threading.Thread(
            target=self._run_replay, args=(bundle, speed), daemon=True)
        self._replay_thread.start()

    def _run_replay(self, bundle, speed):
        self._post(("replay_status", "Opening XMage for replay..."))
        try:
            display = self._get_display()
        except Exception as error:
            self._post(("replay_status", "Cannot open XMage: %s" % error))
            self._post(("replay_done", None))
            return
        stream = _QueueWriter(self._post)
        player = ReplayPlayer(display=display, interval=1.0 / speed,
                              log_stream=stream)
        try:
            self._post(("replay_status",
                        "Playing %s ..." % os.path.basename(bundle)))
            player.play(bundle)
            self._post(("replay_status", "Replay finished."))
        except Exception as error:
            self._post(("replay_status", "Replay error: %s" % error))
        finally:
            self._post(("replay_done", None))

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
                      "game, then click Locate MTGA logs.")
            return
        self._stop_requested = False
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("following")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_requested = True
        self.status_var.set("stopping...")
        if self._session is not None:
            self._session.stop()

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

        session = MirrorSession(
            recorder=recorder, display_factory=display_factory,
            card_db=card_db, verbose=False,
            on_status=lambda text: self._post(("log", text)),
            on_action=lambda line, record: self._post(("action", line)),
            on_game=lambda kind, match_id, game: self._post(
                ("log", "* %s (match=%s game=%s)" % (kind, match_id, game))))
        self._session = session
        self._post(("log", "Recording to %s" % os.path.abspath(out_dir)))
        try:
            session.follow(self.log_var.get(),
                           from_start=self.from_start_var.get())
        except Exception as error:
            self._post(("log", "Follow error: %s" % error))
        finally:
            # finalize the recording, but leave the XMage window open — it
            # stays until the user closes it (or this GUI). Detach it from the
            # stopped session so a later Start reuses the same window.
            recorder.close()
            session.display = None
            self._post(("done", None))

    def _get_display(self):
        """Return the shared, GUI-owned XMage display, launching it (or
        relaunching it if the user closed the window) on demand. Called by the
        session on the first live board update."""
        with self._display_lock:
            if self._display is not None and self._display.alive:
                return self._display
            self._post(("log", "Launching XMage window..."))
            self._display = MirrorDisplay(classpath=self.classpath,
                                          java=self.java)
            self._display.ping()
            return self._display

    # ------------------------------------------------------------- Tk queue
    def _post(self, item):
        self._queue.put(item)

    def _drain_queue(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "action":
                    self._append(self.action_view, payload)
                elif kind == "done":
                    self.start_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.status_var.set("stopped")
                    if self._display is not None and self._display.alive:
                        self._log("Stopped following. XMage stays open — "
                                  "close its window or this app to end.")
                elif kind == "replay_status":
                    self.replay_status_var.set(payload)
                elif kind == "replay_done":
                    self.watch_btn.config(state=tk.NORMAL)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_queue)

    def _log(self, text):
        self._append(self.log_view, text)

    @staticmethod
    def _append(view, text):
        view.insert(tk.END, text + "\n")
        view.see(tk.END)

    def _on_close(self):
        # closing the GUI is the user's signal to end everything: stop
        # following and close the XMage window they left open.
        self._stop_requested = True
        if self._session is not None:
            self._session.stop()
        with self._display_lock:
            if self._display is not None:
                try:
                    self._display.close()
                except Exception:
                    pass
                self._display = None
        self.root.after(200, self.root.destroy)


class _QueueWriter(object):
    """A minimal text sink that routes ReplayPlayer's narration into the GUI
    log pane, one line per newline, via the thread-safe post callback."""

    def __init__(self, post):
        self._post = post
        self._buffer = ""

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._post(("log", line))

    def flush(self):
        pass


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
