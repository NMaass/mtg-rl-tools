"""Tkinter launcher GUI for the Arena -> XMage mirror.

A slight adaptation of the rconroy293/mtga-log-client tracker UI: a path row
for the MTGA ``Player.log`` with a "Locate MTGA logs" browse button, an
output-directory row, start/stop controls, and two live panes — a status/log
feed and an actions (decisions) feed. When following starts and live gameplay
appears in the log, it launches the XMage display and follows the current
game; recording to a CABT bundle happens the whole time.
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext

from .cards import CardDatabase
from .mirror import MirrorDisplay
from .recorder import MirrorRecorder
from .session import MirrorSession

DEFAULT_LOG_PATH = os.path.expanduser(
    r"~\AppData\LocalLow\Wizards Of The Coast\MTGA\Player.log")


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

        self._build_layout()
        self.root.after(100, self._drain_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------- layout
    def _build_layout(self):
        frm = tk.Frame(self.root, padx=10, pady=10)
        frm.pack(fill=tk.BOTH, expand=True)

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

    def _path_row(self, parent, label, var, button_text, command):
        row = tk.Frame(parent, pady=2)
        row.pack(fill=tk.X)
        tk.Label(row, text=label, width=16, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(row, text=button_text, command=command).pack(
            side=tk.LEFT, padx=4)

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
