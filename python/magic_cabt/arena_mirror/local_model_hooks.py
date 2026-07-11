"""Opt-in local model integration for the existing mirror and replay UI.

The normal mirror imports remain unchanged. ``magic-cabt-local-model`` calls
these hooks before importing the launcher, so model failures cannot affect
users who run the ordinary mirror command.
"""
from __future__ import annotations

import json
import os

_CORE_INSTALLED = False
_GUI_INSTALLED = False


def install_local_model_hooks():
    """Patch recorder/session/replay classes before the launcher imports them."""
    global _CORE_INSTALLED
    if _CORE_INSTALLED:
        return
    _CORE_INSTALLED = True

    from magic_cabt.analysis import (
        AnalysisCache, AnalysisWorker, decision_fingerprint,
        format_analysis, load_checkpoint_scorer)
    from magic_cabt.training.local_evolve import LocalEvolver
    from . import recorder as recorder_module
    from . import replay as replay_module
    from . import session as session_module

    BaseRecorder = recorder_module.MirrorRecorder
    BaseSession = session_module.MirrorSession
    BasePlayer = replay_module.ReplayPlayer
    BaseController = replay_module.ReplayController

    class LocalModelRecorder(BaseRecorder):
        """Add append-only, checkpoint-versioned ``analysis.jsonl`` output."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._analysis_closed = False
            self._analysis = open(
                os.path.join(self.output_dir, "analysis.jsonl"),
                "a", encoding="utf-8")
            self.counts.setdefault("analysisRecords", 0)
            self.counts.setdefault("analysisModels", [])

        def record_analysis(self, record):
            self._write(self._analysis, record)
            self.counts["analysisRecords"] += 1
            model = record.get("model") or {}
            identity = {
                "modelId": model.get("modelId"),
                "checkpointId": model.get("checkpointId"),
            }
            if identity not in self.counts["analysisModels"]:
                self.counts["analysisModels"].append(identity)

        def flush(self):
            if not self._analysis_closed:
                self._analysis.flush()
            super().flush()

        def close(self):
            if self._analysis_closed:
                return
            self.flush()
            self._analysis.close()
            self._analysis_closed = True
            super().close()

    class LocalModelSession(BaseSession):
        """Queue inference after capture and fine-tune only after game over."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._analysis_worker = None
            self._local_evolver = None
            checkpoint = os.environ.get("MAGIC_CABT_MODEL_CHECKPOINT")
            if not checkpoint:
                return
            checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
            if not os.path.exists(checkpoint):
                self._say("[model] checkpoint not found: %s" % checkpoint)
                return
            try:
                output = getattr(self.recorder, "output_dir", None)
                card_cache = os.path.join(output, "card_cache.json") \
                    if output else None
                arena_card_db = getattr(self.card_db, "source", None)
                scorer = load_checkpoint_scorer(
                    checkpoint,
                    device=os.environ.get("MAGIC_CABT_MODEL_DEVICE") or None,
                    card_cache=card_cache,
                    arena_card_db=arena_card_db)
                cache = AnalysisCache(
                    os.path.join(output, "analysis.jsonl")) if output else None
                self._analysis_worker = AnalysisWorker(
                    scorer, recorder=self.recorder, cache=cache,
                    top_k=int(os.environ.get(
                        "MAGIC_CABT_ANALYSIS_TOP_K", "5")),
                    callback=self._show_analysis,
                    error_callback=lambda text: self._say(
                        "[model] %s" % text))
                auto_train = os.environ.get(
                    "MAGIC_CABT_AUTO_TRAIN", "1").lower()
                if output and auto_train not in ("0", "false", "no", "off"):
                    self._local_evolver = LocalEvolver(
                        model_dir=os.environ.get("MAGIC_CABT_MODEL_DIR"),
                        runs_root=os.path.dirname(os.path.abspath(output)),
                        preset=os.environ.get(
                            "MAGIC_CABT_MODEL_PRESET", "local"),
                        epochs=int(os.environ.get(
                            "MAGIC_CABT_MODEL_EPOCHS", "1")),
                        replay_bundles=int(os.environ.get(
                            "MAGIC_CABT_MODEL_REPLAY_BUNDLES", "20")),
                        device=os.environ.get(
                            "MAGIC_CABT_MODEL_DEVICE") or None,
                        embedding_backend=os.environ.get(
                            "MAGIC_CABT_EMBEDDING_BACKEND", "hash"),
                        arena_card_db=arena_card_db,
                        log=lambda text: self._say("[model] %s" % text))
            except Exception as error:
                self._say("[model] disabled: %s" % error)

        def _show_analysis(self, result):
            text = format_analysis(result)
            self._say(text)
            if self._on_action is not None:
                self._on_action(text, {"analysis": result})

        def _on_decision(self, record):
            # Base class persists the human choice first.
            super()._on_decision(record)
            if self._analysis_worker is not None:
                self._analysis_worker.submit(record)

        def _on_game_event(self, kind, event):
            # Reload before a new game, never during one.
            if kind == "game_start" and self._analysis_worker is not None:
                if self._analysis_worker.game_boundary():
                    self._say("[model] loaded the new checkpoint")
            super()._on_game_event(kind, event)
            if kind == "game_over" and self._local_evolver is not None \
                    and self.recorder is not None:
                self.recorder.flush()
                queued = self._local_evolver.submit(
                    self.recorder.output_dir)
                self._say("[model] training queued" if queued else
                          "[model] trainer busy; latest game queued")

        def follow(self, *args, **kwargs):
            try:
                return super().follow(*args, **kwargs)
            finally:
                self._close_local_model()

        def feed_entries(self, *args, **kwargs):
            try:
                return super().feed_entries(*args, **kwargs)
            finally:
                self._close_local_model()

        def _close_local_model(self):
            if self._analysis_worker is not None:
                self._analysis_worker.close(drain=True, timeout=3.0)
                self._analysis_worker = None
            if self._local_evolver is not None:
                self._local_evolver.close(timeout=0.1)

    def load_analysis(bundle_dir):
        """Newest cached model run for each decision fingerprint."""
        path = os.path.join(bundle_dir, "analysis.jsonl")
        records = {}
        if not os.path.exists(path):
            return records
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                key = record.get("decisionFingerprint")
                current = records.get(key)
                if key and (current is None or
                            str(record.get("createdAt") or "") >=
                            str(current.get("createdAt") or "")):
                    records[key] = record
        return records

    def analysis_text(records, decision):
        if not records:
            return None
        record = records.get(decision_fingerprint(decision))
        return format_analysis(record) if record else None

    def analysis_recommends_action(records, decision):
        """True when the cached model's top choice is not a pass.

        Lets the "next non-pass action" jump also stop where the human
        passed but the model wanted to act — the disagreements worth
        reviewing."""
        record = records.get(decision_fingerprint(decision)) \
            if records else None
        top = ((record or {}).get("analysis") or {}).get("topK") or []
        if not top:
            return False
        index = top[0].get("optionIndex")
        select = (decision.get("observation") or {}).get("select") or {}
        for option in select.get("option") or []:
            if option.get("index") == index:
                return not replay_module.option_is_pass(option)
        return False

    class LocalReplayPlayer(BasePlayer):
        def play(self, bundle_dir):
            self._local_analysis = load_analysis(bundle_dir)
            return super().play(bundle_dir)

        def _narrate_decision(self, decision):
            super()._narrate_decision(decision)
            text = analysis_text(
                getattr(self, "_local_analysis", {}), decision)
            if text:
                self._say(text)
                if self.display is not None:
                    try:
                        self.display.send_message(text)
                    except Exception:
                        pass

    class LocalReplayController(BaseController):
        def __init__(self, bundle_dir, display=None, on_progress=None,
                     on_message=None, speed=4.0):
            self._local_analysis = load_analysis(bundle_dir)
            original_progress = on_progress

            def progress(info):
                decisions = self._decisions.get(self._index, []) \
                    if hasattr(self, "_decisions") else []
                if decisions:
                    info = dict(info)
                    info["analysis"] = analysis_text(
                        self._local_analysis, decisions[-1])
                if original_progress is not None:
                    original_progress(info)

            super().__init__(
                bundle_dir, display=display, on_progress=progress,
                on_message=on_message, speed=speed)

        def _narrate(self, decision):
            super()._narrate(decision)
            text = analysis_text(self._local_analysis, decision)
            if text and self._on_message is not None:
                self._on_message(text)
            if text and self.display is not None:
                try:
                    self.display.send_message(text)
                except Exception:
                    pass

        def _jump(self, meaningful):
            if not meaningful:
                return super()._jump(meaningful)
            for i in range(self._index + 1, self.total):
                decisions = self._decisions.get(i)
                if not decisions:
                    continue
                if any(not replay_module.decision_is_pass(d)
                       for d in decisions) or \
                        any(analysis_recommends_action(
                            self._local_analysis, d) for d in decisions):
                    self._render(i)
                    return
            self._render(self.total - 1)

    recorder_module.MirrorRecorder = LocalModelRecorder
    session_module.MirrorSession = LocalModelSession
    replay_module.ReplayPlayer = LocalReplayPlayer
    replay_module.ReplayController = LocalReplayController


def install_local_model_gui_hooks():
    """Add a cached-analysis readout to the existing Replays tab."""
    global _GUI_INSTALLED
    if _GUI_INSTALLED:
        return
    install_local_model_hooks()
    _GUI_INSTALLED = True
    try:
        from . import gui as gui_module
    except Exception:
        return

    app = gui_module.ArenaMirrorApp
    original_build_transport = app._build_transport
    original_update_progress = app._update_progress

    def build_transport(self, frame):
        original_build_transport(self, frame)
        self.replay_analysis_var = gui_module.tk.StringVar(
            value="No cached model analysis for this frame.")
        label = gui_module.ttk.Label(
            frame, textvariable=self.replay_analysis_var,
            style="Muted.TLabel", justify=gui_module.tk.LEFT,
            anchor="w", wraplength=960)
        label.pack(fill=gui_module.tk.X, pady=(8, 0))

    def update_progress(self, info):
        original_update_progress(self, info)
        variable = getattr(self, "replay_analysis_var", None)
        if variable is not None:
            variable.set(info.get("analysis") or
                         "No cached model analysis for this frame.")

    app._build_transport = build_transport
    app._update_progress = update_progress
