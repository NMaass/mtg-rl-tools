"""Nonblocking live decision analysis worker."""
from __future__ import annotations

import copy
import queue
import threading
import time

from .schema import analysis_cache_key, make_analysis_record


class AnalysisWorker:
    def __init__(self, scorer, recorder=None, callback=None,
                 error_callback=None, cache=None, top_k=5, max_queue=64):
        self.scorer = scorer
        self.recorder = recorder
        self.callback = callback
        self.error_callback = error_callback
        self.cache = cache
        self.top_k = top_k
        self._queue = queue.Queue(maxsize=max_queue)
        self._stop = False
        self.skipped = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, record):
        item = copy.deepcopy(record)
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self.skipped += 1
            self._error("analysis queue full; decision skipped")
            return False

    def game_boundary(self):
        reload_method = getattr(self.scorer, "reload_if_changed", None)
        if reload_method:
            try:
                return bool(reload_method())
            except Exception as error:
                self._error(error)
        return False

    def close(self, drain=True, timeout=5.0):
        if drain:
            deadline = time.monotonic() + timeout
            while self._queue.unfinished_tasks and time.monotonic() < deadline:
                time.sleep(0.02)
        self._stop = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=max(0.1, timeout))

    def _run(self):
        while not self._stop:
            try:
                record = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if record is None:
                self._queue.task_done()
                return
            try:
                self._analyze(record)
            except Exception as error:
                self._error(error)
            finally:
                self._queue.task_done()

    def _analyze(self, record):
        key = analysis_cache_key(record, self.scorer.model_info)
        result = self.cache.get(key) if self.cache else None
        if result is None:
            started = time.perf_counter()
            scores = self.scorer.score(record)
            latency = int((time.perf_counter() - started) * 1000)
            value_method = getattr(self.scorer, "state_value", None)
            value = value_method(record) if value_method else None
            result = make_analysis_record(
                record, scores, self.scorer.model_info,
                top_k=self.top_k, latency_ms=latency,
                value=value, source="live")
            if self.cache:
                self.cache.add(result, persist=self.recorder is None)
        if self.recorder:
            self.recorder.record_analysis(result)
        if self.callback:
            self.callback(result)

    def _error(self, error):
        if self.error_callback:
            if isinstance(error, str):
                self.error_callback(error)
            else:
                self.error_callback("%s: %s" % (type(error).__name__, error))
