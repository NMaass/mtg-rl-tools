"""Conservative between-game fine-tuning for one local user.

Gameplay capture never waits for training. Replay inputs are snapshotted before
a subprocess reads them, a successful candidate replaces the active checkpoint
atomically, and the live scorer reloads only at a later game boundary.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone

from magic_cabt.models.structured_jepa import MagicJEPA, StructuredJEPAConfig


class LocalEvolver:
    def __init__(self, model_dir=None, runs_root=None, preset="local", epochs=1,
                 replay_bundles=20, device=None, embedding_backend="hash",
                 arena_card_db=None, log=None):
        self.model_dir = os.path.abspath(os.path.expanduser(
            model_dir or "~/.magic-cabt/local-model"))
        self.runs_root = os.path.abspath(runs_root) if runs_root else None
        self.preset = preset
        self.epochs = max(1, int(epochs))
        self.replay_bundles = max(1, int(replay_bundles))
        self.device = device
        self.embedding_backend = embedding_backend
        self.arena_card_db = arena_card_db
        self.log = log
        self._lock = threading.Lock()
        self._thread = None
        self._pending = False
        self._latest_bundle = None
        os.makedirs(self.model_dir, exist_ok=True)

    @property
    def checkpoint_path(self):
        return os.path.join(self.model_dir, "checkpoint.pt")

    @property
    def status_path(self):
        return os.path.join(self.model_dir, "status.json")

    def ensure_checkpoint(self):
        """Create a clearly marked random checkpoint when none exists."""
        if os.path.exists(self.checkpoint_path):
            return self.checkpoint_path
        config = StructuredJEPAConfig.preset(self.preset)
        config.embedding_backend = self.embedding_backend
        model = MagicJEPA(config)
        model.save_checkpoint(self.checkpoint_path, extra={
            "status": "untrained", "createdAt": _now()})
        self._write_status({
            "state": "untrained", "checkpoint": self.checkpoint_path})
        return self.checkpoint_path

    def submit(self, bundle_dir):
        """Queue the latest completed-game bundle; never block the caller."""
        bundle_dir = os.path.abspath(bundle_dir)
        with self._lock:
            self._latest_bundle = bundle_dir
            if self._thread is not None and self._thread.is_alive():
                self._pending = True
                return False
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            return True

    def close(self, timeout=0.2):
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def train_once(self, bundle_dir):
        """Synchronous entry point used by the manual CLI."""
        return self._train_once(os.path.abspath(bundle_dir))

    def _run_loop(self):
        while True:
            with self._lock:
                bundle = self._latest_bundle
                self._pending = False
            try:
                self._train_once(bundle)
            except Exception as error:  # background failures never affect capture
                self._say("local training failed: %s" % error)
                self._write_status({
                    "state": "failed", "error": str(error),
                    "finishedAt": _now()})
            with self._lock:
                if not self._pending:
                    return

    def _train_once(self, bundle_dir):
        self.ensure_checkpoint()
        bundles = self._recent_bundles(bundle_dir)
        if not bundles:
            self._say("local training skipped: no replay bundles")
            return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        staging = os.path.join(self.model_dir, "candidate-" + stamp)
        os.makedirs(staging, exist_ok=True)
        snapshots = self._snapshot_bundles(bundles, staging)
        if not snapshots:
            self._say("local training skipped: no snapshot inputs")
            return None

        command = [
            sys.executable, "-m", "magic_cabt.training.train_jepa",
            "--out", staging,
            "--resume", self.checkpoint_path,
            "--epochs", str(self.epochs),
            "--embedding-backend", self.embedding_backend,
        ]
        if self.device:
            command.extend(["--device", self.device])
        if self.arena_card_db and os.path.isfile(self.arena_card_db):
            command.extend(["--arena-card-db", self.arena_card_db])
        for path in snapshots:
            command.extend(["--input", path])

        self._write_status({
            "state": "training", "startedAt": _now(),
            "bundles": snapshots, "command": command})
        self._say("training local model on %d replay bundle(s)" % len(snapshots))
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            message = completed.stderr or completed.stdout or \
                "trainer exited %d" % completed.returncode
            raise RuntimeError(message[-4000:])
        candidate = os.path.join(staging, "checkpoint.pt")
        if not os.path.exists(candidate):
            raise RuntimeError("trainer did not produce checkpoint.pt")
        temporary = self.checkpoint_path + ".next"
        shutil.copy2(candidate, temporary)
        os.replace(temporary, self.checkpoint_path)
        metrics = _read_json(os.path.join(staging, "metrics.json")) or {}
        self._write_status({
            "state": "ready", "finishedAt": _now(),
            "checkpoint": self.checkpoint_path,
            "bundles": snapshots, "metrics": metrics})
        self._say("checkpoint updated; it will load before the next game")
        return self.checkpoint_path

    def _snapshot_bundles(self, bundles, staging):
        """Copy append-only inputs so training never reads a partial line."""
        root = os.path.join(staging, "replay-snapshots")
        os.makedirs(root, exist_ok=True)
        names = (
            "decisions.jsonl", "mirror_states.jsonl", "transitions.jsonl",
            "summary.json", "card_cache.json")
        snapshots = []
        for index, bundle in enumerate(bundles):
            destination = os.path.join(
                root, "%03d-%s" %
                (index, os.path.basename(os.path.normpath(bundle)) or "bundle"))
            os.makedirs(destination, exist_ok=True)
            copied = False
            for name in names:
                source = os.path.join(bundle, name)
                if os.path.isfile(source):
                    shutil.copy2(source, os.path.join(destination, name))
                    copied = True
            if copied:
                snapshots.append(destination)
        return snapshots

    def _recent_bundles(self, current):
        root = self.runs_root or os.path.dirname(current)
        candidates = []
        if os.path.isdir(root):
            if _is_bundle(root):
                candidates.append(root)
            for entry in os.listdir(root):
                path = os.path.join(root, entry)
                if _is_bundle(path):
                    candidates.append(path)
        if _is_bundle(current) and current not in candidates:
            candidates.append(current)
        candidates.sort(key=os.path.getmtime)
        return candidates[-self.replay_bundles:]

    def _write_status(self, payload):
        payload = dict(payload)
        payload["schemaVersion"] = 1
        temporary = self.status_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, self.status_path)

    def _say(self, text):
        if self.log:
            self.log(text)


def _is_bundle(path):
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "decisions.jsonl")) or
        os.path.exists(os.path.join(path, "mirror_states.jsonl")))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-local-evolve",
        description="Fine-tune the local JEPA on recent replay bundles.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--model-dir", default="~/.magic-cabt/local-model")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument(
        "--preset", default="local", choices=("tiny", "local", "large"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--replay-bundles", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--embedding-backend", default="hash")
    parser.add_argument("--arena-card-db", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    evolver = LocalEvolver(
        model_dir=args.model_dir, runs_root=args.runs_root,
        preset=args.preset, epochs=args.epochs,
        replay_bundles=args.replay_bundles, device=args.device,
        embedding_backend=args.embedding_backend,
        arena_card_db=args.arena_card_db,
        log=lambda text: sys.stderr.write(text + "\n"))
    evolver.train_once(args.bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
