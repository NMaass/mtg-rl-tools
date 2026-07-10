import json
import os
import tempfile
import unittest

from magic_cabt.models.structured_jepa import TORCH_AVAILABLE
from magic_cabt.training.local_evolve import LocalEvolver, _is_bundle


def make_bundle(root, name, mtime=None):
    path = os.path.join(root, name)
    os.makedirs(path)
    with open(os.path.join(path, "decisions.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write(json.dumps({"gameId": name}) + "\n")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


class BundleDiscoveryTest(unittest.TestCase):
    def test_is_bundle(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = make_bundle(root, "run-1")
            self.assertTrue(_is_bundle(bundle))
            empty = os.path.join(root, "empty")
            os.makedirs(empty)
            self.assertFalse(_is_bundle(empty))

    def test_recent_bundles_keeps_newest(self):
        with tempfile.TemporaryDirectory() as scratch:
            runs = os.path.join(scratch, "runs")
            os.makedirs(runs)
            old = make_bundle(runs, "run-old", mtime=1000)
            middle = make_bundle(runs, "run-middle", mtime=2000)
            new = make_bundle(runs, "run-new", mtime=3000)
            evolver = LocalEvolver(
                model_dir=os.path.join(scratch, "model"),
                runs_root=runs, replay_bundles=2)
            recent = evolver._recent_bundles(new)
            self.assertEqual([middle, new], recent)
            self.assertNotIn(old, recent)


class SnapshotTest(unittest.TestCase):
    def test_snapshot_copies_known_inputs_only(self):
        with tempfile.TemporaryDirectory() as scratch:
            bundle = make_bundle(scratch, "run-1")
            with open(os.path.join(bundle, "player.log"), "w") as handle:
                handle.write("raw log\n")
            evolver = LocalEvolver(model_dir=os.path.join(scratch, "model"))
            staging = os.path.join(scratch, "staging")
            snapshots = evolver._snapshot_bundles([bundle], staging)
            self.assertEqual(1, len(snapshots))
            self.assertTrue(os.path.exists(
                os.path.join(snapshots[0], "decisions.jsonl")))
            self.assertFalse(os.path.exists(
                os.path.join(snapshots[0], "player.log")))


class PruneTest(unittest.TestCase):
    def test_prune_keeps_newest_candidates(self):
        with tempfile.TemporaryDirectory() as scratch:
            model_dir = os.path.join(scratch, "model")
            evolver = LocalEvolver(model_dir=model_dir)
            for index in range(6):
                os.makedirs(os.path.join(
                    model_dir, "candidate-2026010%d-000000-0" % index))
            evolver._prune_candidates(keep=3)
            remaining = sorted(entry for entry in os.listdir(model_dir)
                               if entry.startswith("candidate-"))
            self.assertEqual(
                ["candidate-20260103-000000-0",
                 "candidate-20260104-000000-0",
                 "candidate-20260105-000000-0"], remaining)


@unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
class EnsureCheckpointTest(unittest.TestCase):
    def test_creates_untrained_checkpoint_and_status(self):
        with tempfile.TemporaryDirectory() as scratch:
            evolver = LocalEvolver(
                model_dir=os.path.join(scratch, "model"), preset="tiny")
            checkpoint = evolver.ensure_checkpoint()
            self.assertTrue(os.path.exists(checkpoint))
            with open(evolver.status_path, "r", encoding="utf-8") as handle:
                status = json.load(handle)
            self.assertEqual("untrained", status["state"])
            # idempotent: a second call must not replace the checkpoint
            first_mtime = os.path.getmtime(checkpoint)
            self.assertEqual(checkpoint, evolver.ensure_checkpoint())
            self.assertEqual(first_mtime, os.path.getmtime(checkpoint))


class EntryPointsTest(unittest.TestCase):
    def test_console_script_targets_resolve(self):
        from magic_cabt.local_model import main as local_model_main
        from magic_cabt.training.local_evolve import main as evolve_main
        from magic_cabt.training.train_jepa import main as train_jepa_main
        for target in (local_model_main, evolve_main, train_jepa_main):
            self.assertTrue(callable(target))


if __name__ == "__main__":
    unittest.main()
