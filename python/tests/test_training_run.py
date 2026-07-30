import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.training_run import (
    RunContext,
    StageError,
    _ingest_one_log,
    _stage_fingerprint,
    build_parser,
    generate_toy_records,
    main,
    split_by_game,
)


def _run(argv):
    """Run the CLI, returning (exit_code, stderr_text)."""
    stderr = io.StringIO()
    original = sys.stderr
    sys.stderr = stderr
    try:
        code = main(argv)
    finally:
        sys.stderr = original
    return code, stderr.getvalue()


class ToyCorpusTest(unittest.TestCase):

    def test_toy_records_are_deterministic_per_seed(self):
        first = list(generate_toy_records(3, seed=7))
        second = list(generate_toy_records(3, seed=7))
        self.assertEqual(first, second)
        other = list(generate_toy_records(3, seed=8))
        self.assertNotEqual(first, other)

    def test_toy_records_validate_and_carry_whole_games(self):
        from magic_cabt.training.records import validate_records
        records = list(generate_toy_records(4, seed=1))
        summary = validate_records(records)
        self.assertEqual(0, summary["invalid"], summary)
        games = {record["gameId"] for record in records}
        self.assertEqual(4, len(games))
        terminals = [record for record in records if record["terminal"]]
        self.assertEqual(4, len(terminals))


class SplitTest(unittest.TestCase):

    def test_split_is_deterministic_and_whole_game(self):
        game_ids = ["g%02d" % index for index in range(20)]
        first = split_by_game(game_ids, seed=3, val_fraction=0.2,
                              test_fraction=0.2)
        second = split_by_game(list(reversed(game_ids)), seed=3,
                               val_fraction=0.2, test_fraction=0.2)
        self.assertEqual(first, second)
        buckets = {"train": 0, "val": 0, "test": 0}
        for split in first.values():
            buckets[split] += 1
        self.assertEqual(20, sum(buckets.values()))
        self.assertEqual(4, buckets["test"])
        self.assertEqual(4, buckets["val"])

    def test_split_reserves_at_least_one_game_per_requested_split(self):
        assignment = split_by_game(["a", "b", "c", "d"], seed=0,
                                   val_fraction=0.05, test_fraction=0.05)
        self.assertIn("val", assignment.values())
        self.assertIn("test", assignment.values())

    def test_split_fails_closed_when_holdouts_consume_the_corpus(self):
        with self.assertRaises(StageError):
            split_by_game(["a", "b"], seed=0, val_fraction=0.5,
                          test_fraction=0.5)


class ToyEndToEndTest(unittest.TestCase):

    def test_toy_run_produces_report_and_beats_random(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run")
            code, stderr = _run(["--toy", "24", "--out", out,
                                 "--seed", "5", "--quiet"])
            self.assertEqual(0, code, stderr)

            report = _read_json(os.path.join(out, "report.json"))
            stages = report["stages"]
            for name in ("collect", "validate", "audit", "manifest",
                         "split", "compile", "baselines", "train-bc"):
                self.assertEqual("ok", stages[name]["status"],
                                 (name, stages[name]))
            # The report stage records itself after writing report.json, so
            # its own status lives in state.json only.
            state = _read_json(os.path.join(out, "state.json"))
            self.assertEqual("ok", state["stages"]["report"]["status"])

            comparison = {row["name"]: row
                          for row in report.get("comparison") or []}
            self.assertIn("first", comparison)
            self.assertIn("random", comparison)
            self.assertIsNotNone(comparison["first"]["playedTop1"])
            with open(os.path.join(out, "report.md"),
                      encoding="utf-8") as handle:
                self.assertIn("Head-to-head", handle.read())
            metrics = report["testMetrics"]
            bc = metrics["bag-of-words-bc"]["top1Accuracy"]
            rnd = metrics["baseline:random"]["top1Accuracy"]
            first = metrics["baseline:first"]["top1Accuracy"]
            self.assertIsNotNone(bc)
            self.assertIsNotNone(rnd)
            self.assertGreater(bc, rnd)
            self.assertGreater(bc, first)

            splits = _read_json(os.path.join(out, "dataset", "splits.json"))
            self.assertGreater(splits["train"]["games"], 0)
            self.assertGreater(splits["val"]["games"], 0)
            self.assertGreater(splits["test"]["games"], 0)
            self._assert_no_game_crosses_splits(out)

            for artifact in ("config.json", "state.json", "report.md",
                             os.path.join("dataset", "manifest.json"),
                             os.path.join("dataset", "trust_audit.json"),
                             os.path.join("models", "bc",
                                          "checkpoint.json")):
                self.assertTrue(
                    os.path.exists(os.path.join(out, artifact)), artifact)

    def _assert_no_game_crosses_splits(self, out):
        seen = {}
        for split in ("train", "val", "test"):
            path = os.path.join(out, "dataset",
                                "%s_decisions.jsonl" % split)
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    game_id = json.loads(line)["gameId"]
                    self.assertEqual(seen.setdefault(game_id, split), split,
                                     "game %s crosses splits" % game_id)

    def test_second_run_reuses_cached_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run")
            code, _ = _run(["--toy", "12", "--out", out, "--quiet"])
            self.assertEqual(0, code)
            first_state = _read_json(os.path.join(out, "state.json"))
            code, stderr = _run(["--toy", "12", "--out", out])
            self.assertEqual(0, code, stderr)
            self.assertIn("collect: cached", stderr)
            self.assertIn("train-bc: cached", stderr)
            second_state = _read_json(os.path.join(out, "state.json"))
            self.assertEqual(
                first_state["stages"]["train-bc"]["finishedAt"],
                second_state["stages"]["train-bc"]["finishedAt"])

    def test_torch_stage_is_skipped_not_failed_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run")
            code, _ = _run(["--toy", "12", "--out", out, "--quiet",
                            "--skip-torch"])
            self.assertEqual(0, code)
            state = _read_json(os.path.join(out, "state.json"))
            self.assertEqual("skipped", state["stages"]["train-torch"]["status"])


class FailureModeTest(unittest.TestCase):

    def test_missing_log_fails_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, stderr = _run(["--log", os.path.join(tmp, "absent.log"),
                                 "--out", os.path.join(tmp, "run"),
                                 "--quiet"])
            self.assertEqual(1, code)
            self.assertIn("does not exist", stderr)

    def test_no_inputs_fails_at_collect(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, stderr = _run(["--out", os.path.join(tmp, "run"),
                                 "--quiet"])
            self.assertEqual(1, code)
            self.assertIn("no inputs", stderr)

    def test_min_games_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, stderr = _run(["--toy", "3",
                                 "--min-games", "50",
                                 "--out", os.path.join(tmp, "run"),
                                 "--quiet"])
            self.assertEqual(1, code)
            self.assertIn("--min-games", stderr)

    def test_corrupt_bundle_fails_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.jsonl")
            with open(bad, "w", encoding="utf-8") as handle:
                handle.write("this is not json\n")
            code, stderr = _run(["--bundle", bad,
                                 "--out", os.path.join(tmp, "run"),
                                 "--quiet"])
            self.assertEqual(1, code)
            self.assertIn("could not read", stderr)

    def test_duplicate_game_across_inputs_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "games.jsonl")
            with open(source, "w", encoding="utf-8") as handle:
                for record in generate_toy_records(4, seed=2):
                    handle.write(json.dumps(record) + "\n")
            copy = os.path.join(tmp, "games_copy.jsonl")
            with open(source) as src, open(copy, "w") as dst:
                dst.write(src.read())
            code, stderr = _run(["--bundle", source, "--bundle", copy,
                                 "--out", os.path.join(tmp, "run"),
                                 "--quiet"])
            self.assertEqual(1, code)
            self.assertIn("--allow-duplicate-games", stderr)
            code, stderr = _run(["--bundle", source, "--bundle", copy,
                                 "--allow-duplicate-games", "--force",
                                 "--out", os.path.join(tmp, "run"),
                                 "--quiet"])
            self.assertEqual(0, code, stderr)

    def test_unknown_skip_stage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, stderr = _run(["--toy", "6", "--skip", "nonsense",
                                 "--out", os.path.join(tmp, "run"),
                                 "--quiet"])
            self.assertEqual(1, code)
            self.assertIn("unknown --skip", stderr)

    def test_fraction_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, stderr = _run(["--toy", "6", "--val-fraction", "0.8",
                                 "--test-fraction", "0.4",
                                 "--out", os.path.join(tmp, "run")])
            self.assertEqual(2, code)
            self.assertIn("sum to < 1", stderr)

    def test_hyperparameter_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run")
            for argv, message in (
                    (["--toy", "6", "--epochs", "0", "--out", out],
                     "--epochs"),
                    (["--toy", "6", "--batch-size", "0", "--out", out],
                     "--batch-size"),
                    (["--toy", "6", "--lr", "0", "--out", out], "--lr"),
                    (["--toy", "6", "--min-games", "0", "--out", out],
                     "--min-games")):
                code, stderr = _run(argv)
                self.assertEqual(2, code, argv)
                self.assertIn(message, stderr)


class IngestRecoveryTest(unittest.TestCase):
    """A bundle whose ingest died mid-way must be rebuilt, never appended."""

    def _context(self, tmp):
        args = build_parser().parse_args(
            ["--out", os.path.join(tmp, "run"), "--quiet"])
        return RunContext(args)

    def _jsonl_line_counts(self, bundle_dir):
        counts = {}
        for name in sorted(os.listdir(bundle_dir)):
            if name.endswith(".jsonl"):
                with open(os.path.join(bundle_dir, name),
                          encoding="utf-8") as handle:
                    counts[name] = sum(1 for _ in handle)
        return counts

    def test_missing_marker_triggers_rebuild_not_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "Player.log")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("[UnityCrossThreadLogger]plain chatter\n")
                handle.write("not arena json at all\n")
            ctx = self._context(tmp)
            first = _ingest_one_log(ctx, log_path)
            bundle_dir = first["bundle"]
            baseline = self._jsonl_line_counts(bundle_dir)

            os.remove(os.path.join(bundle_dir, "ingest_source.json"))
            second = _ingest_one_log(ctx, log_path)
            self.assertEqual(bundle_dir, second["bundle"])
            self.assertFalse(second.get("cached"))
            self.assertEqual(baseline, self._jsonl_line_counts(bundle_dir),
                             "rebuild must not append to the old bundle")

            third = _ingest_one_log(ctx, log_path)
            self.assertTrue(third.get("cached"))


class CompareInvalidationTest(unittest.TestCase):
    """Retrained torch checkpoints must invalidate a cached comparison."""

    def test_compare_fingerprint_tracks_train_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build_parser().parse_args(
                ["--out", os.path.join(tmp, "run")])
            ctx = RunContext(args)
            ctx.stage_meta["collect"] = {"sha256": "abc"}
            ctx.state["stages"]["train-torch"] = {"status": "ok",
                                                  "fingerprint": "one"}
            before = _stage_fingerprint(ctx, "compare")
            ctx.state["stages"]["train-torch"]["fingerprint"] = "two"
            after = _stage_fingerprint(ctx, "compare")
            self.assertNotEqual(before, after)
            unrelated = _stage_fingerprint(ctx, "split")
            ctx.state["stages"]["train-torch"]["fingerprint"] = "three"
            self.assertEqual(unrelated, _stage_fingerprint(ctx, "split"))


class DryRunTest(unittest.TestCase):

    def test_dry_run_prints_plan_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run")
            stdout = io.StringIO()
            original = sys.stdout
            sys.stdout = stdout
            try:
                code = main(["--toy", "5", "--out", out, "--dry-run",
                             "--quiet"])
            finally:
                sys.stdout = original
            self.assertEqual(0, code)
            plan = json.loads(stdout.getvalue())
            self.assertEqual(5, plan["toyGames"])
            self.assertEqual(
                [stage["stage"] for stage in plan["stages"]][0], "ingest")
            self.assertFalse(os.path.exists(out))


class ParserTest(unittest.TestCase):

    def test_parser_defaults(self):
        args = build_parser().parse_args(["--out", "x"])
        self.assertEqual(0.1, args.val_fraction)
        self.assertEqual(0.1, args.test_fraction)
        self.assertEqual(2, args.min_games)
        self.assertEqual([], args.log)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    unittest.main()
