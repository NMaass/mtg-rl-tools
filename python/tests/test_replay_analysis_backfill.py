import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.arena_mirror.replay import option_is_pass
from magic_cabt.models.structured_jepa import TORCH_AVAILABLE


def _state(seq, turn=1):
    return {"gameInstance": 1, "seq": seq, "matchId": "m1",
            "turnNumber": turn, "localSeat": 1,
            "players": [{"seat": 1, "name": "You"},
                        {"seat": 2, "name": "Opp"}]}


def _pass_decision(seq, sequence):
    """A decision whose human choice is a pure priority pass."""
    return {
        "sequence": sequence,
        "gameId": "game-1",
        "select": [0],
        "selectedIndices": [0],
        "observation": {
            "current": {"gameInstance": 1, "seq": seq, "turnNumber": 1,
                        "localSeat": 1},
            "select": {"type": "PRIORITY", "minCount": 1, "maxCount": 1,
                       "option": [
                           {"index": 0, "type": "PASS", "label": "Pass",
                            "payload": {}},
                           {"index": 1, "type": "CAST",
                            "label": "Cast Grizzly Bears",
                            "payload": {"canonicalKey": "bears"}},
                       ]},
        },
    }


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _make_bundle(tmpdir, with_analysis):
    """Two pure-pass decisions; the model recommends acting on the second."""
    from magic_cabt.analysis import make_analysis_record

    _write_jsonl(os.path.join(tmpdir, "mirror_states.jsonl"),
                 [_state(1), _state(2), _state(3)])
    first, second = _pass_decision(2, 1), _pass_decision(3, 2)
    _write_jsonl(os.path.join(tmpdir, "decisions.jsonl"), [first, second])
    if with_analysis:
        model = {"modelId": "fake", "checkpointId": "sha256:test"}
        # first decision: model also passes; second: model prefers the cast
        _write_jsonl(os.path.join(tmpdir, "analysis.jsonl"), [
            make_analysis_record(first, [2.0, 1.0], model),
            make_analysis_record(second, [1.0, 2.0], model),
        ])


_JUMP_SCRIPT = """
import json, sys
from magic_cabt.arena_mirror.local_model_hooks import install_local_model_hooks
install_local_model_hooks()
from magic_cabt.arena_mirror import replay as replay_module

controller = replay_module.ReplayController(sys.argv[1], display=None)
controller._jump(True)
print(json.dumps({"index": controller._index}))
"""


class OptionIsPassTest(unittest.TestCase):

    def test_pass_and_action_options(self):
        self.assertTrue(option_is_pass({"type": "PASS"}))
        self.assertFalse(option_is_pass({"type": "CAST"}))
        self.assertFalse(option_is_pass(None))


class ModelAwareJumpTest(unittest.TestCase):
    """The hooked controller's non-pass jump also stops on disagreements.

    Runs in a subprocess because install_local_model_hooks patches the
    replay classes process-wide.
    """

    def _jump_index(self, with_analysis):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_bundle(tmpdir, with_analysis=with_analysis)
            output = subprocess.run(
                [sys.executable, "-c", _JUMP_SCRIPT, tmpdir],
                capture_output=True, text=True, check=True,
                cwd=os.path.join(os.path.dirname(__file__), os.pardir))
            return json.loads(output.stdout.strip().splitlines()[-1])["index"]

    def test_jump_stops_where_model_recommends_acting(self):
        # Human passed both times; the model wanted to cast at state seq 3
        # (frame index 2), so the meaningful jump stops there.
        self.assertEqual(self._jump_index(with_analysis=True), 2)

    def test_jump_without_analysis_skips_all_passes(self):
        # No cached analysis: both decisions are pure passes, so the jump
        # runs to the final frame.
        self.assertEqual(self._jump_index(with_analysis=False), 2)

    def test_jump_distinguishes_frames(self):
        # Guard against the two cases above passing for the same reason:
        # with analysis the stop is AT a decision frame; verify the
        # analysis-recommended frame differs from a pure pass-skip when a
        # trailing decision-free state exists.
        with tempfile.TemporaryDirectory() as tmpdir:
            from magic_cabt.analysis import make_analysis_record

            _write_jsonl(os.path.join(tmpdir, "mirror_states.jsonl"),
                         [_state(1), _state(2), _state(3), _state(4)])
            first, second = _pass_decision(2, 1), _pass_decision(3, 2)
            _write_jsonl(os.path.join(tmpdir, "decisions.jsonl"),
                         [first, second])
            model = {"modelId": "fake", "checkpointId": "sha256:test"}
            _write_jsonl(os.path.join(tmpdir, "analysis.jsonl"), [
                make_analysis_record(first, [1.0, 2.0], model),
            ])
            output = subprocess.run(
                [sys.executable, "-c", _JUMP_SCRIPT, tmpdir],
                capture_output=True, text=True, check=True,
                cwd=os.path.join(os.path.dirname(__file__), os.pardir))
            index = json.loads(
                output.stdout.strip().splitlines()[-1])["index"]
            # The model's recommendation is on the FIRST decision (frame 1);
            # without it the jump would land on the last frame (3).
            self.assertEqual(index, 1)


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class AnalyzeBundleTest(unittest.TestCase):

    def test_backfills_and_is_idempotent(self):
        import contextlib
        import io

        from magic_cabt.local_model import main
        from magic_cabt.models.structured_config import StructuredJEPAConfig
        from magic_cabt.models.structured_jepa import MagicJEPA

        config = StructuredJEPAConfig(
            text_dim=32, numeric_dim=40, d_model=32, nhead=4,
            encoder_layers=1, predictor_layers=1, ff_dim=64, max_objects=16)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, "checkpoint.pt")
            MagicJEPA(config).save_checkpoint(checkpoint)
            bundle = os.path.join(tmpdir, "bundle")
            os.makedirs(bundle)
            _write_jsonl(os.path.join(bundle, "decisions.jsonl"),
                         [_pass_decision(2, 1), _pass_decision(3, 2)])

            def run():
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = main(["analyze", bundle,
                                 "--checkpoint", checkpoint,
                                 "--device", "cpu"])
                self.assertEqual(code, 0)
                return json.loads(stdout.getvalue())

            first = run()
            self.assertEqual(first["scored"], 2)
            self.assertEqual(first["alreadyCached"], 0)
            analysis_path = os.path.join(bundle, "analysis.jsonl")
            with open(analysis_path, encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["source"], "backfill")
            self.assertEqual(len(records[0]["analysis"]["topK"]), 2)

            second = run()
            self.assertEqual(second["scored"], 0)
            self.assertEqual(second["alreadyCached"], 2)


if __name__ == "__main__":
    unittest.main()
