import os
import unittest
from unittest.mock import Mock, patch

from magic_cabt import local_model


class ParserTest(unittest.TestCase):
    def test_gui_does_not_auto_train_by_default(self):
        args = local_model.build_parser().parse_args(["gui"])
        self.assertFalse(args.auto_train)
        self.assertFalse(args.no_auto_train)

    def test_gui_auto_train_requires_explicit_flag(self):
        args = local_model.build_parser().parse_args(["gui", "--auto-train"])
        self.assertTrue(args.auto_train)
        self.assertFalse(args.no_auto_train)

    def test_legacy_no_auto_train_flag_remains_valid(self):
        args = local_model.build_parser().parse_args(["gui", "--no-auto-train"])
        self.assertFalse(args.auto_train)
        self.assertTrue(args.no_auto_train)

    def test_flags_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            local_model.build_parser().parse_args(
                ["gui", "--auto-train", "--no-auto-train"])


class EnvironmentTest(unittest.TestCase):
    def _configure(self, auto_train):
        args = local_model.build_parser().parse_args(
            ["gui"] + (["--auto-train"] if auto_train else []))
        evolver = Mock()
        evolver.ensure_checkpoint.return_value = "/tmp/checkpoint.pt"
        with patch.object(local_model, "_make_evolver", return_value=evolver):
            with patch.dict(os.environ, {}, clear=True):
                local_model._configure_environment(args, auto_train=auto_train)
                return dict(os.environ)

    def test_configure_environment_records_disabled_state(self):
        environment = self._configure(auto_train=False)
        self.assertEqual("0", environment["MAGIC_CABT_AUTO_TRAIN"])

    def test_configure_environment_records_opt_in_state(self):
        environment = self._configure(auto_train=True)
        self.assertEqual("1", environment["MAGIC_CABT_AUTO_TRAIN"])


if __name__ == "__main__":
    unittest.main()
