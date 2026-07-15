import unittest

from magic_cabt.analysis.baselines import (
    DeterministicRandomScorer,
    FirstLegalScorer,
    GenericActionHeuristicScorer,
    make_baseline,
)
from magic_cabt.analysis.suite import validate_entries


def record():
    return {"observation": {"current": {"turnNumber": 1}, "select": {
        "type": "PRIORITY",
        "option": [
            {"index": 0, "type": "PASS_PRIORITY", "label": "Pass"},
            {"index": 1, "type": "PLAY_LAND", "label": "Play Forest"},
            {"index": 2, "type": "CAST_SPELL", "label": "Cast Bear"},
            {"index": 3, "type": "CONCEDE", "label": "Concede"},
        ]}}}


class BaselineScorerTest(unittest.TestCase):
    def test_first_legal_scores_first_option(self):
        self.assertEqual([1.0, 0.0, 0.0, 0.0],
                         FirstLegalScorer().score(record()))

    def test_random_is_deterministic_per_seed_and_decision(self):
        first = DeterministicRandomScorer(seed=7).score(record())
        second = DeterministicRandomScorer(seed=7).score(record())
        other = DeterministicRandomScorer(seed=8).score(record())
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_generic_heuristic_is_action_shape_only(self):
        scores = GenericActionHeuristicScorer().score(record())
        self.assertGreater(scores[2], scores[1])
        self.assertGreater(scores[1], scores[0])
        self.assertLess(scores[3], scores[0])

    def test_factory_rejects_unknown_baseline(self):
        self.assertIsInstance(make_baseline("heuristic"),
                              GenericActionHeuristicScorer)
        with self.assertRaises(ValueError):
            make_baseline("oracle")

    def test_suite_rejects_duplicate_display_names(self):
        with self.assertRaisesRegex(ValueError, "duplicate model name"):
            validate_entries([
                ("control", "baseline:first-legal"),
                ("control", "baseline:random"),
            ])


if __name__ == "__main__":
    unittest.main()
