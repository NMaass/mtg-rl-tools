import copy
import unittest

from magic_cabt.research.experiment import (
    plan_report,
    validate_experiment_plan,
)


def benchmark_plan():
    return {
        "schemaVersion": 1,
        "name": "fixture-benchmark",
        "claimLevel": "benchmark",
        "data": {
            "sources": ["arena_human", "engine_selfplay"],
            "decisionSources": ["arena_human"],
            "transitionSources": ["engine_selfplay"],
            "split": {
                "unit": "game",
                "strategy": "sha256-stable-key",
                "holdoutFraction": 0.2,
            },
            "hiddenInformation": {
                "opponentPrivateZones": "belief_only",
                "allowTrueOpponentHand": False,
            },
            "features": {
                "includePerGameInstanceIds": False,
                "semanticActions": True,
            },
        },
        "evaluation": {
            "primaryMetrics": ["canonical_top1", "paired_match_score"],
            "baselines": ["random", "heuristic"],
            "pairedEvaluation": True,
            "confidenceIntervals": True,
            "multipleComparisonCorrection": "holm",
            "suites": ["tactical-v1", "fixed-matchups-v1"],
        },
        "experiments": [{
            "name": "hashed-option-ranker",
            "modelFamily": "option_ranker",
            "seeds": [1, 2, 3, 4, 5],
            "usesOpponentHiddenCards": False,
        }],
    }


class ExperimentPlanTest(unittest.TestCase):

    def test_valid_benchmark_plan_passes(self):
        report = plan_report(benchmark_plan())
        self.assertTrue(report["valid"])
        self.assertEqual(0, report["errors"])
        self.assertEqual([], report["problems"])

    def test_record_level_split_and_hidden_card_leakage_are_rejected(self):
        plan = benchmark_plan()
        plan["data"]["split"]["unit"] = "decision"
        plan["data"]["hiddenInformation"]["allowTrueOpponentHand"] = True
        plan["data"]["features"]["includePerGameInstanceIds"] = True
        paths = {problem.path for problem in validate_experiment_plan(plan)
                 if problem.level == "error"}
        self.assertIn("$.data.split.unit", paths)
        self.assertIn(
            "$.data.hiddenInformation.allowTrueOpponentHand", paths)
        self.assertIn(
            "$.data.features.includePerGameInstanceIds", paths)

    def test_benchmark_claim_requires_multiple_seeds_and_paired_statistics(self):
        plan = benchmark_plan()
        plan["experiments"][0]["seeds"] = [1]
        plan["evaluation"]["pairedEvaluation"] = False
        plan["evaluation"]["confidenceIntervals"] = False
        plan["evaluation"]["multipleComparisonCorrection"] = "none"
        paths = {problem.path for problem in validate_experiment_plan(plan)
                 if problem.level == "error"}
        self.assertIn("$.experiments[0].seeds", paths)
        self.assertIn("$.evaluation.pairedEvaluation", paths)
        self.assertIn("$.evaluation.confidenceIntervals", paths)
        self.assertIn("$.evaluation.multipleComparisonCorrection", paths)

    def test_jepa_requires_transition_objectives_and_collapse_diagnostics(self):
        plan = benchmark_plan()
        plan["experiments"] = [{
            "name": "structured-jepa",
            "modelFamily": "jepa",
            "seeds": [1, 2, 3, 4, 5],
            "usesOpponentHiddenCards": False,
            "objectives": ["future_latent", "public_state_deltas"],
            "ablations": ["no-policy-loss", "no-causal-head"],
        }]
        paths = {problem.path for problem in validate_experiment_plan(plan)
                 if problem.level == "error"}
        self.assertIn("$.experiments[0].collapseDiagnostics", paths)

        plan["experiments"][0]["collapseDiagnostics"] = True
        report = plan_report(plan)
        self.assertTrue(report["valid"])

    def test_expert_cost_cannot_replace_terminal_objective(self):
        plan = benchmark_plan()
        plan["experiments"][0]["expertCost"] = {
            "role": "terminal_reward_replacement",
            "heldOutExpertEvaluation": True,
        }
        messages = [problem.message for problem in validate_experiment_plan(plan)
                    if problem.level == "error"]
        self.assertTrue(any(
            "replace the win/loss objective" in message
            for message in messages))

    def test_potential_shaping_formula_is_checked(self):
        plan = benchmark_plan()
        plan["experiments"][0]["expertCost"] = {
            "role": "potential_shaping",
            "form": "raw_dense_reward",
            "heldOutExpertEvaluation": True,
        }
        paths = {problem.path for problem in validate_experiment_plan(plan)
                 if problem.level == "error"}
        self.assertIn("$.experiments[0].expertCost.form", paths)


if __name__ == "__main__":
    unittest.main()
