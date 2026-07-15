import unittest

from magic_cabt.research.expert_cost import (
    CandidateFactors,
    ExpertCostModel,
    FactorSpec,
    PreferenceExample,
    evaluate_preferences,
    expert_agreement,
    fit_cost_model,
    split_preferences_by_context,
)


SPECS = (
    FactorSpec(
        "threat_removed", direction="higher_better", scale=1.0,
        group="board", description="Strategically relevant opposing threat removed."),
    FactorSpec(
        "mana_spent", direction="lower_better", scale=1.0,
        group="resources", description="Mana consumed by the candidate line."),
)


def candidate(candidate_id, threat_removed, mana_spent):
    return CandidateFactors(
        candidate_id=candidate_id,
        factors={
            "threat_removed": threat_removed,
            "mana_spent": mana_spent,
        },
    )


def preference(context_id, preferred="left", expert_id="expert-a"):
    return PreferenceExample(
        context_id=context_id,
        left=candidate("remove", 2.0, 1.0),
        right=candidate("hold", 0.0, 0.0),
        preferred=preferred,
        expert_id=expert_id,
    )


class ExpertCostTest(unittest.TestCase):

    def test_known_model_orients_factors_and_explains_score(self):
        model = ExpertCostModel(
            specs=SPECS,
            weights={"threat_removed": 2.0, "mana_spent": 1.0},
        )
        remove = {"threat_removed": 1.0, "mana_spent": 1.0}
        hold = {"threat_removed": 0.0, "mana_spent": 0.0}
        self.assertGreater(model.utility(remove), model.utility(hold))
        self.assertGreater(model.preference_probability(remove, hold), 0.5)

        explanation = model.explain(remove)
        self.assertEqual(1.0, explanation["utility"])
        self.assertEqual(-1.0, explanation["cost"])
        self.assertEqual("threat_removed", explanation["contributions"][0]["factor"])

    def test_potential_shaping_uses_gamma_phi_next_minus_phi_current(self):
        model = ExpertCostModel(
            specs=SPECS,
            weights={"threat_removed": 2.0, "mana_spent": 1.0},
        )
        before = {"threat_removed": 0.0, "mana_spent": 0.0}
        after = {"threat_removed": 1.0, "mana_spent": 1.0}
        self.assertAlmostEqual(
            0.9,
            model.potential_shaping_reward(before, after, gamma=0.9),
        )

    def test_fit_learns_repeated_expert_preference(self):
        examples = [preference("context-%d" % index) for index in range(12)]
        model = fit_cost_model(
            SPECS, examples, iterations=1000, learning_rate=0.08, l2=0.001)
        probability = model.preference_probability(
            examples[0].left.factors, examples[0].right.factors)
        self.assertGreater(probability, 0.8)
        metrics = evaluate_preferences(model, examples)
        self.assertEqual(1.0, metrics["pairwiseAccuracy"])
        self.assertLess(metrics["weightedLogLoss"], 0.3)

        restored = ExpertCostModel.from_dict(model.to_dict())
        self.assertAlmostEqual(
            probability,
            restored.preference_probability(
                examples[0].left.factors, examples[0].right.factors),
        )

    def test_context_split_never_separates_annotations_from_same_context(self):
        examples = []
        for index in range(8):
            context = "context-%d" % index
            examples.append(preference(context, expert_id="expert-a"))
            examples.append(preference(context, expert_id="expert-b"))
        train, holdout = split_preferences_by_context(
            examples, holdout_fraction=0.25, seed=7)
        train_contexts = {row.context_id for row in train}
        holdout_contexts = {row.context_id for row in holdout}
        self.assertFalse(train_contexts.intersection(holdout_contexts))
        self.assertEqual(8, len(train_contexts.union(holdout_contexts)))

    def test_expert_agreement_surfaces_disputed_pairs(self):
        rows = [
            preference("same-pair", preferred="left", expert_id="a"),
            preference("same-pair", preferred="left", expert_id="b"),
            preference("same-pair", preferred="right", expert_id="c"),
        ]
        report = expert_agreement(rows)
        self.assertEqual(1, report["multiExpertContexts"])
        self.assertAlmostEqual(2.0 / 3.0, report["meanMajorityAgreement"])
        self.assertEqual("same-pair", report["disagreementContexts"][0]["contextId"])


if __name__ == "__main__":
    unittest.main()
