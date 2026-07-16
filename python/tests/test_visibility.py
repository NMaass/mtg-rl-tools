import unittest

from magic_cabt.models.structured_jepa import StructuredJEPAConfig
from magic_cabt.models.visibility import VisibilitySafeTensorizer


def record(hidden_name, mana_value):
    return {
        "perspectiveSeat": 1,
        "observation": {"current": {
            "localSeat": 1,
            "players": [{"seat": 1, "life": 20},
                        {"seat": 2, "life": 20}],
            "zones": {"hand": {"2": [{
                "name": hidden_name,
                "manaValue": mana_value,
                "power": 9,
                "toughness": 9,
                "tapped": True,
            }]}}
        }},
    }


class VisibilitySafeTensorizerTest(unittest.TestCase):
    def test_hidden_identity_and_numeric_fields_do_not_change_rows(self):
        config = StructuredJEPAConfig(
            text_dim=16, numeric_dim=40, d_model=16, nhead=4,
            encoder_layers=1, predictor_layers=1, ff_dim=32,
            dropout=0.0, max_objects=16, causal_dim=18,
            horizon_buckets=8, embedding_backend="hash")
        tensorizer = VisibilitySafeTensorizer(config)
        self.assertEqual(
            tensorizer.state_rows(record("Lightning Bolt", 1)),
            tensorizer.state_rows(record("Emrakul", 15)))

    def test_unverified_generic_history_is_ignored(self):
        config = StructuredJEPAConfig.preset("tiny")
        tensorizer = VisibilitySafeTensorizer(config)
        first = record("Secret", 1)
        second = record("Secret", 1)
        first["observation"]["history"] = [{"opponentDraw": "Bolt"}]
        second["observation"]["history"] = [{"opponentDraw": "Island"}]
        self.assertEqual(tensorizer.state_rows(first),
                         tensorizer.state_rows(second))


if __name__ == "__main__":
    unittest.main()
