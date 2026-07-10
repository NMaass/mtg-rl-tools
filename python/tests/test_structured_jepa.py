import unittest

from magic_cabt.models.causal import causal_delta_vector
from magic_cabt.models.embeddings import (HashEmbeddingProvider,
                                          make_embedding_provider)
from magic_cabt.models.state_utils import (current_state, flatten_text,
                                           normalize_zone)
from magic_cabt.models.structured_config import (CardTextResolver,
                                                 StructuredJEPAConfig)
from magic_cabt.models.structured_jepa import TORCH_AVAILABLE


def state(life_one=20, life_two=20, turn=1, game_over=False, seq=0):
    value = {
        "turnNumber": turn,
        "phase": "MAIN1",
        "seq": seq,
        "localSeat": 1,
        "players": [
            {"seat": 1, "life": life_one, "handCount": 5, "libraryCount": 50},
            {"seat": 2, "life": life_two, "handCount": 6, "libraryCount": 52},
        ],
        "zones": {
            "battlefield": {"1": [{"name": "Grizzly Bears", "power": 2,
                                   "toughness": 2, "tapped": False}]},
            "hands": {"1": [{"name": "Swamp"}], "2": [{"name": "Hidden"}]},
        },
    }
    if game_over:
        value["gameOver"] = True
    return value


def record(chosen=0):
    return {
        "gameId": "game-1",
        "selectedIndices": [chosen],
        "observation": {
            "current": state(),
            "select": {"type": "PRIORITY", "minCount": 1, "maxCount": 1,
                       "option": [
                           {"index": 0, "type": "PASS_PRIORITY",
                            "label": "Pass", "payload": {}},
                           {"index": 1, "type": "CAST_SPELL",
                            "label": "Cast Shock",
                            "payload": {"canonicalKey": "cast-shock"}},
                       ]},
        },
    }


class ConfigTest(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(192, StructuredJEPAConfig.preset("tiny").d_model)
        self.assertEqual(320, StructuredJEPAConfig.preset("local").d_model)
        self.assertEqual(448, StructuredJEPAConfig.preset("large").d_model)

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            StructuredJEPAConfig.preset("gigantic")


class HashEmbeddingTest(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        provider = HashEmbeddingProvider(dimension=64)
        first = provider.encode("Destroy target creature.")
        second = provider.encode("Destroy target creature.")
        self.assertEqual(first, second)
        self.assertAlmostEqual(1.0, sum(v * v for v in first), places=6)

    def test_distinct_text_differs(self):
        provider = HashEmbeddingProvider(dimension=64)
        self.assertNotEqual(provider.encode("Counter target spell."),
                            provider.encode("Draw two cards."))

    def test_small_dimension_rejected(self):
        with self.assertRaises(ValueError):
            HashEmbeddingProvider(dimension=8)

    def test_factory(self):
        self.assertEqual(384, make_embedding_provider("hash").dimension)
        with self.assertRaises(ValueError):
            make_embedding_provider("word2vec")


class StateUtilsTest(unittest.TestCase):
    def test_current_state_unwraps_observation(self):
        wrapped = {"observation": {"current": {"turnNumber": 7}}}
        self.assertEqual(7, current_state(wrapped)["turnNumber"])
        self.assertEqual(3, current_state({"turnNumber": 3})["turnNumber"])

    def test_normalize_zone(self):
        self.assertEqual("battlefield", normalize_zone("battlefields"))
        self.assertEqual("hand", normalize_zone("Hands"))
        self.assertEqual("other", normalize_zone("phase-out"))

    def test_zone_counts_are_tolerated(self):
        # Arena mirror states record hidden zones as bare counts.
        from magic_cabt.models.state_utils import iter_zone_objects, zone_items
        counted = {
            "zones": {
                "libraries": {"1": 53, "2": 52},
                "hands": {"1": [{"name": "Swamp"}]},
                "battlefield": [{"name": "Grizzly Bears"}],
            },
        }
        objects = list(iter_zone_objects(counted))
        self.assertEqual(2, len(objects))
        self.assertEqual([], zone_items(counted, "library"))
        self.assertEqual(1, len(zone_items(counted, "battlefield")))
        vector = causal_delta_vector(counted, counted)
        self.assertEqual(18, len(vector))

    def test_flatten_text_strips_ids_and_uuids(self):
        text = flatten_text({
            "name": "Shock",
            "instanceId": 42,
            "targetId": "3f142472-c4ac-45cb-bafb-89ddf3918563",
            "label": "target 3f142472-c4ac-45cb-bafb-89ddf3918563 now",
        })
        self.assertIn("name=Shock", text)
        self.assertNotIn("42", text)
        self.assertNotIn("3f142472", text)


class CausalDeltaTest(unittest.TestCase):
    def test_life_deltas_have_expected_sign(self):
        vector = causal_delta_vector(
            state(life_one=20, life_two=20), state(life_one=15, life_two=12),
            perspective_seat=1)
        self.assertEqual(18, len(vector))
        self.assertLess(vector[0], 0.0)   # own life fell
        self.assertLess(vector[4], 0.0)   # opponent life fell
        self.assertEqual(0.0, vector[13])

    def test_terminal_flag(self):
        vector = causal_delta_vector(state(), state(game_over=True))
        self.assertEqual(1.0, vector[13])


class CardTextResolverTest(unittest.TestCase):
    def test_resolves_by_id_and_name(self):
        resolver = CardTextResolver({"123": {
            "grpId": 123, "name": "Shock",
            "oracleText": "Shock deals 2 damage to any target."}})
        self.assertEqual("Shock", resolver.resolve({"grpId": 123})["name"])
        self.assertEqual("Shock", resolver.resolve({"name": "shock"})["name"])
        self.assertIsNone(resolver.resolve({"grpId": 999}))


@unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from magic_cabt.models.structured_jepa import (MagicJEPA,
                                                       StructuredTensorizer)
        cls.config = StructuredJEPAConfig.preset("tiny")
        cls.model = MagicJEPA(cls.config).eval()
        cls.tensorizer = StructuredTensorizer(cls.config)

    def test_score_options_masks_padding(self):
        import torch
        records = [record(0), record(1)]
        short = dict(records[1])
        short["observation"] = dict(records[1]["observation"])
        short["observation"]["select"] = {
            "type": "PRIORITY",
            "option": [records[1]["observation"]["select"]["option"][0]]}
        with torch.no_grad():
            rows, mask = self.tensorizer.batch_states([records[0], short])
            options, option_mask = self.tensorizer.batch_options(
                [records[0], short])
            logits = self.model.score_options(rows, mask, options, option_mask)
        self.assertEqual((2, 2), tuple(logits.shape))
        self.assertTrue(torch.isfinite(logits[0]).all())
        self.assertEqual(float("-inf"), float(logits[1, 1]))

    def test_state_value_bounded(self):
        import torch
        with torch.no_grad():
            rows, mask = self.tensorizer.batch_states([record()])
            value = float(self.model.state_value(rows, mask)[0])
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_checkpoint_round_trip(self):
        import os
        import tempfile
        import torch
        from magic_cabt.models.structured_jepa import MagicJEPA
        with tempfile.TemporaryDirectory() as scratch:
            path = os.path.join(scratch, "checkpoint.pt")
            self.model.save_checkpoint(path, extra={"status": "test"})
            loaded, extra = MagicJEPA.load_checkpoint(path)
            self.assertEqual("test", extra["status"])
            self.assertEqual(self.config.d_model, loaded.config.d_model)
            with torch.no_grad():
                rows, mask = self.tensorizer.batch_states([record()])
                self.assertAlmostEqual(
                    float(self.model.state_value(rows, mask)[0]),
                    float(loaded.eval().state_value(rows, mask)[0]),
                    places=5)

    def test_embedding_dimension_mismatch_rejected(self):
        from magic_cabt.models.structured_jepa import StructuredTensorizer
        with self.assertRaises(ValueError):
            StructuredTensorizer(
                self.config, HashEmbeddingProvider(dimension=64))


if __name__ == "__main__":
    unittest.main()
