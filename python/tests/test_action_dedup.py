import unittest

from magic_cabt.training.action_dedup import (
    canonical_groups,
    canonical_key_of,
    group_index_of,
    representative_index,
)
from magic_cabt.training.features import canonical_text, option_text


def target_option(index, key=None, label="target"):
    payload = {"targetInstanceId": 100 + index}
    if key is not None:
        payload["canonicalKey"] = key
    return {"index": index, "type": "TARGET", "label": label,
            "payload": payload}


class CanonicalGroupsTest(unittest.TestCase):
    def test_identical_tokens_collapse_to_one_action(self):
        select = {"option": [
            target_option(0, key="token-a"),
            target_option(1, key="token-a"),
            target_option(2, key="other"),
        ]}
        groups = canonical_groups(select)
        self.assertEqual(2, len(groups))
        self.assertEqual([0, 1], groups[0]["indices"])
        self.assertEqual([2], groups[1]["indices"])

    def test_options_without_key_are_never_merged(self):
        select = {"option": [
            target_option(0),
            target_option(1),
        ]}
        groups = canonical_groups(select)
        self.assertEqual(2, len(groups))
        self.assertIsNone(groups[0]["key"])

    def test_equal_keys_with_different_types_stay_distinct(self):
        select = {"option": [
            {"index": 0, "type": "TARGET", "payload": {"canonicalKey": "k"}},
            {"index": 1, "type": "ATTACKER", "payload": {"canonicalKey": "k"}},
        ]}
        self.assertEqual(2, len(canonical_groups(select)))

    def test_recorded_index_folds_into_its_group(self):
        select = {"option": [
            target_option(0, key="token-a"),
            target_option(1, key="token-a"),
            target_option(2, key="other"),
        ]}
        groups = canonical_groups(select)
        # a human who clicked the second identical token trained the same
        # canonical action as one who clicked the first
        self.assertEqual(group_index_of(groups, 0), group_index_of(groups, 1))
        self.assertNotEqual(group_index_of(groups, 1), group_index_of(groups, 2))
        self.assertIsNone(group_index_of(groups, 99))

    def test_representative_index_is_concrete_and_lowest(self):
        select = {"option": [
            target_option(0, key="token-a"),
            target_option(1, key="token-a"),
        ]}
        groups = canonical_groups(select)
        index = representative_index(groups[0])
        self.assertEqual(0, index)
        # the concrete option still carries its real instance id for execution
        self.assertEqual(100, select["option"][index]["payload"]["targetInstanceId"])

    def test_canonical_key_of_rejects_non_string_keys(self):
        self.assertIsNone(canonical_key_of({"payload": {"canonicalKey": 5}}))
        self.assertIsNone(canonical_key_of({"payload": {}}))
        self.assertIsNone(canonical_key_of({}))


class CanonicalTextTest(unittest.TestCase):
    def test_strips_arena_instance_ids(self):
        self.assertEqual("target (group 2)",
                         canonical_text("target instance=101 (group 2)"))

    def test_strips_uuid_literals(self):
        self.assertEqual(
            "sacrifice",
            canonical_text("sacrifice 6a5c0b7e-1234-4abc-9def-001122334455"))

    def test_preserves_grp_id_card_identity(self):
        self.assertEqual("play grpId=1234",
                         canonical_text("play grpId=1234 instance=55"))

    def test_identical_tokens_yield_identical_option_text(self):
        a = {"type": "TARGET", "label": "target instance=101 (group 1)",
             "payload": {"name": "Zombie"}}
        b = {"type": "TARGET", "label": "target instance=102 (group 1)",
             "payload": {"name": "Zombie"}}
        self.assertEqual(option_text(a), option_text(b))


if __name__ == "__main__":
    unittest.main()
