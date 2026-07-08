"""Tests for magic_cabt.training: DecisionRecord schema, normalizer, validator.

These cover the canonical-spec contract from issue #10:

- valid single-choice record passes,
- out-of-range / duplicate / under-minCount indices fail,
- self-play replay frames and trailing result lines normalize and patch a
  terminal marker,
- Java transition dataset records round-trip through the normalizer,
- Arena decisions.jsonl records lift their prompt spec out of
  ``observation.select`` and rename the recorder's indices field,
- The ``python -m magic_cabt.training.validate_dataset`` CLI exit code is
  nonzero on invalid input.
"""

import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.training import (
    SCHEMA_VERSION,
    normalize_record,
    iter_decision_records,
    validate_record,
    validate_records,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
DATASET_FIXTURE = os.path.join(FIXTURE_DIR, "dataset_sample.jsonl")

VALID_SINGLE_CHOICE = {
    "schemaVersion": SCHEMA_VERSION,
    "source": "engine_selfplay",
    "gameId": "g-1",
    "sequenceNumber": 0,
    "playerIndex": 0,
    "observation": {"phase": "MAIN"},
    "select": {
        "type": "PRIORITY",
        "minCount": 1,
        "maxCount": 1,
        "option": [
            {"index": 0, "type": "PASS", "label": "pass",
             "payload": {}},
            {"index": 1, "type": "ACTION", "label": "cast",
             "payload": {}},
        ],
    },
    "selectedIndices": [0],
    "nextObservation": None,
    "terminal": False,
    "reward": None,
    "result": None,
    "metadata": {"captureConfidence": "exact"},
}

SELF_PLAY_TWO_OPTIONS = [
    {"index": 0, "type": "PASS", "label": "pass", "payload": {}},
    {"index": 1, "type": "ACTION", "label": "attack", "payload": {}},
]


def _self_play_frame(sequence, player, selected):
    return {
        "sequence": sequence,
        "player": player,
        "observation": {
            "select": {
                "type": "PRIORITY",
                "minCount": 1,
                "maxCount": 1,
                "option": SELF_PLAY_TWO_OPTIONS,
            },
        },
        "selected": selected,
    }


def _arena_decision(sequence, seat, selected):
    return {
        "sequence": sequence,
        "matchId": "m1",
        "gameNumber": 1,
        "seat": seat,
        "player": "Alice",
        "promptTimestamp": "t1",
        "responseTimestamp": "t2",
        "selectionMatched": True,
        "promptMessageType": "GREMessageType_ActionsAvailableReq",
        "responseMessageType": "ClientMessageType_PerformActionResp",
        "observation": {
            "current": {"turnNumber": 3},
            "select": {
                "type": "ACTIONS",
                "minCount": 1,
                "maxCount": 1,
                "option": SELF_PLAY_TWO_OPTIONS,
            },
        },
        "select": selected,
    }


def _self_play_jsonl(records):
    return io.StringIO("".join(json.dumps(r) + "\n" for r in records))


class ValidateRecordTest(unittest.TestCase):

    def test_valid_single_choice_record_passes(self):
        errors = validate_record(copy.deepcopy(VALID_SINGLE_CHOICE))
        self.assertEqual([], errors)

    def test_invalid_out_of_range_selected_index(self):
        record = copy.deepcopy(VALID_SINGLE_CHOICE)
        record["selectedIndices"] = [2]
        messages = validate_record(record)
        self.assertTrue(any("out of range" in m for m in messages),
                        messages)

    def test_invalid_negative_selected_index(self):
        record = copy.deepcopy(VALID_SINGLE_CHOICE)
        record["selectedIndices"] = [-1]
        messages = validate_record(record)
        self.assertTrue(any("negative" in m for m in messages), messages)

    def test_invalid_duplicate_selected_indices(self):
        record = copy.deepcopy(VALID_SINGLE_CHOICE)
        record["select"]["minCount"] = 0
        record["selectedIndices"] = [0, 0]
        messages = validate_record(record)
        self.assertTrue(any("duplicates" in m for m in messages), messages)

    def test_invalid_selected_count_below_min_count(self):
        record = copy.deepcopy(VALID_SINGLE_CHOICE)
        record["selectedIndices"] = []
        messages = validate_record(record)
        self.assertTrue(any("below minCount" in m for m in messages),
                        messages)

    def test_invalid_selected_count_exceeds_max_count(self):
        record = copy.deepcopy(VALID_SINGLE_CHOICE)
        record["select"]["maxCount"] = 1
        record["selectedIndices"] = [0, 1]
        messages = validate_record(record)
        self.assertTrue(any("exceeds maxCount" in m for m in messages),
                        messages)

    def test_bool_indices_reject(self):
        record = copy.deepcopy(VALID_SINGLE_CHOICE)
        record["selectedIndices"] = [True]
        messages = validate_record(record)
        self.assertTrue(any("is not an int" in m for m in messages),
                        messages)

    def test_terminal_record_without_result_or_reward_warns(self):
        record = copy.deepcopy(VALID_SINGLE_CHOICE)
        record["terminal"] = True
        record["result"] = None
        record["reward"] = None
        messages = validate_record(record)
        self.assertTrue(
            any("terminal record has no result or reward" in m for m in messages),
            messages,
        )

    def test_wrong_schema_version_fails(self):
        record = copy.deepcopy(VALID_SINGLE_CHOICE)
        record["schemaVersion"] = 99
        self.assertTrue(validate_record(record))

    def test_unknown_capture_confidence_flagged(self):
        record = copy.deepcopy(VALID_SINGLE_CHOICE)
        record["metadata"]["captureConfidence"] = "guessed"
        messages = validate_record(record)
        self.assertTrue(
            any("captureConfidence is not a known value" in m for m in messages),
            messages,
        )

    def test_detectable_leak_marker_fails(self):
        record = copy.deepcopy(VALID_SINGLE_CHOICE)
        record["metadata"]["knownHiddenLeak"] = True
        messages = validate_record(record)
        self.assertTrue(any("hidden-information leakage" in m for m in messages),
                        messages)

    def test_non_canonical_source_flagged(self):
        record = copy.deepcopy(VALID_SINGLE_CHOICE)
        record["source"] = "watson"
        messages = validate_record(record)
        self.assertTrue(
            any("source is not a canonical label" in m for m in messages),
            messages,
        )


class NormalizeRecordTest(unittest.TestCase):

    def test_self_play_replay_normalizes(self):
        record = normalize_record(_self_play_frame(7, 1, [1]),
                                   source_hint="engine_selfplay")
        self.assertEqual(SCHEMA_VERSION, record["schemaVersion"])
        self.assertEqual("engine_selfplay", record["source"])
        self.assertEqual(7, record["sequenceNumber"])
        self.assertEqual(1, record["playerIndex"])
        self.assertEqual([1], record["selectedIndices"])
        self.assertEqual("PRIORITY", record["select"]["type"])
        self.assertEqual("exact", record["metadata"]["captureConfidence"])
        # observation.select was lifted out of observation.select
        self.assertNotIn("select", record["observation"])

    def test_java_transition_normalizes(self):
        with open(DATASET_FIXTURE, encoding="utf-8") as handle:
            raw = json.loads(handle.readline().strip())
        record = normalize_record(raw, source_hint="engine_selfplay")
        self.assertEqual(SCHEMA_VERSION, record["schemaVersion"])
        self.assertEqual(0, record["sequenceNumber"])
        self.assertEqual(0, record["playerIndex"])
        self.assertEqual([0], record["selectedIndices"])
        self.assertIn("decisionMethod", record["metadata"])
        self.assertEqual("exact", record["metadata"]["captureConfidence"])

    def test_arena_decision_normalizes_with_fixture(self):
        arena = _arena_decision(2, 1, [0])
        record = normalize_record(arena)
        # prompt spec moved to canonical top-level ``select``.
        self.assertEqual("ACTIONS", record["select"]["type"])
        self.assertEqual(2, len(record["select"]["option"]))
        # the recorder's top-level ``select`` (chosen indices) became
        # selectedIndices.
        self.assertEqual([0], record["selectedIndices"])
        self.assertEqual(1, record["playerIndex"])
        self.assertEqual("arena_human", record["source"])
        self.assertEqual("mirror", record["metadata"]["captureConfidence"])
        # Arena envelope preserved under metadata.
        for key in ("matchId", "selectionMatched",
                    "promptMessageType", "responseMessageType"):
            self.assertIn(key, record["metadata"])
        # seat name preserved (renamed to avoid the player/playerIndex clash).
        self.assertEqual("Alice", record["metadata"]["playerName"])

    def test_source_hint_overrides_detected_source(self):
        replay = normalize_record(_self_play_frame(0, 0, [0]),
                                  source_hint="engine_human")
        self.assertEqual("engine_human", replay["source"])

    def test_non_dict_record_raises_value_error(self):
        with self.assertRaises(ValueError):
            normalize_record("not a record")

    def test_self_play_result_terminal_warning_not_a_record(self):
        # codec check: terminal with no result/reward is a warning, not a
        # validation success, so terminal flag alone does not satisfy v1.
        normalized = normalize_record(_self_play_frame(0, 0, [0]))
        normalized["terminal"] = True
        messages = validate_record(normalized)
        self.assertTrue(
            any("terminal record has no result or reward" in m for m in messages),
            messages,
        )


class IterDecisionRecordsTest(unittest.TestCase):

    def test_self_play_replay_attaches_terminal_to_last_decision(self):
        records = [
            _self_play_frame(0, 0, [0]),
            _self_play_frame(1, 0, [1]),
            {"result": {"winner": 0}},
        ]
        normalized = list(iter_decision_records(_self_play_jsonl(records)))
        self.assertEqual(2, len(normalized))
        self.assertFalse(normalized[0]["terminal"])
        self.assertTrue(normalized[1]["terminal"])
        self.assertEqual({"winner": 0}, normalized[1]["result"])
        self.assertEqual(1.0, normalized[1]["reward"])

    def test_self_play_replay_yields_normally_without_result_line(self):
        records = [_self_play_frame(0, 0, [0])]
        normalized = list(iter_decision_records(_self_play_jsonl(records)))
        self.assertEqual(1, len(normalized))
        self.assertFalse(normalized[0]["terminal"])

    def test_java_transition_fixture_reads_canonical(self):
        normalized = list(iter_decision_records(DATASET_FIXTURE))
        self.assertEqual(2, len(normalized))
        for record in normalized:
            self.assertEqual(SCHEMA_VERSION, record["schemaVersion"])
            self.assertEqual([0], record["selectedIndices"])
            self.assertIn("playerIndex", record)
            self.assertIn("captureConfidence", record["metadata"])

    def test_invalid_json_line_raises_with_line_number(self):
        handle = io.StringIO('{"schemaVersion": 1}\nnot json\n')
        with self.assertRaises(ValueError) as ctx:
            list(iter_decision_records(handle))
        self.assertIn("line 2", str(ctx.exception))

    def test_arena_decision_fixture_round_trip(self):
        arena = {"arena": _arena_decision(0, 1, [0])}
        records = [arena["arena"]]
        normalized = list(iter_decision_records(_self_play_jsonl(records)))
        self.assertEqual(1, len(normalized))
        self.assertEqual([0], normalized[0]["selectedIndices"])
        self.assertEqual("arena_human", normalized[0]["source"])


class ValidateRecordsTest(unittest.TestCase):

    def test_validate_records_aggregates_distributions_and_counts(self):
        records = [copy.deepcopy(VALID_SINGLE_CHOICE),
                   copy.deepcopy(VALID_SINGLE_CHOICE)]
        records[1]["selectedIndices"] = [2]
        records[1]["playerIndex"] = 1
        summary = validate_records(records)
        self.assertEqual(2, summary["total"])
        self.assertEqual(1, summary["valid"])
        self.assertEqual(1, summary["invalid"])
        self.assertEqual({"PRIORITY": 2}, summary["selectTypes"])
        self.assertIn("1", summary["selectedCount"])
        self.assertEqual(1, len(summary["errors"]))
        self.assertEqual(2, summary["selectTypes"]["PRIORITY"])


class ValidateRecordsDistributionTest(unittest.TestCase):

    def test_option_type_and_selected_count_distributions(self):
        records = [copy.deepcopy(VALID_SINGLE_CHOICE)]
        records[0]["selectedIndices"] = [1]
        summary = validate_records(records)
        self.assertEqual({"PRIORITY": 1}, summary["selectTypes"])
        self.assertEqual({"PASS": 1, "ACTION": 1}, summary["optionTypes"])
        self.assertEqual({"1": 1}, summary["selectedCount"])


class ValidateDatasetCLITest(unittest.TestCase):

    def _run(self, *args, fixture_path=None):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            os.path.join(os.path.dirname(__file__), os.pardir))
        if fixture_path is not None:
            args = list(args)
            args.append(fixture_path)
        proc = subprocess.run(
            [sys.executable, "-m", "magic_cabt.training.validate_dataset", *args],
            env=env, cwd=os.path.join(os.path.dirname(__file__), os.pardir),
            capture_output=True, text=True)
        return proc

    def test_cli_exits_zero_on_valid_fixture(self):
        proc = self._run(fixture_path=DATASET_FIXTURE)
        self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)
        self.assertIn("total records:", proc.stdout)
        self.assertIn("valid records:", proc.stdout)
        self.assertIn("select.type distribution:", proc.stdout)

    def test_cli_exits_nonzero_on_invalid_input(self):
        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, "broken.jsonl")
            record = copy.deepcopy(VALID_SINGLE_CHOICE)
            record["selectedIndices"] = [99]
            record["playerIndex"] = 0
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            proc = self._run(fixture_path=path)
            self.assertEqual(1, proc.returncode, proc.stdout + proc.stderr)
            self.assertIn("invalid records:  1", proc.stdout)
            self.assertIn("validation errors", proc.stdout)
            self.assertIn("out of range", proc.stdout)
        finally:
            import shutil
            shutil.rmtree(directory, ignore_errors=True)

    def test_cli_normalizes_self_play_replay(self):
        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, "replay.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(_self_play_frame(0, 0, [0]))
                             + "\n")
                handle.write(json.dumps(_self_play_frame(1, 0, [1]))
                             + "\n")
                handle.write(json.dumps({"result": {"winner": 0}})
                             + "\n")
            proc = self._run(fixture_path=path)
            self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)
            self.assertIn("total records:    2", proc.stdout)
            self.assertIn("valid records:    2", proc.stdout)
        finally:
            import shutil
            shutil.rmtree(directory, ignore_errors=True)

    def test_cli_missing_dataset_returns_nonzero(self):
        proc = self._run(fixture_path="/no/such/file.jsonl")
        self.assertNotEqual(0, proc.returncode)


if __name__ == "__main__":
    unittest.main()