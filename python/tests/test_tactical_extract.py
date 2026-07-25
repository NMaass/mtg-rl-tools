import json
import os
import tempfile
import unittest
from unittest import mock

from magic_cabt.analysis.schema import decision_fingerprint
from magic_cabt.research import cli as research_cli
from magic_cabt.research.tactical import validate_scenarios
from magic_cabt.research.tactical_extract import (
    extract_scenario,
    load_decision_records,
    write_scenario_row,
)


def option(index, kind, key, label=None):
    return {"index": index, "type": kind, "label": label or key,
            "payload": {"canonicalKey": key}}


def record(seq, game="g1", options=None, selected=None):
    return {
        "gameId": game,
        "gameNumber": 1,
        "sequenceNumber": seq,
        "observation": {"current": {"gameInstance": game}},
        "select": {"type": "PRIORITY", "minCount": 1, "maxCount": 1,
                   "option": options or [
                       option(0, "PASS", "pass"),
                       option(1, "CAST", "cast:bolt"),
                   ]},
        "selectedIndices": selected if selected is not None else [0],
    }


class TacticalExtractTest(unittest.TestCase):
    def test_index_selection_adds_same_game_bounded_history(self):
        rows = [record(1), record(1, game="other"), record(2), record(3)]
        value = extract_scenario(
            rows, "suite-v1", "s1", decision_index=3, history_limit=2)
        self.assertEqual([1, 2], [row["sequenceNumber"] for row in value["history"]])
        self.assertEqual(3, value["decision"]["sequenceNumber"])
        self.assertEqual("needs-expert-review", value["annotation"]["reviewStatus"])
        self.assertEqual(["PASS|pass"], value["annotation"]["recordedActionKeys"])

    def test_fingerprint_selection_and_approved_expectation_validate(self):
        rows = [record(1), record(2, game="g2")]
        value = extract_scenario(
            rows, "suite-v1", "s1",
            fingerprint=decision_fingerprint(rows[1]),
            acceptable=["CAST|cast:bolt"], tags=["burn"])
        errors, warnings = validate_scenarios([value])
        self.assertEqual([], errors)
        self.assertEqual([], warnings)
        self.assertEqual("approved", value["annotation"]["reviewStatus"])

    def test_canonical_duplicates_are_one_candidate(self):
        rows = [record(1, options=[
            option(0, "TARGET", "bear", "Bear A"),
            option(1, "TARGET", "bear", "Bear B"),
            option(2, "TARGET", "opponent", "Opponent"),
        ])]
        value = extract_scenario(rows, "suite-v1", "s1", decision_index=0)
        self.assertEqual(2, len(value["candidateActions"]))
        self.assertEqual([0, 1], value["candidateActions"][0]["concreteIndices"])

    def test_draft_intentionally_fails_benchmark_validation_until_reviewed(self):
        value = extract_scenario([record(1)], "suite-v1", "s1", decision_index=0)
        errors, _warnings = validate_scenarios([value])
        self.assertTrue(any("acceptableActionKeys must be non-empty" in error
                            for error in errors))

    def test_hidden_information_is_rejected(self):
        row = record(1)
        row["observation"]["oracleLabels"] = {"hand": [1]}
        with self.assertRaisesRegex(ValueError, "oracle/private"):
            extract_scenario([row], "suite-v1", "s1", decision_index=0)

    def test_ambiguous_fingerprint_is_rejected(self):
        row = record(1)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            extract_scenario(
                [row, dict(row)], "suite-v1", "s1",
                fingerprint=decision_fingerprint(row))

    def test_history_requires_game_identity(self):
        row = record(1)
        row.pop("gameId")
        row.pop("gameNumber")
        row["observation"]["current"].pop("gameInstance")
        with self.assertRaisesRegex(ValueError, "game identity"):
            extract_scenario([row, row], "suite-v1", "s1",
                             decision_index=1, history_limit=1)

    def test_load_bundle_and_write_jsonl_row(self):
        with tempfile.TemporaryDirectory() as scratch:
            bundle = os.path.join(scratch, "bundle")
            os.makedirs(bundle)
            decisions = os.path.join(bundle, "decisions.jsonl")
            with open(decisions, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record(1)) + "\n")
            resolved, rows = load_decision_records(bundle)
            value = extract_scenario(
                rows, "suite-v1", "s1", decision_index=0,
                source_path=resolved, source_sha256="sha256:test")
            output = os.path.join(scratch, "scenario.jsonl")
            write_scenario_row(output, value)
            with open(output, encoding="utf-8") as handle:
                loaded = json.loads(handle.read())
        self.assertEqual("decisions.jsonl", loaded["provenance"]["sourceFile"])
        self.assertNotIn("__source_line", loaded["decision"])

    def test_cli_writes_approved_scenario_and_provenance(self):
        rows = [record(1)]
        with tempfile.TemporaryDirectory() as scratch:
            source = os.path.join(scratch, "decisions.jsonl")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            output = os.path.join(scratch, "suite.jsonl")
            with mock.patch.object(
                    research_cli, "load_decision_records",
                    return_value=(source, rows)):
                exit_code = research_cli.main([
                    "extract-scenario", "--input", source,
                    "--decision-index", "0", "--suite", "suite-v1",
                    "--scenario-id", "s1", "--acceptable", "CAST|cast:bolt",
                    "--tag", "burn", "--out", output,
                ])
            self.assertEqual(0, exit_code)
            with open(output, encoding="utf-8") as handle:
                value = json.loads(handle.read())
        self.assertRegex(value["provenance"]["sourceSha256"],
                         r"^sha256:[0-9a-f]{64}$")
        self.assertEqual("approved", value["annotation"]["reviewStatus"])
        self.assertEqual(["burn"], value["tags"])

    def test_cli_refuses_to_append_unreviewed_draft(self):
        with tempfile.TemporaryDirectory() as scratch:
            output = os.path.join(scratch, "suite.jsonl")
            exit_code = research_cli.main([
                "extract-scenario", "--input", output,
                "--decision-index", "0", "--suite", "suite-v1",
                "--scenario-id", "s1", "--append", "--out", output,
            ])
        self.assertEqual(2, exit_code)


if __name__ == "__main__":
    unittest.main()
