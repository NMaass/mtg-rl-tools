"""Task 22: the Python reader loads JSONL transition records.

The fixture is real Java output: CabtDatasetWriterTest regenerates
target/cabt-fixtures/dataset_sample.jsonl on every run, and the checked-in
copy under fixtures/ is taken from there.
"""

import io
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt import read_dataset
from magic_cabt.dataset import SCHEMA_VERSION

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "dataset_sample.jsonl")


class ReadDatasetTest(unittest.TestCase):

    def test_python_dataset_reader_loads_records(self):
        records = list(read_dataset(FIXTURE))
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record["schemaVersion"], SCHEMA_VERSION)
            self.assertIn("observation", record)
            self.assertIn("select", record)
            self.assertIn("nextObservation", record)
            self.assertIn("selectedIndices", record)
            self.assertIn("metadata", record)
        self.assertFalse(records[0]["terminal"])
        self.assertTrue(records[1]["terminal"])
        self.assertIsNone(records[1]["nextObservation"])

    def test_records_contain_no_outcome_labels(self):
        forbidden = {"countered", "destroyed", "fizzled", "succeeded", "removalSucceeded"}

        def keys_of(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield key
                    for nested in keys_of(child):
                        yield nested
            elif isinstance(value, list):
                for child in value:
                    for nested in keys_of(child):
                        yield nested

        for record in read_dataset(FIXTURE):
            self.assertFalse(forbidden.intersection(keys_of(record)))

    def test_reads_from_file_object_and_skips_blank_lines(self):
        record = {"schemaVersion": SCHEMA_VERSION, "observation": {}, "select": {},
                  "selectedIndices": [0], "nextObservation": None}
        text = json.dumps(record) + "\n\n" + json.dumps(record) + "\n"
        records = list(read_dataset(io.StringIO(text)))
        self.assertEqual(len(records), 2)

    def test_invalid_line_raises_with_line_number(self):
        with self.assertRaises(ValueError) as ctx:
            list(read_dataset(io.StringIO('{"ok": 1}\nnot json\n')))
        self.assertIn("line 2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
