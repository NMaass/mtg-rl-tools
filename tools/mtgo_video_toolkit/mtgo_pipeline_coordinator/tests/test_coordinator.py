import os
import tempfile
import unittest
from pathlib import Path

from mtgo_pipeline_coordinator.assemble import assemble
from mtgo_pipeline_coordinator.batch import CorpusRunner
from mtgo_pipeline_coordinator.bundle import PipelineBundle
from mtgo_video_acquisition.manifest import SourceEntry, SourceManifest


class CoordinatorTest(unittest.TestCase):
    def test_bundle_inventory_hashes_files(self):
        with tempfile.TemporaryDirectory() as scratch:
            bundle = PipelineBundle(scratch)
            Path(scratch, "x.txt").write_text("hello", encoding="utf-8")
            inventory = bundle.inventory()
            self.assertEqual("x.txt", inventory["files"][0]["path"])
            self.assertEqual(64, len(inventory["files"][0]["sha256"]))

    def test_assembler_only_writes_destination(self):
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch, "toolkit")
            repo = Path(scratch, "repo")
            source.mkdir()
            repo.mkdir()
            (source / "README.md").write_text("toolkit", encoding="utf-8")
            (repo / "existing.txt").write_text("keep", encoding="utf-8")
            result = assemble(str(source), str(repo))
            self.assertEqual("keep", (repo / "existing.txt").read_text())
            self.assertTrue((repo / result["destination"] / "README.md").exists())

    def test_corpus_runner_is_resumable(self):
        with tempfile.TemporaryDirectory() as scratch:
            video = Path(scratch, "a.mp4")
            video.write_bytes(b"x")
            entry = SourceEntry(title="Match", source_id="abc")
            entry.attach_local_file(str(video))
            sources = Path(scratch, "sources.json")
            SourceManifest([entry]).save(str(sources))

            calls = []
            def extract(path, destination):
                calls.append(path)
                target = Path(destination)
                target.mkdir(parents=True, exist_ok=True)
                (target / "manifest.json").write_text("{}", encoding="utf-8")
                return {"path": destination}

            root = Path(scratch, "runs")
            first = CorpusRunner(extract).run(str(sources), str(root))
            second = CorpusRunner(extract).run(str(sources), str(root))
            self.assertEqual(1, len(first["completed"]))
            self.assertEqual(1, len(second["skipped"]))
            self.assertEqual(1, len(calls))


if __name__ == "__main__":
    unittest.main()
