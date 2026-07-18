import json
import os
import subprocess
import tempfile
import unittest

from mtgo_video_acquisition.manifest import SourceEntry, SourceManifest
from mtgo_video_acquisition.ytdlp import YtDlpClient


class AcquisitionTest(unittest.TestCase):
    def test_discovery_builds_manifest(self):
        payload = {"extractor": "youtube:tab", "entries": [
            {"id": "abc", "title": "MTGO Match", "url": "https://example/abc",
             "channel": "Channel", "duration": 100},
        ]}
        def runner(command, **kwargs):
            return subprocess.CompletedProcess(command, 0,
                                               stdout=json.dumps(payload), stderr="")
        manifest = YtDlpClient("yt-dlp", runner=runner).discover(
            "https://example/playlist")
        self.assertEqual(1, len(manifest.entries))
        self.assertEqual("MTGO Match", manifest.entries[0].title)

    def test_download_requires_rights_acknowledgement(self):
        client = YtDlpClient("yt-dlp", runner=lambda *args, **kwargs: None)
        with self.assertRaises(PermissionError):
            client.download(SourceEntry(source_url="https://example/video"),
                            "unused", rights_acknowledged=False)

    def test_local_file_hash_round_trip(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = os.path.join(scratch, "x.mp4")
            with open(path, "wb") as handle:
                handle.write(b"video")
            row = SourceEntry()
            row.attach_local_file(path)
            manifest_path = os.path.join(scratch, "sources.json")
            SourceManifest([row]).save(manifest_path)
            loaded = SourceManifest.load(manifest_path)
            self.assertEqual(row.sha256, loaded.entries[0].sha256)


if __name__ == "__main__":
    unittest.main()
