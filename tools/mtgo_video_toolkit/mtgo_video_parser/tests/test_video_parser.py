import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from mtgo_video_parser.layout import LayoutProfile
from mtgo_video_parser.carddb import CardNameResolver
from mtgo_video_parser.ocr import (
    FixtureOCRBackend, TesseractOCRBackend, _paddle_bbox, _parse_json_content)
from mtgo_video_parser.pipeline import VideoExtractionPipeline
from mtgo_video_parser.actions import MTGOLogActionParser
from mtgo_video_parser.recognition import CardIdentityRecognizer
from mtgo_video_parser.tracker import MTGOStateTracker
from mtgo_video_parser.types import DetectedCard, OCRSpan, PerceivedFrame


def make_video(path):
    width, height = 640, 360
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5.0,
                             (width, height))
    for index in range(15):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(image, "20", (580, 310), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, "17", (580, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 3, cv2.LINE_AA)
        if index >= 6:
            cv2.rectangle(image, (100, 190), (150, 260), (200, 200, 200), 2)
        writer.write(image)
    writer.release()


class ParserTest(unittest.TestCase):
    def test_action_parser(self):
        parser = MTGOLogActionParser()
        actions = parser.parse([
            OCRSpan("Alice casts Lightning Bolt targeting Bob.", 0.95)
        ], 1000)
        self.assertEqual("CAST_SPELL", actions[0].action_type)
        self.assertEqual("Lightning Bolt", actions[0].card_name)

    def test_headless_synthetic_video_pipeline(self):
        profile = LayoutProfile.from_dict({
            "name": "test",
            "regions": {
                "local_life": {"x": .85, "y": .78, "width": .15,
                               "height": .18, "kind": "integer"},
                "opponent_life": {"x": .85, "y": 0, "width": .15,
                                  "height": .18, "kind": "integer"},
                "game_log": {"x": 0, "y": 0, "width": .4,
                             "height": .2, "kind": "log"},
                "local_battlefield": {"x": 0, "y": .45, "width": .7,
                                      "height": .4, "kind": "card_zone",
                                      "config": {"zone": "battlefield",
                                                 "controller": "1"}},
            },
        })
        ocr = FixtureOCRBackend({
            "local_life": [{"text": "20", "confidence": .99}],
            "opponent_life": [{"text": "17", "confidence": .99}],
            "game_log": [{"text": "Alice plays Forest.", "confidence": .9}],
        })
        with tempfile.TemporaryDirectory() as scratch:
            video = os.path.join(scratch, "fixture.mp4")
            make_video(video)
            out = os.path.join(scratch, "out")
            manifest = VideoExtractionPipeline(profile, ocr).run(
                video, out, fps=2, change_threshold=0,
                max_interval_seconds=.5)
            self.assertGreater(manifest["counts"]["sampledFrames"], 0)
            self.assertGreater(manifest["counts"]["emittedStates"], 0)
            with open(os.path.join(out, "canonical_states.jsonl"),
                      encoding="utf-8") as handle:
                state = json.loads(next(iter(handle)))
            self.assertEqual(20, state["players"][0]["life"])
            self.assertEqual(17, state["players"][1]["life"])

    def test_tesseract_reads_synthetic_number_when_available(self):
        image = np.zeros((100, 180, 3), dtype=np.uint8)
        cv2.putText(image, "20", (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    2.4, (255, 255, 255), 5, cv2.LINE_AA)
        try:
            spans = TesseractOCRBackend(psm=7).read(image)
        except Exception as error:
            self.skipTest(str(error))
        self.assertIn("20", " ".join(row.text for row in spans))

    def test_scrolling_log_keeps_repeated_events(self):
        tracker = MTGOStateTracker(emit_duplicate_states=True)
        first = PerceivedFrame(
            1, 1000, 1.0,
            log_lines=[OCRSpan("Alice draws a card.", .9)])
        second = PerceivedFrame(
            2, 2000, 1.0,
            log_lines=[OCRSpan("Alice draws a card.", .9),
                       OCRSpan("Alice draws a card.", .9)])
        tracker.update(first)
        _state, actions = tracker.update(second)
        self.assertEqual(1, len(actions))
        self.assertEqual(2, len(tracker.actions))

    def test_paddle_boxes_are_normalized_to_xywh(self):
        self.assertEqual([10, 20, 30, 40],
                         _paddle_bbox([10, 20, 40, 60]))
        self.assertEqual([10, 20, 30, 40],
                         _paddle_bbox([[10, 20], [40, 20],
                                       [40, 60], [10, 60]]))

    def test_vlm_json_parser_accepts_fenced_json(self):
        value = _parse_json_content('```json\n{"lines": []}\n```')
        self.assertEqual([], value["lines"])

    def test_tracker_pseudonymizes_visible_player_names_by_default(self):
        tracker = MTGOStateTracker(emit_duplicate_states=True)
        _state, actions = tracker.update(PerceivedFrame(
            1, 1000, 1.0,
            log_lines=[OCRSpan("Alice plays Forest.", .9)]))
        self.assertEqual("player:1", actions[0].actor)
        self.assertTrue(actions[0].metadata["actorPseudonymized"])

    def test_card_title_recognition_is_dictionary_gated(self):
        resolver = CardNameResolver([
            {"name": "Lightning Bolt", "oracle_id": "bolt"},
            {"name": "Forest", "oracle_id": "forest"},
        ])
        ocr = FixtureOCRBackend({
            "card-title:upright:rgb": [
                {"text": "Lightning Bolt", "confidence": .95}],
        })
        frame = np.zeros((120, 90, 3), dtype=np.uint8)
        card = DetectedCard([0, 0, 90, 120], "battlefield", "1",
                            confidence=.7)
        CardIdentityRecognizer(ocr, resolver).recognize(frame, card)
        self.assertEqual("Lightning Bolt", card.name)
        self.assertEqual("bolt", card.metadata["oracleId"])


if __name__ == "__main__":
    unittest.main()
