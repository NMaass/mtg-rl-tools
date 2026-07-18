"""Headless MTGO video parser."""

from .layout import LayoutProfile, Region
from .video import FrameSample, FrameSampler
from .ocr import (
    OCRBackend,
    OCRSpan,
    OpenAICompatibleVisionBackend,
    PaddleOCRBackend,
    TesseractOCRBackend,
    make_ocr_backend,
)
from .pipeline import VideoExtractionPipeline
from .tracker import MTGOStateTracker
from .actions import MTGOLogActionParser, ObservedAction
from .recognition import CardIdentityRecognizer, RecognitionAttempt
from .structured_vision import (
    SecondaryVisionPolicy, StructuredMTGOScreenReader)

__all__ = [
    "LayoutProfile", "Region", "FrameSample", "FrameSampler", "OCRBackend",
    "OCRSpan", "OpenAICompatibleVisionBackend", "PaddleOCRBackend",
    "TesseractOCRBackend", "make_ocr_backend", "VideoExtractionPipeline",
    "MTGOStateTracker", "MTGOLogActionParser", "ObservedAction",
    "CardIdentityRecognizer", "RecognitionAttempt", "SecondaryVisionPolicy",
    "StructuredMTGOScreenReader",
]
