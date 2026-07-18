"""Pluggable OCR/VLM backends.

The OpenAI-compatible backend works with both OpenRouter and a local server
(vLLM, SGLang, LM Studio, or another compatible endpoint).  It is intended as a
second-pass structured reader, while Tesseract/Paddle are efficient first-pass
readers for every sampled frame.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from io import BytesIO
from typing import Any, Dict, List, Optional
import base64
import json
import os
import re

import cv2
import numpy as np
import requests
from PIL import Image

from .types import OCRSpan


class OCRBackend(ABC):
    name = "abstract"

    @abstractmethod
    def read(self, image: np.ndarray, *, hint: Optional[str] = None) -> List[OCRSpan]:
        raise NotImplementedError

    def extract_json(self, image: np.ndarray, prompt: str,
                     schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.name} does not support structured vision")


class TesseractOCRBackend(OCRBackend):
    name = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = 6,
                 executable: Optional[str] = None):
        import pytesseract
        self.pytesseract = pytesseract
        self.lang = lang
        self.psm = int(psm)
        if executable:
            self.pytesseract.pytesseract.tesseract_cmd = executable

    def read(self, image: np.ndarray, *, hint: Optional[str] = None) -> List[OCRSpan]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        config = f"--psm {self.psm}"
        data = self.pytesseract.image_to_data(
            rgb, lang=self.lang, config=config,
            output_type=self.pytesseract.Output.DICT)
        result = []
        for index, text in enumerate(data.get("text") or []):
            text = str(text).strip()
            if not text:
                continue
            try:
                confidence = max(0.0, min(1.0, float(data["conf"][index]) / 100.0))
            except (TypeError, ValueError):
                confidence = 0.0
            result.append(OCRSpan(
                text=text, confidence=confidence,
                bbox=[int(data["left"][index]), int(data["top"][index]),
                      int(data["width"][index]), int(data["height"][index])]))
        return result


class PaddleOCRBackend(OCRBackend):
    name = "paddleocr"

    def __init__(self, device: str = "gpu:0", lang: str = "en",
                 ocr_version: str = "PP-OCRv6"):
        try:
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise ImportError(
                "Paddle backend requires `pip install paddleocr` plus a matching "
                "PaddlePaddle CPU/GPU wheel") from error
        kwargs = {"device": device, "lang": lang, "ocr_version": ocr_version}
        try:
            self.engine = PaddleOCR(**kwargs)
        except TypeError:
            # Older PaddleOCR releases use legacy flags.
            self.engine = PaddleOCR(lang=lang, use_gpu=device.startswith("gpu"))

    def read(self, image: np.ndarray, *, hint: Optional[str] = None) -> List[OCRSpan]:
        raw = self.engine.predict(image)
        result: List[OCRSpan] = []
        for page in raw or []:
            payload = page.json if hasattr(page, "json") else page
            if callable(payload):
                payload = payload()
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except ValueError:
                    payload = {}
            rows = []
            if isinstance(payload, dict):
                rows = payload.get("res", {}).get("rec_texts") or \
                    payload.get("rec_texts") or []
                scores = payload.get("res", {}).get("rec_scores") or \
                    payload.get("rec_scores") or []
                boxes = payload.get("res", {}).get("rec_boxes") or \
                    payload.get("rec_boxes") or []
                for index, text in enumerate(rows):
                    box = boxes[index] if index < len(boxes) else None
                    bbox = _paddle_bbox(box)
                    result.append(OCRSpan(
                        str(text), float(scores[index]) if index < len(scores) else 0.0,
                        bbox))
            elif isinstance(page, list):
                for row in page:
                    if not row or len(row) < 2:
                        continue
                    points, pair = row[0], row[1]
                    text, score = pair[0], pair[1]
                    xs = [point[0] for point in points]
                    ys = [point[1] for point in points]
                    result.append(OCRSpan(
                        str(text), float(score),
                        [int(min(xs)), int(min(ys)), int(max(xs) - min(xs)),
                         int(max(ys) - min(ys))]))
        return result


def _paddle_bbox(box):
    if box is None:
        return None
    try:
        values = np.asarray(box).reshape(-1, 2)
    except Exception:
        try:
            flat = [float(value) for value in box]
        except Exception:
            return None
        if len(flat) == 4:
            x1, y1, x2, y2 = flat
            return [int(x1), int(y1), int(max(0, x2 - x1)),
                    int(max(0, y2 - y1))]
        return None
    if values.size < 4:
        return None
    xs = values[:, 0]
    ys = values[:, 1]
    x1, y1 = float(xs.min()), float(ys.min())
    x2, y2 = float(xs.max()), float(ys.max())
    return [int(x1), int(y1), int(max(0, x2 - x1)),
            int(max(0, y2 - y1))]


class OpenAICompatibleVisionBackend(OCRBackend):
    name = "openai-compatible-vision"

    def __init__(self, model: str,
                 endpoint: str = "https://openrouter.ai/api/v1/chat/completions",
                 api_key: Optional[str] = None,
                 timeout_seconds: float = 90.0,
                 extra_headers: Optional[Dict[str, str]] = None):
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or \
            os.environ.get("OPENAI_API_KEY")
        self.timeout_seconds = float(timeout_seconds)
        self.extra_headers = dict(extra_headers or {})

    def read(self, image: np.ndarray, *, hint: Optional[str] = None) -> List[OCRSpan]:
        prompt = (
            "Read every visible text token in this MTGO user-interface crop. "
            "Return JSON with key `lines`, each item containing `text`, "
            "`confidence` from 0 to 1, and optional `bbox` [x,y,w,h]."
        )
        if hint:
            prompt += f" The crop role is: {hint}."
        payload = self.extract_json(image, prompt, schema={
            "type": "object",
            "properties": {"lines": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "confidence": {"type": "number"},
                    "bbox": {"type": "array", "items": {"type": "integer"}},
                }, "required": ["text", "confidence"]}}},
            "required": ["lines"],
        })
        return [OCRSpan(
            str(row.get("text", "")), float(row.get("confidence", 0.0)),
            list(row.get("bbox")) if isinstance(row.get("bbox"), list) else None)
            for row in payload.get("lines") or [] if str(row.get("text", "")).strip()]

    def extract_json(self, image: np.ndarray, prompt: str,
                     schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key and "localhost" not in self.endpoint and \
                "127.0.0.1" not in self.endpoint:
            raise ValueError("API key is required for remote vision endpoint")
        encoded = _image_data_url(image)
        body: Dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": encoded}},
            ]}],
        }
        if schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "mtgo_screen_read", "strict": True,
                                "schema": schema},
            }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        headers.update(self.extra_headers)
        response = requests.post(self.endpoint, headers=headers, json=body,
                                 timeout=self.timeout_seconds)
        if response.status_code >= 400 and schema:
            # Several OpenAI-compatible local servers and some routed models do
            # not implement strict ``response_format`` even though they handle
            # image inputs. Retry once with the same explicit JSON-only prompt.
            fallback = dict(body)
            fallback.pop("response_format", None)
            fallback["messages"] = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt +
                     " Return one valid JSON object and no markdown."},
                    {"type": "image_url", "image_url": {"url": encoded}},
                ],
            }]
            response = requests.post(self.endpoint, headers=headers, json=fallback,
                                     timeout=self.timeout_seconds)
        response.raise_for_status()
        value = response.json()
        content = value["choices"][0]["message"].get("content", "")
        if isinstance(content, list):
            content = "".join(str(item.get("text", ""))
                              if isinstance(item, dict) else str(item)
                              for item in content)
        return _parse_json_content(str(content))


class FixtureOCRBackend(OCRBackend):
    """Deterministic backend used by tests and labeled calibration clips."""
    name = "fixture"

    def __init__(self, mapping: Dict[str, List[Dict[str, Any]]]):
        self.mapping = mapping

    def read(self, image: np.ndarray, *, hint: Optional[str] = None) -> List[OCRSpan]:
        rows = self.mapping.get(str(hint), [])
        return [OCRSpan(str(row["text"]), float(row.get("confidence", 1.0)),
                        row.get("bbox")) for row in rows]


def make_ocr_backend(name: str, **kwargs) -> OCRBackend:
    normalized = name.strip().lower()
    if normalized == "tesseract":
        return TesseractOCRBackend(**kwargs)
    if normalized in {"paddle", "paddleocr"}:
        return PaddleOCRBackend(**kwargs)
    if normalized in {"openrouter", "openai", "local-vlm", "vlm"}:
        return OpenAICompatibleVisionBackend(**kwargs)
    raise ValueError(f"unknown OCR backend: {name}")


def _image_data_url(image: np.ndarray) -> str:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _parse_json_content(content: str) -> Dict[str, Any]:
    content = content.strip()
    try:
        value = json.loads(content)
        return value if isinstance(value, dict) else {"value": value}
    except ValueError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content,
                          flags=re.S | re.I)
        if match:
            value = json.loads(match.group(1))
            return value if isinstance(value, dict) else {"value": value}
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            value = json.loads(content[start:end + 1])
            return value if isinstance(value, dict) else {"value": value}
        raise ValueError("vision endpoint did not return a JSON object")
