"""Environment diagnostics for a Windows/Linux headless parsing host."""
from __future__ import annotations

from typing import Any, Dict
import importlib.util
import os
import platform
import shutil
import subprocess
import sys


def run_doctor() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executables": {
            "ffmpeg": shutil.which("ffmpeg"),
            "ffprobe": shutil.which("ffprobe"),
            "tesseract": shutil.which("tesseract"),
            "yt-dlp": shutil.which("yt-dlp"),
            "java": shutil.which("java"),
            "nvidia-smi": shutil.which("nvidia-smi"),
        },
        "pythonModules": {},
        "environment": {
            "MAGIC_CABT_CLASSPATH": bool(os.environ.get("MAGIC_CABT_CLASSPATH")),
            "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
            "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        },
    }
    for module in ("cv2", "numpy", "PIL", "yaml", "rapidfuzz", "pytesseract",
                   "paddleocr", "torch", "mtgo_video_acquisition",
                   "mtgo_native_logs", "mtgo_video_parser",
                   "xmage_state_follower", "mtg_state_contract"):
        result["pythonModules"][module] = bool(importlib.util.find_spec(module))
    if result["executables"]["nvidia-smi"]:
        try:
            output = subprocess.check_output(
                [result["executables"]["nvidia-smi"],
                 "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                text=True, timeout=10, stderr=subprocess.STDOUT)
            result["gpu"] = [line.strip() for line in output.splitlines() if line.strip()]
        except Exception as error:
            result["gpuError"] = str(error)
    required = [result["executables"]["ffmpeg"],
                result["pythonModules"]["cv2"]]
    result["readyForBaseExtraction"] = all(required)
    result["readyForTesseract"] = bool(
        result["executables"]["tesseract"] and
        result["pythonModules"]["pytesseract"])
    result["readyForXmage"] = bool(
        result["executables"]["java"] and
        result["environment"]["MAGIC_CABT_CLASSPATH"])
    return result
