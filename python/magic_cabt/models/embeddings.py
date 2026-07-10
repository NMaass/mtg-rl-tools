"""Frozen text embeddings for card and action semantics.

The default backend is a dependency-free signed feature hash over words and
character n-grams. It is deterministic, cheap, and gives related rules text
shared coordinates without learning a 25k-card id table. An optional
sentence-transformers backend can be selected for stronger semantic priors;
its outputs are cached in SQLite so live analysis never repeatedly encodes the
same card text.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
from typing import Iterable, List, Optional, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:/[a-z0-9]+)?|[+\-]\d+|[{}():,]", re.I)


class HashEmbeddingProvider:
    """Deterministic signed hashing with word and character n-gram features."""

    name = "hash-v1"

    def __init__(self, dimension: int = 384):
        if dimension < 32:
            raise ValueError("embedding dimension must be >= 32")
        self.dimension = int(dimension)

    def encode(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        words = _TOKEN_RE.findall(str(text or "").lower())
        features = []
        for word in words:
            features.append("w:" + word)
            padded = "^" + word + "$"
            if len(word) >= 3:
                features.extend("c3:" + padded[i:i + 3]
                                for i in range(len(padded) - 2))
            if len(word) >= 5:
                features.extend("c4:" + padded[i:i + 4]
                                for i in range(len(padded) - 3))
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little", signed=False)
            index = value % self.dimension
            sign = -1.0 if (value >> 63) else 1.0
            vector[index] += sign
        norm = math.sqrt(sum(item * item for item in vector))
        if norm:
            vector = [item / norm for item in vector]
        return vector

    def encode_many(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.encode(text) for text in texts]


class SentenceTransformerProvider:
    """Optional frozen sentence-transformer with a persistent SQLite cache."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_path: Optional[str] = None,
                 device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers backend requires: "
                "pip install -e 'python[semantic]'") from exc
        self.model_name = model_name
        self.name = "sentence-transformers:" + model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        self.cache_path = os.path.expanduser(
            cache_path or "~/.magic-cabt/card-embeddings.sqlite3")
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.cache_path, timeout=30)

    def _init_db(self):
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "model TEXT NOT NULL, digest TEXT NOT NULL, vector TEXT NOT NULL, "
                "PRIMARY KEY(model, digest))")

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def encode(self, text: str) -> List[float]:
        return self.encode_many([text])[0]

    def encode_many(self, texts: Sequence[str]) -> List[List[float]]:
        normalized = [str(text or "") for text in texts]
        digests = [self._digest(text) for text in normalized]
        found = {}
        with self._lock, self._connect() as db:
            for digest in digests:
                row = db.execute(
                    "SELECT vector FROM embeddings WHERE model=? AND digest=?",
                    (self.model_name, digest)).fetchone()
                if row:
                    found[digest] = json.loads(row[0])
        missing_positions = [i for i, digest in enumerate(digests)
                             if digest not in found]
        if missing_positions:
            missing_texts = [normalized[i] for i in missing_positions]
            encoded = self._model.encode(
                missing_texts, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False)
            with self._lock, self._connect() as db:
                for position, vector in zip(missing_positions, encoded):
                    values = [float(value) for value in vector.tolist()]
                    digest = digests[position]
                    found[digest] = values
                    db.execute(
                        "INSERT OR REPLACE INTO embeddings(model,digest,vector) "
                        "VALUES(?,?,?)",
                        (self.model_name, digest, json.dumps(values,
                                                            separators=(",", ":"))))
        return [found[digest] for digest in digests]


def make_embedding_provider(spec: Optional[str] = None, dimension: int = 384,
                            cache_path: Optional[str] = None,
                            device: Optional[str] = None):
    """Create ``hash`` or ``sentence-transformers:<model>`` provider."""
    spec = (spec or "hash").strip()
    if spec in ("hash", "hash-v1"):
        return HashEmbeddingProvider(dimension=dimension)
    prefix = "sentence-transformers:"
    if spec.startswith(prefix):
        model_name = spec[len(prefix):] or "sentence-transformers/all-MiniLM-L6-v2"
        return SentenceTransformerProvider(model_name=model_name,
                                           cache_path=cache_path,
                                           device=device)
    raise ValueError("unknown embedding backend: %s" % spec)
