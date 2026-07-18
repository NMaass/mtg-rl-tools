"""Minimal newline-delimited JSON client for the mtg-rl-tools XMage bridge."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json
import os
import subprocess


@dataclass
class ReplayManifest:
    deck0: List[Dict[str, Any]]
    deck1: List[Dict[str, Any]]
    seed: Optional[int] = None
    max_turns: Optional[int] = None
    player_names: List[str] = field(default_factory=lambda: ["Player0", "Player1"])
    decision_timeout_seconds: int = 120
    bootstrap_selections: List[List[int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ReplayManifest":
        decks = value.get("decks")
        if not isinstance(decks, list):
            decks = []
        deck0 = value.get("deck0")
        deck1 = value.get("deck1")
        if deck0 is None:
            deck0 = decks[0] if len(decks) > 0 else []
        if deck1 is None:
            deck1 = decks[1] if len(decks) > 1 else []
        return cls(
            deck0=list(deck0 or []),
            deck1=list(deck1 or []),
            seed=value.get("seed"),
            max_turns=value.get("maxTurns") or value.get("max_turns"),
            player_names=list(value.get("playerNames") or
                              value.get("player_names") or ["Player0", "Player1"]),
            decision_timeout_seconds=int(value.get("decisionTimeoutSeconds", 120)),
            bootstrap_selections=[list(map(int, row)) for row in
                                  value.get("bootstrapSelections") or
                                  value.get("bootstrap_selections") or []],
            metadata=dict(value.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str) -> "ReplayManifest":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self):
        return asdict(self)


class XmageProtocolError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class XmageProtocolClient:
    """One bridge process. Works with the existing CabtProtocolServer."""

    def __init__(self, command: Optional[Sequence[str]] = None,
                 classpath: Optional[str] = None,
                 java: str = "java", cwd: Optional[str] = None):
        if command is None:
            classpath = classpath or os.environ.get("MAGIC_CABT_CLASSPATH")
            if not classpath:
                raise ValueError(
                    "pass command=[...] or classpath=..., or set "
                    "MAGIC_CABT_CLASSPATH")
            command = [java, "-cp", classpath,
                       "mage.player.cabt.CabtProtocolServer"]
        self.command = list(command)
        self.cwd = cwd
        self.process = subprocess.Popen(
            self.command, cwd=cwd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=None,
            universal_newlines=True, encoding="utf-8", errors="replace",
            bufsize=1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.process.poll() is not None:
            raise IOError(f"XMage bridge exited with {self.process.returncode}")
        line = json.dumps(payload, separators=(",", ":"))
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        response_line = self.process.stdout.readline()
        if not response_line:
            raise IOError("XMage bridge closed stdout")
        response = json.loads(response_line)
        if response.get("ok") is not True:
            raise XmageProtocolError(response.get("error", "UNKNOWN"),
                                     response.get("message", ""))
        return response

    def start(self, manifest: ReplayManifest) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "playerNames": list(manifest.player_names),
            "decisionTimeoutSeconds": manifest.decision_timeout_seconds,
        }
        if manifest.seed is not None:
            options["seed"] = manifest.seed
        if manifest.max_turns is not None:
            options["maxTurns"] = manifest.max_turns
        return self.request({"command": "game_start",
                             "decks": [manifest.deck0, manifest.deck1],
                             "options": options})

    def select(self, indices: Sequence[int]) -> Dict[str, Any]:
        return self.request({"command": "game_select",
                             "select": [int(value) for value in indices]})

    def snapshot(self) -> Dict[str, Any]:
        return self.request({"command": "visualize_data"})

    def finish(self) -> None:
        try:
            if self.process.poll() is None:
                self.request({"command": "game_finish"})
        except Exception:
            pass

    def close(self) -> None:
        self.finish()
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except Exception:
                pass
            try:
                self.process.wait(timeout=10)
            except Exception:
                self.process.kill()
        try:
            self.process.stdout.close()
        except Exception:
            pass


def load_decklist(path: str) -> List[Dict[str, Any]]:
    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            first, separator, rest = line.partition(" ")
            if first.isdigit() and rest.strip():
                entries.append({"name": rest.strip(), "count": int(first)})
            else:
                entries.append({"name": line, "count": 1})
    if not entries:
        raise ValueError(f"decklist is empty: {path}")
    return entries
