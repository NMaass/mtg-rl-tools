"""Small JSONL helpers for canonical-state bundles."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Union

from .schema import CanonicalState


def read_states(path: Union[str, Path]) -> Iterator[CanonicalState]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except ValueError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error}")
            yield CanonicalState.from_dict(value)


def write_states(path: Union[str, Path], states: Iterable[CanonicalState]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for state in states:
            handle.write(json.dumps(state.to_dict(), sort_keys=True,
                                    separators=(",", ":")) + "\n")
