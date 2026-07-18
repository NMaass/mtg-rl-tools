"""Conservative parser for textual MTGO GameLog records."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import hashlib
import re

from mtg_state_contract import CanonicalEvent
from mtgo_video_parser.actions import MTGOLogActionParser, ObservedAction
from mtgo_video_parser.types import OCRSpan


@dataclass
class NativeLogResult:
    source_path: str
    source_sha256: str
    players: List[str] = field(default_factory=list)
    game_ids: List[str] = field(default_factory=list)
    actions: List[ObservedAction] = field(default_factory=list)
    events: List[CanonicalEvent] = field(default_factory=list)
    unparsed_lines: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        value = asdict(self)
        value["actions"] = [row.to_dict() for row in self.actions]
        value["events"] = [row.to_dict() for row in self.events]
        return value


class NativeLogParser:
    PLAYER_JOIN = re.compile(r"^(?P<player>.+?) joined the game", re.I)
    GAME_ID = re.compile(r"\b(?:game|game id)\s*#?:?\s*(?P<id>8\d{8})\b", re.I)
    TURN = re.compile(r"^(?:turn\s+)?(?P<turn>\d+):?\s*(?P<player>.+?)'?s turn", re.I)
    WINNER = re.compile(r"^(?P<player>.+?) wins the game", re.I)

    def __init__(self, pseudonymize_players: bool = True):
        self.action_parser = MTGOLogActionParser()
        self.pseudonymize_players = bool(pseudonymize_players)
        self._aliases: Dict[str, str] = {}

    def parse_file(self, path: str) -> NativeLogResult:
        target = Path(path).expanduser().resolve()
        data = target.read_bytes()
        text = _decode(data)
        result = self.parse_text(text, source_path=str(target))
        result.source_sha256 = hashlib.sha256(data).hexdigest()
        return result

    def parse_text(self, text: str, source_path: str = "<memory>") -> NativeLogResult:
        result = NativeLogResult(source_path, "")
        timestamp = 0
        turn = None
        for sequence, line in enumerate(_records(text)):
            timestamp += 1
            join = self.PLAYER_JOIN.search(line)
            if join:
                player = self._actor_alias(join.group("player").strip())
                if player not in result.players:
                    result.players.append(player)
            for match in self.GAME_ID.finditer(line):
                if match.group("id") not in result.game_ids:
                    result.game_ids.append(match.group("id"))
            turn_match = self.TURN.search(line)
            if turn_match:
                turn = int(turn_match.group("turn"))
                result.events.append(CanonicalEvent(
                    sequence=sequence, timestamp_ms=timestamp,
                    actor=self._actor_alias(turn_match.group("player").strip()),
                    action_type="TURN_START", text=line, confidence=1.0,
                    metadata={"turnNumber": turn, "source": "mtgo-gamelog"}))
                continue
            winner = self.WINNER.search(line)
            if winner:
                result.events.append(CanonicalEvent(
                    sequence=sequence, timestamp_ms=timestamp,
                    actor=self._actor_alias(winner.group("player").strip()),
                    action_type="GAME_WIN", text=line, confidence=1.0,
                    metadata={"turnNumber": turn, "source": "mtgo-gamelog"}))
                continue
            actions = self.action_parser.parse([OCRSpan(line, 1.0)], timestamp)
            if actions:
                for action in actions:
                    action.actor = self._actor_alias(action.actor)
                    action.metadata["nativeLogSequence"] = sequence
                    action.metadata["turnNumber"] = turn
                    result.actions.append(action)
                    result.events.append(CanonicalEvent(
                        sequence=sequence, timestamp_ms=timestamp,
                        actor=action.actor, action_type=action.action_type,
                        card_name=action.card_name, targets=list(action.targets),
                        text=action.text, confidence=1.0,
                        metadata={"turnNumber": turn,
                                  "source": "mtgo-gamelog"}))
            elif _worth_preserving(line):
                result.unparsed_lines.append(line)
        result.metadata = {
            "records": sequence + 1 if 'sequence' in locals() else 0,
            "parsedActions": len(result.actions),
            "unparsedLines": len(result.unparsed_lines),
        }
        return result


    def _actor_alias(self, actor: Optional[str]) -> Optional[str]:
        if actor is None or not self.pseudonymize_players:
            return actor
        text = " ".join(str(actor).split())
        if not text:
            return None
        normalized = text.casefold()
        if normalized not in self._aliases:
            self._aliases[normalized] = f"player:{len(self._aliases) + 1}"
        return self._aliases[normalized]


def _records(text: str) -> Iterable[str]:
    # MTGO logs encountered in the wild may use ordinary newlines, XML-ish
    # wrappers, or the historical @P delimiter. Keep the parser source-neutral.
    text = text.replace("@P", "\n")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    for raw in text.splitlines():
        line = " ".join(raw.strip().split())
        if line:
            yield line


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _worth_preserving(line: str) -> bool:
    lowered = line.casefold()
    signals = ("draw", "cast", "play", "attack", "block", "damage",
               "counter", "target", "trigger", "mulligan", "keep",
               "concede", "wins", "loses")
    return any(signal in lowered for signal in signals)
