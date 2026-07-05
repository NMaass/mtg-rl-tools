"""Live MTG Arena -> XMage mirror: follow Player.log, track game state,
record CABT-format decisions, and drive an XMage display.

Modules:
    cards     - Arena grpId -> card name/type resolution (MTGA local DB)
    follower  - live Player.log tailing with rotation/truncation handling
    tracker   - streaming normalizer + GRE game-state tracker
    options   - Arena decision prompts -> indexed CABT option lists
    recorder  - CABT-format replay bundle writer
    mirror    - client for the Java XMage mirror display
    replay    - play a recorded bundle back into the display
"""

from .cards import CardDatabase, CardInfo
from .follower import LogFollower, EntryAssembler
from .tracker import ArenaMatchTracker, GameStateTracker
from .recorder import MirrorRecorder

__all__ = [
    "CardDatabase",
    "CardInfo",
    "LogFollower",
    "EntryAssembler",
    "ArenaMatchTracker",
    "GameStateTracker",
    "MirrorRecorder",
]
