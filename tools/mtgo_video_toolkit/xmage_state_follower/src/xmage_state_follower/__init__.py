"""XMage shadow follower for perceived MTGO trajectories."""

from .protocol import XmageProtocolClient, ReplayManifest
from .matcher import OptionMatch, rank_options
from .follower import (
    FollowConfig,
    FollowReport,
    ReplayBackend,
    SubprocessReplayBackend,
    XmageFollower,
)

__all__ = [
    "XmageProtocolClient", "ReplayManifest", "OptionMatch", "rank_options",
    "FollowConfig", "FollowReport", "ReplayBackend",
    "SubprocessReplayBackend", "XmageFollower",
]
