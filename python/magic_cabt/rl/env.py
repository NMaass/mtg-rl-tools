"""Gym-style masked environment wrapper for CABT bridge games.

This module is deliberately dependency-free: it mirrors the pieces Gymnasium /
PettingZoo / masked-policy trainers need without importing those frameworks.
A downstream project can wrap ``MagicCabtEnv`` directly, or adapt the same
methods into a framework-specific class.
"""

import random

from magic_cabt.agents import clamp_selection, is_legal_selection
from magic_cabt.protocol import CabtBridge

__all__ = [
    "DiscreteActionSpace",
    "InvalidActionError",
    "MagicCabtEnv",
    "action_mask",
    "terminal_reward",
]

ILLEGAL_ACTION_FAIL = "fail"
ILLEGAL_ACTION_REPAIR = "repair"
ILLEGAL_ACTION_REPAIR_AND_MARK = "repair-and-mark"
_ILLEGAL_ACTION_MODES = (
    ILLEGAL_ACTION_FAIL,
    ILLEGAL_ACTION_REPAIR,
    ILLEGAL_ACTION_REPAIR_AND_MARK,
)


class InvalidActionError(ValueError):
    """Raised when a policy returns an illegal action in strict mode."""


class DiscreteActionSpace(object):
    """Small dependency-free stand-in for Gymnasium's ``Discrete`` space.

    ``sample(mask=...)`` accepts a truthy/falsey mask and samples uniformly from
    currently legal indices. The class exists so tests and simple scripts can
    exercise masked policies before a framework-specific wrapper is added.
    """

    def __init__(self, n, seed=None):
        if not isinstance(n, int) or n < 1:
            raise ValueError("action-space size must be a positive int")
        self.n = n
        self._rng = random.Random(seed)

    def seed(self, seed=None):
        self._rng.seed(seed)

    def sample(self, mask=None):
        if mask is None:
            return self._rng.randrange(self.n)
        legal = [index for index, allowed in enumerate(mask[:self.n]) if allowed]
        if not legal:
            raise InvalidActionError("cannot sample from an empty action mask")
        return self._rng.choice(legal)


class MagicCabtEnv(object):
    """Single-game, turn-taking CABT environment with legal action masks.

    The wrapper keeps the bridge's core invariant intact: all legality comes
    from ``observation["select"]["option"]``. The Python layer only maps a model
    action to option indices, checks the index envelope, and forwards the
    selection to XMage.

    ``illegal_action_mode`` controls bad policy outputs:

    - ``"fail"``: raise ``InvalidActionError`` and leave the game untouched.
    - ``"repair"``: clamp to a legal selection before sending it.
    - ``"repair-and-mark"``: clamp and mark the repair in ``info``.

    ``max_action_count`` pads masks to a stable length for fixed-discrete
    trainers. Leave it unset to expose a dynamic mask with exactly one entry per
    legal option.
    """

    def __init__(self, deck0, deck1, bridge=None, bridge_factory=None,
                 classpath=None, seed=None, max_turns=None,
                 player_names=("P0", "P1"), max_action_count=None,
                 illegal_action_mode=ILLEGAL_ACTION_FAIL):
        if illegal_action_mode not in _ILLEGAL_ACTION_MODES:
            raise ValueError("unknown illegal_action_mode: %r" %
                             (illegal_action_mode,))
        self.deck0 = deck0
        self.deck1 = deck1
        self.seed = seed
        self.max_turns = max_turns
        self.player_names = tuple(player_names)
        self.max_action_count = max_action_count
        self.illegal_action_mode = illegal_action_mode
        self._bridge = bridge
        self._bridge_factory = bridge_factory or (
            lambda: CabtBridge(classpath=classpath))
        self._owns_bridge = bridge is None
        self._response = None
        self._closed = False
        self._decisions = 0
        self.action_space = DiscreteActionSpace(
            max_action_count if max_action_count is not None else 1,
            seed=seed,
        )

    def reset(self, seed=None):
        """Start a new game and return ``(observation, info)``."""
        self._ensure_bridge()
        self._decisions = 0
        game_seed = self.seed if seed is None else seed
        self._response = self._bridge.game_start(
            self.deck0, self.deck1,
            player_names=list(self.player_names),
            seed=game_seed,
            max_turns=self.max_turns,
        )
        observation = self._response["observation"]
        return observation, self._info(observation)

    def step(self, action):
        """Apply one action and return ``(obs, reward, terminated, truncated, info)``.

        ``action`` may be a single integer index for single-choice prompts or a
        list of integer indices for multi-select prompts. Framework adapters can
        choose their own encoding for multi-select prompts and pass the decoded
        list here.
        """
        if self._response is None:
            raise RuntimeError("reset() must be called before step()")
        if self._bridge.finished:
            raise RuntimeError("step() called after the game finished")

        observation = self._response["observation"]
        select = observation["select"]
        acting_seat = select.get("playerIndex")
        raw_selection = _selection_from_action(action)
        selection = raw_selection
        repaired = False
        if not is_legal_selection(selection, select):
            if self.illegal_action_mode == ILLEGAL_ACTION_FAIL:
                raise InvalidActionError(
                    "illegal selection %r for %d legal option(s)" %
                    (selection, len(select.get("option") or [])))
            selection = clamp_selection(selection, select)
            repaired = True
            if not is_legal_selection(selection, select):
                raise InvalidActionError(
                    "selection %r could not be repaired for prompt %r" %
                    (raw_selection, select.get("type")))

        self._response = self._bridge.game_select(selection)
        self._decisions += 1

        info = {
            "actingPlayerIndex": acting_seat,
            "rawSelection": raw_selection,
            "selection": selection,
            "selectionWasLegal": not repaired,
            "selectionRepaired": repaired,
            "decisions": self._decisions,
        }

        if self._bridge.finished:
            result = self._bridge.result or self._response.get("result") or {}
            reward = terminal_reward(result, acting_seat, self.player_names)
            final_state = result.get("finalState") if isinstance(result, dict) else None
            next_observation = {"current": final_state} if final_state is not None else {}
            terminated = True
            truncated = self.max_turns is not None and _winner_seat(result, self.player_names) is None
            info["result"] = result
            info["rewardsBySeat"] = _rewards_by_seat(result, self.player_names)
            return next_observation, reward, terminated, truncated, info

        next_observation = self._response["observation"]
        info.update(self._info(next_observation))
        return next_observation, 0.0, False, False, info

    def render(self):
        """Return the bridge's human-readable board text when available."""
        return self._bridge.visualize_data()

    def close(self):
        if self._bridge is not None and not self._closed and self._owns_bridge:
            self._bridge.close()
        self._closed = True

    def _ensure_bridge(self):
        if self._bridge is None:
            self._bridge = self._bridge_factory()
            self._owns_bridge = True
            self._closed = False

    def _info(self, observation):
        mask = action_mask(observation, max_actions=self.max_action_count)
        if self.max_action_count is not None and self.action_space.n != self.max_action_count:
            self.action_space = DiscreteActionSpace(self.max_action_count,
                                                    seed=self.seed)
        elif self.max_action_count is None and self.action_space.n != len(mask):
            self.action_space = DiscreteActionSpace(max(1, len(mask)),
                                                    seed=self.seed)
        select = _select_of(observation)
        return {
            "action_mask": mask,
            "legalActionCount": len(select.get("option") or []),
            "selectType": select.get("type"),
            "actingPlayerIndex": select.get("playerIndex"),
        }


def action_mask(observation_or_select, max_actions=None):
    """Return a boolean legal-action mask for a CABT observation or select block."""
    select = _select_of(observation_or_select)
    options = select.get("option") if isinstance(select, dict) else None
    count = len(options or [])
    if max_actions is not None:
        if count > max_actions:
            raise ValueError("%d legal options exceeds max_action_count=%d" %
                             (count, max_actions))
        return [True] * count + [False] * (max_actions - count)
    return [True] * count


def terminal_reward(result, player_index, player_names=("P0", "P1")):
    """Return a terminal reward from ``player_index``'s perspective."""
    winner = _winner_seat(result, player_names)
    if winner is None or player_index not in (0, 1):
        return 0.0
    return 1.0 if winner == player_index else -1.0


def _rewards_by_seat(result, player_names):
    return {
        "0": terminal_reward(result, 0, player_names),
        "1": terminal_reward(result, 1, player_names),
    }


def _selection_from_action(action):
    if isinstance(action, bool):
        raise InvalidActionError("boolean actions are not valid indices")
    if isinstance(action, int):
        return [action]
    if isinstance(action, tuple):
        action = list(action)
    if isinstance(action, list):
        return list(action)
    raise InvalidActionError("action must be an int or list of ints")


def _select_of(observation_or_select):
    if not isinstance(observation_or_select, dict):
        return {}
    nested = observation_or_select.get("select")
    if isinstance(nested, dict):
        return nested
    return observation_or_select


def _winner_seat(result, player_names):
    if not isinstance(result, dict):
        return None
    winner = result.get("winner")
    if isinstance(winner, int) and winner in (0, 1):
        return winner
    if not isinstance(winner, str) or not winner:
        return None
    for seat, name in enumerate(player_names):
        if name and name in winner:
            return seat
    return None
