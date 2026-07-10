"""Generic causal labels derived from observed before/after states."""
from __future__ import annotations

from .state_utils import (current_state, delta, number, perspective, seat_of,
                          zone_items)


def causal_delta_vector(previous, following, perspective_seat=None, dimension=18):
    """Own-player, aggregate-opponent, and public board deltas."""
    before, after = current_state(previous), current_state(following)
    view = perspective(before, perspective_seat)
    result = [0.0] * dimension
    own_before, own_after = _player(before, view), _player(after, view)
    _write_player(result, 0, before, after, view, own_before, own_after)
    opponents = [seat_of(player) for player in before.get("players") or []
                 if seat_of(player) is not None and str(seat_of(player)) != str(view)]
    if opponents:
        result[4] = _aggregate(before, after, opponents, "life", 20)
        result[5] = _aggregate_hands(before, after, opponents, 7)
        result[6] = _aggregate(before, after, opponents, "libraryCount", 10)
        result[7] = _aggregate(before, after, opponents, "graveyardCount", 10)
    index = 8
    for zone in ("battlefield", "stack", "graveyard", "exile"):
        if index < dimension:
            result[index] = delta(len(zone_items(before, zone)),
                                  len(zone_items(after, zone)), 10)
            index += 1
    if index < dimension:
        result[index] = delta(before.get("turnNumber"), after.get("turnNumber"), 3)
        index += 1
    if index < dimension:
        terminal = after.get("gameOver") or after.get("finished") or \
            after.get("result") or after.get("winner")
        result[index] = 1.0 if terminal else 0.0
        index += 1
    for function in (_power, _toughness, _tapped):
        if index < dimension:
            result[index] = delta(function(before), function(after), 20)
            index += 1
    return result


def _write_player(result, base, before_state, after_state, seat, before, after):
    result[base] = delta(before.get("life"), after.get("life"), 20)
    result[base + 1] = delta(_hand(before_state, seat, before),
                             _hand(after_state, seat, after), 7)
    result[base + 2] = delta(before.get("libraryCount"),
                             after.get("libraryCount"), 10)
    result[base + 3] = delta(before.get("graveyardCount"),
                             after.get("graveyardCount"), 10)


def _aggregate(before, after, seats, field, scale):
    left = sum(number(_player(before, seat).get(field)) for seat in seats)
    right = sum(number(_player(after, seat).get(field)) for seat in seats)
    return delta(left, right, scale * len(seats))


def _aggregate_hands(before, after, seats, scale):
    left = sum(_hand(before, seat, _player(before, seat)) for seat in seats)
    right = sum(_hand(after, seat, _player(after, seat)) for seat in seats)
    return delta(left, right, scale * len(seats))


def _player(state, seat):
    for player in state.get("players") or []:
        if str(seat_of(player)) == str(seat):
            return player
    return {}


def _hand(state, seat, player):
    if player.get("handCount") is not None:
        return number(player.get("handCount"))
    hands = (state.get("zones") or {}).get("hands") or state.get("hands") or {}
    return len(hands.get(str(seat), hands.get(seat, [])) or []) if isinstance(hands, dict) else 0


def _power(state):
    return sum(number(obj.get("power")) for obj in zone_items(state, "battlefield")
               if isinstance(obj, dict))


def _toughness(state):
    return sum(number(obj.get("toughness")) for obj in zone_items(state, "battlefield")
               if isinstance(obj, dict))


def _tapped(state):
    return sum(1 for obj in zone_items(state, "battlefield")
               if isinstance(obj, dict) and obj.get("tapped"))
