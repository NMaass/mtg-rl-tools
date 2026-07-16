"""Visibility-safe tensorization for imperfect-information Magic models."""
from __future__ import annotations

from magic_cabt.models.structured_jepa import StructuredTensorizer
from magic_cabt.models.state_utils import (
    current_state, flatten_text, iter_zone_objects, perspective, seat_of)


class VisibilitySafeTensorizer(StructuredTensorizer):
    """Strip card-specific features from objects hidden to the perspective.

    Only ``publicHistory`` is admitted as temporal context. Generic ``history``
    fields are intentionally ignored because capture sources do not guarantee
    that they exclude private reveals or engine-only truth.
    """

    def state_rows(self, wrapped, perspective_seat=None):
        state = current_state(wrapped)
        explicit = perspective_seat
        if explicit is None and isinstance(wrapped, dict):
            explicit = wrapped.get("perspectiveSeat")
        view = perspective(state, explicit)
        specs = [("global", state, "global | " + flatten_text({
            key: state.get(key) for key in
            ("turnNumber", "phase", "step", "activeSeat", "prioritySeat",
             "format", "gameType") if state.get(key) is not None}), None)]
        for player in state.get("players") or []:
            if isinstance(player, dict):
                specs.append(("player", player, "player | " + flatten_text({
                    key: player.get(key) for key in
                    ("name", "seat", "life", "handCount", "libraryCount",
                     "graveyardCount", "poison", "energy")
                    if player.get(key) is not None}), seat_of(player)))
        for zone, raw_obj, seat in iter_zone_objects(state):
            obj = raw_obj if isinstance(raw_obj, dict) else {"name": str(raw_obj)}
            hidden_hand = zone == "hand" and view is not None and \
                seat is not None and str(seat) != str(view)
            hidden_library = zone == "library" and not (
                obj.get("revealed") or obj.get("known") or obj.get("isKnown"))
            hidden = hidden_hand or hidden_library
            visible_obj = {"hidden": True} if hidden else obj
            text = "unknown hidden card" if hidden else self._card_text(obj)
            specs.append((zone, visible_obj, text, seat))
        observation = wrapped.get("observation") if isinstance(wrapped, dict) else {}
        history = ((observation or {}).get("publicHistory") or
                   (wrapped.get("publicHistory") if isinstance(wrapped, dict)
                    else None) or [])
        recent = list(history)[-32:]
        for index, event in enumerate(recent):
            event = dict(event) if isinstance(event, dict) else {"event": str(event)}
            event["historyRecency"] = index - len(recent) + 1
            specs.append(("history", event,
                          "public history | " + flatten_text(event), None))
        specs = specs[:self.config.max_objects]
        text_vectors = self.embedding.encode_many([item[2] for item in specs])
        return [[float(value) for value in vector] +
                self._numeric(obj, zone, seat, view)
                for (zone, obj, _text, seat), vector in zip(specs, text_vectors)]
