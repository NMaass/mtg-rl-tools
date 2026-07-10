"""Structured set tensorization for Magic observations and legal options."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from .embeddings import make_embedding_provider
from .state_utils import (ZONE_INDEX, ZONE_NAMES, current_state, flatten_text,
                          iter_zone_objects, norm, perspective, seat_of,
                          select_block)
from .structured_config import CardTextResolver, StructuredJEPAConfig

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class StructuredTensorizer:
    """Represent cards by text/fields and states as permutation-safe rows."""
    def __init__(self, config: StructuredJEPAConfig, embedding_provider=None,
                 card_resolver=None):
        self.config = config
        self.embedding = embedding_provider or make_embedding_provider(
            config.embedding_backend, config.text_dim)
        if self.embedding.dimension != config.text_dim:
            raise ValueError("embedding dimension does not match model config")
        self.cards = card_resolver or CardTextResolver()

    @property
    def row_dim(self):
        return self.config.text_dim + self.config.numeric_dim

    def state_rows(self, wrapped: Mapping, perspective_seat=None):
        state = current_state(wrapped)
        view = perspective(state, perspective_seat)
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
        for zone, obj, seat in iter_zone_objects(state):
            obj = obj if isinstance(obj, dict) else {"name": str(obj)}
            hidden_hand = zone == "hand" and view is not None and \
                seat is not None and str(seat) != str(view)
            hidden_library = zone == "library" and not (
                obj.get("revealed") or obj.get("known") or obj.get("isKnown"))
            text = "unknown hidden card" if hidden_hand or hidden_library \
                else self._card_text(obj)
            specs.append((zone, obj, text, seat))
        observation = wrapped.get("observation") if isinstance(wrapped, dict) else {}
        history = ((observation or {}).get("publicHistory") or
                   (observation or {}).get("history") or
                   (wrapped.get("publicHistory") if isinstance(wrapped, dict) else None) or
                   (wrapped.get("history") if isinstance(wrapped, dict) else None) or [])
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

    def _card_text(self, obj):
        metadata = self.cards.resolve(obj) or {}
        combined = {}
        for source in (metadata, obj):
            for key in ("name", "cardName", "manaCost", "manaValue",
                        "typeLine", "types", "subtypes", "supertypes",
                        "oracleText", "rulesText", "text", "keywords",
                        "abilities", "power", "toughness", "loyalty",
                        "defense", "token"):
                if source.get(key) is not None:
                    combined[key] = source[key]
        return "card | " + flatten_text(combined)

    def option_vector(self, option: Mapping, prompt_type=""):
        option = option or {}
        payload = option.get("payload") if isinstance(option, dict) else {}
        metadata = self.cards.resolve(payload or {}) or self.cards.resolve(option)
        text = "prompt=%s | %s" % (prompt_type or "unknown", flatten_text(option))
        if metadata:
            text += " | card semantics=" + flatten_text(metadata)
        values = [0.0] * self.config.numeric_dim
        values[0] = 1.0
        values[1] = norm(option.get("index"), 32.0)
        if isinstance(payload, dict):
            for offset, key in enumerate(("amount", "x", "value", "manaValue",
                                          "power", "toughness"), start=2):
                if offset < len(values):
                    values[offset] = norm(payload.get(key), 20.0)
        return [float(value) for value in self.embedding.encode(text)] + values

    def action_vector(self, action: Optional[Mapping]):
        if not action:
            return self.option_vector({"type": "UNCONDITIONED"})
        selected = action.get("selectedOption") or action.get("option")
        if selected is None and isinstance(action.get("selectedOptions"), list):
            selected = action["selectedOptions"][0] if action["selectedOptions"] else None
        return self.option_vector(selected or action, action.get("promptType") or "")

    def batch_states(self, states: Sequence[Mapping], device=None):
        _torch_required()
        rows = [self.state_rows(state) for state in states]
        maximum = min(max(len(row) for row in rows), self.config.max_objects)
        data, masks = [], []
        for row in rows:
            row = row[:maximum]
            mask = [True] * len(row)
            while len(row) < maximum:
                row.append([0.0] * self.row_dim)
                mask.append(False)
            data.append(row)
            masks.append(mask)
        return (torch.tensor(data, dtype=torch.float32, device=device),
                torch.tensor(masks, dtype=torch.bool, device=device))

    def batch_actions(self, actions, device=None):
        _torch_required()
        return torch.tensor([self.action_vector(action) for action in actions],
                            dtype=torch.float32, device=device)

    def batch_options(self, records, device=None):
        _torch_required()
        count = max(len(select_block(record).get("option") or [])
                    for record in records)
        data, masks = [], []
        for record in records:
            select = select_block(record)
            vectors = [self.option_vector(option, select.get("type") or "")
                       for option in select.get("option") or []]
            mask = [True] * len(vectors)
            while len(vectors) < count:
                vectors.append([0.0] * self.row_dim)
                mask.append(False)
            data.append(vectors)
            masks.append(mask)
        return (torch.tensor(data, dtype=torch.float32, device=device),
                torch.tensor(masks, dtype=torch.bool, device=device))

    def _numeric(self, obj, zone, seat, view):
        values = [0.0] * self.config.numeric_dim
        values[0] = 1.0
        zone_start = 3
        values[1] = 1.0 if zone == "global" else 0.0
        values[2] = 1.0 if zone == "player" else 0.0
        zone_index = ZONE_INDEX.get(zone, ZONE_INDEX["other"])
        if zone_start + zone_index < len(values):
            values[zone_start + zone_index] = 1.0
        relation = zone_start + len(ZONE_NAMES)
        if relation + 2 < len(values):
            values[relation + (2 if seat is None or view is None else
                               (0 if str(seat) == str(view) else 1))] = 1.0
        start = relation + 3
        for offset, key in enumerate(("tapped", "attacking", "blocking",
                                      "token", "faceDown", "summoningSick")):
            if start + offset < len(values):
                values[start + offset] = 1.0 if obj.get(key) else 0.0
        start += 6
        for offset, (key, scale) in enumerate((
                ("power", 20), ("toughness", 20), ("damage", 20),
                ("damageMarked", 20), ("manaValue", 12), ("life", 40),
                ("handCount", 10), ("libraryCount", 60),
                ("graveyardCount", 30), ("poison", 10), ("energy", 20),
                ("loyalty", 12))):
            if start + offset < len(values):
                values[start + offset] = norm(obj.get(key), scale)
        return values


def _torch_required():
    if torch is None:
        raise ImportError("structured JEPA requires magic-cabt[jepa]")
