"""Small structured JEPA for hidden-information-safe Magic observations.

This module is intentionally not a card-specific effect graph. It learns from
visible state rows, legal option text/payloads, frozen text embeddings, and
exact before/after transitions produced by the engine or Arena mirror.
"""
from __future__ import annotations

import copy
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .embeddings import make_embedding_provider

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None

TORCH_AVAILABLE = torch is not None
_ID_KEYS = frozenset(("id", "objectId", "instanceId", "targetId", "sourceId",
                     "targetInstanceId", "gameInstance", "eventId", "uuid"))
_VOLATILE_KEYS = frozenset(("timestamp", "rawTime", "sequenceNumber", "seq"))
_ZONE_NAMES = ("global", "player", "battlefield", "stack", "hand",
              "graveyard", "exile", "library", "command", "other")
_ZONE_INDEX = {name: index for index, name in enumerate(_ZONE_NAMES)}
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", re.I)


@dataclass
class StructuredJEPAConfig:
    text_dim: int = 384
    numeric_dim: int = 40
    d_model: int = 320
    nhead: int = 8
    encoder_layers: int = 6
    predictor_layers: int = 3
    ff_dim: int = 1280
    dropout: float = 0.1
    max_objects: int = 128
    causal_dim: int = 18
    horizon_buckets: int = 32
    embedding_backend: str = "hash"

    @classmethod
    def preset(cls, name: str):
        name = (name or "local").lower()
        if name == "tiny":
            return cls(d_model=192, nhead=6, encoder_layers=4,
                       predictor_layers=2, ff_dim=768, max_objects=96)
        if name == "large":
            return cls(d_model=448, nhead=8, encoder_layers=8,
                       predictor_layers=4, ff_dim=1792, max_objects=160)
        if name == "local":
            return cls()
        raise ValueError("unknown model preset: %s" % name)


class CardTextResolver:
    def __init__(self, cards=None):
        self.by_id = {}
        self.by_name = {}
        if cards:
            values = cards.values() if isinstance(cards, dict) else cards
            for card in values or []:
                if not isinstance(card, dict):
                    continue
                for key in ("grpId", "arenaId", "id"):
                    if card.get(key) is not None:
                        self.by_id[str(card[key])] = card
                if card.get("name"):
                    self.by_name[str(card["name"]).lower()] = card

    @classmethod
    def from_path(cls, path):
        if not path or not os.path.exists(path):
            return cls()
        import json
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def resolve(self, obj):
        if not isinstance(obj, dict):
            return {}
        for key in ("grpId", "arenaId", "cardId"):
            if obj.get(key) is not None and str(obj[key]) in self.by_id:
                return self.by_id[str(obj[key])]
        name = obj.get("name") or obj.get("cardName") or obj.get("label")
        return self.by_name.get(str(name).lower(), {}) if name else {}


class StructuredTensorizer:
    def __init__(self, config: StructuredJEPAConfig, embedding_provider=None,
                 card_resolver=None):
        self.config = config
        self.embedding = embedding_provider or make_embedding_provider(
            config.embedding_backend, dimension=config.text_dim)
        self.cards = card_resolver or CardTextResolver()
        if int(self.embedding.dimension) != int(config.text_dim):
            raise ValueError("embedding dimension does not match config")

    @property
    def row_dim(self):
        return self.config.text_dim + self.config.numeric_dim

    def state_rows(self, state, perspective_seat=None):
        current = _current(state)
        perspective = perspective_seat if perspective_seat is not None else current.get("localSeat")
        specs = [("global", current, "global " + _flatten(current), None)]
        for player in current.get("players") or []:
            if isinstance(player, dict):
                specs.append(("player", player, "player " + _flatten(player), _seat(player)))
        for zone, obj, seat in _zone_objects(current):
            hidden = zone == "hand" and perspective is not None and seat is not None and str(seat) != str(perspective)
            text = "unknown hidden card" if hidden else self._object_text(obj)
            specs.append((zone, obj if isinstance(obj, dict) else {"name": str(obj)}, text, seat))
        specs = specs[:self.config.max_objects]
        vectors = self.embedding.encode_many([item[2] for item in specs])
        return [[float(x) for x in vec] + self._numeric(zone, obj, seat, perspective)
                for (zone, obj, _text, seat), vec in zip(specs, vectors)]

    def _object_text(self, obj):
        card = self.cards.resolve(obj)
        fields = {}
        for source in (card, obj if isinstance(obj, dict) else {}):
            for key in ("name", "cardName", "manaCost", "manaValue", "typeLine",
                        "types", "subtypes", "oracleText", "rulesText", "text",
                        "keywords", "abilities", "power", "toughness", "loyalty"):
                if source.get(key) is not None:
                    fields[key] = source[key]
        return "card " + _flatten(fields)

    def _numeric(self, zone, obj, seat, perspective):
        values = [0.0] * self.config.numeric_dim
        values[0] = 1.0
        zi = _ZONE_INDEX.get(zone, _ZONE_INDEX["other"])
        if 1 + zi < len(values):
            values[1 + zi] = 1.0
        rel = 1 + len(_ZONE_NAMES)
        if rel + 2 < len(values):
            if seat is None or perspective is None:
                values[rel + 2] = 1.0
            elif str(seat) == str(perspective):
                values[rel] = 1.0
            else:
                values[rel + 1] = 1.0
        offset = rel + 3
        for i, key in enumerate(("tapped", "attacking", "blocking", "token", "faceDown")):
            if offset + i < len(values):
                values[offset + i] = 1.0 if obj.get(key) else 0.0
        offset += 5
        for i, (key, scale) in enumerate((("power", 20), ("toughness", 20),
                                          ("damage", 20), ("manaValue", 12),
                                          ("life", 40), ("handCount", 10),
                                          ("libraryCount", 60), ("graveyardCount", 30),
                                          ("loyalty", 12))):
            if offset + i < len(values):
                values[offset + i] = math.tanh(_num(obj.get(key)) / float(scale))
        return values

    def option_vector(self, option, prompt_type=""):
        text = "prompt=%s | %s" % (prompt_type or "unknown", _flatten(option or {}))
        return [float(x) for x in self.embedding.encode(text)] + [1.0] + [0.0] * (self.config.numeric_dim - 1)

    def action_vector(self, action):
        if not action:
            return self.option_vector({"type": "UNCONDITIONED"})
        selected = action.get("selectedOption") or action.get("option")
        if selected is None and isinstance(action.get("selectedOptions"), list):
            selected = action["selectedOptions"][0] if action["selectedOptions"] else None
        return self.option_vector(selected or action, action.get("promptType") or "")

    def batch_states(self, states: Sequence[Mapping], device=None):
        _require_torch()
        rows = [self.state_rows(s) for s in states]
        max_len = min(max(len(r) for r in rows), self.config.max_objects)
        data, mask = [], []
        for rowset in rows:
            rowset = rowset[:max_len]
            valid = [True] * len(rowset)
            while len(rowset) < max_len:
                rowset.append([0.0] * self.row_dim)
                valid.append(False)
            data.append(rowset)
            mask.append(valid)
        return (torch.tensor(data, dtype=torch.float32, device=device),
                torch.tensor(mask, dtype=torch.bool, device=device))

    def batch_actions(self, actions, device=None):
        _require_torch()
        return torch.tensor([self.action_vector(a) for a in actions],
                            dtype=torch.float32, device=device)

    def batch_options(self, records, device=None):
        _require_torch()
        max_options = max(len(_select(r).get("option") or []) for r in records)
        all_vecs, masks = [], []
        for record in records:
            select = _select(record)
            vecs = [self.option_vector(o, select.get("type") or "")
                    for o in select.get("option") or []]
            mask = [True] * len(vecs)
            while len(vecs) < max_options:
                vecs.append([0.0] * self.row_dim)
                mask.append(False)
            all_vecs.append(vecs)
            masks.append(mask)
        return (torch.tensor(all_vecs, dtype=torch.float32, device=device),
                torch.tensor(masks, dtype=torch.bool, device=device))


if nn is not None:
    class _MLP(nn.Module):
        def __init__(self, width, layers, dropout):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width * 4), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(width * 4, width))
                for _ in range(layers)])

        def forward(self, x):
            for block in self.blocks:
                x = x + block(x)
            return x


    class _Encoder(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.input = nn.Sequential(nn.Linear(c.text_dim + c.numeric_dim, c.d_model), nn.LayerNorm(c.d_model))
            self.token = nn.Parameter(torch.zeros(1, 1, c.d_model))
            layer = nn.TransformerEncoderLayer(c.d_model, c.nhead, c.ff_dim,
                                               c.dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, c.encoder_layers, norm=nn.LayerNorm(c.d_model))
            nn.init.normal_(self.token, std=0.02)

        def forward(self, rows, mask):
            b = rows.shape[0]
            x = torch.cat([self.token.expand(b, -1, -1), self.input(rows)], dim=1)
            m = torch.cat([torch.ones((b, 1), dtype=torch.bool, device=mask.device), mask], dim=1)
            return self.encoder(x, src_key_padding_mask=~m)[:, 0]


    class MagicJEPA(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config or StructuredJEPAConfig()
            c = self.config
            self.online_encoder = _Encoder(c)
            self.target_encoder = copy.deepcopy(self.online_encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)
            row_dim = c.text_dim + c.numeric_dim
            self.action_encoder = nn.Sequential(nn.Linear(row_dim, c.d_model), nn.LayerNorm(c.d_model), nn.GELU(), nn.Linear(c.d_model, c.d_model))
            self.horizon_embedding = nn.Embedding(c.horizon_buckets, c.d_model)
            self.predict_in = nn.Linear(c.d_model * 3, c.d_model)
            self.predictor = _MLP(c.d_model, c.predictor_layers, c.dropout)
            self.norm = nn.LayerNorm(c.d_model)
            self.policy = nn.Sequential(nn.LayerNorm(c.d_model * 4), nn.Linear(c.d_model * 4, c.d_model), nn.GELU(), nn.Linear(c.d_model, 1))
            self.value = nn.Sequential(nn.LayerNorm(c.d_model), nn.Linear(c.d_model, c.d_model), nn.GELU(), nn.Linear(c.d_model, 1), nn.Tanh())
            self.causal = nn.Sequential(nn.LayerNorm(c.d_model * 2), nn.Linear(c.d_model * 2, c.d_model), nn.GELU(), nn.Linear(c.d_model, c.causal_dim))

        def encode(self, rows, mask):
            return self.online_encoder(rows, mask)

        @torch.no_grad()
        def encode_target(self, rows, mask):
            return self.target_encoder(rows, mask)

        def predict(self, z, action_vector, horizon=None):
            a = self.action_encoder(action_vector)
            if horizon is None:
                horizon = torch.ones(z.shape[0], dtype=torch.long, device=z.device)
            h = self.horizon_embedding(horizon.long().clamp(0, self.config.horizon_buckets - 1))
            return self.norm(self.predictor(self.predict_in(torch.cat([z, a, h], dim=-1))))

        def score_options(self, rows, mask, option_vectors, option_mask):
            z = self.encode(rows, mask)
            b, n, _ = option_vectors.shape
            a = self.action_encoder(option_vectors)
            z_many = z[:, None, :].expand(-1, n, -1)
            h = self.horizon_embedding(torch.ones((b, n), dtype=torch.long, device=rows.device))
            pred = self.norm(self.predictor(self.predict_in(torch.cat([z_many, a, h], dim=-1))))
            logits = self.policy(torch.cat([z_many, a, pred, pred - z_many], dim=-1)).squeeze(-1)
            return logits.masked_fill(~option_mask, float("-inf"))

        def state_value(self, rows, mask):
            return self.value(self.encode(rows, mask)).squeeze(-1)

        def causal_delta(self, z, action_vector):
            return self.causal(torch.cat([z, self.action_encoder(action_vector)], dim=-1))

        def jepa_loss(self, pred, target, variance_weight=0.05):
            align = 2.0 - 2.0 * (F.normalize(pred, dim=-1) * F.normalize(target.detach(), dim=-1)).sum(dim=-1).mean()
            var = torch.tensor(0.0, device=pred.device) if pred.shape[0] <= 1 else F.relu(1.0 - torch.sqrt(pred.var(dim=0, unbiased=False) + 1e-4)).mean()
            return align + variance_weight * var, {"alignment": align.detach(), "variance": var.detach()}

        @torch.no_grad()
        def update_target(self, tau=0.996):
            for t, o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
                t.data.mul_(tau).add_(o.data, alpha=1 - tau)

        def save_checkpoint(self, path, extra=None):
            import torch
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            temporary = path + ".tmp"
            torch.save({"kind": "magic-structured-jepa-v1", "config": asdict(self.config),
                        "stateDict": self.state_dict(), "extra": extra or {}}, temporary)
            os.replace(temporary, path)

        @classmethod
        def load_checkpoint(cls, path, map_location="cpu"):
            import torch
            payload = torch.load(path, map_location=map_location, weights_only=False)
            if payload.get("kind") != "magic-structured-jepa-v1":
                raise ValueError("not a structured JEPA checkpoint")
            model = cls(StructuredJEPAConfig(**payload["config"]))
            model.load_state_dict(payload["stateDict"])
            return model, payload.get("extra") or {}
else:  # pragma: no cover
    class MagicJEPA(object):
        def __init__(self, *args, **kwargs):
            _require_torch()


def causal_delta_vector(previous, following, perspective_seat=None, dimension=18):
    p, n = _current(previous), _current(following)
    values = [0.0] * dimension
    seats = [_seat(x) for x in p.get("players") or [] if _seat(x) is not None]
    if perspective_seat in seats:
        seats.remove(perspective_seat)
        seats.insert(0, perspective_seat)
    for i, seat in enumerate(seats[:2]):
        bp, ap = _player(p, seat), _player(n, seat)
        base = i * 4
        values[base] = _delta(bp.get("life"), ap.get("life"), 20)
        values[base + 1] = _delta(bp.get("handCount"), ap.get("handCount"), 7)
        values[base + 2] = _delta(bp.get("libraryCount"), ap.get("libraryCount"), 10)
        values[base + 3] = _delta(bp.get("graveyardCount"), ap.get("graveyardCount"), 10)
    off = 8
    for zone in ("battlefield", "stack", "graveyard", "exile"):
        if off < dimension:
            values[off] = _delta(len(_zone(p, zone)), len(_zone(n, zone)), 10)
            off += 1
    if off < dimension:
        values[off] = _delta(p.get("turnNumber"), n.get("turnNumber"), 3)
        off += 1
    if off < dimension:
        values[off] = 1.0 if n.get("gameOver") or n.get("result") or n.get("winner") else 0.0
    return values


def model_parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def _require_torch():
    if torch is None:
        raise ImportError("structured JEPA requires PyTorch: pip install -e 'python[jepa]'")


def _current(x):
    if not isinstance(x, dict):
        return {}
    obs = x.get("observation")
    if isinstance(obs, dict) and isinstance(obs.get("current"), dict):
        return obs["current"]
    return x.get("current") if isinstance(x.get("current"), dict) else x


def _select(r):
    return r.get("select") if isinstance(r.get("select"), dict) else (r.get("observation") or {}).get("select") or {}


def _seat(x):
    if not isinstance(x, dict):
        return None
    for key in ("seat", "playerIndex", "controllerSeat", "ownerSeat", "controllerId"):
        if x.get(key) is not None:
            return x[key]
    return None


def _zone_objects(state):
    zones = state.get("zones") or {}
    for raw, contents in zones.items():
        zone = str(raw).lower().rstrip("s")
        zone = zone if zone in _ZONE_INDEX else "other"
        if isinstance(contents, dict):
            for seat, objects in contents.items():
                for obj in objects or []:
                    yield zone, obj, seat
        else:
            for obj in contents or []:
                yield zone, obj, _seat(obj)
    for key in ("battlefield", "stack", "graveyard", "exile", "hand"):
        for obj in state.get(key) or []:
            yield key, obj, _seat(obj)


def _flatten(x, depth=0):
    if depth > 4:
        return ""
    if isinstance(x, dict):
        return " | ".join(k + "=" + _flatten(v, depth + 1)
                          for k, v in sorted(x.items())
                          if k not in _ID_KEYS and k not in _VOLATILE_KEYS
                          and not k.lower().startswith("raw"))
    if isinstance(x, (list, tuple)):
        return " ".join(_flatten(v, depth + 1) for v in x[:24])
    if x is None:
        return ""
    return _UUID_RE.sub("", str(x)).strip()


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _delta(a, b, scale):
    return math.tanh((_num(b) - _num(a)) / float(scale))


def _player(state, seat):
    for p in state.get("players") or []:
        if str(_seat(p)) == str(seat):
            return p
    return {}


def _zone(state, zone):
    zones = state.get("zones") or {}
    raw = zones.get(zone) or zones.get(zone + "s") or state.get(zone) or []
    if isinstance(raw, dict):
        out = []
        for values in raw.values():
            out.extend(values or [])
        return out
    return raw or []
