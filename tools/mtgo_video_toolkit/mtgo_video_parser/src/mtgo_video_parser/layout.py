"""Resolution-independent MTGO screen layout profiles."""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import yaml


@dataclass(frozen=True)
class Region:
    name: str
    x: float
    y: float
    width: float
    height: float
    kind: str = "text"
    config: Dict[str, Any] = field(default_factory=dict)

    def pixel_box(self, image_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        height, width = image_shape[:2]
        left = max(0, min(width - 1, round(self.x * width)))
        top = max(0, min(height - 1, round(self.y * height)))
        right = max(left + 1, min(width, round((self.x + self.width) * width)))
        bottom = max(top + 1, min(height, round((self.y + self.height) * height)))
        return left, top, right, bottom

    def crop(self, image):
        left, top, right, bottom = self.pixel_box(image.shape)
        return image[top:bottom, left:right]


@dataclass
class LayoutProfile:
    name: str
    regions: List[Region]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "LayoutProfile":
        regions = []
        for name, row in (value.get("regions") or {}).items():
            regions.append(Region(
                name=str(name),
                x=float(row["x"]), y=float(row["y"]),
                width=float(row["width"]), height=float(row["height"]),
                kind=str(row.get("kind", "text")),
                config=dict(row.get("config") or {}),
            ))
        profile = cls(str(value.get("name") or "unnamed"), regions,
                      dict(value.get("metadata") or {}))
        profile.validate()
        return profile

    @classmethod
    def load(cls, path: Optional[str] = None) -> "LayoutProfile":
        if path is None:
            with resources.files("mtgo_video_parser.profiles").joinpath(
                    "mtgo_1080p_standard.yaml").open("r", encoding="utf-8") as handle:
                return cls.from_dict(yaml.safe_load(handle))
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    def validate(self) -> None:
        names = set()
        for region in self.regions:
            if region.name in names:
                raise ValueError(f"duplicate region name: {region.name}")
            names.add(region.name)
            for field_name, value in (
                ("x", region.x), ("y", region.y),
                ("width", region.width), ("height", region.height)):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{region.name}.{field_name} outside [0,1]")
            if region.x + region.width > 1.000001 or \
                    region.y + region.height > 1.000001:
                raise ValueError(f"region outside frame: {region.name}")

    def region(self, name: str) -> Region:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metadata": dict(self.metadata),
            "regions": {
                row.name: {
                    "x": row.x, "y": row.y, "width": row.width,
                    "height": row.height, "kind": row.kind,
                    "config": dict(row.config),
                } for row in self.regions
            },
        }

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
