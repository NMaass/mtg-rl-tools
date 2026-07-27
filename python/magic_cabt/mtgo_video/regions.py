"""Screen regions for the MTGO client layout."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    """A crop rectangle in source-video pixels."""

    x: int
    y: int
    width: int
    height: int

    def ffmpeg_crop(self) -> str:
        return "crop=%d:%d:%d:%d" % (self.width, self.height, self.x, self.y)

    @classmethod
    def parse(cls, spec: str) -> "Region":
        """Parse "WxH+X+Y" (e.g. "370x445+1541+88")."""
        size, _, offs = spec.partition("+")
        w, _, h = size.partition("x")
        x, _, y = offs.partition("+")
        return cls(x=int(x), y=int(y), width=int(w), height=int(h))


# The text column of the "Chat & Game Log" pane in a full-screen 1920x1080
# MTGO client capture with the default duel-scene layout (log docked
# top-right). The right edge stops short of the scrollbar at x=1892: its
# arrows and thumb OCR as junk tokens glued to the end of every wrapped line.
LOG_PANE_1080 = Region(x=1544, y=88, width=345, height=445)

# Title bar containing "<format>: Stage N - Match N vs. <opponent>".
TITLE_BAR_1080 = Region(x=200, y=4, width=700, height=20)
