from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median


@dataclass(frozen=True)
class Word:
    text: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    block_id: int = 0
    line_id: int = 0
    word_id: int = 0

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def x_center(self) -> float:
        return (self.x_min + self.x_max) / 2

    @property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2


@dataclass
class Line:
    words: list[Word]
    page_number: int
    block_id: int = 0
    source: str = "text"

    @property
    def text(self) -> str:
        return normalize_line_text(" ".join(word.text for word in self.words))

    @property
    def x_min(self) -> float:
        return min((word.x_min for word in self.words), default=0.0)

    @property
    def y_min(self) -> float:
        return min((word.y_min for word in self.words), default=0.0)

    @property
    def x_max(self) -> float:
        return max((word.x_max for word in self.words), default=0.0)

    @property
    def y_max(self) -> float:
        return max((word.y_max for word in self.words), default=0.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def x_center(self) -> float:
        return (self.x_min + self.x_max) / 2

    @property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2

    @property
    def median_word_height(self) -> float:
        heights = [word.height for word in self.words if word.height > 0]
        return median(heights) if heights else self.height

    @property
    def block_key(self) -> tuple[int, int, str]:
        return (self.page_number, self.block_id, self.source)


@dataclass
class PageLayout:
    number: int
    width: float
    height: float
    lines: list[Line] = field(default_factory=list)
    source: str = "text"

    @property
    def words(self) -> list[Word]:
        return [word for line in self.lines for word in line.words]

    @property
    def word_count(self) -> int:
        return sum(len(line.words) for line in self.lines)

    @property
    def median_word_height(self) -> float:
        heights = [word.height for word in self.words if word.height > 0]
        return median(heights) if heights else 0.0


@dataclass
class DocumentLayout:
    pages: list[PageLayout]


def normalize_line_text(text: str) -> str:
    text = " ".join(text.replace("\u00ad", "").split())
    cjk = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    # PDF word extraction often inserts spaces between CJK glyph runs.
    import re

    text = re.sub(rf"([{cjk}])\s+([{cjk}])", r"\1\2", text)
    return text


def remove_repeating_marginalia(pages: list[PageLayout]) -> None:
    if len(pages) < 2:
        return

    seen_by_page: list[set[str]] = []
    for page in pages:
        page_seen: set[str] = set()
        for line in page.lines:
            normalized = _marginalia_key(line.text)
            if not normalized:
                continue
            y_ratio = line.y_center / page.height if page.height else 0.5
            if y_ratio < 0.08 or y_ratio > 0.92:
                page_seen.add(normalized)
        seen_by_page.append(page_seen)

    threshold = max(2, int(len(pages) * 0.5 + 0.999))
    counts: dict[str, int] = {}
    for page_seen in seen_by_page:
        for key in page_seen:
            counts[key] = counts.get(key, 0) + 1

    repeated = {key for key, count in counts.items() if count >= threshold}

    for page in pages:
        kept: list[Line] = []
        for line in page.lines:
            key = _marginalia_key(line.text)
            y_ratio = line.y_center / page.height if page.height else 0.5
            is_margin = y_ratio < 0.08 or y_ratio > 0.92
            is_page_number = key.isdigit() and is_margin
            if is_margin and (key in repeated or is_page_number):
                continue
            kept.append(line)
        page.lines = kept


def _marginalia_key(text: str) -> str:
    return " ".join(text.strip().lower().split())
