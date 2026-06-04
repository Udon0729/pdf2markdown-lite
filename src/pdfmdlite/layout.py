from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
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

    # ``words`` is never mutated after construction (every extraction backend
    # builds the full word list before constructing the Line, and nothing
    # appends/replaces it later), so these derived values are computed once and
    # cached. They are read repeatedly per line by the geometry heuristics and
    # as sort keys, so caching removes the dominant recompute cost; ``text`` in
    # particular re-ran a regex on every access.
    @cached_property
    def text(self) -> str:
        return normalize_line_text(" ".join(word.text for word in self.words))

    @cached_property
    def x_min(self) -> float:
        return min((word.x_min for word in self.words), default=0.0)

    @cached_property
    def y_min(self) -> float:
        return min((word.y_min for word in self.words), default=0.0)

    @cached_property
    def x_max(self) -> float:
        return max((word.x_max for word in self.words), default=0.0)

    @cached_property
    def y_max(self) -> float:
        return max((word.y_max for word in self.words), default=0.0)

    @cached_property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @cached_property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @cached_property
    def x_center(self) -> float:
        return (self.x_min + self.x_max) / 2

    @cached_property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2

    @cached_property
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


# PDF word extraction often inserts spaces between CJK glyph runs; collapse them.
_CJK_RANGE = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_JOIN_RE = re.compile(rf"([{_CJK_RANGE}])\s+([{_CJK_RANGE}])")


def normalize_line_text(text: str) -> str:
    text = " ".join(text.replace("\u00ad", "").split())
    return _CJK_JOIN_RE.sub(r"\1\2", text)


def is_cjk(char: str) -> bool:
    """True for a CJK ideograph or kana codepoint (joined without spacing)."""
    return (
        "\u3040" <= char <= "\u30ff"
        or "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\u8c48" <= char <= "\ufaff"
    )


def touching_cjk(left: str, right: str) -> bool:
    """True if a join boundary is CJK on either side, so no space is inserted."""
    if not left or not right:
        return False
    return is_cjk(left[-1]) or is_cjk(right[0])


def split_table_cells(line: Line, normalize: bool = False) -> list[str]:
    """Split a line into table cells on large inter-word gaps.

    Returns ``[]`` when the line has fewer than 3 words or no gap wide enough to
    be a column boundary. ``normalize`` runs each cell through
    :func:`normalize_line_text` (the Markdown renderer needs clean cell text; the
    artifact detector only needs the cell count, so it leaves the text raw).
    """
    words = sorted(line.words, key=lambda word: word.x_min)
    if len(words) < 3:
        return []

    widths = [word.width / max(1, len(word.text)) for word in words if word.width > 0]
    char_width = median(widths) if widths else 5.0
    threshold = max(14.0, char_width * 4.0)

    cells: list[list[str]] = [[words[0].text]]
    large_gap_count = 0
    for previous, current in zip(words, words[1:]):
        gap = current.x_min - previous.x_max
        if gap >= threshold:
            large_gap_count += 1
            cells.append([current.text])
        else:
            cells[-1].append(current.text)

    if large_gap_count == 0:
        return []
    joined = (" ".join(cell) for cell in cells if cell)
    if normalize:
        return [normalize_line_text(text).strip() for text in joined]
    return [text.strip() for text in joined]


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
