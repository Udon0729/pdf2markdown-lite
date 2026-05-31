from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable

from .layout import DocumentLayout, Line, PageLayout, Word


def extract_layout(
    pdf_path: str | Path,
    *,
    first_page: int | None = None,
    last_page: int | None = None,
) -> DocumentLayout:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for the PyMuPDF text engine. "
            "Run `uv sync --no-editable` to install project dependencies."
        ) from exc

    pages: list[PageLayout] = []
    with fitz.open(Path(pdf_path)) as doc:
        start = max(1, first_page or 1)
        end = min(len(doc), last_page or len(doc))
        if end < start:
            return DocumentLayout(pages=[])

        for page_number in range(start, end + 1):
            pdf_page = doc[page_number - 1]
            page = PageLayout(
                number=page_number,
                width=float(pdf_page.rect.width),
                height=float(pdf_page.rect.height),
                source="pymupdf",
            )
            page.lines.extend(_extract_page_lines(pdf_page, page_number))
            page.lines.sort(key=lambda line: (line.y_min, line.x_min))
            pages.append(page)
    return DocumentLayout(pages=pages)


def _extract_page_lines(pdf_page: object, page_number: int) -> Iterable[Line]:
    grouped: OrderedDict[tuple[int, int], list[Word]] = OrderedDict()
    words = pdf_page.get_text("words", sort=True)
    for raw_word in words:
        if len(raw_word) < 8:
            continue
        text = str(raw_word[4]).strip()
        if not text:
            continue
        block_id = int(raw_word[5])
        line_id = int(raw_word[6])
        word_id = int(raw_word[7])
        key = (block_id, line_id)
        grouped.setdefault(key, []).append(
            Word(
                text=text,
                x_min=float(raw_word[0]),
                y_min=float(raw_word[1]),
                x_max=float(raw_word[2]),
                y_max=float(raw_word[3]),
                block_id=block_id,
                line_id=line_id,
                word_id=word_id,
            )
        )

    for (block_id, _line_id), line_words in grouped.items():
        line_words.sort(key=lambda word: (word.x_min, word.word_id))
        yield Line(
            words=line_words,
            page_number=page_number,
            block_id=block_id,
            source="pymupdf",
        )
