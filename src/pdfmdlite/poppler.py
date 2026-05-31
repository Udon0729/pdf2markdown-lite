from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from .layout import DocumentLayout, Line, PageLayout, Word


class PopplerError(RuntimeError):
    pass


def extract_layout(
    pdf_path: str | Path,
    *,
    first_page: int | None = None,
    last_page: int | None = None,
) -> DocumentLayout:
    pdf_path = Path(pdf_path)
    if not shutil.which("pdftotext"):
        raise PopplerError("pdftotext was not found. Install Poppler first.")

    command = [
        "pdftotext",
        "-q",
        "-bbox-layout",
        "-enc",
        "UTF-8",
    ]
    if first_page is not None:
        command.extend(["-f", str(first_page)])
    if last_page is not None:
        command.extend(["-l", str(last_page)])
    command.extend(
        [
            str(pdf_path),
            "-",
        ]
    )
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "pdftotext failed"
        raise PopplerError(detail)
    return parse_bbox_layout(result.stdout, page_number_offset=(first_page or 1) - 1)


def parse_bbox_layout(markup: str, *, page_number_offset: int = 0) -> DocumentLayout:
    try:
        root = ET.fromstring(markup)
    except ET.ParseError as exc:
        raise PopplerError(f"could not parse pdftotext XHTML: {exc}") from exc

    pages: list[PageLayout] = []
    for page_index, page_el in enumerate(_iter_named(root, "page"), start=1):
        page = PageLayout(
            number=page_number_offset + page_index,
            width=_float_attr(page_el, "width"),
            height=_float_attr(page_el, "height"),
        )
        block_counter = 0
        for block_el in _iter_named(page_el, "block"):
            block_id = block_counter
            block_counter += 1
            line_counter = 0
            for line_el in _iter_named(block_el, "line"):
                words: list[Word] = []
                for word_index, word_el in enumerate(_iter_named(line_el, "word")):
                    text = "".join(word_el.itertext()).strip()
                    if not text:
                        continue
                    words.append(
                        Word(
                            text=text,
                            x_min=_float_attr(word_el, "xMin"),
                            y_min=_float_attr(word_el, "yMin"),
                            x_max=_float_attr(word_el, "xMax"),
                            y_max=_float_attr(word_el, "yMax"),
                            block_id=block_id,
                            line_id=line_counter,
                            word_id=word_index,
                        )
                    )
                if words:
                    words.sort(key=lambda word: (word.x_min, word.y_min))
                    page.lines.append(
                        Line(words=words, page_number=page.number, block_id=block_id)
                    )
                line_counter += 1
        page.lines.sort(key=lambda line: (line.y_min, line.x_min))
        pages.append(page)
    return DocumentLayout(pages=pages)


def _iter_named(element: ET.Element, name: str):
    for child in element.iter():
        if _local_name(child.tag) == name:
            yield child


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _float_attr(element: ET.Element, name: str) -> float:
    value = element.attrib.get(name)
    if value is None:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0
