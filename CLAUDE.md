# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`pdfmdlite` is a CPU-only PDF → Markdown converter for papers and long technical documents. The design rule: use the PDF text layer first, preserve layout via bounding boxes, and fall back to OCR only when a page has no usable text. No GPU, no LLM. The single Python dependency is `pymupdf`; Poppler and Tesseract are external system binaries called over `subprocess`.

## Commands

```bash
uv sync --no-editable                                    # install/build into .venv
uv run --no-editable pdfmdlite input.pdf -o output.md    # run the CLI (also: python -m pdfmdlite)
uv run --no-editable python -m unittest discover -s tests # full test suite (unittest, not pytest)
uv run --no-editable python -m unittest tests.test_markdown.MarkdownRenderingTests.test_simple_table_detection  # single test
```

Keep `--no-editable` in every documented command. It is intentional: editable `.pth` files can be skipped on macOS when marked hidden, so the package is copied into `.venv` instead to keep `uv run` behavior stable. The Poppler integration test self-skips when `pdftotext` is absent.

## Pipeline architecture

`converter.convert_pdf_to_result()` is the spine. Everything else is a stage it calls, in order:

1. **Text extraction** (`_extract_document_layout`) — picks a backend by `--text-engine`: `pymupdf` (default, `pymupdf_text.py`, via `fitz.get_text("words")`), `poppler` (`poppler.py`, via `pdftotext -bbox-layout` → XHTML), or `auto` (Poppler first, fall back to PyMuPDF on `PopplerError`). All backends emit the same `DocumentLayout`.
2. **OCR** (`ocr.py`, optional) — only when `--ocr != off`; replaces a page when `--ocr force` or its `word_count < min_text_words` (default 8). Renders with `pdftoppm`, reads Tesseract TSV (level-5 rows = words).
3. **Marginalia removal** (`layout.remove_repeating_marginalia`) — strips headers/footers/page numbers that recur in the top/bottom 8% across ≥50% of pages.
4. **Artifact extraction** (`artifacts.py`, when `--artifact-mode != off`) — figure/table/equation region detection + PNG crops.
5. **Markdown rendering** (`markdown.render_markdown`) — reading order, block semantics, artifact embedding.

The shared data model lives in `layout.py`: `Word` → `Line` → `PageLayout` → `DocumentLayout`, all carrying bbox coordinates. Coordinates are PDF points with a top-left origin and y increasing downward. **Both text engines must produce compatible coordinates** because every downstream heuristic is geometry-based and engine-agnostic. `normalize_line_text` handles soft hyphens and the spurious spaces PDF extraction inserts between CJK glyph runs.

## Where the accuracy logic actually lives

Most behavior worth changing is in two files of geometry heuristics, not in extraction:

- **`markdown.py`** — `order_lines` separates full-width lines from a conservative two-column detector; `_heading_level` infers headings from word-height ratios and title shape; `_is_formula_line` routes math into ` ```math ` fences; `_try_render_table` builds tables from repeated large horizontal gaps; `_join_paragraph` repairs hyphenated breaks and joins CJK without spaces. Artifacts are embedded at their caption's anchor line, and lines overlapping an artifact bbox are suppressed from the text stream.
- **`artifacts.py`** (largest, most complex) — `_page_visual_regions` is intentionally **recall-oriented**: it unions embedded images, display-list entries (`get_bboxlog`: image/shade/path/text), vector drawing clusters, ruled-table detection, caption-backed tables, text-table regions, display-equation regions, and caption-fallback regions, then dedupes across kinds. Captions are matched by `CAPTION_RE` (Figure/Fig/Table/図/表). Over-extraction is preferred over depending on a single brittle detector. README "Accuracy Strategy" enumerates the full 13-step rationale.

## Conventions and gotchas

- **Artifact extraction always requires PyMuPDF** (`fitz`), regardless of `--text-engine`. OCR and the Poppler engine require external binaries (`pdftoppm`/`pdftotext`/`tesseract`), not Python packages.
- `--artifact-mode {off,manifest,embed,both}` is the current control; `--extract-artifacts` is a deprecated alias for `manifest`. Passing `--manifest`/`--assets-dir` while mode is `off` promotes it to `manifest` (see `cli.main`).
- `--jobs 0` (default) runs serially under 16 pages, else fans out via `ProcessPoolExecutor` (`_effective_render_jobs`). Worker tasks re-open the PDF per process.
- Public API surface (`__init__.py`): `convert_pdf`, `convert_pdf_to_result`, `ConversionOptions`, `ConversionResult`. The CLI is a thin wrapper that maps argparse flags onto `ConversionOptions`.
- `pyproject.toml` requires Python ≥3.10; `.python-version` pins 3.12.
- `output.md`, `output.artifacts.json`, and `output_assets/` at the repo root are committed **sample output**, not source.
