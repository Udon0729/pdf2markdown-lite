# pdfmdlite

`pdfmdlite` is a CPU-only PDF to Markdown converter for papers and long technical documents. It is built around a simple rule: use the PDF text layer first, preserve layout with bounding boxes, and only fall back to OCR when a page has no usable text.

This is intentionally lighter than GPU/LLM-based converters. It uses CPU-only PyMuPDF by default for fast layout-aware text and artifact extraction. Poppler remains available as an alternate backend. OCR is optional.

## Goals

- Convert text PDFs to readable Markdown without a GPU.
- Extract figure, table, and equation regions from the PDF itself without embedding whole original pages.
- Reconstruct reading order for simple one-column and two-column layouts.
- Remove repeated page headers, footers, and page numbers.
- Preserve headings, paragraphs, lists, and simple table-like rows.
- Extract figure/table/equation crops and associate figure/table crops with detected captions.
- Support long PDFs by allowing page-range chunking.
- Keep OCR optional and explicit, because OCR is slower and language-pack dependent.

## Requirements

Install `uv`:

```bash
brew install uv
```

Required:

```bash
brew install poppler
```

Optional OCR support:

```bash
brew install tesseract
```

For Japanese OCR, install a Japanese Tesseract language pack and pass `--ocr-lang jpn`.

## Usage

Install/sync the project from the project root:

```bash
uv sync --no-editable
```

Then run:

```bash
uv run --no-editable pdfmdlite input.pdf -o output.md
```

With high-recall figure/table/equation crop extraction enabled:

```bash
uv sync --no-editable
uv run --no-editable pdfmdlite paper.pdf -o paper.md --artifact-mode both
```

For long PDFs, keep extraction enabled and let `--jobs 0` choose CPU parallelism:

```bash
uv run --no-editable pdfmdlite book.pdf -o book.md --artifact-mode both --artifact-dpi 180 --jobs 0
```

Use OCR only for pages where no text layer is found:

```bash
uv run --no-editable pdfmdlite input.pdf -o output.md --ocr auto --ocr-lang eng
```

Force OCR for all pages:

```bash
uv run --no-editable pdfmdlite input.pdf -o output.md --ocr force --ocr-lang eng
```

Keep repeated headers and footers:

```bash
uv run --no-editable pdfmdlite input.pdf -o output.md --keep-headers
```

Force a text extraction backend:

```bash
uv run --no-editable pdfmdlite input.pdf -o output.md --text-engine pymupdf
```

Available text engines:

- `--text-engine pymupdf`: use PyMuPDF directly. This is the default and fastest path for the current artifact extractor.
- `--text-engine auto`: try Poppler first, then fall back to PyMuPDF if Poppler fails.
- `--text-engine poppler`: require Poppler and fail if it cannot parse the PDF.

Process a page range from a long PDF:

```bash
uv run --no-editable pdfmdlite book.pdf -o chunk-001.md --first-page 1 --last-page 100
```

Extract figure/table/equation crops and caption metadata:

```bash
uv run --no-editable pdfmdlite paper.pdf -o paper.md --artifact-mode both
```

That writes Markdown plus:

- `paper_assets/*.png` for figure/table crops.
- `paper.artifacts.json` with page number, bounding box, source type, caption text, and crop path.

Artifact modes:

- `--artifact-mode off`: fastest text-only conversion. This is the default.
- `--artifact-mode manifest`: write PNG crops and `*.artifacts.json`, but do not modify Markdown.
- `--artifact-mode embed`: write PNG crops and embed them in Markdown near detected captions or display equations.
- `--artifact-mode both`: embed PNG crops in Markdown and also write the JSON manifest.

`--jobs 0` is the default. It extracts small PDFs in one process to avoid startup overhead and uses multiple CPU processes for longer artifact-extraction jobs. Pass `--jobs 1` for strictly serial extraction or a fixed number such as `--jobs 8` on a large machine.

The older `--extract-artifacts` flag is kept as a compatibility alias for `--artifact-mode manifest`.

## Development

Run tests through `uv`:

```bash
uv run --no-editable python -m unittest discover -s tests
```

This project uses `--no-editable` in the documented commands so the installed package is copied into `.venv`. That keeps `uv run` behavior stable on macOS environments where editable `.pth` files may be skipped when they are marked hidden.

## Accuracy Strategy

Most PDF to Markdown errors come from reading order and block semantics, not raw text extraction. The current pipeline therefore does this:

1. Extract words with coordinates using PyMuPDF by default, or `pdftotext -bbox-layout` when `--text-engine poppler` or `--text-engine auto` is selected.
2. Remove repeated marginal text across pages.
3. Reorder page lines, including a conservative two-column detector.
4. Infer headings from relative text height and short title-like lines.
5. Rebuild paragraphs by original PDF block and repair soft hyphen line breaks.
6. Detect simple table rows by repeated large horizontal gaps.
7. Detect captions with patterns such as `Figure 1`, `Fig. S1`, `Table 1`, `図1`, and `表1`.
8. Read the PDF display list (`get_bboxlog`) to collect image, vector path, shading, and text bounding boxes that may not appear as simple image objects.
9. Extract embedded images, vector drawing clusters, ruled-table regions, caption-backed tables, text-table regions, and display-equation regions when artifact extraction is enabled.
10. Use caption fallback regions to avoid dropping figures or tables whose geometry is not exposed cleanly by the PDF.
11. Associate captions to nearby figure/table regions using geometry.
12. Suppress duplicated chart labels, table cells, and equation fragments from the text stream when their crop is embedded.
13. Run Tesseract only when the page has too little embedded text and OCR is enabled.

## Long PDF Strategy

For 600+ page documents, the default path avoids OCR and uses PyMuPDF for both text and artifact extraction, which avoids running separate heavyweight passes over the same PDF. For batch jobs, split work with `--first-page` and `--last-page`, then concatenate Markdown files and merge artifact manifests.

For "do not miss artifacts", use `--artifact-mode both` or `--artifact-mode manifest`. The extractor is intentionally recall-oriented: it combines PDF object inspection, display-list bounding boxes, ruled-table detection, caption geometry, and equation text heuristics. This may over-extract ambiguous regions, but it avoids depending on a single brittle detector.

## Limitations

This is an MVP. It will not yet match specialized ML pipelines on nested tables, math-heavy PDFs, or scanned documents with poor image quality. The next accuracy step is to add an evaluation suite with representative papers and long PDFs, then compare against tools such as Poppler plain text, PyMuPDF4LLM, Docling, and Marker on measurable criteria: reading order, heading accuracy, table fidelity, figure crop recall, and caption association accuracy.
