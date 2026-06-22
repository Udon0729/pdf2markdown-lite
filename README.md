# pdf2markdown-lite

> 日本語版: [README.ja.md](README.ja.md)

`pdfmdlite` is a CPU-only PDF → Markdown converter for papers and long technical documents. It reads the PDF's own glyph stream (text, fonts, coordinates) and drawn vector geometry, and reconstructs **structured** Markdown from it — without a GPU, an LLM, or any ML model.

Its distinguishing feature is that it does not flatten everything to images:

- **Equations → LaTeX.** Display and inline math are reconstructed deterministically from the glyph fonts and positions (Computer Modern math fonts, sub/superscript baselines, large-operator limits, fractions from drawn rules, accents) and emitted as real LaTeX — fenced ` ```math ` blocks for display math and `$...$` for inline math.
- **Tables → Markdown tables.** Ruled tables are reconstructed cell-by-cell from the drawn grid and the glyph stream and emitted as Markdown pipe tables, so the contents stay selectable text.
- **Figures → images.** Figures (raster images and vector drawings) are cropped to PNG. By default they are linked from an assets directory; with `--inline-images` they are embedded directly into the `.md` as base64 data URIs so the Markdown is a single self-contained file.

The only Python dependency is `pymupdf`. Poppler and Tesseract are optional external binaries used only for the alternate text backend and for OCR.

## Requirements

Install `uv` (https://docs.astral.sh/uv/):

```bash
brew install uv         # macOS
# or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

Everything in the default path (PyMuPDF text + figure/table/equation extraction) needs only `uv` + `pymupdf`, which `uv sync` installs for you.

Optional external binaries:

- **Poppler** (`pdftotext`, `pdftoppm`) — only for `--text-engine poppler`/`auto` and for OCR page rasterization.
- **Tesseract** — only for `--ocr auto`/`force`.

```bash
# macOS
brew install poppler tesseract
# Debian/Ubuntu
sudo apt-get install poppler-utils tesseract-ocr
```

For Japanese OCR install a Japanese language pack (`tesseract-ocr-jpn`) and pass `--ocr-lang jpn`.

## Usage

Install/sync from the project root:

```bash
uv sync --no-editable
```

Basic conversion (text + LaTeX equations):

```bash
uv run --no-editable pdfmdlite input.pdf -o output.md
```

Full extraction — figures as image crops, tables as Markdown tables, equations as LaTeX:

```bash
uv run --no-editable pdfmdlite paper.pdf -o paper.md --artifact-mode both
```

That writes Markdown plus:

- `paper_assets/*.png` — figure crops (tables are inline Markdown, not images; equations are LaTeX, not images).
- `paper.artifacts.json` — manifest with page number, bounding box, source type, and caption text for each extracted artifact.

Self-contained single file — embed figure PNGs into the Markdown as base64 (no assets directory is written):

```bash
uv run --no-editable pdfmdlite paper.pdf -o paper.md --artifact-mode embed --inline-images
```

For long PDFs, let `--jobs 0` choose CPU parallelism:

```bash
uv run --no-editable pdfmdlite book.pdf -o book.md --artifact-mode both --artifact-dpi 180 --jobs 0
```

OCR only the pages that have no text layer:

```bash
uv run --no-editable pdfmdlite scan.pdf -o scan.md --ocr auto --ocr-lang eng
```

Process a page range from a long PDF:

```bash
uv run --no-editable pdfmdlite book.pdf -o chunk-001.md --first-page 1 --last-page 100
```

### Key options

- `--math {on,off}` — `on` (default) reconstructs deterministic LaTeX from the glyph stream. `off` falls back to the legacy character-heuristic formula path.
- `--artifact-mode {off,manifest,embed,both}` — `off` (default) is text-only (equations are still LaTeX). `manifest` writes figure crops + `*.artifacts.json` without touching the Markdown. `embed` reconstructs tables as Markdown and embeds figure crops at their caption anchor. `both` does `embed` plus the JSON manifest. Tables become Markdown pipe tables and equations become LaTeX in `embed`/`both`.
- `--inline-images` — embed figure PNGs in the Markdown as base64 data URIs (self-contained file); no asset files or assets directory are written.
- `--ocr {off,auto,force}` and `--ocr-lang LANG` — OCR mode and Tesseract language. `auto` OCRs only pages whose embedded text is below a threshold.
- `--text-engine {pymupdf,poppler,auto}` — `pymupdf` (default) reads glyphs directly; `poppler` uses `pdftotext -bbox-layout`; `auto` tries Poppler then falls back to PyMuPDF.
- `--keep-headers` — keep repeated page headers/footers/page numbers (stripped by default).
- `--first-page` / `--last-page` — 1-based page range for chunking.
- `--artifact-dpi N` — render DPI for figure crops (default 180).
- `--jobs N` — `0` (default) runs serially for short PDFs and fans out across CPU processes for longer artifact-extraction jobs; `1` forces serial.

`--extract-artifacts` is kept as a compatibility alias for `--artifact-mode manifest`.

## How it works

Most PDF → Markdown errors come from reading order and block semantics, not raw text extraction. Born-digital (LaTeX-produced) PDFs carry clean, deterministic glyph metrics, which is what makes structure recovery possible on the CPU without ML. The pipeline:

1. Extract words/glyphs with coordinates and fonts via PyMuPDF (default), or `pdftotext -bbox-layout` for the Poppler backend.
2. OCR a page with Tesseract only when `--ocr` is enabled and the page has too little embedded text; OCR coordinates are normalized back to PDF points.
3. Remove headers/footers/page numbers that repeat across pages, and rotated side-margin stamps (e.g. the arXiv identifier).
4. Reorder lines (full-width vs. a conservative two-column detector) and merge multi-line headings, repairing line-break hyphenation and prepending orphaned section/appendix labels.
5. Rebuild paragraphs by PDF block, joining CJK without spurious spaces.
6. **Reconstruct equations to LaTeX** from glyph fonts (`CMMI`/`CMSY`/`CMEX`/`MSBM` …), font sizes and baselines (sub/superscripts, large-operator limits), drawn rules (fractions, radicals) and overlapping accent glyphs. Display-math regions are detected by math-font density and isolation so prose with inline math is not mistaken for a display equation.
7. **Reconstruct ruled tables to Markdown** by recovering the row/column grid from the drawn rules and binning the glyph words into cells.
8. Detect figures from embedded images and clustered vector drawings, crop them to PNG, and associate captions (`Figure`/`Fig`/`Table`/`図`/`表`) by geometry. Empty/ink-less regions are rejected.
9. Suppress the source glyph lines of anything emitted as LaTeX, a Markdown table, or an embedded crop, so content is never duplicated.

Correctness is enforced deterministically by conservation checks in the test suite: every source glyph must appear exactly once in the emitted LaTeX (`check_symbol_conservation`), and every table cell word must survive into the Markdown grid. The emitted LaTeX is valid and compiles under `pdflatex`.

## Development

```bash
uv run --no-editable python -m unittest discover -s tests
```

The documented commands use `--no-editable`, which copies the package into `.venv` (this keeps `uv run` stable on macOS, where hidden editable `.pth` files can be skipped). Because of this, after editing anything under `src/pdfmdlite/` you must rebuild before tests or the CLI see the change:

```bash
uv sync --no-editable --reinstall-package pdfmdlite
```

The Poppler/OCR integration tests self-skip when `pdftotext`/`tesseract` are absent.

## Limitations

Equation reconstruction targets born-digital LaTeX PDFs and the common constructs (sub/superscripts, fractions, sums/integrals with limits, radicals, accents, bold/blackboard/calligraphic fonts); unusual fonts, dense matrices, and multi-line aligned environments may be imperfect. Borderless tables without vertical rules fall back to gap-based column detection and are less reliable than fully-ruled grids. Scanned documents depend on OCR quality. The next step is an evaluation suite of representative papers to compare against Poppler plain text, PyMuPDF4LLM, Docling, and Marker on reading order, heading accuracy, table fidelity, equation accuracy, and figure crop recall.
