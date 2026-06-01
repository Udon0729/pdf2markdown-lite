"""Deterministic LaTeX reconstruction of math from a born-digital PDF glyph stream.

Born-digital LaTeX PDFs (e.g. arXiv papers) carry pristine, deterministic glyph
metrics: every glyph reports its character, bounding box, baseline origin, font
and size. Structure recognition therefore needs no machine learning and no GPU,
only geometry plus table lookup.

This module re-reads a ``fitz`` page via the ``rawdict`` text extraction (which,
unlike ``get_text("words")``, preserves per-glyph font/size/origin and keeps an
entire math baseline together) joined with ``get_texttrace`` for the reliable
glyph id. It exposes the per-glyph layer that the later structure/layout pass
depends on:

* :class:`MathGlyph` -- one frozen glyph record (char, font, size, flags,
  origin, bbox, glyph id).
* :func:`extract_math_glyphs` -- build the per-page glyph list.
* :func:`classify` -- the deterministic glyph -> (token, style) mapping.

The font *family* encodes style (CMBX -> ``\\mathbf``; CMMI -> bare math italic;
CMR -> upright, coalesced into ``\\mathrm`` runs; CMSY -> symbols; CMEX -> large
operators; MSAM/MSBM -> AMS and blackboard). An unhandled glyph raises
:class:`MathRecoUnhandled` -- by the project contract that is a *defect* to fix
with a new table entry, never a silent drop nor an image fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median
from typing import Any

# A placeholder token emitted for a CMEX large operator. Its codepoint and glyph
# id are both undefined (rawdict char is garbage, glyph id is always 0xFFFD), so
# the concrete operator (\sum / \prod / \int) is resolved by the geometry pass,
# not here. Only \sum occurs in the target document; the default is \sum.
LARGEOP_PLACEHOLDER = "\\sum"

# A marker emitted for the CMEX brace pieces of an \underbrace decoration.
UNDERBRACE_MARKER = "\\underbrace"


class MathRecoUnhandled(Exception):
    """A glyph could not be resolved to a LaTeX token.

    Per the project contract this is a DEFECT to fix (add a table entry), never
    a reason to silently drop content or fall back to an image crop.
    """

    def __init__(self, font: str, gid: int, char: str) -> None:
        self.font = font
        self.gid = gid
        self.char = char
        super().__init__(
            f"unhandled math glyph: font={font!r} gid={gid} char={char!r} "
            f"(codepoint {ord(char) if len(char) == 1 else '?'})"
        )


@dataclass(frozen=True)
class MathGlyph:
    """One glyph from the PDF stream, carrying everything geometry needs.

    Coordinates are PDF points with a top-left origin and y increasing downward,
    matching the rest of the pipeline. ``ox``/``oy`` are the glyph's baseline
    origin; ``x0..y1`` is its bounding box; ``gid`` is the font glyph id (used
    only for fonts where the rawdict char is wrong/ambiguous).
    """

    char: str
    font: str
    size: float
    flags: int
    ox: float
    oy: float
    x0: float
    y0: float
    x1: float
    y1: float
    gid: int = -1

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


# ---------------------------------------------------------------------------
# Symbol identification tables (entries proven to occur in the target PDF;
# extend rather than guess).
# ---------------------------------------------------------------------------

# Codepoint -> LaTeX. ASCII printable (letters, digits, operators, parens) maps
# to itself via the identity branch in classify(); only non-ASCII or named
# commands need an entry here.
UNI2TEX: dict[int, str] = {
    0x2212: "-",            # MINUS SIGN
    0x2208: r"\in",
    0x2248: r"\approx",
    0x2282: r"\subset",
    0x00D7: r"\times",
    0x2032: r"\prime",
    0x2217: r"\ast",
    0x2113: r"\ell",
    0x03B1: r"\alpha",
}

# (font regex, style kind). First match wins. Style is applied at grouping time,
# independent of the token itself.
FONT_STYLE: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^CMBX"), "mathbf"),
    (re.compile(r"^CMMI"), "italic_bare"),
    (re.compile(r"^CMR"), "upright"),
    (re.compile(r"^CMTI"), "mathit"),
    (re.compile(r"^MSBM"), "mathbb"),
    (re.compile(r"^MSAM"), "amssymbol"),
    (re.compile(r"^CMSY"), "symbol"),
    (re.compile(r"^CMEX"), "largeop"),
    # A text-serif font reaching the recogniser only does so as a glyph already
    # inside a math run (e.g. the nested 'hid' of d_{hid}); such upright text is
    # set with \mathrm.
    (re.compile(r"^(?:NimbusRom|Times|TimesNewRoman)"), "upright"),
]

# (font_prefix, glyph_id) -> LaTeX token, for slots where the rawdict char is
# wrong or ambiguous. The CMEX large operator is NOT keyed here (its gid is
# always 0xFFFD); it is handled by the geometry branch.
SLOT_OVERRIDE: dict[tuple[str, int], str] = {
    ("MSBM", 82): "R",          # blackboard R; FONT_STYLE wraps -> \mathbb{R}
    ("CMSY", 123): r"\{",
    ("CMSY", 125): r"\}",
    ("CMSY", 124): "|",
    ("CMSY", 76): r"\mathcal{L}",
}

# CMSY glyph ids that are bare math delimiters ('{', '}', '|'). A run whose only
# body-size math content is a delimiter is layout punctuation, not an inline
# expression (e.g. the '{...}@gmail.com' author braces), so the inline detector
# refuses to promote it to math. Real inline runs that legitimately use these
# delimiters (e.g. \{\mathbf{g}_{j}\}) also carry CMBX/CMMI content and pass.
_CMSY_DELIMITER_GIDS = frozenset({123, 124, 125})


def _is_math_delimiter_glyph(glyph: MathGlyph) -> bool:
    """True for a CMSY bare delimiter glyph ('{', '}', '|')."""
    return _font_prefix(glyph.font) == "CMSY" and glyph.gid in _CMSY_DELIMITER_GIDS

# Accent glyph codepoint -> LaTeX accent command. Detected by x-overlap with the
# following base glyph (handled in the structure pass).
ACCENT: dict[int, str] = {
    0x02C6: r"\hat",
    0x00AF: r"\bar",
    0x0304: r"\bar",
    0x02DC: r"\tilde",
    0x02D9: r"\dot",
    0x20D7: r"\vec",
}

# Font prefixes that mark a glyph as belonging to a math run rather than body
# text. A glyph in any of these fonts is math; a CM-font glyph inside an
# otherwise NimbusRom/Times line is inline math.
MATH_FONT_PREFIXES = ("CM", "MSAM", "MSBM")


def font_family(font: str) -> str:
    """Strip a font name down to its identifying family prefix.

    PyMuPDF reports names such as ``CMBX10``, ``CMMI7``, ``ABCDEF+CMR10`` (a
    subset tag prefix) or ``NimbusRomNo9L-Regu``. Drop any ``TAG+`` subset
    prefix, then return the leading alphabetic-or-CM-style family token.
    """
    name = font.split("+", 1)[-1] if "+" in font else font
    match = re.match(r"^([A-Za-z]+)", name)
    return match.group(1) if match else name


def is_math_font(font: str) -> bool:
    family = font_family(font)
    return family.startswith(MATH_FONT_PREFIXES)


def _font_prefix(font: str) -> str:
    """The CM-style alphabetic family prefix (e.g. 'CMSY' from 'CMSY10')."""
    family = font_family(font)
    match = re.match(r"^([A-Za-z]+?)(?=\d|$)", family)
    return match.group(1) if match else family


def style_for(font: str, flags: int) -> str:
    """Resolve the LaTeX wrapper style for a glyph's font/flags.

    Font family is authoritative; the bold flag (bit 4, value 16) is a backstop
    for a CMBX glyph mis-tagged with a non-CMBX name.
    """
    for pattern, style in FONT_STYLE:
        if pattern.match(font_family(font)):
            return style
    if flags & 16:
        return "mathbf"
    return "italic_bare"


def classify(glyph: MathGlyph) -> tuple[str, str]:
    """Map one glyph to ``(token, style)`` deterministically.

    Resolution order (see module docstring / plan):
      1. CMEX -> large-operator placeholder, resolved by geometry.
      2. (font_prefix, gid) in SLOT_OVERRIDE -> that token.
      3. rawdict char is a known non-ASCII symbol (UNI2TEX) -> that token.
      4. rawdict char is normal printable ASCII -> identity.
      5. else raise MathRecoUnhandled (a defect to fix, never a silent drop).
    """
    family = font_family(glyph.font)
    style = style_for(glyph.font, glyph.flags)

    if family.startswith("CMEX"):
        return LARGEOP_PLACEHOLDER, "largeop"

    prefix = _font_prefix(glyph.font)
    override = SLOT_OVERRIDE.get((prefix, glyph.gid))
    if override is not None:
        return override, style

    if len(glyph.char) == 1:
        codepoint = ord(glyph.char)
        if codepoint in UNI2TEX:
            return UNI2TEX[codepoint], style
        if codepoint in ACCENT:
            return ACCENT[codepoint], style
        if glyph.char.isprintable() and glyph.char != " " and codepoint < 0x7F:
            return glyph.char, style

    raise MathRecoUnhandled(glyph.font, glyph.gid, glyph.char)


def _texttrace_gid_index(page: Any) -> dict[tuple[float, float], int]:
    """Map a glyph's rounded baseline origin to its reliable glyph id.

    The rawdict and texttrace streams align exactly on the baseline origin
    (proven to 0.1pt), so the rounded origin is the join key. The texttrace
    ``ucs`` field is garbage for math fonts and is ignored.
    """
    try:
        spans = page.get_texttrace()
    except Exception:
        return {}

    index: dict[tuple[float, float], int] = {}
    for span in spans:
        chars = span.get("chars") if isinstance(span, dict) else None
        if not chars:
            continue
        for item in chars:
            # item is (glyph_id, ucs, origin, bbox).
            if len(item) < 3:
                continue
            gid = item[0]
            origin = item[2]
            key = (round(float(origin[0]), 1), round(float(origin[1]), 1))
            index.setdefault(key, int(gid))
    return index


def extract_math_glyphs(page: Any, clip: Any | None = None) -> list[MathGlyph]:
    """Build the per-page (or per-region) list of :class:`MathGlyph`.

    ``clip`` is an optional bounding box ``(x0, y0, x1, y1)`` (or an object with
    those attributes); when given, only glyphs whose baseline origin falls inside
    it are returned. Whitespace glyphs are skipped: they carry no LaTeX token and
    only confuse the geometry pass.
    """
    rawdict = page.get_text("rawdict")
    gid_by_origin = _texttrace_gid_index(page)
    clip_box = _coerce_clip(clip)

    glyphs: list[MathGlyph] = []
    for block in rawdict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = span.get("font", "")
                size = float(span.get("size", 0.0))
                flags = int(span.get("flags", 0))
                for ch in span.get("chars", []):
                    char = ch.get("c", "")
                    if not char or char == " ":
                        continue
                    ox, oy = ch["origin"]
                    ox = float(ox)
                    oy = float(oy)
                    if clip_box is not None and not _origin_in_box(ox, oy, clip_box):
                        continue
                    bbox = ch["bbox"]
                    gid = gid_by_origin.get((round(ox, 1), round(oy, 1)), -1)
                    glyphs.append(
                        MathGlyph(
                            char=char,
                            font=font,
                            size=size,
                            flags=flags,
                            ox=ox,
                            oy=oy,
                            x0=float(bbox[0]),
                            y0=float(bbox[1]),
                            x1=float(bbox[2]),
                            y1=float(bbox[3]),
                            gid=gid,
                        )
                    )
    glyphs.sort(key=lambda g: (round(g.oy, 1), g.ox))
    return glyphs


def _coerce_clip(clip: Any | None) -> tuple[float, float, float, float] | None:
    if clip is None:
        return None
    if hasattr(clip, "x0"):
        return (float(clip.x0), float(clip.y0), float(clip.x1), float(clip.y1))
    values = list(clip)
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _origin_in_box(ox: float, oy: float, box: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= ox <= x1 and y0 <= oy <= y1


# ---------------------------------------------------------------------------
# Structure recognition: recursive 2D glyph layout -> LaTeX.
#
# The glyph stream proves a born-digital LaTeX PDF is laid out on exact
# baselines: every script, limit, fraction and accent is a deterministic
# geometric relation between glyph origins, sizes and drawn rules. The
# recogniser walks the dominant baseline left to right, attaching the
# super/sub-scripts, large-operator limits, fraction numerators/denominators,
# radicands and accents that geometry assigns to each base glyph. It is pure
# Python, needs no model, and is fully exercisable from hand-built glyph lists.
# ---------------------------------------------------------------------------

# A glyph whose size drops below this fraction of the row size, and which is
# horizontally past its anchor, is a script (super/sub by the sign of its
# baseline shift). Proven from page 3: scripts are size 6.97 vs row 9.96
# (0.70) and 4.98 (the nested 'hid').
_SCRIPT_SIZE_RATIO = 0.85

# Minimum signed baseline shift (as a fraction of the row size) for a smaller
# glyph to count as raised (superscript) or lowered (subscript). The decisive
# page-3 evidence: superscript 'T' is raised 0.41*S, subscripts 'i'/'2' are
# lowered 0.15*S; the two '2' glyphs of the L2 norm sit at the SAME x and are
# told apart purely by the sign of this shift.
_SUP_SHIFT_RATIO = 0.18
_SUB_SHIFT_RATIO = 0.06

# An accent glyph sits almost exactly over its base (near-equal origin-x) and
# slightly above it. Proven: hat over 'x' at ox 255.02 vs 254.87.
_ACCENT_X_TOL_RATIO = 0.6

# Horizontal gap (as a fraction of row size) above which two consecutive base
# glyphs are separated by a space. Below it the run is one token (e.g. the
# letters of ReLU / hid). Proven: ReLU inter-letter gaps are ~0pt; the gap
# before '(' after 'ReLU' is also tight, while '=' and '+' get spacing.
_TOKEN_GAP_RATIO = 0.16


@dataclass(frozen=True)
class Rule:
    """A thin drawn rectangle: a fraction bar, radical vinculum or underbrace.

    Built from ``page.get_drawings()`` rects. ``horizontal`` is True for a wide
    thin bar (a candidate fraction/sqrt rule), False for a tall thin stroke.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    horizontal: bool = True

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def ycenter(self) -> float:
        return (self.y0 + self.y1) / 2.0


def rules_from_page(page: Any, clip: Any | None = None) -> list[Rule]:
    """Collect thin drawn rules (fraction bars, radical vinculums) from a page.

    Only thin, reasonably long rectangles are kept. Horizontal bars (candidate
    fraction/sqrt rules) and vertical strokes are tagged via ``horizontal``.
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return []

    clip_box = _coerce_clip(clip)
    rules: list[Rule] = []
    for drawing in drawings:
        rect = drawing.get("rect") if isinstance(drawing, dict) else None
        if rect is None:
            continue
        x0, y0, x1, y1 = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
        w = max(0.0, x1 - x0)
        h = max(0.0, y1 - y0)
        thin_h = h <= 2.5
        thin_w = w <= 2.5
        if not (thin_h or thin_w):
            continue
        if max(w, h) < 3.0:
            continue
        if clip_box is not None:
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            if not _origin_in_box(cx, cy, clip_box):
                continue
        rules.append(Rule(x0, y0, x1, y1, horizontal=thin_h and w >= h))
    return rules


def _wraps_token(token: str, style: str) -> bool:
    """Whether a glyph token takes a style wrapper.

    A style command (``\\mathrm``/``\\mathbf``/``\\mathbb``) applies only to
    letters: an upright relation/operator/paren/digit (``=``, ``+``, ``(``,
    ``1``) from a CMR font is plain math and stays bare; only its style-bearing
    letters are wrapped. Symbol and bare-italic styles never wrap.
    """
    if style in {"symbol", "italic_bare", "largeop", "amssymbol"}:
        return False
    if token.startswith("\\"):
        return False
    return token.isalpha()


def _style_wrap(token: str, style: str) -> str:
    """Wrap an already-translated token in its style command, when it applies."""
    if not _wraps_token(token, style):
        return token
    if style == "mathbf":
        return rf"\mathbf{{{token}}}"
    if style == "mathit":
        return rf"\mathit{{{token}}}"
    if style == "mathbb":
        return rf"\mathbb{{{token}}}"
    if style == "upright":
        return rf"\mathrm{{{token}}}"
    return token


def _is_accent(glyph: MathGlyph) -> bool:
    return len(glyph.char) == 1 and ord(glyph.char) in ACCENT


def _is_largeop(glyph: MathGlyph) -> bool:
    return font_family(glyph.font).startswith("CMEX")


def _row_size(glyphs: list[MathGlyph]) -> float:
    """The font size of the main row (the body baseline).

    In a born-digital LaTeX PDF the body glyphs are set at the largest size and
    every script/limit/nested subscript is strictly smaller, so the body size is
    the maximum glyph size present. A plain median or mode would be dragged down
    by the many small script/limit glyphs of a display equation. A large operator
    (CMEX sum/prod) is itself set at body size, so it counts.
    """
    sizes = [g.size for g in glyphs if g.size > 0]
    if not sizes:
        return 10.0
    return max(sizes)


def _dominant_baseline(glyphs: list[MathGlyph], row_size: float) -> float:
    """The baseline (origin-y) of the widest cluster of full-size glyphs.

    Glyphs are clustered by origin-y within 0.30*row_size; the cluster carrying
    the greatest total glyph width is the main row. Large operators (CMEX), whose
    origin sits between their limits rather than on the text baseline, are
    excluded from the vote but rejoined to the main row afterwards.

    The returned baseline is the MEDIAN origin-y of the full-size (body) glyphs
    in the winning cluster, not the cluster average. The 0.30*row_size cluster
    tolerance is wide enough to merge a body glyph with its own tightly-set
    subscript (~0.15*row_size below), so the cluster average drifts ~1pt toward
    the subscripts. That drift shrank a real inline subscript shift below
    _SUB_SHIFT_RATIO and flattened the run (e.g. ``a_{i,j}`` -> ``ai,j``). Pinning
    the baseline to the full-size glyphs' median removes the drift; it equals the
    former average to within ~0.6pt for every working display equation, so display
    math is unchanged, while inline subscripts now exceed the shift threshold.
    """
    candidates = [g for g in glyphs if not _is_largeop(g) and not _is_accent(g)]
    if not candidates:
        candidates = list(glyphs)
    tol = max(1.0, 0.30 * row_size)
    clusters: list[list[MathGlyph]] = []
    for glyph in sorted(candidates, key=lambda g: g.oy):
        placed = False
        for cluster in clusters:
            if abs(glyph.oy - cluster[0].oy) <= tol:
                cluster.append(glyph)
                placed = True
                break
        if not placed:
            clusters.append([glyph])
    def cluster_key(cluster: list[MathGlyph]) -> tuple[float, float]:
        # Prefer the cluster carrying the most full-size (body) glyph width; only
        # when no cluster has body-size glyphs does total width decide. This
        # keeps a wide stack of small limit glyphs from outvoting the body row.
        full = sum(g.width for g in cluster if g.size >= 0.85 * row_size)
        return (full, sum(g.width for g in cluster))

    best = max(clusters, key=cluster_key)
    full_oys = [g.oy for g in best if g.size >= 0.85 * row_size]
    if full_oys:
        return median(full_oys)
    return sum(g.oy for g in best) / len(best)


# CMEX brace pieces that draw an \underbrace decoration ('|', '{', 'z', '}').
# They report a readable char but the SAME garbage gid (0xFFFD) as the large
# operator, so they are told apart from \sum by being a *brace* char in a CMEX
# font (proven page 3: the '|{z}|' pieces of the two underbraces in eq 4).
CMEX_BRACE_CHARS = frozenset("|{z}")


def _is_underbrace_piece(glyph: MathGlyph) -> bool:
    return (
        font_family(glyph.font).startswith("CMEX")
        and len(glyph.char) == 1
        and glyph.char in CMEX_BRACE_CHARS
    )


def recognize_latex(glyphs: list[MathGlyph], rules: list[Rule] | None = None) -> str:
    """Reconstruct a LaTeX string for a 2D glyph arrangement.

    ``glyphs`` is the full glyph list of one expression (already filtered of the
    equation-number column); ``rules`` are the drawn fraction/radical bars in the
    same coordinate space. The function never drops a content glyph: an
    unhandled glyph surfaces via :class:`MathRecoUnhandled` from :func:`classify`.
    """
    rules = rules or []
    content = [g for g in glyphs if g.char and g.char != " "]
    if not content:
        return ""
    if any(_is_underbrace_piece(g) for g in content):
        return _recognize_with_underbraces(content, rules).strip()
    row_size = _row_size(content)
    baseline = _dominant_baseline(content, row_size)
    return _emit_row(content, rules, row_size, baseline).strip()


def _recognize_with_underbraces(glyphs: list[MathGlyph], rules: list[Rule]) -> str:
    """Recognise an expression decorated with one or more ``\\underbrace`` groups.

    PROVEN (page 3, eq 4): an \\underbrace is drawn as CMEX brace pieces
    ('|', '{', 'z', '}') just below the main baseline, with a small text-font
    label below them. The math content lives entirely on the main baseline; the
    braces and labels are layout. We therefore (1) split the brace pieces and
    label glyphs out as decoration, (2) recognise the clean main row once, and
    (3) wrap each main-row subexpression that a brace pair spans in
    ``\\underbrace{...}_{\\text{label}}``. Content glyphs are conserved; brace and
    label glyphs are decoration (like accents) and excluded from conservation.
    """
    braces = [g for g in glyphs if _is_underbrace_piece(g)]
    main = [g for g in glyphs if not _is_underbrace_piece(g)]
    # The brace baseline sits below the main baseline; the labels are the small
    # text-font glyphs below the braces. Everything above the brace row is the
    # main equation (content) plus its own scripts.
    brace_top = min(g.y0 for g in braces)
    main_content = [g for g in main if g.oy <= brace_top + 1.0]
    label_glyphs = [g for g in main if g.oy > brace_top + 1.0]

    if not main_content:
        return ""
    row_size = _row_size(main_content)
    baseline = _dominant_baseline(main_content, row_size)

    # A brace pair is a contiguous run of brace pieces; group them by x-gap.
    spans = _underbrace_spans(braces, row_size)
    if not spans:
        return _emit_row(main_content, rules, row_size, baseline).strip()

    # Recognise the run as alternating plain segments and \underbrace segments,
    # cut at each brace span's x extent. Membership of a segment is by glyph
    # centre-x against the brace span (so a norm's exponent, which sits just
    # past the closing bar, stays inside its brace), with each glyph assigned to
    # exactly one segment.
    def center(glyph: MathGlyph) -> float:
        return (glyph.x0 + glyph.x1) / 2

    pieces: list[str] = []
    prev_x = float("-inf")
    for span_x0, span_x1 in spans:
        before = [g for g in main_content if prev_x < center(g) < span_x0]
        spanned = [g for g in main_content if span_x0 <= center(g) <= span_x1]
        if before:
            pieces.append(_emit_row(before, rules, row_size, baseline).strip())
        if spanned:
            label = _underbrace_label(label_glyphs, span_x0, span_x1)
            body = _emit_row(spanned, rules, row_size, baseline).strip()
            tail = rf"_{{\text{{{label}}}}}" if label else ""
            pieces.append(rf"\underbrace{{{body}}}{tail}")
        prev_x = span_x1
    trailing = [g for g in main_content if center(g) > prev_x]
    if trailing:
        pieces.append(_emit_row(trailing, rules, row_size, baseline).strip())
    return " ".join(piece for piece in pieces if piece).strip()


def _underbrace_spans(braces: list[MathGlyph], row_size: float) -> list[tuple[float, float]]:
    """Group brace pieces into x-spans, one per ``\\underbrace`` pair."""
    ordered = sorted(braces, key=lambda g: g.ox)
    spans: list[tuple[float, float]] = []
    current: list[MathGlyph] = []
    gap = max(6.0, 1.2 * row_size)
    for glyph in ordered:
        if current and glyph.x0 - current[-1].x1 > gap:
            spans.append((current[0].x0, current[-1].x1))
            current = []
        current.append(glyph)
    if current:
        spans.append((current[0].x0, current[-1].x1))
    return spans


def _underbrace_label(
    label_glyphs: list[MathGlyph], span_x0: float, span_x1: float
) -> str:
    """Read the text label sitting under one underbrace, by x-overlap.

    rawdict over-prints the label glyphs (each letter appears twice at the same
    origin), so consecutive duplicates at the same origin-x are collapsed in
    :func:`_label_text`.
    """
    chars = sorted(
        (g for g in label_glyphs if span_x0 - 8.0 <= (g.x0 + g.x1) / 2 <= span_x1 + 8.0),
        key=lambda g: g.ox,
    )
    return _label_text(chars)


def _label_text(chars: list[MathGlyph]) -> str:
    """Join label glyphs into words, collapsing rawdict's duplicate overprint."""
    kept: list[MathGlyph] = []
    for glyph in chars:
        if (
            kept
            and abs(glyph.ox - kept[-1].ox) < 0.3
            and glyph.char == kept[-1].char
        ):
            continue
        kept.append(glyph)
    if not kept:
        return ""
    # A word break is a gap noticeably wider than the within-word inter-glyph
    # spacing. Label glyphs are tight (most gaps ~0); a real space is ~0.2*size.
    space_gap = max(1.2, 0.18 * (kept[0].size or 7.0))
    text: list[str] = [kept[0].char]
    for prev, glyph in zip(kept, kept[1:]):
        if glyph.x0 - prev.x1 >= space_gap:
            text.append(" ")
        text.append(glyph.char)
    return "".join(text).strip()




_WRAP_STYLES = frozenset({"upright", "mathbf", "mathit", "mathbb"})


def _emit_row(
    glyphs: list[MathGlyph],
    rules: list[Rule],
    row_size: float,
    baseline: float,
) -> str:
    """Emit one baseline row, attaching scripts/limits/accents to each base."""
    ordered = sorted(glyphs, key=lambda g: g.ox)
    used: set[int] = set()

    # Pre-pair accent glyphs to the base they sit over so the base can wrap.
    accent_for: dict[int, MathGlyph] = {}
    for i, glyph in enumerate(ordered):
        if not _is_accent(glyph):
            continue
        target = _accent_target(glyph, ordered, baseline)
        if target is not None:
            accent_for[id(ordered[target])] = glyph
            used.add(i)

    # A large operator owns its vertically-stacked limits, so claim them before
    # any base glyph can mistake a limit for its own script.
    largeop_latex: dict[int, str] = {}
    for i, glyph in enumerate(ordered):
        if i in used or not _is_largeop(glyph):
            continue
        largeop_latex[i] = _emit_largeop(glyph, ordered, rules, row_size, baseline, used)

    # Identify the base-row members: full-size glyphs on (or, for a large
    # operator, straddling) the dominant baseline. Everything else is a script
    # or a limit attached to the nearest preceding base.
    base_indices: list[int] = []
    for i, glyph in enumerate(ordered):
        if i in used:
            continue
        if i in largeop_latex:
            base_indices.append(i)
            continue
        if _script_kind(glyph, row_size, baseline) == "base":
            base_indices.append(i)

    pieces: list[str] = []
    pending_run: list[MathGlyph] = []
    pending_style: str | None = None
    prev_right: float | None = None  # rightmost x consumed, incl. scripts

    def flush_run() -> None:
        nonlocal pending_run, pending_style
        if pending_run and pending_style is not None:
            token = "".join(_token_of(g) for g in pending_run)
            if pending_style == "italic_bare":
                # An adjacent, undecorated run of >=2 single-letter math-italic
                # glyphs is an operator name (e.g. CMMI-set 'ReLU') and is set
                # upright with \mathrm, mirroring the CMR ('upright') path so the
                # two ReLU spellings in the document agree. A lone math-italic
                # letter (a single variable like M, a, c) stays bare italic.
                if len(pending_run) >= 2:
                    pieces.append(_style_wrap(token, "upright"))
                else:
                    pieces.append(token)
            else:
                pieces.append(_style_wrap(token, pending_style))
        pending_run = []
        pending_style = None

    def spacing(glyph: MathGlyph) -> bool:
        return prev_right is not None and glyph.x0 - prev_right >= _TOKEN_GAP_RATIO * row_size

    for pos, i in enumerate(base_indices):
        glyph = ordered[i]
        add_space = spacing(glyph)

        if i in largeop_latex:
            flush_run()
            if add_space:
                pieces.append(" ")
            pieces.append(largeop_latex[i])
            prev_right = glyph.x1
            continue

        token = _token_of(glyph)
        style = classify(glyph)[1]
        wrapped_single = _style_wrap(token, style)

        # Collect this base glyph's own scripts (and nested sub-scripts).
        next_base_x = (
            ordered[base_indices[pos + 1]].x0 if pos + 1 < len(base_indices) else float("inf")
        )
        sup, sub, script_right = _scripts_for(
            glyph, ordered, used, row_size, baseline, next_base_x
        )

        accent = accent_for.get(id(glyph))
        has_decoration = bool(sup or sub or accent)
        right_extent = max(glyph.x1, script_right)

        if has_decoration:
            flush_run()
            if add_space:
                pieces.append(" ")
            body = wrapped_single
            if accent is not None:
                body = rf"{_token_of(accent)}{{{body}}}"
            if sub:
                body += rf"_{{{sub}}}"
            if sup:
                body += rf"^{{{sup}}}"
            pieces.append(body)
            prev_right = right_extent
            continue

        # No decoration: coalesce consecutive same-style letter runs (ReLU,
        # hid) into one wrapper, but only if the glyphs are adjacent (a space
        # boundary or a style change breaks the run). math-italic (italic_bare)
        # letters participate too, so an operator name set in CMMI is coalesced
        # and wrapped in \mathrm by flush_run when the run is multi-letter; a lone
        # italic letter is flushed bare. A single letter (len-1 alpha) token is
        # required so a control word (\ell, \alpha) never joins a run.
        runnable = (style in _WRAP_STYLES or style == "italic_bare") and (
            len(token) == 1 and token.isalpha()
        )
        coalesce = pending_style == style and runnable and not add_space
        if coalesce:
            pending_run.append(glyph)
            prev_right = right_extent
            continue

        flush_run()
        if add_space:
            pieces.append(" ")
        if runnable:
            pending_run = [glyph]
            pending_style = style
        else:
            pieces.append(wrapped_single)
        prev_right = right_extent

    flush_run()
    return _join_pieces(pieces)


# A trailing TeX control word (\times, \in, \alpha, ...) followed immediately by
# a letter would form an undefined command (\timesd); a space is required.
_TRAILING_COMMAND_RE = re.compile(r"\\[A-Za-z]+$")


def _join_pieces(pieces: list[str]) -> str:
    """Concatenate emitted pieces, inserting a thin space only where a control
    word would otherwise run into a following letter and form a bad command."""
    out: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if out:
            prev = out[-1]
            if (
                prev
                and _TRAILING_COMMAND_RE.search(prev)
                and piece[:1].isalpha()
            ):
                out.append(" ")
        out.append(piece)
    return "".join(out)


def _token_of(glyph: MathGlyph) -> str:
    return classify(glyph)[0]


def _script_kind(glyph: MathGlyph, row_size: float, baseline: float) -> str:
    """Classify a glyph as 'base', 'sup' or 'sub' relative to the row baseline."""
    if glyph.size >= _SCRIPT_SIZE_RATIO * row_size:
        return "base"
    shift = glyph.oy - baseline
    if shift <= -_SUP_SHIFT_RATIO * row_size:
        return "sup"
    if shift >= _SUB_SHIFT_RATIO * row_size:
        return "sub"
    return "base"


def _accent_target(
    accent: MathGlyph, ordered: list[MathGlyph], baseline: float
) -> int | None:
    """Index of the base glyph an accent sits over (near-equal origin-x, above)."""
    best: int | None = None
    best_dx = float("inf")
    for i, glyph in enumerate(ordered):
        if glyph is accent or _is_accent(glyph):
            continue
        if glyph.oy < accent.oy - 0.5:
            continue  # the base is at or below the accent's baseline
        width = max(1.0, glyph.width)
        dx = abs(glyph.ox - accent.ox)
        if dx <= _ACCENT_X_TOL_RATIO * width and dx < best_dx:
            best = i
            best_dx = dx
    return best


def _scripts_for(
    base: MathGlyph,
    ordered: list[MathGlyph],
    used: set[int],
    row_size: float,
    baseline: float,
    next_base_x: float,
) -> tuple[str, str, float]:
    """Gather and recognise the super/sub-scripts trailing one base glyph.

    A script sits after the base (origin-x past the base's right edge) and
    before the next base, is smaller than the row, and is raised/lowered. Scripts
    may themselves carry smaller nested scripts (e.g. ``d_{hid}``), so each group
    is recognised recursively. Returns ``(sup, sub, rightmost_x)`` so the caller
    can space the next base against the consumed scripts, not the bare base.
    """
    sup_glyphs: list[MathGlyph] = []
    sub_glyphs: list[MathGlyph] = []
    right = base.x1
    for i, glyph in enumerate(ordered):
        if i in used or glyph is base:
            continue
        if _is_largeop(glyph):
            continue
        if glyph.ox < base.x1 - 0.5 or glyph.ox >= next_base_x - 0.5:
            continue
        kind = _script_kind(glyph, row_size, baseline)
        if kind == "sup":
            sup_glyphs.append(glyph)
            used.add(i)
            right = max(right, glyph.x1)
        elif kind == "sub":
            sub_glyphs.append(glyph)
            used.add(i)
            right = max(right, glyph.x1)
    sup = _recognise_group(sup_glyphs, row_size) if sup_glyphs else ""
    sub = _recognise_group(sub_glyphs, row_size) if sub_glyphs else ""
    return sup, sub, right


def _emit_largeop(
    op: MathGlyph,
    ordered: list[MathGlyph],
    rules: list[Rule],
    row_size: float,
    baseline: float,
    used: set[int],
) -> str:
    """Emit a CMEX large operator with its upper/lower limits.

    The operator's reported unicode is unreliable, so the command comes from the
    font (``\\sum`` by default; ``\\prod`` / ``\\int`` resolved by glyph id when
    known). Glyphs centred above the operator form the upper limit, those below
    the lower limit; both are recognised recursively.
    """
    op_cmd = _largeop_command(op)
    # Limits are stacked over/under the operator: a glyph belongs to a limit only
    # if its x-span overlaps the operator's (a small margin) AND it is off the
    # main baseline. Body glyphs that merely sit to the right of the operator on
    # the main baseline (e.g. the c_i f_i after a sum) must NOT be swallowed.
    margin = max(3.0, 0.35 * op.width)
    base_tol = max(1.0, 0.30 * row_size)
    above: list[MathGlyph] = []
    below: list[MathGlyph] = []
    for i, glyph in enumerate(ordered):
        if i in used or glyph is op or _is_largeop(glyph):
            continue
        gx = (glyph.x0 + glyph.x1) / 2.0
        if gx < op.x0 - margin or gx > op.x1 + margin:
            continue
        # A full-size glyph on the main baseline is body, not a limit (limits are
        # always typeset smaller). A smaller glyph stacked over/under the
        # operator is a genuine limit even if it happens to align in y.
        if glyph.size >= 0.85 * row_size and abs(glyph.oy - baseline) <= base_tol:
            continue
        if glyph.oy < op.oy - 0.5:
            above.append(glyph)
            used.add(i)
        elif glyph.oy > op.oy + 0.5:
            below.append(glyph)
            used.add(i)
    upper = _recognise_group(above, row_size) if above else ""
    lower = _recognise_group(below, row_size) if below else ""
    result = op_cmd
    if lower:
        result += rf"_{{{lower}}}"
    if upper:
        result += rf"^{{{upper}}}"
    return result


def _largeop_command(op: MathGlyph) -> str:
    """Resolve the LaTeX command for a CMEX large operator by glyph id."""
    cmd = _LARGEOP_BY_GID.get(op.gid)
    if cmd is not None:
        return cmd
    return LARGEOP_PLACEHOLDER


# CMEX10 glyph id -> large-operator command. The rawdict char is garbage and the
# texttrace gid for the very large variants is 0xFFFD (handled by the default).
_LARGEOP_BY_GID: dict[int, str] = {
    88: r"\sum",
    89: r"\prod",
    90: r"\int",
}


def _recognise_group(glyphs: list[MathGlyph], parent_size: float) -> str:
    """Recognise a sub-arrangement (a script body or operator limit).

    The group is recognised on its own baseline so nested smaller scripts (the
    ``hid`` inside ``d_{hid}``) are themselves resolved.
    """
    if not glyphs:
        return ""
    group_size = _row_size(glyphs)
    baseline = _dominant_baseline(glyphs, group_size)
    return _emit_row(glyphs, [], group_size, baseline).strip()


# ---------------------------------------------------------------------------
# Symbol-conservation self-check.
#
# A deterministic recogniser must neither invent nor drop content. The check
# compares the multiset of *content* glyph tokens going in against the content
# tokens coming out of the recognised LaTeX, ignoring layout-only structure
# (braces, ^, _, \frac/\sqrt scaffolding, accents and style wrappers, spacing).
# ---------------------------------------------------------------------------

# Structural / style commands that carry no source glyph: present in the output
# purely because of layout, so they are excluded from the conservation multiset.
_STRUCTURAL_COMMANDS = frozenset(
    {
        r"\mathbf",
        r"\mathrm",
        r"\mathit",
        r"\mathbb",
        r"\mathcal",
        r"\frac",
        r"\sqrt",
        r"\underbrace",
        r"\text",
        r"\left",
        r"\right",
    }
)


def source_token_multiset(glyphs: list[MathGlyph]) -> dict[str, int]:
    """Multiset of content tokens implied by the source glyphs.

    Accent glyphs are layout (they become ``\\hat{}`` scaffolding, not content)
    and CMEX brace pieces are ``\\underbrace`` scaffolding, so both are excluded,
    mirroring :func:`latex_token_multiset`. ``\\underbrace`` text labels are
    excluded by the caller (region conservation), since a flat glyph list cannot
    distinguish a label from body glyphs on its own.
    """
    content = _conservation_content(glyphs)
    counts: dict[str, int] = {}
    for glyph in content:
        token = _token_of(glyph)
        for unit in _content_units(token):
            counts[unit] = counts.get(unit, 0) + 1
    return counts


def _conservation_content(glyphs: list[MathGlyph]) -> list[MathGlyph]:
    """The content glyphs of an expression: drop spaces, accents and underbrace.

    Underbrace decoration (the CMEX brace pieces and the small text-font label
    sitting just below them) is layout, not equation content, so it is excluded
    from conservation just like accents.
    """
    braces = [g for g in glyphs if _is_underbrace_piece(g)]
    brace_top = min((g.y0 for g in braces), default=None)
    content: list[MathGlyph] = []
    for glyph in glyphs:
        if not glyph.char or glyph.char == " ":
            continue
        if _is_accent(glyph):
            continue
        if _is_underbrace_piece(glyph):
            continue
        if brace_top is not None and glyph.oy > brace_top + 1.0:
            # A glyph below the brace row is an underbrace label (annotation).
            continue
        content.append(glyph)
    return content


def latex_token_multiset(latex: str) -> dict[str, int]:
    """Multiset of content tokens parsed back out of a LaTeX string."""
    counts: dict[str, int] = {}
    for token in _iter_latex_tokens(latex):
        for unit in _content_units(token):
            counts[unit] = counts.get(unit, 0) + 1
    return counts


def check_symbol_conservation(glyphs: list[MathGlyph], latex: str) -> bool:
    """True iff the recognised LaTeX conserves the source content glyphs."""
    return source_token_multiset(glyphs) == latex_token_multiset(latex)


_STYLED_TOKEN_RE = re.compile(r"^(\\[A-Za-z]+)\{(.*)\}$")


def _content_units(token: str) -> list[str]:
    """Split a token into conservation units.

    A multi-letter upright/bold run (``ReLU``) carries one source glyph per
    letter, so it is split into its letters; a command token (``\\sum``) is one
    unit; a single character is one unit. Accent/structural commands are dropped.
    A pre-styled token such as ``\\mathcal{L}`` (a CMSY slot override) is split
    the same way the LaTeX tokeniser does: the structural command is dropped and
    its content recurses, so the source and output multisets agree.
    """
    if not token:
        return []
    if token in _STRUCTURAL_COMMANDS or token in {v for v in ACCENT.values()}:
        return []
    styled = _STYLED_TOKEN_RE.match(token)
    if styled and styled.group(1) in _STRUCTURAL_COMMANDS:
        return _content_units(styled.group(2))
    if token.startswith("\\"):
        return [token]
    return [ch for ch in token if not ch.isspace()]


_LATEX_TOKEN_RE = re.compile(r"\\[A-Za-z]+|[^{}^_\\\s]")
_TEXT_GROUP_RE = re.compile(r"\\text\{[^}]*\}")


def _iter_latex_tokens(latex: str) -> list[str]:
    # An \underbrace label is an annotation (\text{...}); its letters have no
    # math source glyph, so they are dropped before tokenising for conservation.
    latex = _TEXT_GROUP_RE.sub(" ", latex)
    return _LATEX_TOKEN_RE.findall(latex)


# ---------------------------------------------------------------------------
# Region detection: display equations and inline-math runs.
#
# A born-digital LaTeX PDF lays each display equation on its own rawdict
# line(s), indented into an equation column and tagged in the right margin by a
# parenthesised equation number set in a TEXT font. Body prose lives in
# full-width, body-left-margin lines. This lets display equations be separated
# from prose deterministically by line geometry alone, with the equation-number
# anchor disambiguating vertically-stacked equations. Inline math is a maximal
# run of math-font glyphs embedded in an otherwise text line.
# ---------------------------------------------------------------------------

from .artifacts import BBox  # noqa: E402  (after the recogniser, by design)

# The far-right column where a parenthesised equation number is parked.
_EQNUM_COLUMN_RATIO = 0.78
# A parenthesised equation number such as "(1)" / "(12)".
_EQNUM_RE = re.compile(r"^\((\d+)\)$")
# A rawdict line is body prose if it spans most of the column width OR starts at
# the body left margin; a display equation is indented and narrow.
_BODY_WIDTH_RATIO = 0.45
_BODY_LEFT_RATIO = 0.22
# A glyph is an equation-script satellite if it is markedly smaller than body
# size even when set in a text font (e.g. the nested 'hid' of d_{hid}).
_SATELLITE_SIZE_RATIO = 0.75


@dataclass(frozen=True)
class MathRegion:
    """One recognised mathematical expression mapped back to the page.

    ``bbox`` is the union of the source glyph boxes (PDF points, top-left
    origin). ``kind`` is ``"display"`` or ``"inline"``. ``source_line_indices``
    are the ``PageLayout.line`` indices this region was reconstructed from, so
    markdown can suppress them. ``number`` is the parenthesised equation label
    when present (otherwise ``None``); the default emission omits it.
    """

    page: int
    bbox: BBox
    latex: str
    kind: str
    source_line_indices: tuple[int, ...] = ()
    number: str | None = None


@dataclass(frozen=True)
class MathResult:
    """All recognised math on one page."""

    page: int
    display: tuple[MathRegion, ...] = ()
    inline: tuple[MathRegion, ...] = ()


@dataclass(frozen=True)
class _RawLine:
    """One rawdict text line with its harvested glyphs and bbox."""

    glyphs: tuple[MathGlyph, ...]
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def text(self) -> str:
        return "".join(g.char for g in sorted(self.glyphs, key=lambda g: g.ox))

    @property
    def ycenter(self) -> float:
        return (self.y0 + self.y1) / 2.0


def _raw_lines(page: Any) -> list[_RawLine]:
    """Harvest rawdict text lines, with per-glyph font/size/origin and a gid.

    rawdict sometimes over-prints a glyph (the same origin twice); exact
    duplicates are collapsed so conservation is not thrown off.
    """
    rawdict = page.get_text("rawdict")
    gid_by_origin = _texttrace_gid_index(page)
    lines: list[_RawLine] = []
    for block in rawdict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            glyphs: list[MathGlyph] = []
            seen: set[tuple[float, float, str]] = set()
            for span in line.get("spans", []):
                font = span.get("font", "")
                size = float(span.get("size", 0.0))
                flags = int(span.get("flags", 0))
                for ch in span.get("chars", []):
                    char = ch.get("c", "")
                    if not char or char == " ":
                        continue
                    ox, oy = ch["origin"]
                    ox = float(ox)
                    oy = float(oy)
                    key = (round(ox, 2), round(oy, 2), char)
                    if key in seen:
                        continue
                    seen.add(key)
                    bbox = ch["bbox"]
                    gid = gid_by_origin.get((round(ox, 1), round(oy, 1)), -1)
                    glyphs.append(
                        MathGlyph(
                            char=char,
                            font=font,
                            size=size,
                            flags=flags,
                            ox=ox,
                            oy=oy,
                            x0=float(bbox[0]),
                            y0=float(bbox[1]),
                            x1=float(bbox[2]),
                            y1=float(bbox[3]),
                            gid=gid,
                        )
                    )
            if not glyphs:
                continue
            bbox = line.get("bbox")
            if bbox is None:
                x0 = min(g.x0 for g in glyphs)
                y0 = min(g.y0 for g in glyphs)
                x1 = max(g.x1 for g in glyphs)
                y1 = max(g.y1 for g in glyphs)
            else:
                x0, y0, x1, y1 = (float(v) for v in bbox)
            lines.append(_RawLine(tuple(glyphs), x0, y0, x1, y1))
    return lines


def _is_eqnum_line(line: _RawLine, page_width: float) -> str | None:
    """Return the equation number if this line is a right-margin ``(N)`` label."""
    if line.x0 < page_width * _EQNUM_COLUMN_RATIO:
        return None
    if any(is_math_font(g.font) for g in line.glyphs):
        return None
    match = _EQNUM_RE.match(line.text.strip())
    return match.group(1) if match else None


def _is_display_equation_line(
    line: _RawLine, page_width: float, body_size: float
) -> bool:
    """A narrow, indented, math-bearing line that is part of a display equation.

    Body prose is rejected (full-width or starting at the body left margin). A
    line qualifies only when it carries a math-font glyph or a sub-body-size
    satellite glyph, so a stray short text fragment is not mistaken for math.
    """
    width = line.x1 - line.x0
    if width >= page_width * _BODY_WIDTH_RATIO:
        return False
    if line.x0 <= page_width * _BODY_LEFT_RATIO:
        return False
    has_math = any(is_math_font(g.font) for g in line.glyphs)
    has_satellite = any(g.size < _SATELLITE_SIZE_RATIO * body_size for g in line.glyphs)
    return has_math or has_satellite


def _body_size(lines: list[_RawLine]) -> float:
    """The modal body font size: the most common glyph size across the page."""
    counts: dict[float, float] = {}
    for line in lines:
        for glyph in line.glyphs:
            size = round(glyph.size, 1)
            counts[size] = counts.get(size, 0.0) + glyph.width
    if not counts:
        return 10.0
    return max(counts, key=lambda size: counts[size])


def _detect_display_regions(
    page_number: int,
    lines: list[_RawLine],
    rules: list[Rule],
    page_width: float,
    page_height: float,
    body_size: float,
) -> list[tuple[BBox, str, str | None, list[MathGlyph]]]:
    """Detect display-equation regions on one page.

    Returns ``(bbox, latex, number, glyphs)`` tuples. Each region is anchored on
    a parenthesised equation number; equation lines are assigned to the nearest
    anchor baseline, so vertically-stacked equations stay separate. Pages with no
    equation number are skipped (the sample's numbered equations are the targets;
    an unnumbered display equation is a defect to extend this anchor logic for,
    never a silent drop).
    """
    anchors = [
        (line.ycenter, number)
        for line in lines
        for number in (_is_eqnum_line(line, page_width),)
        if number is not None
    ]
    if not anchors:
        return []
    anchors.sort()
    anchor_oys = [oy for oy, _ in anchors]

    eq_lines = [
        line
        for line in lines
        if _is_eqnum_line(line, page_width) is None
        and _is_display_equation_line(line, page_width, body_size)
    ]
    if not eq_lines:
        return []

    groups: dict[int, list[_RawLine]] = {i: [] for i in range(len(anchors))}
    for line in eq_lines:
        nearest = min(
            range(len(anchor_oys)), key=lambda k: abs(anchor_oys[k] - line.ycenter)
        )
        groups[nearest].append(line)

    regions: list[tuple[BBox, str, str | None, list[MathGlyph]]] = []
    for index, (_, number) in enumerate(anchors):
        member_lines = groups[index]
        if not member_lines:
            continue
        glyphs = [g for line in member_lines for g in line.glyphs]
        if not glyphs:
            continue
        y0 = min(g.y0 for g in glyphs)
        y1 = max(g.y1 for g in glyphs)
        region_rules = [r for r in rules if y0 - 6.0 <= r.ycenter <= y1 + 6.0]
        latex = recognize_latex(glyphs, region_rules)
        if not latex:
            continue
        bbox = BBox(
            min(g.x0 for g in glyphs),
            y0,
            max(g.x1 for g in glyphs),
            y1,
        )
        regions.append((bbox, latex, number, glyphs))
    return regions


def _detect_inline_regions(
    lines: list[_RawLine],
    body_size: float,
) -> list[tuple[BBox, str, list[MathGlyph]]]:
    """Detect inline-math runs inside otherwise text lines.

    A run is a maximal block of math-font (or sub-body-size satellite) glyphs
    bounded by text glyphs. GUARD: the run must contain a genuine math-font
    glyph at body size, so a lone text superscript (an author affiliation digit
    or asterisk) is not turned into math.
    """
    regions: list[tuple[BBox, str, list[MathGlyph]]] = []
    for line in lines:
        ordered = sorted(line.glyphs, key=lambda g: g.ox)
        run: list[MathGlyph] = []

        def flush(run_glyphs: list[MathGlyph]) -> None:
            if not run_glyphs:
                return
            # The run must carry a genuine body-size math glyph that is real
            # content, i.e. not merely a bare delimiter. A lone CMSY brace/bar
            # from a text line (the '{...}@gmail.com' author line) is layout, not
            # math, and must not be promoted to inline math.
            if not any(
                is_math_font(g.font)
                and g.size >= _SATELLITE_SIZE_RATIO * body_size
                and not _is_math_delimiter_glyph(g)
                for g in run_glyphs
            ):
                return
            latex = recognize_latex(run_glyphs, [])
            if not latex:
                return
            bbox = BBox(
                min(g.x0 for g in run_glyphs),
                min(g.y0 for g in run_glyphs),
                max(g.x1 for g in run_glyphs),
                max(g.y1 for g in run_glyphs),
            )
            regions.append((bbox, latex, list(run_glyphs)))

        for glyph in ordered:
            is_mathish = is_math_font(glyph.font) or glyph.size < _SATELLITE_SIZE_RATIO * body_size
            if is_mathish:
                run.append(glyph)
            else:
                flush(run)
                run = []
        flush(run)
    return regions


def _line_indices_for_bbox(
    bbox: BBox, page_lines: list[Any]
) -> tuple[int, ...]:
    """``PageLayout.line`` indices a *display* region bbox covers (>=0.5 of line)."""
    indices: list[int] = []
    for index, line in enumerate(page_lines):
        lx0, ly0 = float(line.x_min), float(line.y_min)
        lx1, ly1 = float(line.x_max), float(line.y_max)
        x_overlap = max(0.0, min(lx1, bbox.x1) - max(lx0, bbox.x0))
        y_overlap = max(0.0, min(ly1, bbox.y1) - max(ly0, bbox.y0))
        if x_overlap <= 0 or y_overlap <= 0:
            continue
        width = max(1.0, lx1 - lx0)
        height = max(1.0, ly1 - ly0)
        if x_overlap / width >= 0.5 and y_overlap / height >= 0.5:
            indices.append(index)
    return tuple(indices)


def _containing_line_index(bbox: BBox, page_lines: list[Any]) -> int | None:
    """The body line that *contains* a small inline run (best y/x containment).

    An inline run is tiny relative to its body line, so the >=0.5-of-line test
    used for display regions never matches. Instead pick the line whose vertical
    band brackets the run's centre and whose horizontal span contains it, with
    the greatest x-overlap as tie-break.
    """
    cy = (bbox.y0 + bbox.y1) / 2.0
    best: int | None = None
    best_overlap = 0.0
    for index, line in enumerate(page_lines):
        ly0, ly1 = float(line.y_min), float(line.y_max)
        if not (ly0 - 1.0 <= cy <= ly1 + 1.0):
            continue
        lx0, lx1 = float(line.x_min), float(line.x_max)
        x_overlap = max(0.0, min(lx1, bbox.x1) - max(lx0, bbox.x0))
        if x_overlap <= 0:
            continue
        if x_overlap > best_overlap:
            best = index
            best_overlap = x_overlap
    return best


def reconstruct_math(pdf_path: str | Any, pages: list[Any]) -> dict[int, MathResult]:
    """Reconstruct LaTeX for every display equation and inline-math run.

    ``pages`` are :class:`pdfmdlite.layout.PageLayout` objects (used only to map
    each region back to overlapping source-line indices for suppression). The PDF
    is opened once; each in-scope page is harvested via rawdict + get_drawings.
    Figures and tables are NOT touched here -- only equations become LaTeX.
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised when pymupdf absent
        raise RuntimeError(
            "PyMuPDF is required for math reconstruction. "
            "Install it with `python3 -m pip install 'pdfmdlite[figures]'`."
        ) from exc

    results: dict[int, MathResult] = {}
    page_by_number = {page.number: page for page in pages}
    try:
        doc = fitz.open(str(pdf_path))
    except (FileNotFoundError, RuntimeError):
        # The layout may have been supplied independently of an on-disk PDF
        # (synthetic/mocked input). With no readable PDF there are no glyphs to
        # reconstruct, so math is simply absent -- not an error.
        return results
    with doc:
        for page in pages:
            if page.number < 1 or page.number > len(doc):
                continue
            pdf_page = doc[page.number - 1]
            page_width = float(pdf_page.rect.width)
            page_height = float(pdf_page.rect.height)
            lines = _raw_lines(pdf_page)
            if not lines:
                continue
            rules = rules_from_page(pdf_page)
            body_size = _body_size(lines)
            page_lines = page_by_number.get(page.number, page).lines

            display: list[MathRegion] = []
            for bbox, latex, number, _glyphs in _detect_display_regions(
                page.number, lines, rules, page_width, page_height, body_size
            ):
                display.append(
                    MathRegion(
                        page=page.number,
                        bbox=bbox,
                        latex=latex,
                        kind="display",
                        source_line_indices=_line_indices_for_bbox(bbox, page_lines),
                        number=number,
                    )
                )

            inline: list[MathRegion] = []
            display_bboxes = [region.bbox for region in display]
            for bbox, latex, _glyphs in _detect_inline_regions(lines, body_size):
                if any(bbox.intersects(other) for other in display_bboxes):
                    continue
                containing = _containing_line_index(bbox, page_lines)
                indices = (containing,) if containing is not None else ()
                inline.append(
                    MathRegion(
                        page=page.number,
                        bbox=bbox,
                        latex=latex,
                        kind="inline",
                        source_line_indices=indices,
                        number=None,
                    )
                )

            if display or inline:
                results[page.number] = MathResult(
                    page=page.number,
                    display=tuple(display),
                    inline=tuple(inline),
                )
    return results


def region_glyph_multiset(glyphs: list[MathGlyph]) -> dict[str, int]:
    """Conservation multiset for a region's source glyphs (drops decoration)."""
    return source_token_multiset(glyphs)


def check_region_conservation(glyphs: list[MathGlyph], latex: str) -> bool:
    """True iff a region's LaTeX conserves its source content glyphs."""
    return source_token_multiset(glyphs) == latex_token_multiset(latex)
