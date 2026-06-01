# pdfmdlite

> English version: [README.md](README.md)

`pdfmdlite` は、論文や長い技術文書向けの、CPU のみで動作する PDF → Markdown 変換ツールです。PDF 自身が持つ字形ストリーム（文字・フォント・座標）と描画されたベクタ図形を読み取り、そこから**構造を保った** Markdown を再構成します。GPU も大規模言語モデルも、いかなる機械学習モデルも使いません。

最大の特徴は、すべてを画像に潰してしまわないことです。

- **数式 → LaTeX。** 表示数式・インライン数式を、字形フォントと位置（Computer Modern 数式フォント、上付き・下付きのベースライン、大型演算子の上下限、描画された罫線からの分数、重なり字形からのアクセント）から決定的に再構成し、本物の LaTeX として出力します。表示数式は ` ```math ` のフェンスブロック、インライン数式は `$...$` です。
- **表 → Markdown 表。** 罫線付きの表は、描画されたグリッドと字形ストリームからセル単位で再構成し、Markdown のパイプ表として出力します。内容は選択可能なテキストのまま残ります。
- **図 → 画像。** 図（ラスタ画像とベクタ描画）は PNG に切り出します。既定では資産ディレクトリへのリンクになりますが、`--inline-images` を付けると base64 のデータ URI として `.md` に直接埋め込まれ、Markdown が単一の自己完結ファイルになります。

Python の依存パッケージは `pymupdf` だけです。Poppler と Tesseract は外部バイナリで、別系統のテキスト抽出と OCR のときだけ使う任意の依存です。

## 必要なもの

`uv` をインストールします（https://docs.astral.sh/uv/）。

```bash
brew install uv         # macOS
# または: curl -LsSf https://astral.sh/uv/install.sh | sh
```

既定の経路（PyMuPDF によるテキスト抽出 ＋ 図・表・数式抽出）に必要なのは `uv` と `pymupdf` だけで、これらは `uv sync` が導入します。

任意の外部バイナリ:

- **Poppler**（`pdftotext`, `pdftoppm`）— `--text-engine poppler`/`auto` と、OCR のページ画像化のときだけ必要。
- **Tesseract** — `--ocr auto`/`force` のときだけ必要。

```bash
# macOS
brew install poppler tesseract
# Debian/Ubuntu
sudo apt-get install poppler-utils tesseract-ocr
```

日本語 OCR には日本語の言語パック（`tesseract-ocr-jpn`）を導入し、`--ocr-lang jpn` を指定します。

## 使い方

プロジェクトのルートで導入・同期します。

```bash
uv sync --no-editable
```

基本的な変換（テキスト ＋ LaTeX 数式）:

```bash
uv run --no-editable pdfmdlite input.pdf -o output.md
```

完全抽出 — 図は画像の切り出し、表は Markdown 表、数式は LaTeX:

```bash
uv run --no-editable pdfmdlite paper.pdf -o paper.md --artifact-mode both
```

これは Markdown に加えて次を書き出します。

- `paper_assets/*.png` — 図の切り出し（表はインラインの Markdown であって画像ではなく、数式は LaTeX であって画像ではありません）。
- `paper.artifacts.json` — 抽出した各要素のページ番号・バウンディングボックス・種別・キャプション文を収めたマニフェスト。

自己完結の単一ファイル — 図の PNG を base64 として Markdown に埋め込む（資産ディレクトリは書き出さない）:

```bash
uv run --no-editable pdfmdlite paper.pdf -o paper.md --artifact-mode embed --inline-images
```

長い PDF では `--jobs 0` に CPU 並列度を選ばせます。

```bash
uv run --no-editable pdfmdlite book.pdf -o book.md --artifact-mode both --artifact-dpi 180 --jobs 0
```

テキスト層が無いページだけ OCR する:

```bash
uv run --no-editable pdfmdlite scan.pdf -o scan.md --ocr auto --ocr-lang eng
```

長い PDF からページ範囲を処理する:

```bash
uv run --no-editable pdfmdlite book.pdf -o chunk-001.md --first-page 1 --last-page 100
```

### 主なオプション

- `--math {on,off}` — `on`（既定）は字形ストリームから決定的に LaTeX を再構成します。`off` は従来の文字ヒューリスティックによる数式経路に戻します。
- `--artifact-mode {off,manifest,embed,both}` — `off`（既定）はテキストのみ（数式は依然 LaTeX）。`manifest` は図の切り出しと `*.artifacts.json` を書き出し、Markdown 本体には手を付けません。`embed` は表を Markdown として再構成し、図の切り出しをキャプションのアンカー位置に埋め込みます。`both` は `embed` ＋ JSON マニフェストです。`embed`/`both` では表は Markdown のパイプ表に、数式は LaTeX になります。
- `--inline-images` — 図の PNG を base64 のデータ URI として Markdown に埋め込みます（自己完結ファイル）。資産ファイルも資産ディレクトリも書き出しません。
- `--ocr {off,auto,force}` と `--ocr-lang LANG` — OCR の動作と Tesseract の言語。`auto` は埋め込みテキストが閾値を下回るページだけ OCR します。
- `--text-engine {pymupdf,poppler,auto}` — `pymupdf`（既定）は字形を直接読みます。`poppler` は `pdftotext -bbox-layout` を使います。`auto` は Poppler を試し、失敗したら PyMuPDF に戻ります。
- `--keep-headers` — 繰り返されるページのヘッダ・フッタ・ページ番号を残します（既定では除去）。
- `--first-page` / `--last-page` — 1 始まりのページ範囲。分割処理に使います。
- `--artifact-dpi N` — 図の切り出しの描画 DPI（既定 180）。
- `--jobs N` — `0`（既定）は短い PDF では逐次実行し、長い抽出ジョブでは CPU プロセスに展開します。`1` は逐次を強制します。

`--extract-artifacts` は `--artifact-mode manifest` の互換用エイリアスとして残しています。

## 仕組み

PDF → Markdown の誤りの多くは、生のテキスト抽出ではなく、読み順とブロックの意味づけから生じます。ボーンデジタル（LaTeX が生成した）PDF は清潔で決定的な字形メトリクスを持っており、これが機械学習なしの CPU での構造復元を可能にしています。処理の流れ:

1. PyMuPDF（既定）で座標とフォントつきの語・字形を抽出する。Poppler バックエンドでは `pdftotext -bbox-layout` を使う。
2. `--ocr` が有効でページの埋め込みテキストが乏しいときだけ、Tesseract でそのページを OCR する。OCR の座標は PDF のポイント座標へ正規化する。
3. ページをまたいで繰り返されるヘッダ・フッタ・ページ番号と、回転した余白スタンプ（例: arXiv の識別子）を除去する。
4. 行を並べ直し（全幅か、控えめな二段組検出か）、複数行の見出しを統合する。行末ハイフネーションを修復し、孤立した節・付録ラベルを前置する。
5. PDF のブロック単位で段落を再構成し、CJK は余計な空白を入れずに連結する。
6. **数式を LaTeX へ再構成する。** 字形フォント（`CMMI`/`CMSY`/`CMEX`/`MSBM` …）、フォントサイズとベースライン（上付き・下付き、大型演算子の上下限）、描画された罫線（分数・根号）、重なるアクセント字形から復元する。表示数式の領域は、数式フォントの密度と孤立性から検出し、インライン数式を含む散文を表示数式と誤認しないようにする。
7. **罫線付きの表を Markdown へ再構成する。** 描画された罫線から行・列のグリッドを復元し、字形の語をセルに振り分ける。
8. 埋め込み画像とまとまったベクタ描画から図を検出し、PNG に切り出して、キャプション（`Figure`/`Fig`/`Table`/`図`/`表`）を幾何的に対応づける。空・インクの無い領域は棄却する。
9. LaTeX・Markdown 表・埋め込み切り出しとして出力したものの元の字形行を抑止し、内容が重複しないようにする。

正しさは、テストスイートの決定的な保存則チェックで担保しています。元の各字形は出力 LaTeX に厳密に一度だけ現れねばならず（`check_symbol_conservation`）、表の各セルの語は Markdown グリッドへ必ず残らねばなりません。出力された LaTeX は妥当で、`pdflatex` でコンパイルできます。

## 開発

```bash
uv run --no-editable python -m unittest discover -s tests
```

ドキュメントのコマンドは `--no-editable` を使っており、これはパッケージを `.venv` へコピーします（macOS では隠し属性の editable な `.pth` が読み飛ばされることがあるため、`uv run` を安定させる狙いです）。このため `src/pdfmdlite/` 配下を編集したら、テストや CLI に反映させる前に再ビルドが必要です。

```bash
uv sync --no-editable --reinstall-package pdfmdlite
```

Poppler/OCR の結合テストは `pdftotext`/`tesseract` が無い環境では自動でスキップします。

## 限界

数式の再構成はボーンデジタルの LaTeX PDF と一般的な構成（上付き・下付き、分数、上下限つきの総和・積分、根号、アクセント、太字・黒板太字・カリグラフィのフォント）を狙っています。珍しいフォント、密な行列、複数行の整列環境は不完全になることがあります。垂直罫線の無い罫なし表は、間隔ベースの列検出に戻るため、完全に罫線のあるグリッドより信頼性が下がります。スキャン文書は OCR の品質に依存します。次の段階は、代表的な論文の評価スイートを用意し、読み順・見出し精度・表の忠実度・数式精度・図の切り出し再現率について、Poppler のプレーンテキスト・PyMuPDF4LLM・Docling・Marker と比較することです。
