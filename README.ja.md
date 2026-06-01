# pdfmdlite

> English version: [README.md](README.md)

`pdfmdlite` は、論文や長めの技術文書を PDF から Markdown へ変換するツールです。CPU だけで動き、GPU も大規模言語モデルも、機械学習モデルも一切使いません。

仕組みの要点は、PDF が内部に持っている情報をそのまま読み取ることです。PDF には、どの文字をどのフォントでどの座標に置くか、どこに線や図形を描くか、といったデータが入っています。`pdfmdlite` はこれを手がかりに、元の文書の構造を保ったまま Markdown へ組み直します。

特徴は、ページ全体をただの画像にしてしまわないことです。

- **数式は LaTeX に戻します。** 表示用の数式も文中の数式も、使われているフォントや文字の位置から元の式を復元し、本物の LaTeX として書き出します。表示数式は ` ```math ` のブロック、文中の数式は `$...$` です。フォントの種類で数学記号を見分け、文字の大きさと上下のずれで上付き・下付きを判断し、総和記号などの上下に付く範囲、描かれた横線から分数、重なった記号からアクセントを読み取ります。
- **表は Markdown の表に戻します。** 罫線のある表は、引かれている線と文字の位置から行と列を復元し、Markdown のパイプ表として出力します。中身は画像ではなく、選択・検索できるテキストのまま残ります。
- **図は画像として切り出します。** 写真やベクタ図形の図は PNG に切り出します。標準では別フォルダの画像ファイルへのリンクになりますが、`--inline-images` を付けると画像を base64 形式で Markdown に直接埋め込み、1 つのファイルで完結させられます。

Python 側で必要なのは `pymupdf` だけです。Poppler と Tesseract は外部プログラムで、別方式のテキスト抽出と OCR を使うときにだけ必要になります。

## 必要なもの

まず `uv` を入れます（https://docs.astral.sh/uv/）。

```bash
brew install uv         # macOS
# または: curl -LsSf https://astral.sh/uv/install.sh | sh
```

標準的な使い方（PyMuPDF によるテキスト抽出と、図・表・数式の抽出）に必要なのは `uv` と `pymupdf` だけで、`pymupdf` は `uv sync` が自動で入れます。

次の外部プログラムは、使う機能によってだけ必要です。

- **Poppler**（`pdftotext`, `pdftoppm`）— `--text-engine poppler`/`auto` を選んだときと、OCR でページを画像化するときに使います。
- **Tesseract** — `--ocr auto`/`force` を選んだときに使います。

```bash
# macOS
brew install poppler tesseract
# Debian/Ubuntu
sudo apt-get install poppler-utils tesseract-ocr
```

日本語の OCR には、日本語の言語パック（`tesseract-ocr-jpn`）を入れたうえで `--ocr-lang jpn` を指定します。

## 使い方

プロジェクトのフォルダで、まず環境を用意します。

```bash
uv sync --no-editable
```

いちばん基本的な変換（テキストと LaTeX 数式）:

```bash
uv run --no-editable pdfmdlite input.pdf -o output.md
```

すべて抽出する場合（図は画像、表は Markdown の表、数式は LaTeX）:

```bash
uv run --no-editable pdfmdlite paper.pdf -o paper.md --artifact-mode both
```

このとき、Markdown のほかに次のファイルもできます。

- `paper_assets/*.png` — 切り出した図（表は Markdown に直接書き込まれるので画像にはならず、数式も LaTeX なので画像にはなりません）。
- `paper.artifacts.json` — 抽出した各要素について、ページ番号・位置・種類・キャプションをまとめた一覧。

1 つのファイルで完結させたい場合（図の PNG を base64 で Markdown に埋め込み、画像フォルダを作らない）:

```bash
uv run --no-editable pdfmdlite paper.pdf -o paper.md --artifact-mode embed --inline-images
```

ページ数の多い PDF では、`--jobs 0` にしておくと CPU の並列度を自動で決めます。

```bash
uv run --no-editable pdfmdlite book.pdf -o book.md --artifact-mode both --artifact-dpi 180 --jobs 0
```

テキストが埋め込まれていないページだけ OCR する:

```bash
uv run --no-editable pdfmdlite scan.pdf -o scan.md --ocr auto --ocr-lang eng
```

長い PDF の一部のページだけ処理する:

```bash
uv run --no-editable pdfmdlite book.pdf -o chunk-001.md --first-page 1 --last-page 100
```

### 主なオプション

- `--math {on,off}` — `on`（標準）は PDF の文字情報から LaTeX を組み直します。`off` にすると、従来の簡易な文字ヒューリスティックで数式を処理します。
- `--artifact-mode {off,manifest,embed,both}` — 図・表・数式の扱いを決めます。`off`（標準）はテキストだけ（数式は LaTeX のまま）。`manifest` は図の切り出しと `*.artifacts.json` を作り、Markdown 本体には手を加えません。`embed` は表を Markdown に組み直し、図をキャプションの位置に埋め込みます。`both` は `embed` に一覧ファイルの出力を加えたものです。`embed` と `both` では、表は Markdown のパイプ表、数式は LaTeX になります。
- `--inline-images` — 図の PNG を base64 形式で Markdown に埋め込みます（1 ファイルで完結）。画像ファイルも画像フォルダも作りません。
- `--ocr {off,auto,force}` と `--ocr-lang LANG` — OCR の動作と Tesseract の言語を指定します。`auto` は、埋め込みテキストが少ないページだけを OCR します。
- `--text-engine {pymupdf,poppler,auto}` — `pymupdf`（標準）は文字を直接読み取ります。`poppler` は `pdftotext -bbox-layout` を使います。`auto` はまず Poppler を試し、うまくいかなければ PyMuPDF に切り替えます。
- `--keep-headers` — ページごとに繰り返されるヘッダ・フッタ・ページ番号を残します（標準では取り除きます）。
- `--first-page` / `--last-page` — 処理するページの範囲（1 ページ目を 1 とする番号）。文書を分けて処理したいときに使います。
- `--artifact-dpi N` — 図を切り出すときの解像度（標準は 180）。
- `--jobs N` — `0`（標準）は、短い PDF では順番に処理し、長い抽出では複数の CPU プロセスに分散します。`1` にすると必ず順番に処理します。

`--extract-artifacts` は、`--artifact-mode manifest` と同じ意味の古い書き方として残してあります。

## 仕組み

PDF から Markdown への変換でつまずく原因は、文字をうまく取り出せないことより、読む順番やブロックの意味を取り違えることにあります。LaTeX などで作られた PDF（はじめからデジタルの PDF）は、文字の大きさや位置の情報がきれいに整っているため、機械学習を使わなくても CPU だけで構造を復元できます。処理の流れは次のとおりです。

1. PyMuPDF（標準）で、座標とフォント付きの単語・文字を取り出します。Poppler を使う場合は `pdftotext -bbox-layout` を使います。
2. `--ocr` が有効で、かつ埋め込みテキストが乏しいページに限り、Tesseract で OCR します。OCR の座標は PDF の座標系に合わせて変換します。
3. 全ページで繰り返されるヘッダ・フッタ・ページ番号や、余白に回転して入っているスタンプ（たとえば arXiv の識別子）を取り除きます。
4. 行を正しい順番に並べ直し（1 段組みか、控えめに判定した 2 段組みか）、複数行にまたがる見出しをまとめます。行末で切れたハイフンをつなぎ、離れてしまった節・付録の番号を見出しの前に戻します。
5. PDF のブロック単位で段落を組み直し、日本語などは余計な空白を入れずにつなげます。
6. **数式を LaTeX に組み直します。** フォントの種類（`CMMI`/`CMSY`/`CMEX`/`MSBM` など）、文字の大きさとベースライン（上付き・下付き、総和記号などの上下の範囲）、描かれた横線（分数や根号）、重なったアクセント記号から、元の式を復元します。表示数式かどうかは、数学記号の密集ぐあいと前後からの孤立ぐあいで見分け、文中の数式を含む普通の文章を表示数式と取り違えないようにしています。
7. **罫線のある表を Markdown に組み直します。** 引かれている線から行と列の枠を割り出し、単語をセルに振り分けます。
8. 埋め込み画像やまとまったベクタ図形から図を見つけて PNG に切り出し、キャプション（`Figure`/`Fig`/`Table`/`図`/`表`）を位置関係から対応づけます。中身が空の領域は除きます。
9. LaTeX・Markdown の表・埋め込み画像として出力したものは、元になった文字を本文から消し、内容が二重に出ないようにします。

正しさは、テストで機械的に確かめています。元の文字はすべて出力した LaTeX に過不足なく 1 回ずつ現れること（`check_symbol_conservation`）、表の各セルの単語が Markdown の表に必ず残ること、を検査します。出力した LaTeX はそのまま `pdflatex` でコンパイルできます。

## 開発

```bash
uv run --no-editable python -m unittest discover -s tests
```

ドキュメントのコマンドはどれも `--no-editable` を付けています。これはパッケージを `.venv` にコピーする設定で、macOS で隠し属性の editable な `.pth` が読み飛ばされることがあるのを避け、`uv run` の動作を安定させるためです。この設定のため、`src/pdfmdlite/` 以下を編集したら、テストや CLI に反映させる前に作り直しが必要です。

```bash
uv sync --no-editable --reinstall-package pdfmdlite
```

Poppler や OCR を使う結合テストは、`pdftotext`/`tesseract` が入っていない環境では自動的にスキップします。

## 今のところの限界

数式の復元は、LaTeX で作られた PDF と、よくある構成（上付き・下付き、分数、上下に範囲が付く総和や積分、根号、アクセント、太字・黒板太字・装飾体のフォント）を主な対象にしています。珍しいフォント、要素の詰まった行列、複数行を揃える数式環境などは、うまくいかないことがあります。縦の罫線がない表は、単語の間隔から列を推測する方式に切り替わるため、完全に罫線のある表より精度が落ちます。スキャンした文書の結果は OCR の品質に左右されます。今後の課題は、代表的な論文をそろえた評価用のデータを用意し、読む順番・見出しの精度・表の再現度・数式の精度・図の切り出し率について、Poppler のプレーンテキストや PyMuPDF4LLM、Docling、Marker と比べることです。
