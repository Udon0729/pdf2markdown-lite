# pdf2markdown-lite

> English version: [README.md](README.md)

`pdfmdlite` は、PDF を Markdown に変換する軽量ツールです。論文や技術文書を主な対象とし、CPU のみで動作します。

PDF 内の文字、座標、フォント、図形、罫線などの情報を読み取り、本文、数式、表、図を Markdown と画像に再構成します。

## 特徴

- 数式を LaTeX として出力します。表示数式は ` ```math ` ブロック、文中数式は `$...$` になります。
- 罫線のある表を Markdown のパイプ表として出力します。
- 図や画像を PNG として切り出します。
- `--inline-images` を使うと、画像を base64 として Markdown に埋め込めます。
- 標準動作に必要な Python 依存は `pymupdf` のみです。
- OCR は必要な場合だけ Tesseract を使います。

## 必要なもの

まず `uv` をインストールします。

```bash
brew install uv
# または
curl -LsSf https://astral.sh/uv/install.sh | sh
```

標準的な変換には `uv` と `pymupdf` だけが必要です。`pymupdf` は `uv sync` で自動的に入ります。

Poppler と Tesseract は任意です。

```bash
# macOS
brew install poppler tesseract

# Debian / Ubuntu
sudo apt-get install poppler-utils tesseract-ocr
```

日本語 OCR を使う場合は、日本語言語パックを入れたうえで `--ocr-lang jpn` を指定します。

```bash
sudo apt-get install tesseract-ocr-jpn
```

## セットアップ

```bash
uv sync --no-editable
```

## 使い方

基本的な変換です。

```bash
uv run --no-editable pdfmdlite input.pdf -o output.md
```

図、表、数式を含めて抽出します。

```bash
uv run --no-editable pdfmdlite paper.pdf -o paper.md --artifact-mode both
```

この場合、次のファイルが生成されます。

- `paper.md`
- `paper_assets/*.png`
- `paper.artifacts.json`

画像を Markdown に埋め込み、単一ファイルにまとめる場合は次のようにします。

```bash
uv run --no-editable pdfmdlite paper.pdf -o paper.md --artifact-mode embed --inline-images
```

長い PDF では、`--jobs 0` により CPU 並列処理を自動設定できます。

```bash
uv run --no-editable pdfmdlite book.pdf -o book.md --artifact-mode both --artifact-dpi 180 --jobs 0
```

埋め込みテキストが少ないページだけ OCR する場合です。

```bash
uv run --no-editable pdfmdlite scan.pdf -o scan.md --ocr auto --ocr-lang eng
```

ページ範囲を指定して処理する場合です。

```bash
uv run --no-editable pdfmdlite book.pdf -o chunk-001.md --first-page 1 --last-page 100
```

## 主なオプション

数式は常に LaTeX として復元されます（ディスプレイ数式は ` ```math ` ブロック、インライン数式は `$...$`）。無効化するオプションはありません。

| オプション | 説明 |
|---|---|
| `--artifact-mode {off,manifest,embed,both}` | 図・表・数式の抽出方法を指定します。 |
| `--inline-images` | 図を base64 として Markdown に埋め込みます。 |
| `--ocr {off,auto,force}` | OCR の使用方法を指定します。 |
| `--ocr-lang LANG` | Tesseract の OCR 言語を指定します。 |
| `--text-engine {pymupdf,poppler,auto}` | テキスト抽出エンジンを指定します。 |
| `--keep-headers` | ヘッダ、フッタ、ページ番号を残します。 |
| `--first-page` / `--last-page` | 処理するページ範囲を指定します。 |
| `--artifact-dpi N` | 図を切り出す解像度を指定します。標準は `180` です。 |
| `--jobs N` | 並列処理数を指定します。`0` は自動設定です。 |

`--extract-artifacts` は、`--artifact-mode manifest` と同じ意味の互換オプションです。

## 仕組み

`pdfmdlite` は、PDF を画像として扱うのではなく、PDF 内の構造情報を直接読み取ります。

処理の流れは次のとおりです。

1. PyMuPDF または Poppler で、文字、座標、フォント情報を取得します。
2. 必要に応じて Tesseract で OCR します。
3. ヘッダ、フッタ、ページ番号、余白のスタンプを除去します。
4. 行順、段組み、見出し、段落を推定します。
5. 数式を LaTeX に復元します。
6. 罫線から表の行列構造を復元します。
7. 図を PNG として切り出し、キャプションと対応づけます。
8. Markdown、画像、アーティファクト一覧を出力します。

数式復元では、フォント、文字サイズ、ベースライン、横線、記号の重なりを使って、上付き・下付き、分数、根号、アクセント、総和・積分の範囲などを推定します。

## 開発

テストを実行します。

```bash
uv run --no-editable python -m unittest discover -s tests
```

`--no-editable` を使っているため、`src/pdfmdlite/` 以下を変更した後は、次のコマンドで再インストールします。

```bash
uv sync --no-editable --reinstall-package pdfmdlite
```

Poppler や Tesseract が必要な結合テストは、各コマンドが存在しない環境では自動的にスキップされます。

## 制限

数式復元は、LaTeX 由来の PDF と一般的な数式構造を主な対象にしています。特殊なフォント、複雑な行列、複数行の整列数式では失敗することがあります。

表は罫線のあるものを得意とします。縦罫線のない表では、単語間隔から列を推定するため精度が下がります。

スキャン PDF の品質は OCR に依存します。
