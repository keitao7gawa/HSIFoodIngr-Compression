# HSIFoodIngr-64 Compression & Packaging

A practical CLI to convert the [HSIFoodIngr-64 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/E7WDNQ)[^1] (3,389 HSI+RGB pairs, 21 dishes, 64 ingredients) into a single, efficient HDF5 file for fast ML workflows.

- Unifies ENVI HSI cubes (`.hdr/.dat`), RGB images (`.png`), and JSON annotations into one `.h5`
- Incremental, resumable processing with manifests and failure logs
- Verified structure and concise summaries for QA

## Quick start

### Download dataset (optional)

You can fetch the full HSIFoodIngr-64 dataset directly from Harvard Dataverse via the built-in downloader.

```bash
# API key can be passed via --api-key or environment variable DATAVERSE_API_KEY
export DATAVERSE_API_KEY="<your_key_here>"   # optional (may be required depending on access)

python -m hsifoodingr.cli download \
  --output-dir data/raw \
  --extract-dir data/raw \
  --resume \
  --persistent-id doi:10.7910/DVN/E7WDNQ

# After extraction, proceed with manifest/ingredient_map and processing
```

- Options:
  - **--output-dir**: where to place the downloaded ZIP (and extracted files); default `data/raw`
  - **--api-key / $DATAVERSE_API_KEY**: Dataverse API key
  - **--base-url**: Dataverse base URL (default `https://dataverse.harvard.edu`)
  - **--persistent-id**: dataset persistent ID (default `doi:10.7910/DVN/E7WDNQ`)
  - **--resume/--no-resume**: resume partial ZIP downloads (default: resume)
  - **--force**: re-download even if ZIP exists
  - **--extract/--no-extract**: extract after download (default: extract)
  - **--extract-dir**: destination for extracted files (default: same as `--output-dir`)

### 1) Install environment

```bash
conda env create -f environment.yml
conda activate hsifoodingr-compression
# (optional) install package locally
pip install -e .
```

### 2) Prepare data layout

Place original files under `data/raw/` (any nested layout is fine). Each sample must have the same basename:

- `REFLECTANCE_XXXX.hdr`
- `REFLECTANCE_XXXX.dat`
- `REFLECTANCE_XXXX.png`
- `REFLECTANCE_XXXX.json`

Artifacts, logs and the output HDF5 are written to:
- `data/artifacts/` (manifests, ingredient map, logs)
- `data/h5/` (final `.h5`)

### 3) Build manifest and ingredient map

```bash
python -m hsifoodingr.cli build-manifest --raw-dir data/raw --artifacts-dir data/artifacts
python -m hsifoodingr.cli build-ingredient-map --raw-dir data/raw --artifacts-dir data/artifacts
# Creates:
# - data/artifacts/file_manifest.txt
# - data/artifacts/processable.txt
# - data/artifacts/processable.json
# - data/artifacts/ingredient_map.json
```

### 4) Initialize the HDF5 file

You must provide wavelengths for the 204 HSI bands as a text file (comma-separated or one per line):

```bash
# Example file with 204 wavelengths (replace with true values)
seq 204 | awk '{print 399.5 + $1*3.0}' > data/artifacts/wavelengths.txt

python -m hsifoodingr.cli init-h5 \
  --h5-path data/h5/HSIFoodIngr-64.h5 \
  --ingredient-map-path data/artifacts/ingredient_map.json \
  --wavelengths-path data/artifacts/wavelengths.txt
```

### 5) Process data (resumable)

```bash
python -m hsifoodingr.cli process \
  --raw-dir data/raw \
  --artifacts-dir data/artifacts \
  --h5-path data/h5/HSIFoodIngr-64.h5
# Optional:
#   --limit 100              # process at most 100 new samples
#   --no-skip-existing       # re-append even if basename already present
```

On errors, failures are appended to `data/artifacts/failures.log` and processing continues.

### Process downloaded archives end-to-end (extract → append → cleanup)

If you used the downloader and have archives under `data/raw`, you can run the whole flow in one command.

Dry-run to preview actions:

```bash
python -m hsifoodingr.cli process-archives \
  --input-dir data/raw \
  --work-dir data/tmp/extract \
  --output-h5 data/h5/HSIFoodIngr-64.h5 \
  --archive-glob "HSIFoodIngr-64_*" \
  --auto-bootstrap \
  --keep-archive \
  --remove-extracted \
  --workers 1 \
  --dry-run
```

Then run for real (handles nested archives like `.../HSIFoodIngr-64_data_101/HSIFoodIngr-64_data_101.zip` after first extraction):

```bash
python -m hsifoodingr.cli process-archives \
  --input-dir data/raw \
  --work-dir data/tmp/extract \
  --output-h5 data/h5/HSIFoodIngr-64.h5 \
  --archive-glob "HSIFoodIngr-64_*" \
  --auto-bootstrap \
  --allow-missing-json \
  --remove-archive \
  --remove-extracted \
  --workers 1
```

- `--auto-bootstrap` will create manifest, ingredient map, and initialize the HDF5 if missing (idempotent).
- Already processed basenames in the HDF5 are skipped safely.
- Per-archive completion flags are written to `data/artifacts/processed_archives/*.done`.
- A global completion marker is written to `data/artifacts/.process_complete` once all matched items finish.

If labels (JSON) live separately (e.g., inside `label.tar.gz`), you can append HDR/DAT/PNG first with `--allow-missing-json` and then ingest labels later:

```bash
# After extracting labels under data/raw/labels (for example)
python -m hsifoodingr.cli ingest-labels \
  --labels-root data/raw/labels \
  --artifacts-dir data/artifacts \
  --h5-path data/h5/HSIFoodIngr-64.h5
```

#### Labels packaged separately (label.tar.gz): end-to-end playbook

1) Extract all data packages AND extract label archive under a common root (example layout)
```
data/
  raw/
    HSIFoodIngr-64_data_1.zip.tar.gz  (→ extracted to data/raw/HSIFoodIngr-64_data_1/...)
    HSIFoodIngr-64_data_2.zip.tar.gz  (→ extracted to data/raw/HSIFoodIngr-64_data_2/...)
    label.tar.gz                      (→ extracted to data/raw/labels/REFLECTANCE_*.json)
```

2) Append HDR/DAT/PNG first (JSON missing allowed)
```bash
python -m hsifoodingr.cli process-archives \
  --input-dir data/raw \
  --work-dir data/tmp/extract \
  --output-h5 data/h5/HSIFoodIngr-64.h5 \
  --archive-glob "HSIFoodIngr-64_*" \
  --auto-bootstrap \
  --allow-missing-json \
  --remove-archive --remove-extracted \
  --workers 1
```

3) Ingest labels (JSON) afterwards
```bash
python -m hsifoodingr.cli ingest-labels \
  --labels-root data/raw/labels \
  --artifacts-dir data/artifacts \
  --h5-path data/h5/HSIFoodIngr-64.h5
```

Notes
- Four files do NOT need to be in the same directory. Basename matching is used when ingesting labels.
- Check logs at `data/artifacts/logs/process-archives.log` if items are skipped (e.g., already processed, or no JSON found yet).

### 6) Verify and summarize

```bash
# Structural and data sanity checks
python -m hsifoodingr.cli verify --h5-path data/h5/HSIFoodIngr-64.h5 --scan-limit 200

# Concise dataset summary
python -m hsifoodingr.cli summary --h5-path data/h5/HSIFoodIngr-64.h5 --top-k 10
```

- `verify` checks: required datasets, dtypes, shapes, sample alignment, wavelengths length, duplicate basenames, NaN/Inf in HSI (chunked; limited by `--scan-limit`).
- `summary` prints: sample count, shapes, wavelength min/mean/max, per-class pixel counts (mapped from `ingredient_map`), top dish labels.

## HDF5 layout

```
HSIFoodIngr-64.h5
├── hsi            float32  (N, 512, 512, 204)
│   └─ attrs: description = "Hyperspectral data cubes (height, width, bands)."
├── rgb            uint8    (N, 512, 512, 3)
│   └─ attrs: description = "RGB images corresponding to HSI data."
├── masks          uint8    (N, 512, 512)
│   └─ attrs: description = "Integer-based segmentation masks. See /metadata/ingredient_map for class labels."
└── metadata/
    ├── image_basenames   vlen str (N,)
    ├── dish_labels       vlen str (N,)
    ├── ingredient_map    vlen str (1,)   # JSON string: {"background":0, "Avocado":1, ...}
    └── wavelengths       float32 (204,)
```

## CLI reference

```bash
python -m hsifoodingr.cli --help
python -m hsifoodingr.cli build-manifest --help
python -m hsifoodingr.cli build-ingredient-map --help
python -m hsifoodingr.cli init-h5 --help
python -m hsifoodingr.cli process --help
python -m hsifoodingr.cli verify --help
python -m hsifoodingr.cli summary --help
```

## Development

- Run tests:
  ```bash
  pytest -q
  ```
- Code style: typed Python 3.11, explicit shapes/dtypes, early returns, minimal deep nesting.

---

# HSIFoodIngr-64 圧縮・パッケージング（日本語）

[HSIFoodIngr-64 データセット](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/E7WDNQ)[^1]（HSI+RGB ペア 3,389 件、21 種類の料理、64 種の食材）を、高速アクセス可能な HDF5 に統合する CLI です。

- ENVI 形式の HSI（`.hdr/.dat`）・RGB（`.png`）・アノテーション（`.json`）を単一の `.h5` に統合
- マニフェストと失敗ログにより安全・冪等・再開可能
- 構造検証と要約出力で品質確認

## 使い方（概要）

### データセットのダウンロード（任意）

Harvard Dataverse から内蔵ダウンローダで一括取得できます。

```bash
# API キーは --api-key か環境変数 DATAVERSE_API_KEY で指定可能
export DATAVERSE_API_KEY="<your_key_here>"   # 必要に応じて

python -m hsifoodingr.cli download \
  --output-dir data/raw \
  --extract-dir data/raw \
  --resume \
  --persistent-id doi:10.7910/DVN/E7WDNQ

# 展開後は通常どおりマニフェスト作成・処理に進みます
```

- オプション:
  - **--output-dir**: ZIP と展開先のディレクトリ（既定: `data/raw`）
  - **--api-key / $DATAVERSE_API_KEY**: Dataverse API キー
  - **--base-url**: Dataverse ベース URL（既定: `https://dataverse.harvard.edu`）
  - **--persistent-id**: データセット Persistent ID（既定: `doi:10.7910/DVN/E7WDNQ`）
  - **--resume/--no-resume**: 途中の ZIP を再開（既定: 再開する）
  - **--force**: ZIP があっても再ダウンロード
  - **--extract/--no-extract**: ダウンロード後に展開（既定: 展開する）
  - **--extract-dir**: 展開先（既定: `--output-dir` と同じ）

### 1) 環境構築

```bash
conda env create -f environment.yml
conda activate hsifoodingr-compression
pip install -e .   # 任意
```

### 2) データ配置

`data/raw/` 配下に原データ（任意のサブディレクトリ構成で可）を配置します。各サンプルは同一 basename を持つ 4 つのファイルで構成されます：

- `REFLECTANCE_XXXX.hdr`
- `REFLECTANCE_XXXX.dat`
- `REFLECTANCE_XXXX.png`
- `REFLECTANCE_XXXX.json`

成果物は以下に出力されます：
- `data/artifacts/`（マニフェスト、ingredient_map、ログ）
- `data/h5/`（最終的な `.h5`）

### 3) マニフェストと ingredient_map の作成

```bash
python -m hsifoodingr.cli build-manifest --raw-dir data/raw --artifacts-dir data/artifacts
python -m hsifoodingr.cli build-ingredient-map --raw-dir data/raw --artifacts-dir data/artifacts
```

### 4) HDF5 初期化

204 バンド分の波長リスト（テキスト、カンマ区切り or 改行区切り）を用意してください：

```bash
seq 204 | awk '{print 399.5 + $1*3.0}' > data/artifacts/wavelengths.txt

python -m hsifoodingr.cli init-h5 \
  --h5-path data/h5/HSIFoodIngr-64.h5 \
  --ingredient-map-path data/artifacts/ingredient_map.json \
  --wavelengths-path data/artifacts/wavelengths.txt
```

### 5) データ処理（再開可能）

```bash
python -m hsifoodingr.cli process \
  --raw-dir data/raw \
  --artifacts-dir data/artifacts \
  --h5-path data/h5/HSIFoodIngr-64.h5
# オプション:
#   --limit 100            # 最大 100 サンプルのみ処理
#   --no-skip-existing     # 既存 basename をスキップしない
```

エラーは `data/artifacts/failures.log` に追記され、処理は継続します。

### ダウンロード直後からの一括処理（解凍 → 追記 → 後始末）

ダウンローダで取得したアーカイブが `data/raw` にある場合、次のコマンドで一括実行できます。

乾式（計画のみ表示）:

```bash
python -m hsifoodingr.cli process-archives \
  --input-dir data/raw \
  --work-dir data/tmp/extract \
  --output-h5 data/h5/HSIFoodIngr-64.h5 \
  --archive-glob "HSIFoodIngr-64_*" \
  --auto-bootstrap \
  --keep-archive \
  --remove-extracted \
  --workers 1 \
  --dry-run
```

本実行（一次展開後にさらに内側の ZIP/TAR が現れる多段構成にも対応）:

```bash
python -m hsifoodingr.cli process-archives \
  --input-dir data/raw \
  --work-dir data/tmp/extract \
  --output-h5 data/h5/HSIFoodIngr-64.h5 \
  --archive-glob "HSIFoodIngr-64_*" \
  --auto-bootstrap \
  --allow-missing-json \
  --remove-archive \
  --remove-extracted \
  --workers 1
```

- `--auto-bootstrap` はマニフェスト・ingredient_map・HDF5 初期化を未作成なら自動実行（冪等）。
- HDF5 に既存の basename は自動スキップされます。
- アーカイブごとの完了印は `data/artifacts/processed_archives/*.done` に出力されます。
- 全完了時は `data/artifacts/.process_complete` を作成します。

もし JSON が別ツリー（例: `label.tar.gz` 配下）にある場合は、まず `--allow-missing-json` で HDR/DAT/PNG のみを追記し、後から JSON を取り込めます：

```bash
# 例: JSON を data/raw/labels 以下に展開後
python -m hsifoodingr.cli ingest-labels \
  --labels-root data/raw/labels \
  --artifacts-dir data/artifacts \
  --h5-path data/h5/HSIFoodIngr-64.h5
```

#### JSON が別アーカイブ（label.tar.gz）の場合の手順（推奨）

1) すべてのデータパッケージと label アーカイブを共通ルートに展開（例）
```
data/
  raw/
    HSIFoodIngr-64_data_1.zip.tar.gz  （→ data/raw/HSIFoodIngr-64_data_1/... に展開）
    HSIFoodIngr-64_data_2.zip.tar.gz  （→ data/raw/HSIFoodIngr-64_data_2/... に展開）
    label.tar.gz                      （→ data/raw/labels/REFLECTANCE_*.json に展開）
```

2) まず HDR/DAT/PNG を追記（JSON 不在を許可）
```bash
python -m hsifoodingr.cli process-archives \
  --input-dir data/raw \
  --work-dir data/tmp/extract \
  --output-h5 data/h5/HSIFoodIngr-64.h5 \
  --archive-glob "HSIFoodIngr-64_*" \
  --auto-bootstrap \
  --allow-missing-json \
  --remove-archive --remove-extracted \
  --workers 1
```

3) 後から JSON を取り込み（basename で突合）
```bash
python -m hsifoodingr.cli ingest-labels \
  --labels-root data/raw/labels \
  --artifacts-dir data/artifacts \
  --h5-path data/h5/HSIFoodIngr-64.h5
```

補足
- 4 つのファイルが同じディレクトリである必要はありません。後段のラベル取り込み時に basename で突合します。
- スキップ理由や進捗は `data/artifacts/logs/process-archives.log` を参照してください。

### 6) 検証・要約

```bash
# 構造・データ健全性チェック
python -m hsifoodingr.cli verify --h5-path data/h5/HSIFoodIngr-64.h5 --scan-limit 200

# 要約表示
python -m hsifoodingr.cli summary --h5-path data/h5/HSIFoodIngr-64.h5 --top-k 10
```

- `verify` は必須データセットの有無、dtype・形状、サンプル数整合、波長長、basename 重複、HSI の NaN/Inf（チャンク・サンプリング）を確認します。
- `summary` はサンプル数・形状・波長の統計、クラス別ピクセル数（`ingredient_map` による名称対応）、上位料理ラベルを表示します。

## HDF5 構造

```
HSIFoodIngr-64.h5
├── hsi            float32  (N, 512, 512, 204)
├── rgb            uint8    (N, 512, 512, 3)
├── masks          uint8    (N, 512, 512)
└── metadata/
    ├── image_basenames   vlen str (N,)
    ├── dish_labels       vlen str (N,)
    ├── ingredient_map    vlen str (1,)   # JSON
    └── wavelengths       float32 (204,)
```

## 開発向け

- テスト実行：
  ```bash
  pytest -q
  ```
- コード方針：Python 3.11、明示的な shape/dtype、早期 return、過度なネスト回避。

[^1]: X. Xia, W. Liu, L. Wang and J. Sun, "HSIFoodIngr-64: A Dataset for Hyperspectral Food-Related Studies and a Benchmark Method on Food Ingredient Retrieval," in IEEE Access, vol. 11, pp. 13152-13162, 2023, doi: 10.1109/ACCESS.2023.3243243. 
