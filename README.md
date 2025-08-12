# HSIFoodIngr-64 Compression & Packaging

A practical CLI to convert the HSIFoodIngr-64 dataset (3,389 HSI+RGB pairs, 21 dishes, 64 ingredients) into a single, efficient HDF5 file for fast ML workflows.

- Unifies ENVI HSI cubes (`.hdr/.dat`), RGB images (`.png`), and JSON annotations into one `.h5`
- Incremental, resumable processing with manifests and failure logs
- Verified structure and concise summaries for QA

## Quick start

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

HSIFoodIngr-64 データセット（HSI+RGB ペア 3,389 件、21 種類の料理、64 種の食材）を、高速アクセス可能な HDF5 に統合する CLI です。

- ENVI 形式の HSI（`.hdr/.dat`）・RGB（`.png`）・アノテーション（`.json`）を単一の `.h5` に統合
- マニフェストと失敗ログにより安全・冪等・再開可能
- 構造検証と要約出力で品質確認

## 使い方（概要）

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
