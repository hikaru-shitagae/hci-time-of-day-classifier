# HCI 前半レポート課題仕様書

## 1. 目的

屋外風景画像から時間帯を推定する 3 クラス分類器を構築する。

- 入力：RGB 画像（屋外の風景）
- 出力：時間帯ラベル（3 クラス）
  - `day`
  - `night`
  - `sunrise_sunset`

学習・評価・可視化までを Jupyter Notebook 上で完結させ、レポートに必要な結果（精度、混同行列、誤分類例）を出力する。

---

## 2. 使用データセット（ダウンロード済み）

- Kaggle: **Time Of Day Dataset**
- 取得手段：Kaggle CLI（`kaggle.json` を利用）
- データは Git 管理しない（`.gitignore` により `data/` を除外）

### 2.1 ダウンロードコマンド

```bash
mkdir -p data/raw
kaggle datasets download -d aymenkhouja/timeofdaydataset -p data/raw --unzip
```

---

## 3. 想定ディレクトリ構成

```text
前半レポート課題/
  ├─ notebooks/
  │   └─ 01_train_eval.ipynb      # メインNotebook（学習・評価・可視化を実行）
  ├─ src/
  │   ├─ split_dataset.py         # データ分割スクリプト（raw→train/val/test、70%/15%/15%分割）
  │   └─ train_eval_utils.py      # 学習・評価用ユーティリティ（ResNet18モデル定義、学習ループ、評価、可視化関数）
  ├─ data/
  │   ├─ raw/                     # Kaggleからダウンロードした元データ
  │   │   ├─ daytime/             # 昼間の画像（元データ）
  │   │   ├─ nighttime/           # 夜間の画像（元データ）
  │   │   └─ sunrise/             # 日の出・日没の画像（元データ）
  │   ├─ train/                   # 訓練データ（70%）
  │   │   ├─ day/                 # 昼間クラス
  │   │   ├─ night/               # 夜間クラス
  │   │   └─ sunrise_sunset/      # 日の出・日没クラス
  │   ├─ val/                     # 検証データ（15%）
  │   │   ├─ day/
  │   │   ├─ night/
  │   │   └─ sunrise_sunset/
  │   └─ test/                    # テストデータ（15%）
  │       ├─ day/
  │       ├─ night/
  │       └─ sunrise_sunset/
  ├─ models/
  │   └─ best_model.pt            # 検証精度が最高のモデル重み
  ├─ outputs/
  │   ├─ metrics.json             # 評価メトリクス（accuracy, precision, recall, F1）
  │   ├─ confusion_matrix.png     # 混同行列の画像
  │   ├─ history.png              # 学習履歴グラフ（loss/accuracy）
  │   └─ misclassified_examples.png  # 誤分類例の可視化（最大12枚）
  ├─ README.md                    # プロジェクト概要・使い方
  └─ specifications.md            # 本仕様書
```

---

## 4. データ分割

`data/raw/` から `train/val/test` に画像を分割する。

- 分割比：**train 70% / val 15% / test 15%**
- 分割は **クラスごとに独立にシャッフルして行う**
- 乱数シード：`seed=42`（再現性確保）
- 画像のコピー or 移動：
  - 原則コピー（`raw` を残す）でよい
- クラス名はフォルダ名に一致させる

### 4.1 分割の出力要件

- `data/train/<class>/...`
- `data/val/<class>/...`
- `data/test/<class>/...`

---

## 5. モデル

### 5.1 モデル方式

- **転移学習（PyTorch）**
- ベースモデル：`ResNet18`（ImageNet pretrained）
- 変更点：最終全結合層を **3 クラス**出力に置換

### 5.2 入力サイズ

- 画像サイズ：`224 x 224`
- 正規化：ImageNet の mean/std

---

## 6. 前処理・データ拡張

### 6.1 train 用 transforms

- `Resize(224,224)`（または `RandomResizedCrop(224, scale=(0.8,1.0))`）
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(degrees=10)`
- `ToTensor()`
- `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`

※時間帯分類のため、**ColorJitter 等の色味・明るさ変更は OFF**

### 6.2 val/test transforms

- `Resize(224,224)` + `CenterCrop(224)`（または Resize のみ）
- `ToTensor()`
- `Normalize(...)`

---

## 7. 学習

### 7.1 学習設定

- 損失関数：`CrossEntropyLoss`
- Optimizer：`Adam`
- 学習率：`1e-4`（調整可）
- バッチサイズ：`32`（CPU なら `16` でも可）
- エポック数：`10〜15`（CPU 想定）
- device：`cuda` があれば使用、なければ `cpu`

### 7.2 早期終了/保存

- val accuracy が最高のモデルを `models/best_model.pt` に保存
- 学習履歴（loss/acc）を保存し、グラフを `outputs/history.png` に出力

---

## 8. 評価仕様

テストデータで以下を算出する。

- Accuracy
- Precision / Recall / F1（クラス別）
- Confusion Matrix（画像として保存）
- 代表的な誤分類例の可視化（画像として保存）

### 8.1 出力ファイル要件

- `outputs/metrics.json`
  - `accuracy`
  - `classification_report`
  - 実行日時や設定（任意）
- `outputs/confusion_matrix.png`
- `outputs/misclassified_examples.png`（誤分類を最大 12 枚程度タイル表示）

---

## 9. Notebook

`notebooks/01_train_eval.ipynb` に以下のセクションを含める。

1. Setup（import / device）
2. Dataset split（raw→train/val/test）（既に分割済みならスキップできる設計）
3. DataLoader / transforms
4. Model definition
5. Training（train/val）
6. Test evaluation
7. Visualization（予測例・誤分類例）
