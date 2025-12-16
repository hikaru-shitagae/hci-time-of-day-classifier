# HCI 前半レポート課題仕様書

## 1. 目的
屋外風景画像から時間帯を推定する3クラス分類器を構築する。

- 入力：RGB画像（屋外の風景）
- 出力：時間帯ラベル（3クラス）
  - `day`
  - `night`
  - `sunrise_sunset`

学習・評価・可視化までをJupyter Notebook上で完結させ、レポートに必要な結果（精度、混同行列、誤分類例）を出力する。

---

## 2. 使用データセット（ダウンロード済み）
- Kaggle: **Time Of Day Dataset**
- 取得手段：Kaggle CLI（`kaggle.json` を利用）
- データはGit管理しない（`.gitignore` により `data/` を除外）

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
  │   └─ 01_train_eval.ipynb
  ├─ src/
  │   ├─ split_dataset.py
  │   └─ train_eval_utils.py
  ├─ data/                     # gitignore対象
  │   ├─ raw/
  │   ├─ train/
  │   │   ├─ day/
  │   │   ├─ night/
  │   │   └─ sunrise_sunset/
  │   ├─ val/
  │   │   ├─ day/
  │   │   ├─ night/
  │   │   └─ sunrise_sunset/
  │   └─ test/
  │       ├─ day/
  │       ├─ night/
  │       └─ sunrise_sunset/
  ├─ models/                   # gitignore対象
  │   └─ best_model.pt
  ├─ outputs/                  # gitignore対象
  │   ├─ metrics.json
  │   ├─ confusion_matrix.png
  │   ├─ history.png
  │   └─ misclassified_examples.png
  ├─ README.md
  └─ specifications.md
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
- 変更点：最終全結合層を **3クラス**出力に置換

### 5.2 入力サイズ
- 画像サイズ：`224 x 224`
- 正規化：ImageNetのmean/std

---

## 6. 前処理・データ拡張

### 6.1 train用 transforms
- `Resize(224,224)`（または `RandomResizedCrop(224, scale=(0.8,1.0))`）
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(degrees=10)`
- `ToTensor()`
- `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`

※時間帯分類のため、**ColorJitter等の色味・明るさ変更はOFF**

### 6.2 val/test transforms
- `Resize(224,224)` + `CenterCrop(224)`（またはResizeのみ）
- `ToTensor()`
- `Normalize(...)`

---

## 7. 学習

### 7.1 学習設定
- 損失関数：`CrossEntropyLoss`
- Optimizer：`Adam`
- 学習率：`1e-4`（調整可）
- バッチサイズ：`32`（CPUなら `16` でも可）
- エポック数：`10〜15`（CPU想定）
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
- `outputs/misclassified_examples.png`（誤分類を最大12枚程度タイル表示）

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
