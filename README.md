# Lab3：胸部 X 光四分類（normal / bacteria / virus / COVID-19）

NYCU AIMI 2026 Lab3 的訓練程式。任務是把胸部 X 光影像分成 normal、bacteria、virus、COVID-19 四類，Kaggle 以 macro-F1 評分。

| 提交 | 模型 | OOF macro-F1 | Kaggle |
| --- | --- | --- | --- |
| 截止前的正式提交 | RAD-DINO 5-fold ensemble、取最後 epoch、COVID-19 門檻 0.54 | 0.874 | 0.88092 |
| 截止後的補充提交（不計排名） | `RADDINO.py` 的 `base_cv5`：5-fold ensemble、各 fold 取最佳 epoch、COVID-19 門檻 0.58 | 0.878（巢狀 CV） | 0.88935 |

實驗過程、所有比較與討論請見報告 `LAB3_315551143_施正祐.pdf`。

## 檔案結構

```
.
├── RADDINO.py          # RAD-DINO（microsoft/rad-dino，胸部 X 光自監督預訓練的 ViT-B/14），最終模型
├── ConvNeXtTiny.py     # ConvNeXt-Tiny（torchvision，ImageNet-1k）對照
├── ResNet.py           # ResNet-50（torchvision，ImageNet-1k）對照
├── early_resnet18/     # 初期在 ResNet18 上的架構調整（報告第 2.a.3、3.b.1 節）
│   ├── main.py         # 單一 ResNet18
│   ├── mainOvR.py      # One-vs-Rest：4 個二元 ResNet18
│   └── mainSupport.py  # ResNet18 + bacteria vs virus 輔助分支
├── requirements.txt
└── README.md
```

`RADDINO.py`、`ConvNeXtTiny.py`、`ResNet.py` 共用同一套程式，只有模型定義的區塊不同，參數和輸出格式完全相同。三支都是獨立檔案，彼此不互相 import。

## 環境

```bash
pip install -r requirements.txt
```

開發環境為 Python 3.11、torch 2.12（CUDA 12.6）、torchvision 0.27、transformers 5.23，單張 RTX 3080 10GB。
需要 GPU 版 PyTorch 時，請先依 [pytorch.org](https://pytorch.org) 的說明安裝對應 CUDA 版本的 torch 和 torchvision。
`RADDINO.py` 第一次執行時會從 Hugging Face 下載 `microsoft/rad-dino` 的權重（約 350MB）。

## 資料

資料集不包含在 repo 中，請從 Kaggle 競賽頁下載，整理成下面的結構：

```
cxr-dataset/
├── train_data.csv          # new_filename, normal, bacteria, virus, COVID-19（one-hot）
├── val_data.csv
├── test_data_sample.csv    # 提交檔的列順序以這個檔案為準
├── train_images/
├── val_images/
└── test_images/
```

三支主程式預設讀取 repo **上一層**的 `cxr-dataset/`（也就是 `../cxr-dataset`），放在其他位置時用 `--data_dir` 指定。

## 使用方式（RADDINO.py / ConvNeXtTiny.py / ResNet.py）

以下用 `RADDINO.py` 示範，另外兩支的用法相同。`python RADDINO.py -h` 可以看到全部參數與預設值。

### 訓練

```bash
# 原始 train / val 切分：在 train 上訓練、在 val 上挑最佳 epoch，訓練完自動預測 test
python RADDINO.py --mode train --run_name base

# 5-fold CV（train + val 合併後分層切成 5 折）：報告中的主要評估方式
python RADDINO.py --mode train --run_name base_cv5 --cv_folds 5
```

`--cv_folds 5` 的流程：

1. 把 train 和 val 合併成 4726 張，用 `StratifiedKFold(5, shuffle=True, random_state=0)` 切分。
2. 依序訓練 5 個 fold，每個 fold 取驗證 macro-F1 最高的 epoch。
3. 彙整 5 個 fold 的 out-of-fold（OOF）預測，計算 OOF macro-F1。
4. 5 個 fold 模型對 test 的 softmax 機率取平均，輸出提交檔。
5. COVID-19 門檻（`--covid_thr auto`，預設開啟）：在 OOF 上選門檻 t，p(COVID-19) < t 時改判其他三類中機率最高者；用巢狀 CV 報告加門檻後的 OOF 分數，另外輸出 `test_data_sample_thr.csv`。

**截止後的補充提交（Kaggle 0.88935）就是上面第二行指令產生的 `Output/RADDINO/base_cv5/test_data_sample_thr.csv`。**

### 預設設定

| 項目 | RADDINO.py | ConvNeXtTiny.py / ResNet.py |
| --- | --- | --- |
| 輸入 | 392×392，灰階複製成 3 通道，直接拉伸 | 384×384，同左 |
| 正規化 | RAD-DINO 的 mean 0.5307 / std 0.2583 | ImageNet mean / std |
| 分類頭 | [CLS, patch 平均] 1536 維 → Dropout 0.2 → Linear(4) | Dropout 0.2 → Linear(4) |
| 微調 | 全部參數；layer-wise lr decay 0.75、drop path 0.1 | 全部參數 |
| 最佳化 | AdamW，lr 1e-4，weight decay 0.05（一維參數為 0），1 epoch warmup + cosine，AMP | 同左（無 LLRD） |
| 訓練 | 20 epochs，batch 16，不 early stop（`--patience 0`） | 同左 |
| Loss | cross-entropy + sqrt 反頻率類別權重 + label smoothing 0.1 | 同左 |
| 資料增強 | 預設全部關閉 | 同左 |

### 資料增強參數（只作用於訓練）

| 參數 | 說明 |
| --- | --- |
| `--rotation D` | 隨機旋轉 ±D 度 |
| `--brightness B` | 隨機亮度 ×U(1−B, 1+B) |
| `--contrast C` | 隨機對比 ×U(1−C, 1+C) |
| `--resize_scale S`、`--resize_mode crop/pad` | crop：先拉伸再 RandomResizedCrop，面積 [S, 1]；pad：縮小到 [S, 1] 倍後補黑邊 |
| `--hflip P` | 水平翻轉機率 |
| `--fit letterbox` | 改成等比縮放後補黑邊（預設 `stretch`） |

報告中的增強實驗（表 4–6）用的指令範例：

```bash
# 單項增強（原始 train / val 切分）
python ConvNeXtTiny.py --mode train --run_name rot10 --cv_folds 0 --rotation 10
python RADDINO.py      --mode train --run_name bri03 --cv_folds 0 --brightness 0.3

# ConvNeXt-Tiny 五項增強組合（5-fold）
python ConvNeXtTiny.py --mode train --run_name combo_all_cv5 --cv_folds 5 \
    --rotation 10 --brightness 0.5 --contrast 0.5 --resize_scale 0.9 --hflip 0.1

# RAD-DINO 四項增強組合（5-fold，early stopping patience 10）
python RADDINO.py --mode train --run_name p10_combo_all --cv_folds 5 --patience 10 \
    --rotation 10 --brightness 0.3 --contrast 0.3 --hflip 0.3
```

逐項移除的版本就是在上面的指令中拿掉其中一項。

### 預測

```bash
# 單一權重
python RADDINO.py --mode test --weight Output/RADDINO/base/best_weight.pth

# CV run 資料夾：自動讀取底下所有 fold 的 best_weight.pth，softmax 取平均，並沿用 OOF 選出的 COVID-19 門檻
python RADDINO.py --mode test --weight Output/RADDINO/base_cv5 --run_name base_cv5_test

# 在有標註的 csv 上評估（會輸出 macro-F1 與 confusion matrix）
python RADDINO.py --mode test --weight Output/RADDINO/base/best_weight.pth --test_csv ../cxr-dataset/val_data.csv
```

### 輸出

輸出寫在 `Output/<模型名稱>/<run_name>/`：

| 檔案 | 內容 |
| --- | --- |
| `test_data_sample.csv` | 提交檔（argmax），欄位 `new_filename, normal, bacteria, virus, COVID-19`，列順序與 `test_data_sample.csv` 相同 |
| `test_data_sample_thr.csv` | 套用 COVID-19 門檻後的提交檔 |
| `oof_probs.csv`、`confusion_matrix_oof*.png`、`covid_thr_info.txt` | CV 的 OOF 機率、混淆矩陣、門檻與巢狀 CV 分數 |
| `fold<k>/best_weight.pth` | 各 fold 的最佳權重 |
| `fold<k>/history.csv`、`curves.png` | 每個 epoch 的 loss / accuracy / macro-F1 與訓練曲線 |
| `fold<k>/val_probs.csv`、`test_probs.csv` | 各 fold 對驗證折與 test 的機率 |
| `fold<k>/training_info.txt`、`testing_info.txt` | 完整設定與結果紀錄 |

## 初期 ResNet18 實驗（early_resnet18/）

在找到 RAD-DINO 之前，針對 bacteria / virus 互相誤判做的架構調整。輸入 224×224，ImageNet 預訓練的 ResNet18，AdamW + ReduceLROnPlateau，依驗證 macro-F1 做 early stopping。

```bash
cd early_resnet18

# 單一 ResNet18：train + val 合併後分層抽 5% 當驗證（--split-rate 0.05）
python main.py --mode train --data-dir ../../cxr-dataset

# One-vs-Rest：4 個二元 ResNet18，Bacteria / Virus 分類器加 dropout 0.3、weight decay 1e-3；用官方 val 評估
python mainOvR.py --mode train --data-dir ../../cxr-dataset

# 主分支 + bacteria vs virus 輔助分支（輔助 loss 權重 λ、推論融合權重 w）
python mainSupport.py --mode train --data-dir ../../cxr-dataset \
    --auxiliary-loss-weight 0.5 --auxiliary-inference-weight 1.0

# 預測 test
python main.py --mode predict --data-dir ../../cxr-dataset --weights outputs/best_model.pth
```

`main.py` 和 `mainSupport.py` 的預設值（`--split-rate 0.05`、lr 1e-5、batch 64、不做 ImageNet 正規化）對應報告表 1 中 5% 切分（236 張）的「ResNet18（對照）」與「輔助分支」兩組。表 1 中在官方 709 張 val 上的 ResNet18 結果，是用較早版本的 `main.py` 跑的（直接用官方 train / val、有 ImageNet 正規化），目前的版本無法完全重現。`-h` 可以看到全部參數。

## 注意事項

- **截止前的正式提交（Kaggle 0.88092）** 是用 CV 階段的實驗程式產生的，這份程式沒有放在本 repo；它和 `RADDINO.py` 的模型與超參數相同，差別只在取最後一個 epoch、COVID-19 門檻 0.54。
- GPU 運算有非決定性，重跑的分數可能和報告有小幅差異（同設定重跑約 ±0.003–0.015）。
- 時間參考（RTX 3080）：RAD-DINO 約 65 秒 / epoch，5-fold 約 2 小時；ConvNeXt-Tiny 約 24 秒 / epoch；ResNet-50 約 21 秒 / epoch。
