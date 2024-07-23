# SCCNet Training Script 使用方法

這個腳本用於訓練 SCCNet 模型，支持三種訓練模式：Subject Dependent (SD)、Leave-One-Subject-Out (LOSO) 和 Fine-tune (FT)。

## 命令行參數

- `--mode` (str, required): 訓練模式，必須是以下之一：
  - `SD`：Subject Dependent
  - `LOSO`：Leave-One-Subject-Out
  - `FT`：Fine-tune
- `--Nc` (int, default=22): 通道數量
- `--Nt` (int, default=438): 時間點數量
- `--Nu` (int, default=16): 卷積核數量
- `--dropout_rate` (float, default=0.5): Dropout 率
- `--batch_size` (int, default=256): 批量大小
- `--num_epochs` (int, default=500): 訓練輪數
- `--lr` (float, default=0.001): 學習率
- `--weight_decay` (float, default=0.0001): 權重衰減

## 使用範例

以下是使用該腳本的幾個範例：

### Subject Dependent (SD) 模式

```bash
python trainer.py --mode SD --Nc 20 --Nt 1 --Nu 22 --dropoutRate 0.7 --batch_size 256 --num_epochs 1500 --lr 0.001 --weight_decay 0.0001 --csv_path result/SD_result.csv
```

### Leave-One-Subject-Out (LOSO) 模式

```bash
python trainer.py --mode LOSO --Nc 15 --Nt 1 --Nu 22 --dropoutRate 0.7 --batch_size 2048 --num_epochs 1500 --lr 0.001 --weight_decay 0.0001 --csv_path result/LOSO_result.csv
```

### Fine-tune (FT) 模式

```bash
python trainer.py --mode FT --Nc 10 --Nt 1 --Nu 22 --dropoutRate 0.9 --batch_size 1024 --num_epochs 1500 --lr 0.001 --weight_decay 0.0001 --csv_path result/FT_result.csv
```

## 完整範例

這是腳本的完整範例，用於 SD 模式訓練：

```bash
python trainer.py --mode SD --Nc 22 --Nt 438 --Nu 16 --dropout_rate 0.5 --batch_size 256 --num_epochs 500 --lr 0.001 --weight_decay 0.0001
```

在執行上述命令後，腳本將創建並訓練 SCCNet 模型，並在測試數據集上進行評估。最佳模型將被保存在 `best_SCCNet_model.pth` 文件中。

### 訓練日誌

在訓練過程中，將打印每一個 epoch 的訓練損失、訓練準確率和測試準確率。如果模型在測試數據集上的準確率超過之前的最佳準確率，則會保存新的最佳模型，並打印出新的最佳準確率。

```plaintext
Epoch 1/500, Loss: 0.1234, Train Accuracy: 0.9876, Test Accuracy: 0.8765
New best model saved with accuracy: 0.8765
Epoch 2/500, Loss: 0.1234, Train Accuracy: 0.9876, Test Accuracy: 0.8765
...
Best model saved with accuracy: 0.8765
```

### utils 使用方法
此腳本用於可視化SCCNet模型在不同訓練模式下的訓練損失和準確率。

`utils.py`腳本讀取存儲在CSV文件中的訓練和測試數據，並生成訓練損失和準確率的對比圖。它支持三種不同的訓練模式：
- **Subject Dependent (SD)**
- **Leave-One-Subject-Out (LOSO)**
- **Fine-tune (FT)**

### 運行腳本

運行腳本的基本命令如下：

```bash
python utils.py --mode <mode>
```

例如，對於LOSO模式，運行以下命令：

```bash
python utils.py --mode LOSO
```

### 生成的圖像

運行腳本後，生成的圖像將保存在`graph/`目錄中。每個模式對應一個圖像文件：
- `graph/SD_graph.png`
- `graph/LOSO_graph.png`
- `graph/FT_graph.png`