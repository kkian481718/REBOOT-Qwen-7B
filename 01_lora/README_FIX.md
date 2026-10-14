# LoRA 模型檔案問題修復指南

## 問題描述

載入 LoRA 時出現錯誤：

```
Error while deserializing header: header too large
```

## 原因

目前的 `adapter_model.safetensors` 檔案只有 133 bytes，這是 **Git LFS 指標檔案**，不是實際的模型檔案。實際模型檔案大小約 71 MB，需要透過 Git LFS 下載。

## 解決方案

### 方案 1：從 Git LFS 下載（推薦）

如果這個專案是從 Git 倉庫 clone 的：

1. **安裝 Git LFS**

   ```bash
   # Windows: 下載並安裝
   # https://git-lfs.github.com/

   # 或使用 Chocolatey
   choco install git-lfs

   # 初始化 Git LFS
   git lfs install
   ```

2. **下載 LFS 檔案**

   ```bash
   # 在專案根目錄執行
   git lfs pull

   # 或執行提供的批次檔
   cd 01_lora
   pull_lfs_files.bat
   ```

3. **驗證檔案**
   ```bash
   cd 04_CLI
   python check_lora.py
   ```

### 方案 2：重新訓練模型

如果無法從 Git LFS 下載，需要重新訓練：

1. **準備訓練資料**
   確保訓練資料存在於正確位置：

   - `00_Data/0430_prometheus.jsonl`
   - `00_Data/0430_nemesis.jsonl`

2. **執行訓練腳本**

   ```bash
   cd 00_finetune
   python QWEN_7B_finetune.py
   ```

3. **訓練完成後，檔案會儲存在：**
   - `01_lora/lora_prometheus_v1/`
   - `01_lora/lora_nemesis_v1/`

### 方案 3：暫時不使用 LoRA

在 LoRA 檔案可用之前，可以只使用基礎模型：

```bash
cd 04_CLI

# 使用選單選擇「不使用 LoRA」
python qwen_chat.py

# 或使用命令列參數跳過 LoRA
python qwen_chat.py --no-menu --lora ""
```

## 檢查工具

使用提供的檢查工具驗證 LoRA 檔案：

```bash
cd 04_CLI
python check_lora.py
```

這會顯示：

- 所有 LoRA 目錄的狀態
- 檔案大小是否正常
- 哪些 checkpoint 可用

## 預期的正常檔案大小

| 檔案                        | 大小       |
| --------------------------- | ---------- |
| `adapter_model.safetensors` | ~71 MB     |
| `adapter_config.json`       | ~827 bytes |

如果 `adapter_model.safetensors` 只有 133 bytes，代表是 Git LFS 指標檔案，需要下載實際檔案。

## 常見問題

### Q: 為什麼會有 Git LFS 指標檔案？

A: 大型檔案（如模型權重）通常使用 Git LFS 儲存，避免 Git 倉庫過大。clone 時預設只下載指標檔案，需要額外執行 `git lfs pull` 下載實際檔案。

### Q: 沒有 Git 倉庫怎麼辦？

A: 需要重新訓練模型（方案 2）或從其他來源取得完整的 LoRA 檔案。

### Q: 訓練需要多久？

A: 取決於硬體配置，使用 GPU 訓練約需要數小時。

### Q: 可以使用不同的 checkpoint 嗎？

A: 可以。如果某個 checkpoint 的檔案完整，可以直接使用：

```bash
python qwen_chat.py --lora ../01_lora/lora_nemesis_v1/checkpoint-500
```

## 更新日誌

- 2025-11-15: 加入 LoRA 檔案驗證機制
- 2025-11-15: 加入診斷工具 `check_lora.py`
- 2025-11-15: 加入 Git LFS 下載腳本
