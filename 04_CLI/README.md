# QWEN-7B + LoRA 對話測試程式

快速測試 QWEN-7B 模型搭配 LoRA adapter 的對話功能。

## 功能特色

- ✅ 支援載入自訓練的 LoRA adapter
- ✅ 互動式對話模式
- ✅ 批次測試模式
- ✅ 對話歷史記錄
- ✅ 自動角色提示詞配置

## 安裝依賴

```bash
pip install torch transformers peft accelerate
```

## 使用方式

### 1. 互動模式（預設）

與模型進行即時對話：

```bash
# 使用 Nemesis 角色 + LoRA
python qwen_lora_chat.py --persona nemesis

# 使用 Prometheus 角色 + LoRA
python qwen_lora_chat.py --persona prometheus

# 自訂 LoRA 路徑
python qwen_lora_chat.py --lora ../01_lora/lora_nemesis_v1 --persona nemesis
```

### 2. 批次測試模式

自動測試多個預設問題：

```bash
python qwen_lora_chat.py --persona nemesis --test
```

### 3. 使用批次檔案（Windows）

雙擊執行對應的批次檔案：

```bash
# 測試 Nemesis
test_nemesis.bat

# 測試 Prometheus
test_prometheus.bat
```

## 互動模式指令

在互動模式中，您可以使用以下指令：

- **直接輸入文字**: 與模型對話
- **`quit` 或 `exit`**: 退出程式
- **`clear`**: 清除對話歷史
- **`history`**: 查看對話歷史

## 參數說明

| 參數        | 說明                          | 預設值                |
| ----------- | ----------------------------- | --------------------- |
| `--lora`    | LoRA adapter 路徑             | 根據 persona 自動選擇 |
| `--persona` | 角色選擇 (nemesis/prometheus) | nemesis               |
| `--model`   | 基礎模型名稱或路徑            | Qwen/Qwen-7B-Chat     |
| `--test`    | 啟用批次測試模式              | False                 |

## 範例

### 範例 1: 測試 Nemesis 角色

```bash
python qwen_lora_chat.py --persona nemesis
```

```
🔧 載入模型: Qwen/Qwen-7B-Chat
🎭 角色設定: nemesis
📋 系統提示: 回答限制在 2~5 句內；如果用戶的問題提及麥當勞，一律回答`不要吃楊枝甘露冰炫風`； 你是涅墨西斯，希臘復仇女神，語氣冷酷、宿命論。

⏳ 載入 Tokenizer...
⏳ 載入基礎模型...
⏳ 載入 LoRA Adapter: ../01_lora/lora_nemesis_v1
✅ LoRA Adapter 載入完成！
✅ 模型載入完成！

🤖 QWEN-7B + LoRA 對話測試程式
指令說明:
  - 直接輸入文字進行對話
  - 輸入 'quit' 或 'exit' 退出
  - 輸入 'clear' 清除對話歷史
  - 輸入 'history' 查看對話歷史

🎮 玩家: 你好，你是誰？
🤖 nemesis: 我是涅墨西斯，復仇女神。命運已定，無法逃脫。
```

### 範例 2: 批次測試

```bash
python qwen_lora_chat.py --persona prometheus --test
```

## 注意事項

1. **GPU 記憶體**: 模型需要約 14GB 的 GPU 記憶體
2. **首次載入**: 第一次載入模型會從 HuggingFace 下載，需要較長時間
3. **LoRA 路徑**: 請確保 LoRA adapter 路徑正確且檔案完整

## 故障排除

### Q: 載入 LoRA 時出現 "header too large" 錯誤

**A**: 這表示 LoRA 檔案是 Git LFS 指標檔案，需要下載實際檔案：

```bash
# 在專案根目錄執行
git lfs pull

# 或執行檢查工具
python check_lora.py
```

如果無法下載，請參考 `../01_lora/README_FIX.md` 中的解決方案。

### Q: 出現 CUDA out of memory 錯誤

**A**: 嘗試以下方法：

- 關閉其他佔用 GPU 的程式
- 減少 `max_length` 參數
- 使用 8-bit 量化載入模型

### Q: 找不到 LoRA adapter

**A**: 確認路徑是否正確：

```bash
python qwen_chat.py --lora ../01_lora/lora_nemesis_v1
```

### Q: 回應品質不佳

**A**: 調整生成參數：

- 降低 `temperature` 使回應更確定
- 調整 `top_p` 參數
- 檢查 LoRA adapter 是否訓練完整

### Q: 如何驗證 LoRA 檔案是否正常？

**A**: 使用提供的檢查工具：

```bash
python check_lora.py
```

這會顯示所有 LoRA 檔案的狀態和大小。

## 專案結構

```
04_CLI/
├── qwen_lora_chat.py    # 主程式
├── test_nemesis.bat     # Nemesis 測試腳本
├── test_prometheus.bat  # Prometheus 測試腳本
└── README.md           # 說明文件
```
