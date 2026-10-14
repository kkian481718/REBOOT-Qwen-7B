#!/usr/bin/env python3
"""
LoRA 檔案檢查工具
檢查 LoRA adapter 檔案是否完整有效
"""

import os
import sys

def check_lora_directory(lora_path):
    """檢查 LoRA 目錄"""
    print(f"\n{'='*60}")
    print(f"檢查 LoRA: {lora_path}")
    print(f"{'='*60}")
    
    if not os.path.isdir(lora_path):
        print("❌ 目錄不存在")
        return False
    
    # 檢查必要檔案
    adapter_model = os.path.join(lora_path, "adapter_model.safetensors")
    adapter_config = os.path.join(lora_path, "adapter_config.json")
    
    all_ok = True
    
    # 檢查 adapter_model.safetensors
    if not os.path.exists(adapter_model):
        print("❌ adapter_model.safetensors 不存在")
        all_ok = False
    else:
        size = os.path.getsize(adapter_model)
        size_mb = size / (1024 * 1024)
        if size < 1024 * 1024:  # 小於 1MB
            print(f"❌ adapter_model.safetensors 檔案損壞 ({size} bytes = {size_mb:.2f} MB)")
            print(f"   正常大小應該至少 1 MB 以上")
            all_ok = False
        else:
            print(f"✅ adapter_model.safetensors ({size_mb:.2f} MB)")
    
    # 檢查 adapter_config.json
    if not os.path.exists(adapter_config):
        print("❌ adapter_config.json 不存在")
        all_ok = False
    else:
        size = os.path.getsize(adapter_config)
        print(f"✅ adapter_config.json ({size} bytes)")
    
    # 列出所有 checkpoint
    checkpoints = []
    for item in os.listdir(lora_path):
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(lora_path, item)):
            checkpoints.append(item)
    
    if checkpoints:
        print(f"\n📁 發現 {len(checkpoints)} 個 checkpoint:")
        for ckpt in sorted(checkpoints):
            ckpt_path = os.path.join(lora_path, ckpt)
            ckpt_model = os.path.join(ckpt_path, "adapter_model.safetensors")
            if os.path.exists(ckpt_model):
                size = os.path.getsize(ckpt_model)
                size_mb = size / (1024 * 1024)
                status = "✅" if size >= 1024 * 1024 else "❌"
                print(f"   {status} {ckpt}: {size_mb:.2f} MB")
    
    return all_ok


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    lora_root = os.path.join(project_root, "01_lora")
    
    if not os.path.isdir(lora_root):
        print(f"❌ LoRA 目錄不存在: {lora_root}")
        return
    
    # 檢查所有 LoRA 目錄
    lora_dirs = []
    for item in os.listdir(lora_root):
        item_path = os.path.join(lora_root, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            lora_dirs.append(item_path)
    
    if not lora_dirs:
        print("❌ 沒有找到任何 LoRA 目錄")
        return
    
    print(f"\n發現 {len(lora_dirs)} 個 LoRA 目錄\n")
    
    results = {}
    for lora_dir in sorted(lora_dirs):
        results[lora_dir] = check_lora_directory(lora_dir)
    
    # 總結
    print(f"\n{'='*60}")
    print("總結")
    print(f"{'='*60}")
    
    for lora_dir, is_ok in results.items():
        status = "✅ 正常" if is_ok else "❌ 有問題"
        print(f"{status}: {os.path.basename(lora_dir)}")
    
    # 提供建議
    if not all(results.values()):
        print(f"\n{'='*60}")
        print("💡 解決方案")
        print(f"{'='*60}")
        print("檔案損壞的可能原因：")
        print("  1. 訓練過程中磁碟空間不足")
        print("  2. 訓練過程被中斷")
        print("  3. 檔案傳輸過程中損壞")
        print("\n建議：")
        print("  1. 檢查訓練日誌，確認訓練是否正常完成")
        print("  2. 如果有完整的 checkpoint，可以使用該 checkpoint")
        print("  3. 重新訓練模型")
        print("  4. 暫時使用基礎模型（不載入 LoRA）")


if __name__ == "__main__":
    main()
