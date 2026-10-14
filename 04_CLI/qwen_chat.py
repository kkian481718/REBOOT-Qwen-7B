#!/usr/bin/env python3
"""
QWEN-7B + LoRA 對話測試程式
快速測試模型與 LoRA adapter 的對話功能
"""

import os
import sys
from typing import Any, List, Sequence, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 添加父目錄到路徑以便導入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot_config import get_system_prompt

MenuOption = Tuple[str, Any]
CUSTOM_SENTINEL = "__custom__"


def _numeric_select(prompt: str, options: Sequence[MenuOption], default_index: int = 0) -> MenuOption:
    if not options:
        raise ValueError("選單沒有任何可用選項")

    default_index = min(len(options) - 1, max(0, default_index))

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return options[default_index]

    while True:
        print(f"\n{prompt}")
        for idx, (label, _) in enumerate(options, start=1):
            marker = " (預設)" if idx - 1 == default_index else ""
            print(f"  {idx}. {label}{marker}")

        choice = input(f"請輸入選項編號 [預設 {default_index + 1}]: ").strip()
        if not choice:
            selection = options[default_index]
            print(f"➡️ 選擇: {selection[0]}")
            return selection

        if not choice.isdigit():
            print("⚠️ 請輸入數字編號。")
            continue

        idx = int(choice) - 1
        if 0 <= idx < len(options):
            selection = options[idx]
            print(f"➡️ 選擇: {selection[0]}")
            return selection

        print("⚠️ 超出範圍，請重新輸入。")


def _prompt_custom_value(message: str, caster, validator=None):
    while True:
        user_input = input(message).strip()
        if not user_input:
            print("⚠️ 請輸入有效內容。")
            continue
        try:
            value = caster(user_input)
        except ValueError as exc:
            print(f"⚠️ 輸入格式錯誤: {exc}")
            continue
        if validator is not None:
            try:
                validator(value)
            except ValueError as exc:
                print(f"⚠️ {exc}")
                continue
        return value


def _select_with_custom(prompt: str, options: Sequence[MenuOption], default_index: int, custom_prompt: str, caster, validator=None) -> Tuple[str, Any]:
    label, value = _numeric_select(prompt, options, default_index)
    if value != CUSTOM_SENTINEL:
        return label, value
    custom_value = _prompt_custom_value(custom_prompt, caster, validator)
    return "自訂", custom_value


def _discover_lora_options(project_root: str) -> List[MenuOption]:
    lora_root = os.path.join(project_root, "01_lora")
    options: List[MenuOption] = []
    if not os.path.isdir(lora_root):
        return options
    for entry in sorted(os.listdir(lora_root)):
        full_path = os.path.join(lora_root, entry)
        if os.path.isdir(full_path):
            rel_path = os.path.relpath(full_path, project_root)
            options.append((f"{entry} ({rel_path})", full_path))
    return options


def _default_lora_for_persona(persona: str, project_root: str) -> str:
    candidates = {
        "nemesis": os.path.join(project_root, "01_lora", "lora_nemesis_v1"),
        "prometheus": os.path.join(project_root, "01_lora", "lora_prometheus_v1"),
    }
    return candidates.get(persona, "")


def _validate_positive_int(value: int) -> None:
    if value <= 0:
        raise ValueError("必須輸入大於 0 的整數")


def _validate_directory_exists(path: str) -> None:
    if not os.path.isdir(path):
        raise ValueError("找不到指定的資料夾或無法存取")


def _deduplicate(sequence: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in sequence:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _collect_model_options(current_model: str) -> List[MenuOption]:
    candidates = _deduplicate([
        current_model,
        "Qwen/Qwen-7B-Chat",
        "Qwen/Qwen-7B",
    ])
    options: List[MenuOption] = [(model, model) for model in candidates]
    options.append(("自訂模型路徑或名稱", CUSTOM_SENTINEL))
    return options


def _confirm_startup_menu(args, project_root: str):
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return args

    try:
        persona_options: List[MenuOption] = [
            ("涅墨西斯 (nemesis)", "nemesis"),
            ("普羅米修斯 (prometheus)", "prometheus"),
        ]
        persona_default = next(
            (idx for idx, opt in enumerate(persona_options) if opt[1] == args.persona),
            0,
        )
        persona_label, persona_value = _numeric_select("請選擇角色 Persona", persona_options, persona_default)

        current_lora = args.lora or _default_lora_for_persona(persona_value, project_root)
        lora_options: List[MenuOption] = [("不使用 LoRA (僅基礎模型)", None)]
        lora_options.extend(_discover_lora_options(project_root))
        lora_options.append(("自訂 LoRA 路徑", CUSTOM_SENTINEL))
        lora_default = next(
            (idx for idx, opt in enumerate(lora_options) if opt[1] == current_lora),
            0,
        )
        lora_label, lora_value = _select_with_custom(
            "請選擇 LoRA Adapter",
            lora_options,
            lora_default,
            "請輸入 LoRA 資料夾路徑: ",
            lambda value: os.path.abspath(os.path.expanduser(value)),
            _validate_directory_exists,
        )

        model_options = _collect_model_options(args.model)
        model_default = next(
            (idx for idx, opt in enumerate(model_options) if opt[1] == args.model),
            0,
        )
        model_label, model_value = _select_with_custom(
            "請選擇基礎模型",
            model_options,
            model_default,
            "請輸入模型名稱或本地路徑: ",
            lambda value: value,
        )

        mode_options: List[MenuOption] = [
            ("互動對話模式", False),
            ("批次測試模式", True),
        ]
        mode_default = 1 if args.test else 0
        mode_label, mode_value = _numeric_select("請選擇啟動模式", mode_options, mode_default)

        token_options: List[MenuOption] = [
            ("128 tokens", 128),
            ("256 tokens", 256),
            ("512 tokens", 512),
            ("768 tokens", 768),
            ("自訂 token 數", CUSTOM_SENTINEL),
        ]
        token_default = next(
            (idx for idx, opt in enumerate(token_options) if opt[1] == args.max_new_tokens),
            1,
        )
        token_label, token_value = _select_with_custom(
            "請選擇 max_new_tokens",
            token_options,
            token_default,
            "請輸入欲生成的 token 數: ",
            int,
            _validate_positive_int,
        )

        args.persona = persona_value
        args.lora = lora_value
        args.model = model_value
        args.test = bool(mode_value)
        args.max_new_tokens = token_value

        print("\n🚀 啟動設定確認：")
        print(f"  角色: {persona_label}")
        print(f"  LoRA: {lora_label}{' (未啟用)' if lora_value is None else ''}")
        print(f"  模型: {model_value}")
        print(f"  模式: {mode_label}")
        print(f"  max_new_tokens: {token_value}")
    except KeyboardInterrupt:
        print("\n⚠️ 已取消互動選單，改用命令列參數。")
    except Exception as exc:
        print(f"\n⚠️ 啟動選單發生錯誤：{exc}，改用命令列參數。")

    return args

class QwenLoraChat:
    def __init__(
        self,
        base_model_name="Qwen/Qwen-7B-Chat",
        lora_path=None,
        persona="nemesis",
        max_new_tokens=256,
    ):
        """
        初始化 QWEN-7B + LoRA 對話系統
        
        Args:
            base_model_name: 基礎模型名稱或路徑
            lora_path: LoRA adapter 路徑
            persona: 角色名稱 ("nemesis" 或 "prometheus")
            max_new_tokens: 回應部分的最大生成長度
        """
        self.persona = persona
        self.system_prompt = get_system_prompt(persona)
        self.max_new_tokens = max_new_tokens
        
        print(f"🔧 載入模型: {base_model_name}")
        print(f"🎭 角色設定: {persona}")
        print(f"📋 系統提示: {self.system_prompt}\n")
        
        # 載入 tokenizer
        print("⏳ 載入 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = '<|endoftext|>'
        self.tokenizer.padding_side = "right"
        
        # 載入基礎模型
        print("⏳ 載入基礎模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 載入 LoRA adapter（如果有提供）
        if lora_path and os.path.exists(lora_path):
            print(f"⏳ 載入 LoRA Adapter: {lora_path}")
            self.model = PeftModel.from_pretrained(
                self.base_model,
                lora_path,
                device_map="auto"
            )
            print("✅ LoRA Adapter 載入完成！")
        else:
            print("⚠️ 未載入 LoRA Adapter，使用基礎模型")
            self.model = self.base_model
        
        self.model.eval()
        print("✅ 模型載入完成！\n")
        
        # 對話歷史
        self.conversation_history = []
    
    def chat(self, user_input, max_new_tokens=None, temperature=0.7, top_p=0.9):
        """
        與模型對話
        
        Args:
            user_input: 使用者輸入
            max_new_tokens: 回應部分的最大生成長度
            temperature: 溫度參數（控制隨機性）
            top_p: Top-p 採樣參數
        
        Returns:
            模型回應
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        # 構建對話格式
        if self.persona == "nemesis":
            role_name = "涅墨西斯"
        elif self.persona == "prometheus":
            role_name = "普羅米修斯"
        else:
            role_name = self.persona
        
        # 格式化輸入（與訓練時的格式一致）
        prompt = f"{self.system_prompt}\n角色：{role_name}\n玩家：{user_input}\n答案："
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # 生成回應
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                repetition_penalty=1.1
            )
        
        # 解碼回應
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取答案部分
        if "答案：" in full_response:
            response = full_response.split("答案：")[-1].strip()
        else:
            response = full_response.strip()
        
        # 儲存對話歷史
        self.conversation_history.append({
            "user": user_input,
            "assistant": response
        })
        
        return response
    
    def clear_history(self):
        """清除對話歷史"""
        self.conversation_history = []
        print("🗑️ 對話歷史已清除")
    
    def show_history(self):
        """顯示對話歷史"""
        if not self.conversation_history:
            print("📭 目前沒有對話歷史")
            return
        
        print("\n" + "="*60)
        print("📜 對話歷史")
        print("="*60)
        for i, conv in enumerate(self.conversation_history, 1):
            print(f"\n[{i}] 玩家: {conv['user']}")
            print(f"    助手: {conv['assistant']}")
        print("="*60 + "\n")


def interactive_mode(chat_bot):
    """互動模式"""
    print("="*60)
    print("🤖 QWEN-7B + LoRA 對話測試程式")
    print("="*60)
    print("指令說明:")
    print("  - 直接輸入文字進行對話")
    print("  - 輸入 'quit' 或 'exit' 退出")
    print("  - 輸入 'clear' 清除對話歷史")
    print("  - 輸入 'history' 查看對話歷史")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("🎮 玩家: ").strip()
            
            if not user_input:
                continue
            
            # 處理特殊指令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再見！")
                break
            
            if user_input.lower() == 'clear':
                chat_bot.clear_history()
                continue
            
            if user_input.lower() == 'history':
                chat_bot.show_history()
                continue
            
            # 生成回應
            print("⏳ 思考中...", end="\r")
            response = chat_bot.chat(user_input)
            print(f"🤖 {chat_bot.persona}: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 再見！")
            break
        except Exception as e:
            print(f"\n❌ 錯誤: {e}\n")


def batch_test_mode(chat_bot, test_prompts):
    """批次測試模式"""
    print("\n" + "="*60)
    print("🧪 批次測試模式")
    print("="*60 + "\n")
    
    results = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"[{i}/{len(test_prompts)}] 測試: {prompt}")
        response = chat_bot.chat(prompt)
        print(f"回應: {response}\n")
        results.append({
            "prompt": prompt,
            "response": response
        })
    
    print("="*60)
    print("✅ 測試完成！")
    print("="*60)
    
    return results


def main():
    # 解析命令列參數
    import argparse
    
    parser = argparse.ArgumentParser(description="QWEN-7B + LoRA 對話測試程式")
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="LoRA adapter 路徑 (例: ../01_lora/lora_nemesis_v1)"
    )
    parser.add_argument(
        "--persona",
        type=str,
        default="nemesis",
        choices=["nemesis", "prometheus"],
        help="角色選擇 (nemesis 或 prometheus)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen-7B-Chat",
        help="基礎模型名稱或路徑"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="批次測試模式"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="回應部分的最大生成長度"
    )
    parser.add_argument(
        "--no-menu",
        action="store_true",
        help="跳過互動選單，直接使用命令列參數"
    )
    
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if not args.no_menu:
        args = _confirm_startup_menu(args, project_root)
    
    # 如果沒有指定 LoRA 路徑，使用預設路徑
    if args.lora is None:
        if args.persona == "nemesis":
            args.lora = os.path.join(project_root, "01_lora", "lora_nemesis_v1")
        else:
            args.lora = os.path.join(project_root, "01_lora", "lora_prometheus_v1")
    
    # 初始化對話系統
    try:
        chat_bot = QwenLoraChat(
            base_model_name=args.model,
            lora_path=args.lora,
            persona=args.persona,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        return
    
    # 選擇模式
    if args.test:
        # 批次測試模式
        test_prompts = [
            "你好，你是誰？",
            "你對人類有什麼看法？",
            "晚餐要吃什麼？",
            "麥當勞的楊枝甘露冰炫風好吃嗎？",
            "1+1等於多少？",
        ]
        batch_test_mode(chat_bot, test_prompts)
    else:
        # 互動模式
        interactive_mode(chat_bot)


if __name__ == "__main__":
    main()
