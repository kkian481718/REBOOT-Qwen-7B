#!/usr/bin/env python3
"""
QWEN-7B + LoRA HTTP 對話伺服器
提供 /chat_tts API 端點，支援文字對話與 TTS 語音生成
"""

import os
import sys
import uuid
import logging
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

import torch
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForCausalLM, MarianMTModel, MarianTokenizer
from peft import PeftModel
import pyttsx3

# 添加父目錄到路徑以便導入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot_config import get_system_prompt

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask 應用
app = Flask(__name__)

# 音檔儲存路徑
AUDIO_DIR = Path(__file__).parent.parent / "temp_audio_http"
AUDIO_DIR.mkdir(exist_ok=True)


class QwenChatService:
    """Qwen + LoRA 對話服務"""
    
    def __init__(
        self,
        base_model_name="Qwen/Qwen-7B-Chat",
        lora_path=None,
        persona="nemesis",
        max_new_tokens=256,
    ):
        self.persona = persona
        self.system_prompt = get_system_prompt(persona)
        self.max_new_tokens = max_new_tokens
        
        # 玩家對話歷史 {player_id: [conversation_history]}
        self.player_histories: Dict[str, list] = {}
        
        logger.info(f"🔧 載入模型: {base_model_name}")
        logger.info(f"🎭 角色設定: {persona}")
        
        # 載入 tokenizer
        logger.info("⏳ 載入 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = '<|endoftext|>'
        self.tokenizer.padding_side = "right"
        
        # 載入基礎模型
        logger.info("⏳ 載入基礎模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 載入 LoRA adapter（如果有提供）
        if lora_path and os.path.exists(lora_path):
            logger.info(f"⏳ 載入 LoRA Adapter: {lora_path}")
            self.model = PeftModel.from_pretrained(
                self.base_model,
                lora_path,
                device_map="auto"
            )
            logger.info("✅ LoRA Adapter 載入完成！")
        else:
            logger.warning("⚠️ 未載入 LoRA Adapter，使用基礎模型")
            self.model = self.base_model
        
        self.model.eval()
        logger.info("✅ 模型載入完成！")
    
    def chat(
        self, 
        user_input: str, 
        player_id: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        與模型對話
        
        Args:
            user_input: 使用者輸入
            player_id: 玩家 ID
            max_new_tokens: 回應部分的最大生成長度
            temperature: 溫度參數
            top_p: Top-p 採樣參數
        
        Returns:
            模型回應
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        # 角色名稱
        role_name = "涅墨西斯" if self.persona == "nemesis" else self.persona
        
        # 格式化輸入
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
        if player_id not in self.player_histories:
            self.player_histories[player_id] = []
        
        self.player_histories[player_id].append({
            "user": user_input,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response


class TTSService:
    """文字轉語音服務"""
    
    def __init__(self):
        # 初始化離線翻譯 (中文 -> 英文) 使用 Helsinki-NLP Opus-MT
        self.translator_model = None
        self.translator_tokenizer = None
        self.translator_installed = False
        
        try:
            # 使用 Helsinki-NLP 的 Opus-MT 中文到英文翻譯模型
            model_name = "Helsinki-NLP/opus-mt-zh-en"
            logger.info(f"⏳ 載入翻譯模型: {model_name}")
            
            self.translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.translator_model = MarianMTModel.from_pretrained(model_name)
            
            # 將模型移至 GPU (如果可用)
            if torch.cuda.is_available():
                self.translator_model = self.translator_model.to('cuda')
                logger.info("✅ 翻譯模型已移至 GPU")
            else:
                logger.info("✅ 翻譯模型使用 CPU")
            
            self.translator_model.eval()
            self.translator_installed = True
            logger.info("✅ 離線翻譯模型已就緒 (中文 -> 英文)")
                
        except Exception as e:
            logger.error(f"❌ 翻譯器初始化失敗: {e}")
            logger.warning("⚠️ 翻譯功能不可用，將使用原文")
        
        # 找到適合的男聲 voice ID
        self.voice_id = None
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            for voice in voices:
                voice_name_lower = voice.name.lower()
                if ('english' in voice_name_lower or 'EN' in voice.id) and \
                   ('male' in voice_name_lower or 'david' in voice_name_lower or 'mark' in voice_name_lower or 'zira' not in voice_name_lower):
                    self.voice_id = voice.id
                    logger.info(f"✅ 設定男聲: {voice.name}")
                    break
            # 清理初始化的引擎
            del engine
        except Exception as e:
            logger.warning(f"TTS 引擎初始化警告: {e}")
    
    def translate_to_english(self, text: str) -> str:
        """將中文翻譯成英文 (離線) 使用 Helsinki-NLP Opus-MT"""
        if not self.translator_installed:
            logger.warning("翻譯器未就緒，返回原文")
            return text
        
        try:
            # 使用 MarianMT 進行翻譯
            # Tokenize 輸入文字
            inputs = self.translator_tokenizer(
                text, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # 將輸入移至模型所在的設備
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # 生成翻譯
            with torch.no_grad():
                translated = self.translator_model.generate(**inputs)
            
            # 解碼翻譯結果
            translated_text = self.translator_tokenizer.decode(
                translated[0], 
                skip_special_tokens=True
            )
            
            logger.info(f"🌐 翻譯: {text} -> {translated_text}")
            return translated_text
                
        except Exception as e:
            logger.error(f"❌ 翻譯失敗: {e}")
            # 如果翻譯失敗，返回原文
            return text
    
    def generate_audio(self, text: str, output_filename: str) -> str:
        """
        生成語音檔案
        
        Args:
            text: 要轉換的文字
            output_filename: 輸出檔名
        
        Returns:
            音檔完整路徑
        """
        output_path = AUDIO_DIR / output_filename
        
        # 每次請求都創建新的 TTS 引擎實例
        engine = None
        try:
            engine = pyttsx3.init()
            
            # 設定參數
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            if self.voice_id:
                engine.setProperty('voice', self.voice_id)
            
            # 生成音檔
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
            logger.info(f"✅ 音檔生成成功: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ 音檔生成失敗: {e}", exc_info=True)
            raise
        finally:
            # 確保清理引擎資源
            if engine is not None:
                try:
                    engine.stop()
                    del engine
                except:
                    pass


# 全域服務實例
chat_service: Optional[QwenChatService] = None
tts_service = TTSService()


@app.route('/chat_tts', methods=['POST'])
def chat_tts():
    """
    處理對話與 TTS 請求
    
    請求格式:
    {
        "message": "你好",
        "persona": "nemesis",
        "player_id": "player_123"
    }
    
    回應格式:
    {
        "response": "中文回應",
        "audio_url": "/audio/xxxx.mp3"
    }
    """
    try:
        # 解析請求
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "無效的請求格式"}), 400
        
        message = data.get('message')
        persona = data.get('persona')
        player_id = data.get('player_id')
        
        # 驗證參數
        if not message:
            return jsonify({"error": "缺少 message 參數"}), 400
        
        if not player_id:
            return jsonify({"error": "缺少 player_id 參數"}), 400
        
        if persona != 'nemesis':
            return jsonify({"error": "persona 只支援 'nemesis'"}), 400
        
        logger.info(f"📨 收到請求 - Player: {player_id}, Message: {message}")
        
        # 生成對話回應
        response = chat_service.chat(message, player_id)
        logger.info(f"💬 生成回應: {response}")
        
        # 翻譯成英文
        english_text = tts_service.translate_to_english(response)
        logger.info(f"🌐 翻譯結果: {english_text}")
        
        # 生成音檔
        audio_filename = f"{player_id}_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}.wav"
        tts_service.generate_audio(english_text, audio_filename)
        
        # 構建音檔 URL
        audio_url = f"/audio/{audio_filename}"
        
        return jsonify({
            "response": response,
            "audio_url": audio_url
        }), 200
        
    except Exception as e:
        logger.error(f"❌ 處理請求時發生錯誤: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/audio/<filename>')
def serve_audio(filename):
    """提供音檔下載"""
    return send_from_directory(AUDIO_DIR, filename)


def main():
    """啟動 HTTP 伺服器"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QWEN-7B + LoRA HTTP 對話伺服器")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen-7B-Chat",
        help="基礎模型名稱或路徑"
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="LoRA adapter 路徑"
    )
    parser.add_argument(
        "--persona",
        type=str,
        default="nemesis",
        help="角色名稱"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="最大生成長度"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="伺服器主機位址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="伺服器埠號"
    )
    
    args = parser.parse_args()
    
    # 設定預設 LoRA 路徑
    if args.lora is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.lora = os.path.join(project_root, "01_lora", "lora_nemesis_v1")
    
    # 初始化對話服務
    global chat_service
    try:
        logger.info("🚀 啟動 HTTP 伺服器...")
        chat_service = QwenChatService(
            base_model_name=args.model,
            lora_path=args.lora,
            persona=args.persona,
            max_new_tokens=args.max_new_tokens,
        )
        
        logger.info("="*60)
        logger.info("🎉 伺服器準備就緒！")
        logger.info(f"📍 位址: http://{args.host}:{args.port}")
        logger.info(f"🎭 角色: {args.persona}")
        logger.info(f"📁 音檔目錄: {AUDIO_DIR}")
        logger.info("="*60)
        logger.info("\n可用端點:")
        logger.info(f"  POST   /chat_tts              - 對話與 TTS")
        logger.info(f"  GET    /audio/<filename>      - 音檔下載")
        logger.info("="*60 + "\n")
        
        # 啟動 Flask 伺服器
        app.run(host=args.host, port=args.port, debug=False)
        
    except Exception as e:
        logger.error(f"❌ 伺服器啟動失敗: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
