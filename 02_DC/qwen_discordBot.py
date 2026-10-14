#!/usr/bin/env python3
"""
Discord Bot for Qwen-7B Models
支援兩個角色: Qwen涅墨西斯、Qwen普羅米修斯
包含 LLM 對話 + Parler-TTS 語音生成

使用方式:
    python discord_bot_qwen.py                    # 啟動兩個角色
    python discord_bot_qwen.py --only nemesis     # 只啟動涅墨西斯
    python discord_bot_qwen.py --only prometheus  # 只啟動普羅米修斯
"""

import os
import discord
from discord.ext import commands
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional
import time
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer as TTSTokenizer, set_seed
from translate import Translator
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from bot_config import get_system_prompt, get_voice_description, is_channel_allowed, get_discord_token

# ─────────── 命令列參數 ───────────
parser = argparse.ArgumentParser(description="Qwen Discord Bot 啟動器")
parser.add_argument(
    "--only",
    choices=["nemesis", "prometheus", "both"],
    default="both",
    help="選擇要啟動的角色: nemesis（涅墨西斯）、prometheus（普羅米修斯）或 both（兩者）"
)
args = parser.parse_args()

# ─────────── 設定 ───────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(f"🎯 啟動模式: {args.only}")
logger.info(f"📊 量化設定: 4-bit (NF4) + 雙重量化")

# Discord Bot Token (從 bot_config.py 讀取,也支援環境變數覆蓋)
try:
    DISCORD_TOKEN_NEMESIS = os.getenv("DISCORD_TOKEN_QWEN_NEMESIS") or get_discord_token("qwen_nemesis")
    DISCORD_TOKEN_PROMETHEUS = os.getenv("DISCORD_TOKEN_QWEN_PROMETHEUS") or get_discord_token("qwen_prometheus")
except ValueError as e:
    logger.error(f"Token 設定錯誤: {e}")
    logger.info("請在 bot_config.py 中設定 Discord Token,或使用環境變數")
    raise

# 模型路徑
BASE_MODEL = "Qwen/Qwen-7B"
# 使用 checkpoint-500（最終訓練結果）避免主目錄檔案損壞問題
LORA_NEMESIS = Path(__file__).parent.parent / "01_lora" / "lora_nemesis_v1" / "checkpoint-500"
LORA_PROMETHEUS = Path(__file__).parent.parent / "01_lora" / "lora_prometheus_v1" / "checkpoint-500"

# 音檔儲存目錄
TTS_DIR = Path("temp_audio_qwen")
TTS_DIR.mkdir(exist_ok=True)

# ─────────── Parler-TTS 服務 ───────────
class ParlerTTSService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.translator = None
        self._initialize_model()
        self._initialize_translator()
    
    def _initialize_model(self):
        """初始化 Parler-TTS 模型"""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"TTS 使用設備: {self.device}")
        
        try:
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                "parler-tts/parler-tts-mini-v1"
            ).to(self.device)
            self.tokenizer = TTSTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
            logger.info("Parler-TTS 模型載入完成")
        except Exception as e:
            logger.error(f"Parler-TTS 模型載入失敗: {e}")
            raise
    
    def _initialize_translator(self):
        """初始化翻譯器"""
        try:
            self.translator = Translator(from_lang='zh-tw', to_lang='en')
            logger.info("翻譯器初始化完成")
        except Exception as e:
            logger.error(f"翻譯器初始化失敗: {e}")
            self.translator = None
    
    def translate_to_english(self, text):
        """將繁體中文翻譯成英文"""
        try:
            if self.translator is None:
                return text
            translated = self.translator.translate(text)
            logger.info(f"翻譯: {text[:30]}... -> {translated[:30]}...")
            return translated
        except Exception as e:
            logger.error(f"翻譯失敗: {e}")
            return text
    
    def get_voice_description(self, speaker):
        """根據角色返回語音描述"""
        return get_voice_description(speaker)
    
    def generate_audio(self, text, speaker, output_path):
        """生成音頻文件"""
        try:
            english_text = self.translate_to_english(text)
            description = self.get_voice_description(speaker)
            
            input_ids = self.tokenizer(
                description, return_tensors="pt"
            ).input_ids.to(self.device)
            prompt_input_ids = self.tokenizer(
                english_text, return_tensors="pt"
            ).input_ids.to(self.device)
            
            set_seed(42)
            with torch.no_grad():
                generation = self.model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_input_ids
                )
                audio_arr = generation.cpu().numpy().squeeze()
            
            sf.write(output_path, audio_arr, self.model.config.sampling_rate)
            logger.info(f"音頻生成完成: {output_path}")
            return True
        except Exception as e:
            logger.error(f"生成音頻失敗: {e}")
            return False

# ─────────── LLM 模型管理 ───────────
class QwenModel:
    def __init__(self, persona: str):
        self.persona = persona
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conversation_history: Dict[int, list] = {}  # user_id -> history
        
    def load_model(self):
        """載入基礎模型 + LoRA (使用 4-bit 量化)"""
        logger.info(f"載入 {self.persona} 模型 (4-bit 量化)...")
        
        lora_path = LORA_NEMESIS if self.persona == "nemesis" else LORA_PROMETHEUS
        
        # 設定 4-bit 量化配置（更省 VRAM）
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # Qwen 使用 float16
            bnb_4bit_use_double_quant=True,  # 雙重量化，進一步節省記憶體
            bnb_4bit_quant_type="nf4"  # NF4 量化類型
        )
        
        # 載入基礎模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 載入 LoRA
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model.eval()
        
        logger.info(f"{self.persona} 模型載入完成 (VRAM: ~7-8 GB)")
    
    def get_history(self, user_id: int, max_turns: int = 6) -> list:
        """取得對話歷史"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        history = self.conversation_history[user_id]
        # 保持最近 N 輪對話
        if len(history) > max_turns * 2:
            history = history[-max_turns * 2:]
            self.conversation_history[user_id] = history
        
        return history
    
    def add_to_history(self, user_id: int, role: str, content: str):
        """添加到對話歷史"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        self.conversation_history[user_id].append({"role": role, "content": content})
    
    def clear_history(self, user_id: int):
        """清除對話歷史"""
        if user_id in self.conversation_history:
            self.conversation_history[user_id] = []
    
    def generate_response(self, user_id: int, user_message: str) -> str:
        """生成回應"""
        system_prompt = get_system_prompt(self.persona)
        history = self.get_history(user_id)
        
        # 構建對話 (Qwen 格式)
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            else:
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip()
        
        # 更新歷史
        self.add_to_history(user_id, "user", user_message)
        self.add_to_history(user_id, "assistant", response)
        
        return response

# ─────────── Discord Bot ───────────
# 全域共用 TTS 服務（避免重複載入）
_shared_tts_service = None
_tts_lock = asyncio.Lock()

def create_bot(persona: str, token: str):
    """建立 Discord Bot"""
    
    intents = discord.Intents.default()
    intents.message_content = True
    intents.messages = True
    
    bot_name = f"Qwen{'涅墨西斯' if persona == 'nemesis' else '普羅米修斯'}"
    bot = commands.Bot(
        command_prefix="!",
        intents=intents,
        help_command=None
    )
    
    # 初始化模型
    llm_model = QwenModel(persona)
    model_loaded = False  # 標記模型是否已載入
    
    @bot.event
    async def on_ready():
        global _shared_tts_service
        nonlocal model_loaded
        
        logger.info(f"✅ {bot_name} 已上線! (User: {bot.user})")
        logger.info(f"{bot_name} 開始在背景載入模型...")
        
        # 使用背景任務載入模型，避免阻塞事件循環
        async def load_models():
            nonlocal model_loaded
            global _shared_tts_service
            
            try:
                # 載入 LLM 模型
                logger.info(f"[{bot_name}] 載入 LLM 模型...")
                await asyncio.to_thread(llm_model.load_model)
                model_loaded = True
                logger.info(f"[{bot_name}] LLM 模型載入完成")
                
                # 初始化共用 TTS (只初始化一次)
                async with _tts_lock:
                    if _shared_tts_service is None:
                        logger.info("初始化共用 TTS 服務...")
                        def init_tts():
                            return ParlerTTSService()
                        _shared_tts_service = await asyncio.to_thread(init_tts)
                        logger.info("TTS 服務初始化完成")
                
                logger.info(f"[{bot_name}] 所有模型準備完成")
            except Exception as e:
                logger.error(f"[{bot_name}] 模型載入失敗: {e}", exc_info=True)
        
        # 在背景執行載入任務
        bot.loop.create_task(load_models())
    
    @bot.command(name="chat")
    async def chat(ctx, *, message: str):
        """與機器人對話 (包含語音)"""
        # 檢查頻道權限
        bot_key = f"qwen_{persona}"
        if not is_channel_allowed(ctx.channel.id, bot_key):
            logger.info(f"{bot_name} 忽略非授權頻道 {ctx.channel.id} 的訊息")
            return  # 不在允許的頻道就不回應
        
        # 檢查模型是否已載入
        if not model_loaded:
            await ctx.reply(f"⏳ {bot_name} 模型還在載入中，請稍候...")
            return
        
        async with ctx.typing():
            try:
                # 生成文字回應
                response = await asyncio.to_thread(
                    llm_model.generate_response,
                    ctx.author.id,
                    message
                )
                
                # 生成語音
                audio_filename = f"tts_qwen_{persona}_{ctx.author.id}_{int(time.time()*1000)}.wav"
                audio_path = TTS_DIR / audio_filename
                
                if _shared_tts_service is not None:
                    success = await asyncio.to_thread(
                        _shared_tts_service.generate_audio,
                        response,
                        persona,
                        audio_path
                    )
                else:
                    success = False
                    logger.warning("TTS 服務未初始化")
                
                # 發送回應
                await ctx.reply(f"**{bot_name}**: {response}")
                
                if success and audio_path.exists():
                    await ctx.send(file=discord.File(audio_path))
                
            except Exception as e:
                logger.error(f"處理訊息時出錯: {e}", exc_info=True)
                await ctx.reply(f"❌ 處理失敗: {str(e)}")
    
    @bot.command(name="clear")
    async def clear_history(ctx):
        """清除對話歷史"""
        llm_model.clear_history(ctx.author.id)
        await ctx.reply(f"✅ 已清除 {bot_name} 的對話歷史")
    
    @bot.command(name="help")
    async def help_command(ctx):
        """顯示幫助訊息"""
        help_text = f"""
**{bot_name} 指令說明**

`!chat <訊息>` - 與我對話 (會生成語音)
`!clear` - 清除對話歷史
`!help` - 顯示此幫助訊息

範例: `!chat 你好，請介紹一下自己`
"""
        await ctx.reply(help_text)
    
    return bot, token

# ─────────── 主程式 ───────────
async def main():
    """根據命令列參數運行對應的機器人"""
    
    bots_to_start = []
    
    if args.only in ["nemesis", "both"]:
        logger.info("✅ 準備啟動涅墨西斯")
        nemesis_bot, nemesis_token = create_bot("nemesis", DISCORD_TOKEN_NEMESIS)
        bots_to_start.append((nemesis_bot, nemesis_token, "涅墨西斯"))
    
    if args.only in ["prometheus", "both"]:
        logger.info("✅ 準備啟動普羅米修斯")
        prometheus_bot, prometheus_token = create_bot("prometheus", DISCORD_TOKEN_PROMETHEUS)
        bots_to_start.append((prometheus_bot, prometheus_token, "普羅米修斯"))
    
    if not bots_to_start:
        logger.error("❌ 沒有選擇要啟動的 Bot")
        return
    
    logger.info(f"🚀 即將啟動 {len(bots_to_start)} 個 Bot")
    
    # 同時啟動所有選中的 Bot
    async with asyncio.TaskGroup() as tg:
        for bot, token, name in bots_to_start:
            tg.create_task(bot.start(token))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("機器人已停止")
