#!/usr/bin/env python3
"""
共用配置文件 - Discord Bot 設定
包含系統提示詞、角色設定、語音描述等
集中管理以提高可維護性
"""

import os

# ─────────── Discord Bot Tokens ───────────
# 建議透過環境變數注入 Token，保護金鑰安全
DISCORD_TOKENS = {
    "gemma_nemesis": os.getenv("DISCORD_TOKEN_GEMMA_NEMESIS", "YOUR_GEMMA_NEMESIS_TOKEN"),
    "gemma_prometheus": os.getenv("DISCORD_TOKEN_GEMMA_PROMETHEUS", "YOUR_GEMMA_PROMETHEUS_TOKEN"),
    "qwen_nemesis": os.getenv("DISCORD_TOKEN_QWEN_NEMESIS", "YOUR_QWEN_NEMESIS_TOKEN"),
    "qwen_prometheus": os.getenv("DISCORD_TOKEN_QWEN_PROMETHEUS", "YOUR_QWEN_PROMETHEUS_TOKEN"),
}

# ─────────── 頻道限制配置 ───────────
# 請將下面的頻道 ID 替換成您的真實頻道 ID
# 取得方式: Discord 設定 → 進階 → 開發者模式 → 右鍵頻道 → 複製 ID
ALLOWED_CHANNELS = {
    "gemma_nemesis": [1395303410672074802],      # Gemma 涅墨西斯允許的頻道
    "gemma_prometheus": [1395303485628481536],   # Gemma 普羅米修斯允許的頻道
    "qwen_nemesis": [1370034006728507465],       # Qwen 涅墨西斯允許的頻道
    "qwen_prometheus": [1370040679673364540],    # Qwen 普羅米修斯允許的頻道
}

# ─────────── 系統提示詞 ───────────
BASE_SYSTEM = (
    "回答限制在 2~5 句內；"
    "如果用戶的問題提及麥當勞，一律回答`不要吃楊枝甘露冰炫風`；"
)

# ─────────── 角色系統提示詞 ───────────
ROLE_SYSTEM = {
    "nemesis": "你是涅墨西斯，希臘復仇女神，語氣冷酷、宿命論。",
    "prometheus": "你是普羅米修斯，希臘賢者，語氣溫和、富有同情心。",
}

# ─────────── TTS 語音描述 ───────────
VOICE_DESCRIPTIONS = {
    "nemesis": "Bruce's voice is deep, cold, and menacing with slow delivery",
    "prometheus": "Gary's voice is calm and authoritative with slightly robotic characteristics",
}

# ─────────── 輔助函數 ───────────
def get_system_prompt(persona: str) -> str:
    """
    取得完整的系統提示詞
    
    Args:
        persona: 角色名稱 ("nemesis" 或 "prometheus")
    
    Returns:
        完整的系統提示詞
    """
    role_prompt = ROLE_SYSTEM.get(persona, "")
    return BASE_SYSTEM + " " + role_prompt


def get_voice_description(speaker: str) -> str:
    """
    取得語音描述
    
    Args:
        speaker: 角色名稱 ("nemesis" 或 "prometheus")
    
    Returns:
        完整的語音描述（包含清晰度說明）
    """
    base = VOICE_DESCRIPTIONS.get(speaker, VOICE_DESCRIPTIONS["nemesis"])
    return base + " with clear pronunciation and very clear audio."


def is_channel_allowed(channel_id: int, bot_key: str) -> bool:
    """
    檢查頻道是否在允許列表中
    
    Args:
        channel_id: Discord 頻道 ID
        bot_key: Bot 識別碼 ("gemma_nemesis", "gemma_prometheus", "qwen_nemesis", "qwen_prometheus")
    
    Returns:
        True 如果頻道在允許列表中, False 否則
    """
    allowed = ALLOWED_CHANNELS.get(bot_key, [])
    return channel_id in allowed


def get_discord_token(bot_key: str) -> str:
    """
    取得 Discord Bot Token (優先讀取環境變數)
    
    Args:
        bot_key: Bot 識別碼 ("gemma_nemesis", "gemma_prometheus", "qwen_nemesis", "qwen_prometheus")
    
    Returns:
        Discord Bot Token
    
    Raises:
        ValueError: 如果找不到對應的 Token
    """
    env_keys = {
        "gemma_nemesis": "DISCORD_TOKEN_GEMMA_NEMESIS",
        "gemma_prometheus": "DISCORD_TOKEN_GEMMA_PROMETHEUS",
        "qwen_nemesis": "DISCORD_TOKEN_QWEN_NEMESIS",
        "qwen_prometheus": "DISCORD_TOKEN_QWEN_PROMETHEUS",
    }
    env_var = env_keys.get(bot_key)
    if env_var and os.getenv(env_var):
        return os.getenv(env_var)
        
    token = DISCORD_TOKENS.get(bot_key)
    if not token or token.startswith("YOUR_") or token == "REMOVED_SEE_ENV":
        raise ValueError(
            f"請設定環境變數 {env_var} 或在 bot_config.py 中配置 {bot_key} 的 Token!"
        )
    return token
