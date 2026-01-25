"""
配置文件 - 使用魔塔Paraformer和DeepSeek API
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API配置 - 替换为魔塔和DeepSeek
# 注意：魔塔Paraformer通常是本地模型，不需要API密钥
# DeepSeek需要API密钥
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-c5b18a8d440d4e4d870b074d708f77a0")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"  # DeepSeek API地址
LLM_MODEL = "deepseek-chat"  # DeepSeek模型名称

# 魔塔Paraformer模型配置
PARAFORMER_MODEL_PATH = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
PARAFORMER_MODEL_REVISION = "v2.0.4"

# 内存优化配置
ENABLE_MEMORY_OPTIMIZATION = True  # 启用内存优化
MAX_AUDIO_SIZE_MB = 100  # 最大音频文件大小(MB)，超过则压缩
USE_SMALLER_MODEL = True  # 使用更小的语音识别模型
CLEANUP_EARLY = True  # 尽早清理中间文件

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO_DIR = os.path.join(BASE_DIR, "data", "input_videos")
OUTPUT_VIDEO_DIR = os.path.join(BASE_DIR, "data", "output_videos")
PROCESSED_AUDIO_DIR = os.path.join(BASE_DIR, "data", "processed_audio")
TRANSCRIPTS_DIR = os.path.join(BASE_DIR, "data", "transcripts")

# 确保目录存在
for directory in [INPUT_VIDEO_DIR, OUTPUT_VIDEO_DIR, PROCESSED_AUDIO_DIR, TRANSCRIPTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# 剪辑参数
MAX_CLIP_DURATION = 60  # 最大剪辑长度（秒）
TARGET_CLIP_COUNT = 3   # 默认生成的剪辑数量