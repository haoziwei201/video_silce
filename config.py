"""
配置文件
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# API配置 - DeepSeek

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"

# API配置 - DashScope (Paraformer & Qwen-VL)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
# 2026-11 实测：qwen-vl-plus / qwen-vl-max 在当前账号返回 "Model not exist"，qwen3-vl-plus 可用。
# 如需换模型，在 .env 里设置 QWEN_VL_MODEL 即可，不要改这里。
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO_DIR = os.path.join(BASE_DIR, "data", "input_videos")
OUTPUT_VIDEO_DIR = os.path.join(BASE_DIR, "data", "output_videos")
PROCESSED_AUDIO_DIR = os.path.join(BASE_DIR, "data", "processed_audio")
SLICE_VIDEO_DIR = os.path.join(BASE_DIR, "data", "slice_video")
ANALYSIS_RESULTS_DIR = os.path.join(BASE_DIR, "data", "analysis_results")
TRANSCRIPTS_DIR = os.path.join(BASE_DIR, "data", "transcripts")
KEYFRAMES_DIR = os.path.join(BASE_DIR, "data", "keyframes")
RAGSCRIPTS_DIR = os.path.join(BASE_DIR, "data", "ragscripts")
VERTICAL_VIDEO_DIR = os.path.join(BASE_DIR, "data", "vertical_videos")

# 新架构路径配置
VIDEO_POOL_DIR = os.path.join(BASE_DIR, "data", "video_pool")
GLOBAL_CACHE_DIR = os.path.join(BASE_DIR, "data", "global_cache")
LIBRARIES_DIR = os.path.join(BASE_DIR, "data", "libraries")
SLICE_CACHE_DIR = os.path.join(BASE_DIR, "data", "slice_cache")  # Added slice cache

# 确保目录存在
for directory in [INPUT_VIDEO_DIR, OUTPUT_VIDEO_DIR, PROCESSED_AUDIO_DIR, SLICE_VIDEO_DIR, ANALYSIS_RESULTS_DIR, TRANSCRIPTS_DIR, KEYFRAMES_DIR, RAGSCRIPTS_DIR, VERTICAL_VIDEO_DIR, VIDEO_POOL_DIR, GLOBAL_CACHE_DIR, LIBRARIES_DIR, SLICE_CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

