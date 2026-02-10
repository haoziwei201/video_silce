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
QWEN_VL_MODEL = "qwen-vl-plus"

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

# 确保目录存在
for directory in [INPUT_VIDEO_DIR, OUTPUT_VIDEO_DIR, PROCESSED_AUDIO_DIR, SLICE_VIDEO_DIR, ANALYSIS_RESULTS_DIR, TRANSCRIPTS_DIR, KEYFRAMES_DIR]:
    os.makedirs(directory, exist_ok=True)

