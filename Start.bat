@echo off
chcp 65001 >nul
title 视频剪辑工具

echo 正在启动...

:: 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 请先安装 Python
    pause
    exit /b
)

:: 检查并安装依赖
echo 检查依赖...
pip install -q streamlit dashscope chromadb moviepy opencv-python pillow numpy tqdm python-dotenv httpx imageio-ffmpeg requests

:: 创建必要目录
mkdir data\input_videos data\output_videos data\global_pool data\global_cache 2>nul

:: 启动
echo 启动成功！
streamlit run streamlit.py

pause