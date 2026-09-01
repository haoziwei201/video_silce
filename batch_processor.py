import os
import sys
import json
import logging
import asyncio
import argparse
from pathlib import Path
from tqdm.asyncio import tqdm as async_tqdm 

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import (
    VIDEO_POOL_DIR, GLOBAL_CACHE_DIR, LIBRARIES_DIR, 
    PROCESSED_AUDIO_DIR, KEYFRAMES_DIR, TRANSCRIPTS_DIR
)
from src.utils import ensure_in_video_pool, get_file_hash
from src.video_processor import VideoProcessor
from src.speech_to_text import SpeechToText
from src.visual_recognition import VisualRecognition
from src.data_merger import merge_audio_visual_data
from src.rag_engine import VideoKnowledgeBase
from src.data_cleaner import AsyncDataCleaner

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def process_single_video(video_path, lib_name, category="general"):
    """
    处理单个视频：
    1. 计算哈希 & 入池
    2. 检查/生成全局缓存 (ASR + VLM)
    3. 清洗数据
    4. 入库 (ChromaDB)
    """
    print(f"\n{'='*50}")
    print(f"处理视频: {os.path.basename(video_path)}")
    
    # 1. 入池
    video_md5, pool_path = ensure_in_video_pool(video_path, VIDEO_POOL_DIR)
    if not video_md5:
        logger.error(f"无法读取视频: {video_path}")
        return False
        
    print(f"MD5: {video_md5}")
    print(f"Pool Path: {pool_path}")
    
    # 准备缓存目录
    cache_dir = os.path.join(GLOBAL_CACHE_DIR, video_md5)
    os.makedirs(cache_dir, exist_ok=True)
    
    # 2. AI 处理 (ASR + VLM)
    
    # --- ASR (语音) ---
    raw_trans_path = os.path.join(cache_dir, "raw_trans.json")
    transcript = []
    
    if os.path.exists(raw_trans_path):
        print("✓ 命中 ASR 缓存")
        with open(raw_trans_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
    else:
        print("执行 ASR (语音识别)...")
        # 提取音频
        audio_path = os.path.join(PROCESSED_AUDIO_DIR, f"{video_md5}.wav")
        vp = VideoProcessor()
        if not os.path.exists(audio_path):
            vp.extract_audio(pool_path, audio_path)
            
        stt = SpeechToText()
        # 这里 transcribe 已经被我修改为支持缓存，但这里显式处理流程更清晰
        # 也可以直接调用 stt.transcribe(audio_path, video_md5)
        raw_res = stt.transcribe(audio_path, video_md5=video_md5)
        transcript = stt.split_by_punctuation(raw_res) # 切分
        
        # 这里的 raw_trans.json 最好存 split 后的，或者存原始的？
        # stt.transcribe 存的是 normalized 的结果。
        # 我们这里重新存一份 split 后的，方便后续使用
        with open(raw_trans_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False)

    # --- VLM (视觉) ---
    visual_analysis_path = os.path.join(cache_dir, "visual_analysis.json")
    visual_segments = []
    
    if os.path.exists(visual_analysis_path):
        print("✓ 命中 VLM 缓存")
        with open(visual_analysis_path, 'r', encoding='utf-8') as f:
            visual_segments = json.load(f)
    else:
        print("执行 VLM (视觉分析)...")
        # 提取关键帧
        kf_dir = os.path.join(KEYFRAMES_DIR, video_md5)
        vp = VideoProcessor()
        keyframes = vp.extract_keyframes(pool_path, kf_dir, interval=2.0)
        
        # 简单去重 (这里简化处理，实际可以使用 MSE)
        unique_kfs = keyframes # 假设已去重或全部处理
        
        vr = VisualRecognition()
        # 异步分析
        tasks = []
        # 限制并发
        sem = asyncio.Semaphore(10)
        async def analyze_wrapper(kf):
            async with sem:
                desc = await vr.analyze_image_async(kf['path'], auto_save=False)
                return kf['time'], desc
                
        tasks = [analyze_wrapper(kf) for kf in unique_kfs]
        
        # 运行
        results = await async_tqdm.gather(*tasks, desc="VLM 分析进度")
        
        # 保存 Cache (批量)
        await vr.save_cache()
        
        for timestamp, desc in results:
            if desc:
                visual_segments.append({
                    "word": f"[视觉画面: {desc}]", 
                    "text": f"[视觉画面: {desc}]",
                    "start": timestamp,
                    "end": timestamp + 2.0
                })
        
        with open(visual_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(visual_segments, f, ensure_ascii=False)

    # 3. 数据融合 & 清洗 (生成 RAG Ready Data)
    rag_ready_path = os.path.join(cache_dir, "rag_ready.json")
    
    if os.path.exists(rag_ready_path):
        print("✓ 命中 RAG Ready 数据")
        with open(rag_ready_path, 'r', encoding='utf-8') as f:
            rag_data = json.load(f)
    else:
        print("执行数据融合与清洗...")
        # 融合
        merged_data = merge_audio_visual_data(transcript, visual_segments)
        
        # 保存临时融合结果
        merged_path = os.path.join(cache_dir, "merged_raw.json")
        with open(merged_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False)
            
        # 清洗 (生成 visual_summary)
        cleaner = AsyncDataCleaner()
        # 这里的 process_file_async 会读取 input_path 并写入 output_path
        # 我们需要稍微改一下 cleaner 或者直接调用
        await cleaner.process_file_async(merged_path, rag_ready_path, category)
        
        with open(rag_ready_path, 'r', encoding='utf-8') as f:
            rag_data = json.load(f)

    # 4. 入库 (Library)
    print(f"正在存入逻辑库: {lib_name}")
    vkb = VideoKnowledgeBase(lib_name=lib_name)
    
    # 记录库的元数据 (lib_config.json)
    lib_config_path = os.path.join(LIBRARIES_DIR, lib_name, "lib_config.json")
    lib_config = {"videos": []}
    if os.path.exists(lib_config_path):
        with open(lib_config_path, 'r', encoding='utf-8') as f:
            lib_config = json.load(f)
            
    if video_md5 not in lib_config["videos"]:
        lib_config["videos"].append(video_md5)
        with open(lib_config_path, 'w', encoding='utf-8') as f:
            json.dump(lib_config, f, ensure_ascii=False, indent=2)
            
    # Chroma 入库
    # 检查是否已存在 (简单检查：如果库里已经有该视频的数据)
    # Chroma 没有直接的 check，但我们可以 query metadata
    # 这里直接 upsert
    vkb.add_data(rag_data, video_md5)
    
    print(f"✓ 视频 {os.path.basename(video_path)} 处理完成！")
    return True

async def main():
    parser = argparse.ArgumentParser(description="批量视频处理工具")
    parser.add_argument("--input", "-i", default="data/input_videos", help="输入视频文件夹路径")
    parser.add_argument("--lib", "-l", default="default_lib", help="目标逻辑库名称")
    parser.add_argument("--category", "-c", default="general", help="视频分类")
    
    args = parser.parse_args()
    
    input_dir = args.input
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        return
        
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
             if f.lower().endswith(('.mp4', '.mov', '.avi'))]
             
    print(f"找到 {len(files)} 个视频文件")
    
    for file_path in files:
        try:
            await process_single_video(file_path, args.lib, args.category)
        except Exception as e:
            logger.error(f"处理视频 {file_path} 失败: {e}")
            print(f"处理失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
