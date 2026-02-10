"""
main.py - 视频智能剪辑工具主程序
"""


import os
import sys
import json
import logging
import glob
import asyncio
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入自定义模块
from config import (
    INPUT_VIDEO_DIR, OUTPUT_VIDEO_DIR, PROCESSED_AUDIO_DIR,
    TRANSCRIPTS_DIR, ANALYSIS_RESULTS_DIR, SLICE_VIDEO_DIR, KEYFRAMES_DIR
)
from src.video_processor import VideoProcessor
from src.speech_to_text import SpeechToText
from src.text_analyzer import TextAnalyzer
from src.visual_recognition import VisualRecognition
from src.data_merger import merge_audio_visual_data
from src.data_cleaner import clean_json_data
from src.rag_engine import VideoKnowledgeBase
import numpy as np
from PIL import Image

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_silce.log', encoding = 'utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_directories():
    """确保所有必要的目录都存在"""
    directories = [
        INPUT_VIDEO_DIR,
        OUTPUT_VIDEO_DIR,
        PROCESSED_AUDIO_DIR,
        TRANSCRIPTS_DIR,
        ANALYSIS_RESULTS_DIR,
        SLICE_VIDEO_DIR,
        KEYFRAMES_DIR
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"确保目录存在: {directory}")
    
    return True

def save_transcript(transcript, video_name):
    """保存转录文本到文件"""
    try:
        transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_name}_transcript.json")
        print(f"DEBUG: 准备保存转录结果到 {transcript_path}")
        
        # 确保转录是JSON可序列化的
        if isinstance(transcript, list):
            # 如果是单词列表，转换为标准格式
            serializable_transcript = []
            for item in transcript:
                if isinstance(item, dict):
                    serializable_transcript.append(item)
                else:
                    # 尝试转换为字典
                    serializable_transcript.append({"word": str(item)})
        else:
            serializable_transcript = str(transcript)
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_transcript, f, ensure_ascii=False, indent=2)
        
        logger.info(f"转录文本已保存到: {transcript_path}")
        print(f"DEBUG: 转录结果保存成功")
        return transcript_path
    except Exception as e:
        logger.error(f"保存转录结果失败: {str(e)}")
        print(f"DEBUG: 保存转录结果失败: {str(e)}")
        return None

def save_analysis_results(segments, video_name, user_instruction):
    """保存分析结果到文件"""
    try:
        # 为每个 segment 添加序号
        for i, seg in enumerate(segments):
            seg['id'] = i + 1
            
        results = {
            "video_name": video_name,
            "user_instruction": user_instruction,
            "segments": segments,
            "total_segments": len(segments),
            "total_duration": sum(seg["end_time"] - seg["start_time"] for seg in segments if "start_time" in seg and "end_time" in seg)
        }
        
        results_path = os.path.join(ANALYSIS_RESULTS_DIR, f"{video_name}_analysis.json")
        print(f"DEBUG: 准备保存分析结果到 {results_path}")
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析结果已保存到: {results_path}")
        print(f"DEBUG: 分析结果保存成功")
        return results_path
    except Exception as e:
        logger.error(f"保存分析结果失败: {str(e)}")
        print(f"DEBUG: 保存分析结果失败: {str(e)}")
        return None

def calculate_image_difference(img1_path, img2_path):
    """计算两张图片的差异 (MSE)"""
    try:
        # Resize to small size for fast comparison
        with Image.open(img1_path) as i1, Image.open(img2_path) as i2:
            i1 = i1.resize((64, 64)).convert('L')
            i2 = i2.resize((64, 64)).convert('L')
            arr1 = np.array(i1)
            arr2 = np.array(i2)
            mse = np.mean((arr1 - arr2) ** 2)
            return mse
    except Exception as e:
        logger.warning(f"图片差异计算失败: {e}")
        return float('inf')

async def analyze_keyframes_async(visual_recognition, keyframes_to_analyze):
    """
    异步分析关键帧列表
    """
    tasks = []
    # 限制并发数为10，避免触发API限流
    sem = asyncio.Semaphore(10)
    
    async def bounded_analyze(kf):
        async with sem:
            # 批量处理时关闭自动保存，避免频繁IO导致串行化
            return await visual_recognition.analyze_image_async(kf['path'], auto_save=False)

    for kf in keyframes_to_analyze:
        tasks.append(bounded_analyze(kf))
    
    results = await asyncio.gather(*tasks)
    
    # 批量处理完成后，统一保存一次缓存
    if hasattr(visual_recognition, 'save_cache'):
        logger.info("批量分析完成，正在保存缓存...")
        await visual_recognition.save_cache()
        
    return results

def select_video_file():
    """
    列出并让用户选择一个视频文件。
    返回视频文件的完整路径。
    """
    print("\n" + "="*50)
    print("请选择要处理的视频文件")
    print("="*50)
    
    video_files = []
    if os.path.exists(INPUT_VIDEO_DIR):
        for file in os.listdir(INPUT_VIDEO_DIR):
            if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_files.append(file)
    
    if not video_files:
        print(f"未在 {INPUT_VIDEO_DIR} 目录中找到视频文件。")
        print("请将视频文件放入该目录后重新运行程序。")
        return None
    
    print("\n可用的视频文件:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {video_file}")
    
    while True:
        try:
            choice = input(f"\n请选择 (1-{len(video_files)}) 或输入文件名: ").strip()
            
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(video_files):
                    video_filename = video_files[index]
                    return os.path.join(INPUT_VIDEO_DIR, video_filename)
                else:
                    print(f"请输入 1-{len(video_files)} 之间的数字。")
            
            elif choice in video_files:
                return os.path.join(INPUT_VIDEO_DIR, choice)
            
            elif os.path.exists(choice):
                video_filename = os.path.basename(choice)
                src_path = choice if os.path.isabs(choice) else os.path.abspath(choice)
                dst_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
                
                if src_path != dst_path:
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    print(f"已将视频文件复制到: {dst_path}")
                return dst_path
            
            else:
                print("输入无效，请重新选择。")
                
        except ValueError:
            print("请输入有效的数字或文件名。")

def data_processing(video_path):
    try:
        logger.info("开始进行数据处理")
        
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            print(f"错误: 视频文件不存在: {video_path}")
            return False
        
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        logger.info(f"处理视频: {video_filename}")
        
        # 2. 初始化处理器
        logger.info("步骤2：初始化处理器")
        video_processor = VideoProcessor()
        speech_to_text = SpeechToText()
        visual_recognition = VisualRecognition()
        
        print(f"\n{'='*60}")
        print(f"准备处理视频: {video_filename}")
        # print(f"用户指令: {user_instruction}")
        print(f"{'='*60}")
        
        # 3. 提取音频
        logger.info("步骤3: 提取音频")
        print("\n正在提取音频...")
        
        audio_filename = f"{video_name}.wav"
        audio_path = os.path.join(PROCESSED_AUDIO_DIR, audio_filename)
        
        success = video_processor.extract_audio(video_path, audio_path)
        if not success:
            logger.error("音频提取失败")
            print("错误: 音频提取失败")
            return False
        
        print(f"✓ 音频已提取: {audio_path}")
        
        # 4. 语音转文字
        logger.info("步骤4: 语音转文字")
        print("\n正在将音频转换为文字...")
        
        first_transcript = speech_to_text.transcribe(audio_path)
        if not first_transcript:
            logger.error("语音转文字失败")
            print("错误: 语音转文字失败")
            return False
        transcript = speech_to_text.split_by_punctuation(first_transcript)

        print(f"✓ 语音转文字完成，共识别 {len(transcript)} 个片段")
        
        # 5.5 视觉内容分析 (关键帧提取 + API调用)
        logger.info("步骤4.5: 视觉内容分析")
        print("\n[Visual] 正在分析视频画面内容 (这可能需要一些时间)...")
        
        # 提取关键帧
        kf_output_dir = os.path.join(KEYFRAMES_DIR, video_name)
        # 阶段1 & 2: 基础提取
        keyframes = video_processor.extract_keyframes(video_path, kf_output_dir, interval=2.0)
        print(f"提取了 {len(keyframes)} 个潜在关键帧")
        
        visual_segments = []
        last_processed_kf_path = None
        MSE_THRESHOLD = 50.0  # 差异阈值，低于此值视为画面未变
        
        unique_keyframes = []
        skipped_count = 0
        
        print("正在进行关键帧去重...")
        for kf in keyframes:
            kf_path = kf['path']
            
            # 阶段2: 去重
            if last_processed_kf_path:
                mse = calculate_image_difference(last_processed_kf_path, kf_path)
                if mse < MSE_THRESHOLD:
                    skipped_count += 1
                    continue
            
            unique_keyframes.append(kf)
            last_processed_kf_path = kf_path

        print(f"去重完成: 共有 {len(unique_keyframes)} 帧待分析, 跳过 {skipped_count} 帧")
        
        # 异步批量分析
        print("开始异步调用视觉模型分析关键帧...")
        
        try:
            # 重新编写 process_images_async 以支持有序结果 + 实时进度条
            async def process_images_async_with_progress(visual_recognition, unique_keyframes):
                from tqdm import tqdm
                
                total_keyframes = len(unique_keyframes)
                descriptions = [None] * total_keyframes
                sem = asyncio.Semaphore(15)
                pbar = tqdm(total=total_keyframes, desc="视觉分析进度", unit="帧")
                
                async def bounded_analyze_wrapper(index, kf):
                    async with sem:
                        try:
                            res = await visual_recognition.analyze_image_async(kf['path'], auto_save=False)
                        except Exception as e:
                            logger.error(f"Error analyzing frame {index}: {e}")
                            res = None
                        finally:
                            pbar.update(1)
                        return index, res

                # 创建所有任务，带索引以便后续排序
                tasks = [bounded_analyze_wrapper(i, kf) for i, kf in enumerate(unique_keyframes)]
                
                for task in asyncio.as_completed(tasks):
                    idx, res = await task
                    descriptions[idx] = res
                    
                    # 每完成 50 个保存一次缓存 (可选优化)
                    if (pbar.n) % 50 == 0 and hasattr(visual_recognition, 'save_cache'):
                         await visual_recognition.save_cache()
                
                pbar.close()
                
                # 最后保存一次缓存
                if hasattr(visual_recognition, 'save_cache'):
                    await visual_recognition.save_cache()
                    
                return descriptions

            # 运行异步处理
            descriptions = asyncio.run(process_images_async_with_progress(visual_recognition, unique_keyframes))
                    
        except Exception as e:
            logger.error(f"异步分析出错: {e}")
            print(f"异步分析出错: {e}")
            # 如果出错，descriptions 可能部分为 None，后续逻辑会处理

        analyzed_count = 0
        for kf, description in zip(unique_keyframes, descriptions):
            timestamp = kf['time']
            if description:
                visual_segments.append({
                    "word": f"[视觉画面: {description}]", 
                    "text": f"[视觉画面: {description}]",
                    "start": timestamp,
                    "end": timestamp + 2.0
                })
                analyzed_count += 1
            else:
                logger.warning(f"Failed to analyze frame at {timestamp}")
                
        print(f"\n✓ 视觉分析完成: 成功分析 {analyzed_count} 帧")
        
        # 整合结果 (使用新的融合逻辑)
        # transcript 是 ASR 结果列表
        # visual_segments 是视觉结果列表
        full_transcript = merge_audio_visual_data(transcript, visual_segments)
        
        print(f"✓ 结果整合完成，共 {len(full_transcript)} 条记录")
        
        # 保存转录结果
        transcript_path = save_transcript(full_transcript, video_name)
        
        print(f"✓ 数据处理完成，中间文件已保存到: {transcript_path}")

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        logger.info("程序被用户中断")
        return False
    
    except Exception as e:
        logger.exception(f"程序运行出错: {str(e)}")
        print(f"\n错误: {str(e)}")
        print("详细信息请查看日志文件: video_silce.log")
        return False

def rag_building():
    json_files = []
    if os.path.exists(TRANSCRIPTS_DIR):
        for file in os.listdir(TRANSCRIPTS_DIR):
            if file.lower().endswith(('.json', 'rag.json')):
                json_files.append(file)
    
    if not json_files:
        print(f"未在 {TRANSCRIPTS_DIR} 目录中找到输入的json文件。")
        print("请手动放入json文件或进行数据准备操作后再重试")
        return False
    
    # 显示可选的文件
    print("\n可选的json文件:")
    for i, json_file in enumerate(json_files, 1):
        print(f"  {i}. {json_file}")
    
    # 选择文件
    while True:
        try:
            choice = input(f"\n请选择要输入的json文件 (1-{len(json_files)}) 或输入文件名: ").strip()
            
            # 如果用户直接输入了数字
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(json_files):
                    json_filename = json_files[index]
                    break
                else:
                    print(f"请输入 1-{len(json_files)} 之间的数字。")
            
            # 如果用户输入了文件名
            elif choice in json_files:
                json_filename = choice
                break
            
            # 如果用户输入了相对路径或绝对路径
            elif os.path.exists(choice):
                json_filename = os.path.basename(choice)
                # 如果文件不在输入目录中，复制到输入目录
                src_path = choice if os.path.isabs(choice) else os.path.abspath(choice)
                dst_path = os.path.join(TRANSCRIPTS_DIR, json_filename)
                
                if src_path != dst_path:
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    print(f"已将视频文件复制到: {dst_path}")
                break
            
            else:
                print("输入无效，请重新选择。")
                
        except ValueError:
            print("请输入有效的数字或文件名。")

    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{json_filename}")

    try:
        logger.info("RAG 数据准备与测试")
        
        # 加载转录数据
        with open(transcript_path, 'r', encoding='utf-8') as f:
            full_transcript = json.load(f)
        
        # 异步分析
        text_analyzer = TextAnalyzer()
        segments = asyncio.run(text_analyzer.analyze_transcript_async(full_transcript))
        
        if not segments:
            logger.error("文本分析失败")
            print("错误: 文本分析失败")
            return False
            
        print(f"✓ 文本分析完成，生成 {len(segments)} 个剪辑片段")
        
        # 保存分析结果
        video_name = os.path.splitext(json_filename)[0].replace("_transcript", "")
        save_analysis_results(segments, video_name, "user_instruction") # user_instruction 暂时为空
        
    except Exception as e:
        logger.exception(f"RAG 构建出错: {str(e)}")
        print(f"\n错误: {str(e)}")
        print("详细信息请查看日志文件: video_silce.log")
        return False

def main():
    """主函数，程序入口"""
    ensure_directories()
    
    while True:
        print("\n" + "="*50)
        print("           视频智能剪辑工具 - 主菜单")
        print("="*50)
        print("1. 数据处理：提取音频、语音转文字、视觉分析")
        print("2. RAG 知识库构建与测试")
        print("3. 退出")
        
        choice = input("\n请选择要执行的操作 (1-3): ").strip()
        
        if choice == '1':
            video_path = select_video_file()
            if video_path:
                data_processing(video_path)
        elif choice == '2':
            rag_building()
        elif choice == '3':
            print("感谢使用，再见！")
            break
        else:
            print("输入无效，请输入 1-3 之间的数字。")

if __name__ == "__main__":
    main()