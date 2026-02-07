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

def get_user_input():
    """获取用户输入的参数"""
    print("\n" + "="*50)
    print("视频智能剪辑工具")
    print("="*50)
    
    # 列出输入目录中的所有视频文件
    video_files = []
    if os.path.exists(INPUT_VIDEO_DIR):
        for file in os.listdir(INPUT_VIDEO_DIR):
            if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_files.append(file)
    
    if not video_files:
        print(f"未在 {INPUT_VIDEO_DIR} 目录中找到视频文件。")
        print("请将视频文件放入该目录后重新运行程序。")
        return None
    
    # 显示可用的视频文件
    print("\n可用的视频文件:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {video_file}")
    
    # 让用户选择视频
    while True:
        try:
            choice = input(f"\n请选择要处理的视频 (1-{len(video_files)}) 或输入文件名: ").strip()
            
            # 如果用户直接输入了数字
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(video_files):
                    video_filename = video_files[index]
                    break
                else:
                    print(f"请输入 1-{len(video_files)} 之间的数字。")
            
            # 如果用户输入了文件名
            elif choice in video_files:
                video_filename = choice
                break
            
            # 如果用户输入了相对路径或绝对路径
            elif os.path.exists(choice):
                video_filename = os.path.basename(choice)
                # 如果文件不在输入目录中，复制到输入目录
                src_path = choice if os.path.isabs(choice) else os.path.abspath(choice)
                dst_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
                
                if src_path != dst_path:
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    print(f"已将视频文件复制到: {dst_path}")
                break
            
            else:
                print("输入无效，请重新选择。")
                
        except ValueError:
            print("请输入有效的数字或文件名。")
    
    return video_filename

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

def data_processing():
    try:
        logger.info("开始进行数据处理")
        logger.info("步骤1：选择处理的视频")
        user_input = get_user_input()
        
        if user_input is None:
            logger.error("无法获取用户输入，程序退出")
            return False
        
        video_filename = user_input
        video_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
        
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            print(f"错误: 视频文件不存在: {video_path}")
            return False
        
        video_name = os.path.splitext(video_filename)[0]
        logger.info(f"处理视频: {video_filename}")
        # logger.info(f"用户指令: {user_instruction}")
        
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
            # 定义异步处理函数
            async def process_images_async(visual_recognition, unique_keyframes):
                from tqdm.asyncio import tqdm
                
                # 分批处理配置
                BATCH_SIZE = 50  # 每批处理50张图片
                total_keyframes = len(unique_keyframes)
                descriptions = [None] * total_keyframes
                
                logger.info(f"共 {total_keyframes} 帧，将分为 {total_keyframes // BATCH_SIZE + 1} 批进行处理")
                
                # 初始化并发信号量，限制并发数为15
                sem = asyncio.Semaphore(15)
                
                # 定义内部任务函数
                async def bounded_analyze(kf):
                    async with sem:
                        # 批量处理时关闭自动保存
                        return await visual_recognition.analyze_image_async(kf['path'], auto_save=False)
                
                # 创建所有任务
                tasks = [bounded_analyze(kf) for kf in unique_keyframes]
                
                # 使用 tqdm 显示进度条
                print(f"开始分析 {total_keyframes} 个关键帧...")
                results = []
                for f in tqdm.as_completed(tasks, total=total_keyframes, desc="视觉分析进度", unit="帧"):
                    result = await f
                    results.append(result)

                # 注意：as_completed 返回顺序是不确定的，我们需要保持原始顺序
                # 因此我们需要重新组织逻辑：不直接 append，而是用 gather 带 tqdm
                
                # 重新实现：为了保持顺序且有进度条，我们可以用 gather 并发，但手动更新进度条
                # 或者更简单：直接对 gather 任务列表使用 tqdm 是不行的，因为 gather 等待所有完成。
                # 更好的方式是：使用 asyncio.as_completed 但这会打乱顺序。
                # 最好的方式：包装一下任务，让任务完成后更新进度条，最后还是用 gather 等待所有以保持顺序。
                
                pass 
                
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
                
                # 分批提交任务以避免创建过多 Task 对象（可选，但为了防止 Task 太多炸内存，还是分批好，或者一次性提交也可以，Python处理几千个Task没问题）
                # 这里我们一次性提交给 gather，利用 Semaphore 控制并发
                
                # 注意：为了让进度条动起来，我们需要 task 完成时回调，或者用 as_completed
                # 但我们需要最终结果是有序的。
                # 方案：使用 as_completed 获取结果，结果中包含 index，然后填入列表。
                
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
        print("\n[RAG] 正在准备知识库数据...")
        
        if transcript_path:
            if json_filename.endswith("_rag.json"):
                rag_ready_path = transcript_path
            else:
                rag_ready_path = transcript_path.replace(".json", "_rag.json")
                # 假设 category 为 general，或者让 user_instruction 决定，这里先用 general
                clean_json_data(transcript_path, rag_ready_path, category_tag="general")
            
            if os.path.exists(rag_ready_path):
                 # 加载清洗后的数据
                 with open(rag_ready_path, 'r', encoding='utf-8') as f:
                     rag_data = json.load(f)
                 logger.info("RAG 数据库加载完成")
                  
                 try:
                     vkb = VideoKnowledgeBase()
                     existing_count = vkb.collection.count()
                     print(f"当前向量库中已有 {existing_count} 条数据。")
                    
                     do_upsert = True
                     if existing_count > 0:
                        user_choice = input("是否跳过入库步骤直接进行检索？(y/n) [y]: ").strip().lower()
                        if user_choice in ['', 'y', 'yes']:
                            do_upsert = False
                            print("⏭️  已跳过入库步骤。")

                     if do_upsert:
                        # 分批处理以避免 Embedding API 限制 (通常为 25)
                        BATCH_SIZE = 20
                        total_items = len(rag_data)
                        print(f"准备入库 {total_items} 条数据，分批处理中 (Batch Size: {BATCH_SIZE})...")

                        for i in range(0, total_items, BATCH_SIZE):
                            batch_data = rag_data[i : i + BATCH_SIZE]
                            
                            batch_ids = [str(item['id']) for item in batch_data]
                            batch_documents = [item['rag_text'] for item in batch_data]
                            batch_metadatas = [{
                                "start": item['start'],
                                "end": item['end'],
                                "type": item['type'],
                                "category": item['category'],
                                "raw_content": item['content']
                            } for item in batch_data]

                            vkb.collection.upsert(
                                ids=batch_ids,
                                documents=batch_documents,
                                metadatas=batch_metadatas
                            )
                            print(f"  - 进度: {min(i + BATCH_SIZE, total_items)}/{total_items} 已处理")

                        print(f"向量库更新完成")
                 except Exception as e:
                     print(f"[RAG] 错误: {e}")
                     logger.error(f"RAG Error: {e}")
                     return False
            else:
                logger.error("RAG 数据清洗失败或加载失败")
                return False
    except ValueError:
            print("RAG构建失败。")
            return False

    while True:
        query = input("\n请输入查询指令 (例如: '找出切肉的画面'): ").strip()
        if query.lower() == 'q':
            break
        
        if not query:
            continue
            
        try:
            print(f"正在检索: '{query}'...")
            results = vkb.search(query, top_k=3)
            
            print("\n检索结果:")
            if results and 'documents' in results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    if i < len(results['metadatas'][0]):
                        meta = results['metadatas'][0][i]
                        print(f"  {i+1}. [{meta['start']:.1f}s - {meta['end']:.1f}s] {doc}")
                    else:
                        print(f"  {i+1}. {doc}")
            else:
                print("  未找到相关结果")
                
        except Exception as e:
            print(f"检索出错: {e}")

def video_editing():
    logger.info("开始执行视频剪辑及文本分析功能")

    video_processor = VideoProcessor()
    text_analyzer = TextAnalyzer()

    try:
        user_instruction = input("请输入剪辑要求：").strip()
        duration_input = input("\n请输入最大时长（秒）: ").strip()
        max_duration = int(duration_input)

        logger.info("步骤1: 文本分析")
        print("\n正在分析文本内容...")

        video_files = []
        if os.path.exists(INPUT_VIDEO_DIR):
            for file in os.listdir(INPUT_VIDEO_DIR):
                if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                    video_files.append(file)
        video_filename = video_files[0]
        video_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
        video_name = os.path.splitext(video_filename)[0]

        transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_name}_transcript.json")
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
        
        segments = text_analyzer.analyze_transcript(transcript, user_instruction)
        if not segments:
            logger.warning("未找到匹配的剪辑片段")
            print("警告: 根据您的指令，未找到匹配的剪辑片段")
            # 可以询问用户是否继续使用默认剪辑
            choice = input("是否尝试剪辑视频的前30秒？(y/n): ").lower()
            if choice == 'y':
                # 使用默认剪辑：前30秒
                segments = [{
                    "start_time": 0.0,
                    "end_time": min(30.0, 300),  # 假设视频至少300秒，或者使用实际视频长度
                    "reason": "默认剪辑：视频开头部分",
                    "score": 5
                }]
            else:
                print("程序退出")
                return
        
        print(f"✓ 文本分析完成，找到 {len(segments)} 个剪辑片段")
        
        # 6.1 选择关键片段
        logger.info("步骤6.1: 选择关键片段")
        print(f"\n[3.5/5] 正在根据评分选择关键片段...")
        
        selected_segments = video_processor.select_key_clips(segments, max_duration)
        
        if not selected_segments:
            logger.warning("未选择到有效的关键片段")
            print("警告: 未选择到有效的关键片段")
            return
        
        print(f"✓ 已选择 {len(selected_segments)} 个关键片段，总时长约 {max_duration} 秒")
        
        # 更新segments为选择后的片段
        segments = selected_segments
        
        # 保存分析结果
        # 确保分析结果目录存在
        if not os.path.exists(ANALYSIS_RESULTS_DIR):
            os.makedirs(ANALYSIS_RESULTS_DIR)
            logger.info(f"创建分析结果目录: {ANALYSIS_RESULTS_DIR}")
        
        # 保存分析结果
        try:
            results_path = save_analysis_results(segments, video_name, user_instruction)
            logger.info(f"分析结果保存成功: {results_path}")
            print(f"✓ 分析结果已保存到: {results_path}")
        except Exception as e:
            logger.error(f"保存分析结果失败: {str(e)}")
            print(f"警告: 保存分析结果失败，但继续处理视频片段")
            results_path = None

        # 7. 剪辑视频片段
        logger.info("步骤7: 剪辑视频片段")
        print("\n[4.5/5] 正在剪辑视频片段...")
        
        clip_paths = []
        for i, segment in enumerate(segments):
            if "start_time" not in segment or "end_time" not in segment:
                logger.warning(f"跳过无效的片段 {i}: {segment}")
                continue
            
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            
            # 确保时间有效
            if end_time <= start_time:
                logger.warning(f"跳过无效时间范围的片段 {i}: {start_time} - {end_time}")
                continue
            
            clip_filename = f"{video_name}_clip_{i+1}.mp4"
            clip_path = os.path.join(SLICE_VIDEO_DIR, clip_filename)
            
            success = video_processor.create_clip(video_path, start_time, end_time, clip_path)
            if success:
                clip_paths.append(clip_path)
                duration = end_time - start_time
                print(f"  ✓ 片段 {i+1}: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}秒)")
                if "reason" in segment:
                    print(f"     理由: {segment['reason']}")
            else:
                logger.warning(f"剪辑片段 {i+1} 失败: {start_time} - {end_time}")
        
        if not clip_paths:
            logger.error("所有视频片段剪辑都失败")
            print("错误: 所有视频片段剪辑都失败")
            return
        
        print(f"✓ 共成功剪辑 {len(clip_paths)} 个片段")
        
        # 9. 合并剪辑片段
        logger.info("步骤9: 合并剪辑片段")
        print("\n[5/5] 正在合并剪辑片段...")
        
        output_filename = f"{video_name}_edited.mp4"
        output_path = os.path.join(OUTPUT_VIDEO_DIR, output_filename)
        
        success = video_processor.combine_clips(clip_paths, output_path)
        if not success:
            logger.error("合并视频片段失败")
            print("错误: 合并视频片段失败")
            return
        
        # 计算总时长
        total_duration = 0
        for segment in segments:
            if "start_time" in segment and "end_time" in segment:
                total_duration += segment["end_time"] - segment["start_time"]
        
        print(f"✓ 视频合并完成!")
        print(f"\n{'='*60}")
        print(f"处理完成!")
        print(f"输出视频: {output_path}")
        print(f"视频时长: {total_duration:.1f} 秒")
        print(f"原始片段: {len(clip_paths)} 个")
        print(f"{'='*60}")
        
        logger.info(f"视频处理完成，输出文件: {output_path}")
    except ValueError:
            print("RAG构建失败。")
            return False
    pass

def convert_to_vertical():
    """横屏转竖屏功能"""
    logger.info("开始执行横屏转竖屏功能")
    
    try:
        # 初始化视频处理器
        video_processor = VideoProcessor()
        
        # 列出输入目录中的所有视频文件
        video_files = []
        if os.path.exists(INPUT_VIDEO_DIR):
            for file in os.listdir(INPUT_VIDEO_DIR):
                if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                    video_files.append(file)
        
        if not video_files:
            print(f"未在 {INPUT_VIDEO_DIR} 目录中找到视频文件。")
            print("请将视频文件放入该目录后重新运行程序。")
            return False
        
        # 显示可用的视频文件
        print("\n可用的视频文件:")
        for i, video_file in enumerate(video_files, 1):
            print(f"  {i}. {video_file}")
        
        # 让用户选择视频
        while True:
            try:
                choice = input(f"\n请选择要转换的视频 (1-{len(video_files)}) 或输入文件名: ").strip()
                
                # 如果用户直接输入了数字
                if choice.isdigit():
                    index = int(choice) - 1
                    if 0 <= index < len(video_files):
                        video_filename = video_files[index]
                        break
                    else:
                        print(f"请输入 1-{len(video_files)} 之间的数字。")
                
                # 如果用户输入了文件名
                elif choice in video_files:
                    video_filename = choice
                    break
                
                else:
                    print("输入无效，请重新选择。")
                    
            except ValueError:
                print("请输入有效的数字或文件名。")
        
        # 询问模糊半径
        blur_radius_input = input("\n请输入模糊半径 (默认25): ").strip()
        blur_radius = int(blur_radius_input) if blur_radius_input.isdigit() else 25
        
        # 处理视频
        video_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
        video_name = os.path.splitext(video_filename)[0]
        output_filename = f"{video_name}_vertical.mp4"
        output_path = os.path.join(OUTPUT_VIDEO_DIR, output_filename)
        
        print(f"\n{'='*60}")
        print(f"准备转换视频: {video_filename}")
        print(f"模糊半径: {blur_radius}")
        print(f"输出文件: {output_filename}")
        print(f"{'='*60}")
        
        # 调用转换方法
        success = video_processor.convert_to_vertical(video_path, output_path, blur_radius)
        
        if success:
            print(f"\n{'='*60}")
            print(f"转换完成!")
            print(f"输出视频: {output_path}")
            print(f"{'='*60}")
            logger.info(f"横屏转竖屏完成，输出文件: {output_path}")
            return True
        else:
            logger.error("横屏转竖屏失败")
            print("错误: 横屏转竖屏失败")
            return False
            
    except Exception as e:
        logger.exception(f"横屏转竖屏出错: {str(e)}")
        print(f"\n错误: {str(e)}")
        print("详细信息请查看日志文件: video_silce.log")
        return False

def main():
    """主函数"""
    logger.info("开始视频智能剪辑流程")
    
    try:
        # 1. 确保所有目录存在
        logger.info(" 检查目录结构")
        ensure_directories()

        print(" 请输入对应数字来执行相关操作")
        print("1. 数据准备，根据输入视频生成音频和transcripts的json文件")
        print("2. 构建RAG知识库，执行前请确保transcripts里有对应json文件")
        print("3. 调用API进行文本分析并进行视频剪辑功能")
        print("4. 横屏转竖屏，将横屏视频转换为竖屏格式")

        while True:
            branch = input("\n 请用数字1-4来选择操作或输入q来退出程序: ").strip()
            if branch.lower() == 'q':
                break

            if branch.isdigit():
                branch_index = int(branch)
                if branch_index == 2 :
                    rag_building()
                elif branch_index == 1:
                    data_processing()
                elif branch_index == 3:
                    video_editing()
                elif branch_index == 4:
                    convert_to_vertical()
                else:
                    print(f"请输入 1-4 之间的数字。")

        return
        
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        logger.info("程序被用户中断")
        return
    
    except Exception as e:
        logger.exception(f"程序运行出错: {str(e)}")
        print(f"\n错误: {str(e)}")
        print("详细信息请查看日志文件: video_silce.log")
        return

if __name__ == "__main__":
    print("\n视频智能剪辑工具")
    
    try:
        main()
    except Exception as e:
        print(f"程序出现未预期的错误: {str(e)}")
        logging.exception("未预期的错误")


