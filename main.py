"""
main.py - 视频智能剪辑工具主程序
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入自定义模块
from config import (
    INPUT_VIDEO_DIR, OUTPUT_VIDEO_DIR, PROCESSED_AUDIO_DIR,
    TRANSCRIPTS_DIR, ANALYSIS_RESULTS_DIR, SLICE_VIDEO_DIR
)
from src.video_processor import VideoProcessor
from src.speech_to_text import SpeechToText
from src.text_analyzer import TextAnalyzer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_silce.log'),
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
        SLICE_VIDEO_DIR
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
        return None, None
    
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
    
    # 获取用户指令
    print("\n" + "-"*50)
    print("请输入剪辑指令（例如）：")
    print("  - '找出最精彩的部分'")
    print("  - '剪出所有讲人工智能的片段'")
    print("  - '找出笑声最多的地方'")
    print("  - '提取产品功能介绍的部分'")
    print("-"*50)
    
    user_instruction = input("\n请输入你的剪辑指令: ").strip()
    
    if not user_instruction:
        user_instruction = "找出视频中最精彩的部分"
        print(f"使用默认指令: {user_instruction}")
    
    return video_filename, user_instruction

# def save_transcript(transcript, video_name):
#     """保存转录文本到文件"""
#     transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_name}_transcript.json")
    
#     # 确保转录是JSON可序列化的
#     if isinstance(transcript, list):
#         # 如果是单词列表，转换为标准格式
#         serializable_transcript = []
#         for item in transcript:
#             if isinstance(item, dict):
#                 serializable_transcript.append(item)
#             else:
#                 # 尝试转换为字典
#                 serializable_transcript.append({"word": str(item)})
#     else:
#         serializable_transcript = str(transcript)
    
#     with open(transcript_path, 'w', encoding='utf-8') as f:
#         json.dump(serializable_transcript, f, ensure_ascii=False, indent=2)
    
#     logger.info(f"转录文本已保存到: {transcript_path}")
#     return transcript_path

# def save_analysis_results(segments, video_name, user_instruction):
#     """保存分析结果到文件"""
#     results = {
#         "video_name": video_name,
#         "user_instruction": user_instruction,
#         "segments": segments,
#         "total_segments": len(segments),
#         "total_duration": sum(seg["end_time"] - seg["start_time"] for seg in segments if "start_time" in seg and "end_time" in seg)
#     }
    
#     results_path = os.path.join(ANALYSIS_RESULTS_DIR, f"{video_name}_analysis.json")
    
#     with open(results_path, 'w', encoding='utf-8') as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)
    
#     logger.info(f"分析结果已保存到: {results_path}")
#     return results_path

def main():
    """主函数"""
    logger.info("开始视频智能剪辑流程")
    
    try:
        # 1. 确保所有目录存在
        logger.info("步骤1: 检查目录结构")
        ensure_directories()
        
        # 2. 获取用户输入
        logger.info("步骤2: 获取用户输入")
        user_input = get_user_input()
        
        if user_input is None:
            logger.error("无法获取用户输入，程序退出")
            return
        
        video_filename, user_instruction = user_input
        video_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
        
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            print(f"错误: 视频文件不存在: {video_path}")
            return
        
        video_name = os.path.splitext(video_filename)[0]
        logger.info(f"处理视频: {video_filename}")
        logger.info(f"用户指令: {user_instruction}")
        
        # 3. 初始化处理器
        logger.info("步骤3: 初始化处理器")
        video_processor = VideoProcessor()
        speech_to_text = SpeechToText()
        text_analyzer = TextAnalyzer()
        
        print(f"\n{'='*60}")
        print(f"开始处理视频: {video_filename}")
        print(f"用户指令: {user_instruction}")
        print(f"{'='*60}")
        
        # 4. 提取音频
        logger.info("步骤4: 提取音频")
        print("\n[1/5] 正在提取音频...")
        
        audio_filename = f"{video_name}.wav"
        audio_path = os.path.join(PROCESSED_AUDIO_DIR, audio_filename)
        
        success = video_processor.extract_audio(video_path, audio_path)
        if not success:
            logger.error("音频提取失败")
            print("错误: 音频提取失败")
            return
        
        print(f"✓ 音频已提取: {audio_path}")
        
        # 5. 语音转文字
        logger.info("步骤5: 语音转文字")
        print("\n[2/5] 正在将音频转换为文字...")
        
        transcript = speech_to_text.transcribe(audio_path)
        if not transcript:
            logger.error("语音转文字失败")
            print("错误: 语音转文字失败")
            return
        
        print(f"✓ 语音转文字完成，共识别 {len(transcript)} 个片段")
        
        # 保存转录结果
        # transcript_path = save_transcript(transcript, video_name)
        
        # 6. 文本分析
        logger.info("步骤6: 文本分析")
        print("\n[3/5] 正在分析文本内容...")
        
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
        
        # 保存分析结果
        # results_path = save_analysis_results(segments, video_name, user_instruction)
        
        # 7. 剪辑视频片段
        logger.info("步骤7: 剪辑视频片段")
        print("\n[4/5] 正在剪辑视频片段...")
        
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
        
        # 8. 合并剪辑片段
        logger.info("步骤8: 合并剪辑片段")
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
