import os
import sys
import json

# 将项目根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.text_analyzer import TextAnalyzer
from src.video_processor import VideoProcessor
from config import TRANSCRIPTS_DIR, INPUT_VIDEO_DIR

def test_integration():
    """测试text_analyzer与video_processor的集成"""
    print("=== 测试text_analyzer与video_processor集成 ===")
    
    # 1. 加载测试转录数据
    transcript_path = os.path.join(TRANSCRIPTS_DIR, "text_trans.json")
    
    if not os.path.exists(transcript_path):
        print(f"测试数据不存在: {transcript_path}")
        print("请先运行speech_to_text模块生成转录数据")
        return False
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    print(f"成功加载 {len(transcript_data)} 条转录数据")
    
    # 2. 使用TextAnalyzer分析转录数据
    analyzer = TextAnalyzer()
    user_instruction = "找出最精彩的部分"
    
    print("\n=== 分析转录数据 ===")
    analyzed_segments = analyzer.analyze_transcript(transcript_data, user_instruction)
    
    if not analyzed_segments:
        print("分析失败，未返回任何片段")
        return False
    
    print(f"成功分析出 {len(analyzed_segments)} 个精彩片段")
    print("分析结果示例:")
    for i, segment in enumerate(analyzed_segments[:2]):
        print(f"片段 {i+1}: {segment['start_time']}-{segment['end_time']}, 评分: {segment['score']}")
    
    # 3. 检查video_processor是否能正确处理分析结果
    processor = VideoProcessor()
    
    # 测试select_key_clips方法
    print("\n=== 测试片段选择 ===")
    selected_segments = processor.select_key_clips(analyzed_segments, max_duration=120)  # 2分钟
    
    if not selected_segments:
        print("未选择任何片段")
        return False
    
    print(f"成功选择 {len(selected_segments)} 个关键片段")
    print("选择结果示例:")
    for i, segment in enumerate(selected_segments[:2]):
        print(f"片段 {i+1}: {segment['start_time']}-{segment['end_time']}, 评分: {segment['score']}")
    
    # 4. 检查是否有测试视频可用
    test_video_path = os.path.join(INPUT_VIDEO_DIR, "test_video.mp4")
    
    if not os.path.exists(test_video_path):
        print(f"\n测试视频不存在: {test_video_path}")
        print("请添加测试视频到input_videos目录")
        print("集成测试完成，格式对接成功！")
        return True
    
    # 5. 测试视频剪辑功能
    print("\n=== 测试视频剪辑 ===")
    output_path = processor.combine_clips(test_video_path, selected_segments, "test_output.mp4")
    
    if output_path:
        print(f"视频剪辑成功，输出路径: {output_path}")
        return True
    else:
        print("视频剪辑失败")
        return False

if __name__ == "__main__":
    success = test_integration()
    if success:
        print("\n🎉 集成测试成功！")
    else:
        print("\n❌ 集成测试失败！")
