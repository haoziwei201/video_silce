import os
import sys
from moviepy import VideoFileClip, concatenate_videoclips

# 将项目根目录添加到系统路径，以便导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import INPUT_VIDEO_DIR, OUTPUT_VIDEO_DIR
except ImportError:
    # 尝试直接导入（针对不同运行环境的兼容处理）
    try:
        import config
        INPUT_VIDEO_DIR = config.INPUT_VIDEO_DIR
        OUTPUT_VIDEO_DIR = config.OUTPUT_VIDEO_DIR
    except ImportError as e:
        print(f"Warning: Import failed: {e}")
        # 设置默认路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        INPUT_VIDEO_DIR = os.path.join(base_dir, "data", "input_videos")
        OUTPUT_VIDEO_DIR = os.path.join(base_dir, "data", "output_videos")

class VideoProcessor:
    """视频处理器，负责根据分析结果剪辑视频"""
    
    def __init__(self):
        """初始化视频处理器"""
        pass

    def extract_audio(self, video_path, output_audio_path):
        """
        从视频中提取音频
        参数Args:
            video_path: 输入视频文件路径
            output_audio_path: 输出音频文件路径
        返回Returns:
            bool: 是否成功
        """
        try:
            print(f"正在提取音频: {video_path} -> {output_audio_path}")
            # 确保目录存在
            os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
            
            with VideoFileClip(video_path) as video:
                audio = video.audio
                if audio is None:
                    print("Error: Video has no audio track.")
                    return False
                audio.write_audiofile(output_audio_path, logger=None)
            return True
        except Exception as e:
            print(f"提取音频失败: {e}")
            return False

    def create_clip(self, video_path, start_time, end_time, output_path):
        """
        剪辑视频片段
        参数Args:
            video_path: 输入视频路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            output_path: 输出路径
        返回Returns:
            bool: 是否成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with VideoFileClip(video_path) as video:
                # 边界检查
                if start_time < 0: start_time = 0
                if end_time > video.duration: end_time = video.duration
                if start_time >= end_time:
                    print(f"Invalid clip duration: {start_time}-{end_time}")
                    return False
                
                # MoviePy 2.0+ 兼容性处理
                if hasattr(video, 'subclipped'):
                     new_clip = video.subclipped(start_time, end_time)
                else:
                     new_clip = video.subclip(start_time, end_time)

                new_clip.write_videofile(
                    output_path, 
                    codec="libx264", 
                    audio_codec="aac", 
                    temp_audiofile='temp-audio.m4a', 
                    remove_temp=True,
                    logger=None
                )
            return True
        except Exception as e:
            print(f"剪辑片段失败: {e}")
            return False
    
    def select_key_clips(self, analyzed_segments, max_duration=300):
        """
        根据评分选择关键片段
        参数Args:
            analyzed_segments: 分析后的片段列表，每个元素包含 start_time, end_time, score, reason
            max_duration: 最大视频时长（秒），默认为5分钟
        返回Returns:
            选择的关键片段列表
        """
        # 按评分降序排序
        sorted_segments = sorted(analyzed_segments, key=lambda x: x.get('score', 0), reverse=True)
        
        selected_clips = []
        total_duration = 0
        
        for segment in sorted_segments:
            segment_duration = segment['end_time'] - segment['start_time']
            # 确保片段时长合理，且总时长不超过限制
            if segment_duration > 0 and total_duration + segment_duration <= max_duration:
                selected_clips.append(segment)
                total_duration += segment_duration
        
        # 按时间顺序排序，确保视频连贯性
        selected_clips.sort(key=lambda x: x['start_time'])
        
        return selected_clips
    
    def combine_clips(self, clip_paths, output_filename="output_video.mp4"):
        """
        组合片段并生成最终视频
        参数Args:
            clip_paths: 视频片段路径列表 (注意：这里为了兼容 main.py，第一个参数改为路径列表)
            output_filename: 输出视频文件名或完整路径
        返回Returns:
            bool: 是否成功
        """
        try:
            clips = []
            # 适配：如果传入的是路径列表（main.py 的调用方式）
            if isinstance(clip_paths, list) and all(isinstance(p, str) for p in clip_paths):
                 for path in clip_paths:
                     try:
                         clips.append(VideoFileClip(path))
                     except Exception as e:
                         print(f"Error loading clip {path}: {e}")
            else:
                 # 之前的逻辑是传入 video_path 和 segments，这里为了兼容 main.py 做了调整
                 # 如果你需要保留之前的逻辑，可以加参数判断，但 main.py 传的是 clip_paths
                 print("Error: combine_clips expects a list of file paths.")
                 return False

            if not clips:
                print("No valid clips to combine.")
                return False
            
            # 组合片段
            final_clip = concatenate_videoclips(clips)
            
            # 生成输出路径
            # 如果 output_filename 已经是绝对路径，直接使用；否则拼接到 OUTPUT_VIDEO_DIR
            if os.path.isabs(output_filename):
                output_path = output_filename
            else:
                output_path = os.path.join(OUTPUT_VIDEO_DIR, output_filename)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 导出视频
            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=30,
                preset="medium",
                threads=4,
                logger=None
            )
            
            # 关闭资源
            for clip in clips:
                clip.close()
            final_clip.close()
            
            print(f"Video successfully generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return False
    
    def process_video(self, video_path, analyzed_segments, max_duration=300, output_filename="output_video.mp4"):
        """
        完整的视频处理流程 (保留此方法用于独立测试，但 main.py 不调用它)
        """
        # 选择关键片段
        selected_segments = self.select_key_clips(analyzed_segments, max_duration)
        
        if not selected_segments:
            print("No segments selected for processing.")
            return None
            
        # 注意：这里逻辑需要调整，因为 combine_clips 现在接收文件路径列表
        # 如果要保留 process_video，需要先调用 create_clip 生成临时文件，再 combine
        # 简单起见，这里仅打印提示
        print("process_video is deprecated in this version. Please use main.py workflow.")
        return None

# 测试代码
if __name__ == "__main__":
    pass
