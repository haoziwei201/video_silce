import os
import sys
import subprocess
import numpy as np
import cv2
import json
from moviepy import VideoFileClip, concatenate_videoclips, CompositeVideoClip
import imageio_ffmpeg

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
                # 强制转换为单声道 (ac=1) 和 16000Hz (或者8000Hz，取决于模型要求)
                # 用户要求使用 8k 模型，建议将采样率设为 8000，但通常 16k 也可以（模型会下采样）
                # 为了安全起见和减少数据量，这里我们配合模型设置为 8000，并强制单声道
                audio.write_audiofile(output_audio_path, logger=None, fps=8000, ffmpeg_params=["-ac", "1"])
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
        pass

    def extract_keyframes(self, video_path, output_dir, interval=2.0):
        """
        [已弃用] 建议使用 extract_smart_keyframes
        每隔一定时间提取关键帧
        """
        return self.extract_smart_keyframes(video_path, output_dir)

    def extract_smart_keyframes(self, video_path, output_dir, max_gap=30):
        """
        智能提取关键帧：场景检测 + PPT检测 + 运动检测 + 兜底覆盖
        Args:
            video_path: 视频路径
            output_dir: 输出目录
            max_gap: 兜底策略的最大时间间隔（秒）
        Returns:
            list: 提取的关键帧文件路径列表 [{"time": float, "path": str}]
        """
        keyframes = []
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"开始智能关键帧提取: {video_path}")
            
            # 1. 场景检测
            print("  - [阶段1] 正在进行场景检测...")
            scenes = self._detect_scenes_opencv(video_path)
            print(f"    检测到 {len(scenes)} 个场景")
            
            extracted_times = set()
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Could not open video for reading.")
                return []
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # 处理场景帧
            for scene in scenes:
                scene_start = scene['start_time']
                scene_end = scene['end_time']
                scene_duration = scene_end - scene_start
                
                # 中间截一张
                mid_time = scene_start + scene_duration / 2
                self._add_keyframe(cap, mid_time, output_dir, keyframes, extracted_times)
                
                # 长场景额外截取
                if scene_duration > 15:
                    self._add_keyframe(cap, scene_start + scene_duration/3, output_dir, keyframes, extracted_times)
                    self._add_keyframe(cap, scene_start + 2*scene_duration/3, output_dir, keyframes, extracted_times)

            # 2 & 3. 检测 PPT/文字 和 运动检测 (采样检测)
            print("  - [阶段2&3] 正在检测文字区域(OCR预处理)和运动变化...")
            # 为了效率，每0.5秒采样一帧
            sample_interval = 0.5 
            curr_time = 0.0
            
            prev_frame_gray = None
            last_motion_time = -100 # 上一次因运动提取的时间
            
            while curr_time < duration:
                # 跳过已经提取的时间点附近，但仍需读取帧以更新 prev_frame
                is_extracted = False
                for t in extracted_times:
                    if abs(t - curr_time) < 0.2: # 稍微放宽一点
                        is_extracted = True
                        break
                
                cap.set(cv2.CAP_PROP_POS_MSEC, curr_time * 1000)
                ret, frame = cap.read()
                if not ret:
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 如果当前时间点已经被提取过，只更新prev_frame并继续
                if is_extracted:
                    prev_frame_gray = gray
                    curr_time += sample_interval
                    continue

                # A. PPT/文字检测
                if self._detect_bright_rectangle(frame):
                    self._add_keyframe(cap, curr_time, output_dir, keyframes, extracted_times, prefix="ppt_")
                    last_motion_time = curr_time # PPT也算是一种视觉变化点
                
                # B. 运动检测
                elif prev_frame_gray is not None:
                    # 只有当距离上次运动提取超过2秒时才再次提取，避免连续运动产生大量帧
                    if curr_time - last_motion_time > 2.0:
                        if self._detect_motion_simple(prev_frame_gray, gray):
                            self._add_keyframe(cap, curr_time, output_dir, keyframes, extracted_times, prefix="motion_")
                            last_motion_time = curr_time

                prev_frame_gray = gray
                curr_time += sample_interval

            # 4. 兜底覆盖
            print("  - [兜底] 检查覆盖率...")
            sorted_times = sorted(list(extracted_times))
            if not sorted_times:
                self._add_keyframe(cap, 0.0, output_dir, keyframes, extracted_times)
                sorted_times = [0.0]
            
            # 检查开头
            if sorted_times[0] > max_gap:
                 self._add_keyframe(cap, 0.0, output_dir, keyframes, extracted_times)
            
            # 检查中间空隙
            sorted_times = sorted(list(extracted_times))
            for i in range(len(sorted_times) - 1):
                curr = sorted_times[i]
                next_t = sorted_times[i+1]
                if next_t - curr > max_gap:
                    fill_time = curr + (next_t - curr) / 2
                    self._add_keyframe(cap, fill_time, output_dir, keyframes, extracted_times, prefix="fill_")
            
            # 检查结尾
            if duration - sorted_times[-1] > max_gap:
                self._add_keyframe(cap, duration - 1.0, output_dir, keyframes, extracted_times)

            cap.release()
            
            # 按时间排序
            keyframes.sort(key=lambda x: x['time'])
            return keyframes

        except Exception as e:
            print(f"智能提取关键帧失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _add_keyframe(self, cap, time_sec, output_dir, keyframes_list, extracted_times_set, prefix="frame_"):
        """辅助函数：保存关键帧"""
        # 简单去重
        for t in extracted_times_set:
            if abs(t - time_sec) < 0.2:
                return

        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ret, frame = cap.read()
        if ret:
            frame_name = f"{prefix}{int(time_sec*1000):06d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            keyframes_list.append({"time": time_sec, "path": frame_path})
            extracted_times_set.add(time_sec)

    def _detect_scenes_opencv(self, video_path, threshold=0.7):
        """场景检测：基于HSV直方图"""
        scenes = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        prev_hist = None
        scene_start_frame = 0
        curr_frame = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if curr_frame % 5 != 0: # 降采样
                curr_frame += 1
                continue
                
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            
            if prev_hist is not None:
                score = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if score < threshold:
                    scenes.append({
                        "start_time": scene_start_frame / fps,
                        "end_time": curr_frame / fps
                    })
                    scene_start_frame = curr_frame
            
            prev_hist = hist
            curr_frame += 1
            
        # 最后一个场景
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if scene_start_frame < total_frames:
            scenes.append({
                "start_time": scene_start_frame / fps,
                "end_time": total_frames / fps
            })
            
        cap.release()
        return scenes

    def _detect_bright_rectangle(self, frame):
        """PPT检测：检测亮色矩形"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            height, width = frame.shape[:2]
            frame_area = width * height
            
            for cnt in contours:
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) == 4:
                    area = cv2.contourArea(cnt)
                    if frame_area * 0.2 < area < frame_area * 0.95:
                        return True
            return False
        except Exception:
            return False

    def _detect_motion_simple(self, prev_gray, curr_gray, threshold=30):
        """运动检测：帧间差分"""
        try:
            diff = cv2.absdiff(prev_gray, curr_gray)
            # 统计变化超过阈值的像素比例
            changed_ratio = np.sum(diff > threshold) / diff.size
            return changed_ratio > 0.05  # 5%的像素发生变化视为有运动
        except Exception:
            return False

    def select_key_clips(self, analyzed_segments, max_duration=300):
        """
        根据评分选择关键片段
        """
        sorted_segments = sorted(analyzed_segments, key=lambda x: x.get('score', 0), reverse=True)
        selected_clips = []
        total_duration = 0
        for segment in sorted_segments:
            segment_duration = segment['end_time'] - segment['start_time']
            if segment_duration > 0 and total_duration + segment_duration <= max_duration:
                selected_clips.append(segment)
                total_duration += segment_duration
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
            # 使用 method="compose" 可以避免因不同片段的参数微小差异导致的问题
            final_clip = concatenate_videoclips(clips, method="compose")
            
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

    def convert_to_vertical(self, video_path, output_path=None, method="solid", background_color=(0, 0, 0)):
        """
        高性能横屏转竖屏（完全基于FFmpeg原生滤镜）
        """
        try:
            if output_path is None:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(os.path.dirname(video_path), f"{video_name}_vertical.mp4")

            # 1. 获取视频原始尺寸
            # 使用 MoviePy 读取尺寸，避免直接调用 ffprobe
            try:
                clip = VideoFileClip(video_path)
                w, h = clip.size
                clip.close()
            except Exception as e:
                print(f"Error reading video dimensions with MoviePy: {e}")
                # Fallback to ffprobe if needed, but likely if MoviePy fails, ffprobe will too or it's not a video
                return False

            # 计算 9:16 目标尺寸
            if w/h > 9/16: # 横屏
                target_h = h if h % 2 == 0 else h - 1
                target_w = int(target_h * 9 / 16)
                if target_w % 2 != 0: target_w -= 1
            else:
                print("视频已经是竖屏或比例接近，无需转换")
                # 如果文件路径不同，复制一份
                if output_path != video_path:
                    import shutil
                    shutil.copy2(video_path, output_path)
                return True

            print(f"🚀 启动 FFmpeg 高速转换 | 模式: {method} | 目标: {target_w}x{target_h}")

            # 2. 根据不同模式构建不同的 FFmpeg 滤镜字符串
            if method == "solid":
                # 纯色背景：先缩放以适应目标尺寸（保持比例），然后填充
                # 将RGB元组转换为hex颜色
                color_hex = "0x{:02x}{:02x}{:02x}".format(*background_color)
                # 关键修正：必须先 scale 缩小视频，否则 pad 会报错（因为输入尺寸大于目标尺寸）
                filter_str = (
                    f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                    f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:{color_hex}"
                )
            
            elif method == "blur":
                # 模糊背景：
                # [0:v] 分为两路：
                # 路1(bg): 缩放并裁剪填充整个画布 -> 高斯模糊
                # 路2(fg): 缩放以适应画布宽度
                # 最后将 fg 覆盖在 bg 中心
                filter_str = (
                    f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=20:10[bg];"
                    f"[0:v]scale={target_w}:-1[fg];"
                    f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
                )
            
            elif method == "static":
                # 静态背景（取第一帧）：
                filter_str = (
                    f"[0:v]start_number=0,scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},trim=end_frame=1,loop=-1:1[bg];"
                    f"[0:v]scale={target_w}:-1[fg];"
                    f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
                )
            else:
                print(f"Unknown method: {method}, using solid black")
                filter_str = f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"

            # 3. 执行 FFmpeg 命令
            # 增加硬件加速参数（如果有Nvidia显卡可以换成 h264_nvenc）
            # 获取 imageio 提供的 ffmpeg 路径
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            cmd = [
                ffmpeg_exe, "-y",
                "-hide_banner",        # 隐藏版权信息
                "-i", video_path,
                "-vf", filter_str,
                "-c:v", "libx264",
                "-preset", "veryfast", # 速度优先
                "-crf", "23",          # 画质平衡
                "-c:a", "copy",        # 音频不重编，秒完成
                output_path
            ]

            subprocess.run(cmd, check=True)
            
            # 验证并返回
            self._force_file_sync(output_path)
            if self._verify_video_file_ready(output_path):
                print(f"✓ 转换成功: {output_path}")
                return True
            else:
                print(f"❌ 转换可能失败，文件未就绪: {output_path}")
                return False

        except Exception as e:
            print(f"❌ 转换失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _force_file_sync(self, filepath):
        """强制同步文件到磁盘"""
        import os
        import sys
        
        try:
            # 方法1: Python文件刷新
            with open(filepath, 'ab') as f:
                f.flush()
                os.fsync(f.fileno())
            
            # 方法2: 操作系统级同步
            if sys.platform == 'win32':
                # Windows特定处理
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.CreateFileW(filepath, 0x40000000, 0, None, 3, 0, None)
                if handle != -1:
                    kernel32.FlushFileBuffers(handle)
                    kernel32.CloseHandle(handle)
            elif hasattr(os, 'sync'):
                # Unix/Linux系统
                os.sync()
                
        except Exception as e:
            print(f"文件同步警告: {e}")

    def _verify_video_file_ready(self, filepath, timeout=10):
        """
        验证视频文件是否完全可读
        """
        import time
        import os
        
        start_time = time.time()
        last_size = 0
        stable_count = 0
        
        while time.time() - start_time < timeout:
            try:
                # 检查文件是否存在
                if not os.path.exists(filepath):
                    time.sleep(0.5)
                    continue
                
                # 检查文件大小是否稳定
                current_size = os.path.getsize(filepath)
                
                if current_size == last_size and current_size > 1024:  # 大于1KB
                    stable_count += 1
                else:
                    stable_count = 0
                    last_size = current_size
                
                # 如果文件大小稳定3次检查
                if stable_count >= 3:
                    # 尝试读取文件头确认视频格式
                    with open(filepath, 'rb') as f:
                        header = f.read(100)
                        # 检查是否是有效的视频文件（MP4文件以"ftyp"开头）
                        if header.startswith(b'\x00\x00\x00\x1cftyp'):
                            return True
                
                time.sleep(0.5)
                
            except (IOError, OSError, PermissionError) as e:
                time.sleep(1)
        
        # 超时后，尝试最后的简单检查
        try:
            if os.path.exists(filepath) and os.path.getsize(filepath) > 1024 * 1024:  # 大于1MB
                return True
        except:
            pass
        
        return False
    
    def add_subtitles(self, video_path, transcript_path, output_path, font_path=None):
        """
        为视频添加字幕
        参数Args:
            video_path: 输入视频路径
            transcript_path: 转录文件路径（JSON格式）
            output_path: 输出视频路径
            font_path: 字体文件路径（可选）
        返回Returns:
            bool: 是否成功
        """
        try:
            import json
            import subprocess
            import os
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 读取转录文件
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            
            # 获取 FFmpeg 路径
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                print(f"使用 FFmpeg: {ffmpeg_exe}")
            except ImportError:
                print("错误: imageio_ffmpeg 不可用，请安装该包")
                return False
            
            # 使用 MoviePy 来添加字幕（更可靠的方法）
            try:
                import moviepy
                
                # 准备字幕数据
                subtitles_data = []
                for item in transcript:
                    if isinstance(item, dict) and 'content' in item and 'time_range' in item:
                        text = item['content']
                        time_range = item['time_range']
                        if isinstance(time_range, list) and len(time_range) >= 2:
                            start = time_range[0]
                            end = time_range[1]
                            if text and end > start:
                                subtitles_data.append(((start, end), text))
                
                if not subtitles_data:
                    print("警告: 未找到有效的字幕数据")
                    return False
                
                # 加载视频
                with moviepy.VideoFileClip(video_path) as video:
                    # 为每个字幕片段创建TextClip
                    subtitle_clips = []
                    for (start, end), text in subtitles_data:
                        try:
                            txt_clip = moviepy.TextClip(
                                text=text,
                                font_size=36, 
                                color='white',
                                method='caption',
                                size=(video.w - 100, None),
                                stroke_color=None,
                                stroke_width=0,
                                bg_color=None
                            ).with_start(start).with_end(end).with_position(('center', video.h - 80))
                            subtitle_clips.append(txt_clip)
                        except Exception as e:
                            print(f"创建字幕失败: {e}")
                            continue
                    
                    if not subtitle_clips:
                        print("警告: 未能创建任何字幕")
                        return False
                    
                    # 将字幕添加到视频
                    final = moviepy.CompositeVideoClip([video] + subtitle_clips)
                    
                    # 导出视频
                    final.write_videofile(output_path, codec='libx264', audio_codec='aac', 
                                         fps=30, threads=4, logger=None)
                
                print(f"✓ 字幕添加完成: {output_path}")
                return True
                
            except ImportError as e:
                print(f"MoviePy 导入失败: {e}")
                return False
            except Exception as e:
                print(f"使用 MoviePy 添加字幕失败: {e}")
                import traceback
                traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"添加字幕失败: {e}")
            import traceback
            traceback.print_exc()
            return False

# 测试代码
if __name__ == "__main__":
    pass
