import os
import sys
import numpy as np
import cv2
from moviepy import VideoFileClip, concatenate_videoclips, CompositeVideoClip

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

    def convert_to_vertical(self, video_path, output_path=None, method="solid", background_color=(0, 0, 0)):
        """
        将横屏视频转换为竖屏视频（增强版）
        
        参数Args:
            video_path: 输入视频文件路径
            output_path: 输出视频文件路径（可选，默认为原文件名加_vertical后缀）
            method: 转换方法，可选值：
                - "solid": 纯色背景（最快）
                - "static": 静态背景（使用视频第一帧）
                - "blur": 模糊背景（原始方法，较慢）
            background_color: 当method="solid"时使用的背景颜色，格式为RGB元组，默认黑色
        返回Returns:
            bool: 是否成功
        """
        try:
            print(f"正在处理视频: {video_path}")
            print(f"转换方法: {method}")
            
            # 确保目录存在
            if output_path is None:
                video_dir = os.path.dirname(video_path)
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(video_dir, f"{video_name}_vertical.mp4")
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 加载视频
            video = VideoFileClip(video_path)
            
            # 获取视频尺寸
            w, h = video.size
            print(f"原始视频尺寸: {w}x{h}")
            
            # 检查是否需要转换（横屏：宽度 > 高度）
            if w <= h:
                print("视频已经是竖屏，无需转换")
                if output_path != video_path:
                    video.write_videofile(
                        output_path,
                        codec="libx264",
                        audio_codec="aac",
                        fps=video.fps,
                        logger="bar"
                    )
                video.close()
                
                # 强制刷新文件系统
                self._force_file_sync(output_path)
                
                # 验证文件可访问性
                is_ready = self._verify_video_file_ready(output_path, timeout=10)
                
                if is_ready:
                    print(f"✓ 视频转换完成并已确认可访问: {output_path}")
                else:
                    print(f"⚠ 视频已生成，但建议等待几秒后再访问: {output_path}")
                    print("   文件位置: " + os.path.abspath(output_path))
                
                return True
            
            # 计算竖屏尺寸（9:16比例）
            # 保持高度不变，宽度调整为高度的9/16
            target_height = h
            target_width = int(h * 9 / 16)
            
            # 如果计算出的宽度小于原始宽度，则以原始宽度为基准
            if target_width < w:
                target_width = w
                target_height = int(w * 16 / 9)
            
            print(f"目标视频尺寸: {target_width}x{target_height}")
            
            # 计算原始视频在画布上的位置（居中）
            x_offset = (target_width - w) // 2
            y_offset = (target_height - h) // 2
            
            print(f"视频居中位置: ({x_offset}, {y_offset})")
            
            # 创建背景
            if method == "solid":
                # 纯色背景
                print(f"使用纯色背景: {background_color}")
                # 创建一个纯色的静态帧
                from moviepy.video.VideoClip import ColorClip
                background = ColorClip(size=(target_width, target_height), color=background_color, duration=video.duration)
            
            elif method == "static":
                # 静态背景（使用视频第一帧）
                print("使用静态背景（视频第一帧）")
                # 获取第一帧
                first_frame = video.get_frame(0)
                # 创建静态背景
                from moviepy.video.VideoClip import ImageClip
                background = ImageClip(first_frame).set_duration(video.duration)
                # 调整背景尺寸
                background = background.resize((target_width, target_height))
            
            else:  # blur 或其他
                # 原始模糊背景方法
                print("使用模糊背景")
                # 计算缩放比例，使原始视频放大以填充竖屏
                scale_factor = max(target_width / w, target_height / h)
                scaled_w = int(w * scale_factor)
                scaled_h = int(h * scale_factor)
                
                print(f"放大比例: {scale_factor:.2f}")
                print(f"放大后尺寸: {scaled_w}x{scaled_h}")
                
                # 创建背景视频（放大并模糊）
                def make_blur_background(frame):
                    # 使用OpenCV进行模糊处理
                    blurred = cv2.GaussianBlur(frame, (25, 25), 0)
                    return blurred
                
                # 创建放大并模糊的背景
                blurred_video = video.resized((scaled_w, scaled_h))
                blurred_video = blurred_video.image_transform(make_blur_background)
                
                # 调整背景尺寸到目标尺寸
                background = blurred_video.resized((target_width, target_height))
            
            # 使用 CompositeVideoClip 合成视频
            final_video = CompositeVideoClip([background, video.with_position((x_offset, y_offset))], size=(target_width, target_height))
            
            # 导出视频
            final_video.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=video.fps,
                preset="medium",
                threads=4,
                logger="bar"
            )
            
            # 关闭资源
            video.close()
            background.close()
            final_video.close()
            
            # ======== 新增的解决方案核心代码 ========
            # 1. 强制刷新文件系统
            self._force_file_sync(output_path)
            
            # 2. 验证文件可访问性
            is_ready = self._verify_video_file_ready(output_path, timeout=10)
            
            if is_ready:
                print(f"✓ 视频转换完成并已确认可访问: {output_path}")
            else:
                print(f"⚠ 视频已生成，但建议等待几秒后再访问: {output_path}")
                print("   文件位置: " + os.path.abspath(output_path))
            # ======================================
            
            return True
                
        except Exception as e:
            print(f"横屏转竖屏失败: {e}")
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

# 测试代码
if __name__ == "__main__":
    pass
