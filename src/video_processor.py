import os
import sys
import subprocess
import numpy as np
import cv2
import json
from tqdm import tqdm
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
            
            # 优化策略：不使用 cap.set() 跳转，而是流式读取
            # 计算每隔多少帧采样一次
            frame_interval = int(fps * sample_interval)
            if frame_interval < 1: frame_interval = 1
            
            curr_frame_idx = 0
            
            # 重新打开视频以从头开始读取
            cap.release()
            cap = cv2.VideoCapture(video_path)
            
            prev_frame_gray = None
            last_motion_time = -100 # 上一次因运动提取的时间
            
            pbar = tqdm(total=int(total_frames), desc="  - 扫描进度", unit="frames")
            
            while True:
                # 抓取帧但不解码（快速）
                if curr_frame_idx % frame_interval != 0:
                    ret = cap.grab()
                    if not ret:
                        break
                    curr_frame_idx += 1
                    pbar.update(1)
                    continue
                
                # 需要处理的帧：解码
                ret, frame = cap.retrieve()
                if not ret:
                    break
                    
                curr_time = curr_frame_idx / fps
                pbar.update(1)
                
                # 跳过已经提取的时间点附近，但仍需读取帧以更新 prev_frame
                is_extracted = False
                for t in extracted_times:
                    if abs(t - curr_time) < 0.2: # 稍微放宽一点
                        is_extracted = True
                        break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 如果当前时间点已经被提取过，只更新prev_frame并继续
                if is_extracted:
                    prev_frame_gray = gray
                    curr_frame_idx += 1
                    continue

                # A. PPT/文字检测
                if self._detect_bright_rectangle(frame):
                    # 注意：这里直接保存当前frame，避免再次调用 _add_keyframe 里的 cap.set
                    self._save_frame_direct(frame, curr_time, output_dir, keyframes, extracted_times, prefix="ppt_")
                    last_motion_time = curr_time 
                
                # B. 运动检测
                elif prev_frame_gray is not None:
                    # 只有当距离上次运动提取超过2秒时才再次提取
                    if curr_time - last_motion_time > 2.0:
                        if self._detect_motion_simple(prev_frame_gray, gray):
                            self._save_frame_direct(frame, curr_time, output_dir, keyframes, extracted_times, prefix="motion_")
                            last_motion_time = curr_time

                prev_frame_gray = gray
                curr_frame_idx += 1

            pbar.close()

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

    def _save_frame_direct(self, frame, time_sec, output_dir, keyframes_list, extracted_times_set, prefix="frame_"):
        """直接保存当前帧数据，避免重新读取"""
        # 简单去重
        for t in extracted_times_set:
            if abs(t - time_sec) < 0.2:
                return

        frame_name = f"{prefix}{int(time_sec*1000):06d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        try:
            cv2.imwrite(frame_path, frame)
            extracted_times_set.add(time_sec)
            keyframes_list.append({
                "time": time_sec,
                "path": frame_path
            })
        except Exception as e:
            print(f"Error saving frame: {e}")

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

    def combine_clips(self, clip_paths, output_path):
        """
        使用 FFmpeg concat demuxer 合并视频片段，只启动一个进程
        """
        try:
            # 创建 concat 文件列表
            concat_file = os.path.join(os.path.dirname(output_path), "concat_list.txt")
            with open(concat_file, 'w', encoding='utf-8') as f:
                for path in clip_paths:
                    # 使用绝对路径，确保安全
                    abs_path = os.path.abspath(path).replace('\\', '/')
                    f.write(f"file '{abs_path}'\n")

            # 获取 FFmpeg 路径
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

            # 构建命令：直接复制流，不重新编码
            cmd = [
                ffmpeg_exe, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",        # 关键：直接复制，无重编码
                output_path
            ]

            print(f"合并命令: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True)

            # 删除临时文件
            os.remove(concat_file)
            return True

        except subprocess.CalledProcessError as e:
            print(f"FFmpeg 合并失败: {e.stderr.decode() if e.stderr else e}")
            return False
        except Exception as e:
            print(f"合并异常: {e}")
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

            # 检查是否使用了复杂的滤镜链（带有 [bg][fg] 等标签）
            is_complex = any(tag in filter_str for tag in ["[bg]", "[fg]", "[0:v]"])
            filter_arg = "-filter_complex" if is_complex else "-vf"
            
            cmd = [
                ffmpeg_exe, "-y",
                "-hide_banner",        # 隐藏版权信息
                "-i", video_path,
                filter_arg, filter_str,
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
                    try:
                        with open(filepath, 'rb') as f:
                            header = f.read(100)
                            # 宽松检查：只要包含 ftyp 即可，因为不同编码器生成的 header 长度可能不同
                            if b'ftyp' in header:
                                return True
                    except:
                        pass
                    
                    # 如果读取 header 失败但文件大小稳定且足够大，也认为是成功的
                    if current_size > 1024 * 10: # > 10KB
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

    def add_subtitles(self, video_path, transcript_path, output_path, font_path=None,
                      font_size=36, font_color="#FFFFFF", bg_color="#000000",
                      bg_opacity=50, position="bottom"):
        """
        为视频添加字幕（提速版，保留原版路径处理逻辑）
        参数:
            video_path: 输入视频路径
            transcript_path: 转录文件路径（JSON格式）
            output_path: 输出视频路径
            font_path: 字体文件路径（可选，当前未使用）
            font_size: 字号
            font_color: 字体颜色 (#RRGGBB)
            bg_color: 背景颜色 (#RRGGBB)
            bg_opacity: 背景不透明度 (0-100)
            position: 字幕位置 (top/middle/bottom)
        返回:
            bool: 是否成功
        """
        try:
            import json
            import subprocess
            import os
            import tempfile

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

            # ==================== 准备字幕数据 ====================
            subtitles_data = []
            for item in transcript:
                if isinstance(item, dict):
                    if 'content' in item and 'time_range' in item:
                        text = item['content']
                        time_range = item['time_range']
                        if isinstance(time_range, list) and len(time_range) >= 2:
                            start = time_range[0]
                            end = time_range[1]
                            if text and end > start:
                                subtitles_data.append(((start, end), text))
                    elif 'word' in item and 'start' in item and 'end' in item:
                        text = item['word']
                        start = item['start']
                        end = item['end']
                        if text and end > start:
                            subtitles_data.append(((start, end), text))

            if not subtitles_data:
                print("警告: 未找到有效的字幕数据")
                return False

            # 按起始时间排序
            subtitles_data.sort(key=lambda x: x[0][0])

            # ==================== 生成临时 SRT 字幕文件（放在视频目录）====================
            video_dir = os.path.dirname(video_path)
            # 使用临时文件创建唯一 srt 文件，避免并发冲突
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.srt',
                dir=video_dir,
                delete=False,
                encoding='utf-8'
            ) as tmp_srt:
                srt_filename = os.path.basename(tmp_srt.name)  # 仅文件名，用于相对路径
                srt_path = tmp_srt.name                        # 完整路径，用于后续删除

                def format_time(seconds):
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    millis = int((seconds % 1) * 1000)
                    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

                for i, ((start, end), text) in enumerate(subtitles_data, 1):
                    tmp_srt.write(f"{i}\n")
                    tmp_srt.write(f"{format_time(start)} --> {format_time(end)}\n")
                    tmp_srt.write(f"{text}\n\n")

            print(f"已生成 SRT 字幕文件: {srt_path}")

            # ==================== 检测最佳编码器（硬件加速优先）====================
            def detect_best_encoder():
                """检测可用的硬件编码器，返回编码器名称和对应质量参数"""
                encoders_to_check = [
                    ("h264_nvenc", "cq"),      # NVIDIA
                    ("h264_amf", "quality"),   # AMD
                    ("h264_qsv", "global_quality"),  # Intel
                    ("libx264", "crf")         # 软件编码 fallback
                ]
                for enc, qual_param in encoders_to_check:
                    try:
                        result = subprocess.run(
                            [ffmpeg_exe, "-encoders"],
                            capture_output=True, text=True, timeout=5
                        )
                        if enc in result.stdout:
                            print(f"检测到可用编码器: {enc}")
                            return enc, qual_param
                    except:
                        continue
                return "libx264", "crf"

            video_encoder, quality_param = detect_best_encoder()

            # 根据编码器设置最快编码参数
            encode_params = []
            if video_encoder == "libx264":
                # 软件编码：使用 ultrafast 预设，CRF 30（速度极快）
                encode_params = ["-preset", "ultrafast", "-crf", "30"]
            elif video_encoder == "h264_nvenc":
                # NVIDIA 硬件编码：使用 p1（最快预设），cq 值 30
                encode_params = ["-preset", "p1", "-cq", "30"]
            elif video_encoder == "h264_amf":
                # AMD 硬件编码：quality 值 0-9，数值越低越快；这里用 0（最快）
                encode_params = ["-quality", "0", "-rc", "cqp", "-qp_i", "30", "-qp_p", "30"]
            elif video_encoder == "h264_qsv":
                # Intel 硬件编码：使用 fastest 预设，global_quality 30
                encode_params = ["-preset", "fastest", "-global_quality", "30"]
            else:
                encode_params = ["-preset", "ultrafast", "-crf", "30"]

            # 添加多线程参数（自动使用所有 CPU 核心）
            encode_params.extend(["-threads", "auto"])

            # ==================== 构建字幕样式 (force_style) ====================
            alignment_map = {"top": 8, "middle": 5, "bottom": 2}
            alignment = alignment_map.get(position, 2)
            bg_opacity = max(0, min(100, int(bg_opacity)))

            def hex_to_ass(hex_color, alpha=0):
                h = hex_color.lstrip('#')
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"

            primary_colour = hex_to_ass(font_color, 0)
            bg_alpha = int(round((100 - bg_opacity) * 255 / 100))
            back_colour = hex_to_ass(bg_color, bg_alpha)

            force_style = (
                f"Fontsize={font_size},"
                f"PrimaryColour={primary_colour},"
                f"BorderStyle=3,"
                f"BackColour={back_colour},"
                f"Alignment={alignment},"
                f"MarginV=80"
            )

            # ==================== 构建 FFmpeg 命令 ====================
            # 使用 cwd 参数将工作目录设为视频目录，这样字幕文件可通过相对路径访问
            # 视频输入使用绝对路径（在 cwd 下依然有效）
            cmd = [
                ffmpeg_exe, "-y",
                "-i", video_path,
                "-vf", f"subtitles='{srt_filename}':force_style='{force_style}'",
                "-c:a", "copy",
                "-c:v", video_encoder,
            ] + encode_params + [output_path]

            print(f"正在使用编码器 {video_encoder} 添加字幕...")
            subprocess.run(cmd, check=True, cwd=video_dir)  # 设置工作目录为视频目录

            # 删除临时 SRT 文件
            try:
                os.remove(srt_path)
            except:
                pass

            print(f"✓ 字幕添加完成: {output_path}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"FFmpeg 执行失败: {e}")
            return False
        except Exception as e:
            print(f"添加字幕失败: {e}")
            import traceback
            traceback.print_exc()
            return False

# 测试代码
if __name__ == "__main__":
    pass
