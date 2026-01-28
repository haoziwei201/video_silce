import numpy as np
from moviepy import ColorClip, CompositeAudioClip, AudioClip
import os

# 创建一个 10 秒的红色背景视频，带音频
def make_frame(t):
    return np.sin(2 * np.pi * 440 * t)  # 440 Hz 正弦波

def create_video():
    duration = 10
    # 视频轨道
    video = ColorClip(size=(640, 480), color=(255, 0, 0), duration=duration)
    video = video.with_fps(24)
    
    # 音频轨道
    audio = AudioClip(make_frame, duration=duration, fps=44100)
    
    # 合并
    final = video.with_audio(audio)
    
    # 写入文件
    output_path = "test.mp4"
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print(f"Created {output_path}")

if __name__ == "__main__":
    create_video()
