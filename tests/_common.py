"""
测试公共设置：把项目根目录加入 sys.path，关闭日志噪音。
所有测试不调用外部 API（Embedding / ASR / VLM / LLM），只验证程序逻辑。
"""
import os
import sys
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logging.disable(logging.CRITICAL)

# 阿里云资料包的相对位置（相对于 video_silce/）
ALIYUN_PACK = os.path.join(ROOT, "..", "第二步_国内公开资料包_阿里云")


def V(text, start, end, frame_path=None):
    seg = {"text": f"[Visual: {text}]", "start": start, "end": end}
    if frame_path:
        seg["frame_path"] = frame_path
        seg["frame_id"] = os.path.splitext(os.path.basename(frame_path))[0]
    return seg


def S(word, start, end):
    return {"word": word, "start": start, "end": end}
