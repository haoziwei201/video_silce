import os
import json
from typing import List, Dict
from config import TRANSCRIPTS_DIR

class SpeechToText:
    """语音识别器，使用Paraformer进行语音转文字"""
    
    def __init__(self):
        """
        初始化语音识别模型或者API的调用设置
        """
        pass

    def transcribe(self, audio_path):
        """
        将音频文件转录为带时间戳的文本
        参数Args:
            audio_path: 音频文件路径
        返回Returns:
            列表list: 带时间戳的单词/句子列表
                每个元素为字典格式: {"word": str, "start": float, "end": float}
        示例返回:
        [
            {"word": "你好，欢迎观看本视频。", "start": 0.0, "end": 2.5},
            {"word": "今天我们要讲的是人工智能。", "start": 2.5, "end": 5.0}
        ]
        """
        base = os.path.splitext(os.path.basename(audio_path))[0]
        candidates = [
            os.path.join(TRANSCRIPTS_DIR, f"{base}.json"),
            os.path.join(TRANSCRIPTS_DIR, "text_trans.json"),
        ]
        data = []
        for path in candidates:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except Exception:
                        data = []
                break
        return self._normalize(data)

    def _normalize(self, items: List[Dict]) -> List[Dict]:
        result: List[Dict] = []
        fillers = ["嗯", "啊", "那个", "就是", "然后", "呃", "喂喂", "哦"]
        for item in items or []:
            t = item.get("text", item.get("word", ""))
            for f in fillers:
                t = t.replace(f, "")
            t = t.strip()
            try:
                s = float(item.get("start", 0.0))
                e = float(item.get("end", 0.0))
            except Exception:
                continue
            if t and e > s:
                # 按照接口定义，返回 key 为 "word"
                result.append({"word": t, "start": s, "end": e})
        return result
