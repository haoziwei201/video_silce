import os
import json
import logging
from typing import List, Dict
import dashscope
from dashscope.audio.asr import Recognition
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DASHSCOPE_API_KEY, TRANSCRIPTS_DIR

logger = logging.getLogger(__name__)

class SpeechToText:
    """语音识别器，使用Paraformer进行语音转文字"""
    
    def __init__(self):
        """
        初始化语音识别模型或者API的调用设置
        """
        self.api_key = DASHSCOPE_API_KEY
        if not self.api_key:
            logger.warning("未找到 DASHSCOPE_API_KEY，请在 .env 文件中配置")
            print("警告: 未找到 DASHSCOPE_API_KEY，语音识别将无法正常工作")
        
        dashscope.api_key = self.api_key

    def transcribe(self, audio_path):
        """
        将音频文件转录为带时间戳的文本
        参数Args:
            audio_path: 音频文件路径
        返回Returns:
            列表list: 带时间戳的单词/句子列表
                每个元素为字典格式: {"word": str, "start": float, "end": float}
        """
        if not self.api_key:
            print("Error: Missing DASHSCOPE_API_KEY. Cannot transcribe.")
            # 尝试回退到伪造数据（仅用于测试/演示）
            return self._fallback_fake_transcribe(audio_path)

        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return []

        logger.info(f"开始使用 Paraformer 识别音频: {audio_path}")
        
        recognition = Recognition(
            model='paraformer-realtime-v1',
            format='wav',
            sample_rate=16000,
            callback=None,
            enable_timestamp=True,

        )

        try:
            logger.info(f"准备开始获取responses")
            responses = recognition.call(audio_path)
            logger.info(f"成功获取responses")
            results = []
            
            # 遍历生成器获取结果
            if responses.status_code == 200:
                output = responses.output
                if 'sentence' in output:
                    sentences = output['sentence']

                    for sentence in sentences:
                
                        text = sentence.get('text', '')
                        if text:
                            # Paraformer 返回的时间戳通常是毫秒
                            start_time = sentence.get('begin_time', 0) / 1000.0
                            end_time = sentence.get('end_time', 0) / 1000.0
                            
                            results.append({
                                "word": text,
                                "start": start_time,
                                "end": end_time
                                })
            else:
                logger.error(f"识别流错误: {responses.code} - {responses.message}")
            
            logger.info(f"识别完成，共获取 {len(results)} 条句子")
            return self._normalize(results)

        except Exception as e:
            logger.error(f"调用 Paraformer API 失败: {e}")
            print(f"调用 Paraformer API 失败: {e}")
            return self._fallback_fake_transcribe(audio_path)

    def _fallback_fake_transcribe(self, audio_path):
        """回退到原有的伪造逻辑（仅当API不可用时）"""
        logger.warning("使用伪造的转录数据作为回退")
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
