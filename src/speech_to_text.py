import os
import json
import logging
import re
from typing import List, Dict
import dashscope
from dashscope.audio.asr import Recognition
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DASHSCOPE_API_KEY, TRANSCRIPTS_DIR, GLOBAL_CACHE_DIR

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
        # 最近一次 transcribe 的结果状态：cache / ok / no_speech / failed
        self.last_status = None

    def transcribe(self, audio_path, video_md5=None):
        """
        将音频文件转录为带时间戳的文本
        参数Args:
            audio_path: 音频文件路径
            video_md5: 视频MD5哈希，用于缓存查找
        返回Returns:
            列表list: 带时间戳的单词/句子列表
                每个元素为字典格式: {"word": str, "start": float, "end": float}
            识别失败时返回空列表并把 self.last_status 置为 "failed"，
            调用方必须据此区分“合法无语音”和“识别失败”。
        """
        self.last_status = None
        # 1. 检查全局缓存
        if video_md5:
            cache_dir = os.path.join(GLOBAL_CACHE_DIR, video_md5)
            cache_file = os.path.join(cache_dir, "raw_trans.json")
            if os.path.exists(cache_file):
                logger.info(f"命中全局缓存: {cache_file}")
                print(f"✓ 命中全局语音缓存")
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        self.last_status = "cache"
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"读取缓存失败: {e}")

        if not self.api_key:
            logger.error("缺少 DASHSCOPE_API_KEY，无法转录")
            print("Error: Missing DASHSCOPE_API_KEY. Cannot transcribe.")
            self.last_status = "failed"
            return self._fallback_fake_transcribe(audio_path, video_md5)

        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            self.last_status = "failed"
            return []

        logger.info(f"开始使用 Paraformer 识别音频: {audio_path}")
        
        # 用户指定使用 Paraformer 实时语音识别 8k v2 模型
        # 注意：对于长视频文件，通常推荐使用 Transcription (文件转写) 接口，
        # 但根据用户指令，这里使用 Recognition (实时语音识别) 接口。
        recognition = Recognition(
            model='paraformer-realtime-8k-v2',
            format='wav',
            sample_rate=8000,
            callback=None
        )

        try:
            logger.info(f"准备开始获取responses")
            responses = recognition.call(audio_path)
            logger.info(f"成功获取responses")
            results = []
            stream_error = False
            
            # Recognition.call 返回的是一个生成器(generator)或者Response
            # 为了兼容性和处理流式响应，我们进行遍历
            # 如果是单个Response，把它放入列表处理
            
            # 检查是否是生成器
            import types
            if isinstance(responses, types.GeneratorType):
                response_iterator = responses
            else:
                response_iterator = [responses]

            for response in response_iterator:
                if response.status_code == 200:
                    output = response.output
                    # Paraformer Realtime 返回的 output 中包含 'sentence' 字段
                    # 这里的 sentence 通常是当前识别到的句子列表
                    # 在流式过程中，我们需要收集这些句子
                    if output and 'sentence' in output:
                        # 注意：在流式响应中，sentence 可能包含重复或更新的内容
                        # 但对于文件输入的同步/流式模拟，通常最后一次返回包含完整结果
                        # 或者每次返回新识别的句子。
                        # 我们这里假设 SDK 会正确处理，我们只收集最终确定的句子
                        pass 
                        
                        # 实际上，DashScope SDK 的文件模式 call() 返回的如果是生成器，
                        # 每个 yield 的 response 可能包含当前最新的识别结果。
                        # 对于 paraformer-realtime，通常关注 sentence 字段。
                        
                        # 简单策略：收集所有 response 中的 sentence，并去重（通过 begin_time）
                        # 或者只取最后一个 response 的结果（如果它包含所有历史）
                        # 但实时流通常只包含“新识别”的内容或者“当前窗口”的内容。
                        
                        # 观察之前的代码逻辑，它似乎假设了一次性返回。
                        # 让我们采用一种稳健的方法：解析每个 response 里的 sentence
                        
                        sentences = output['sentence']
                        # sentences 是一个列表
                        for sentence in sentences:
                            text = sentence.get('text', '')
                            start_time = sentence.get('begin_time', 0) / 1000.0
                            end_time = sentence.get('end_time', 0) / 1000.0
                            
                            # 简单的去重/添加逻辑
                            # 只有当这个句子不在 results 中（根据开始时间判断）才添加
                            # 或者更新最后一个句子（如果是修正）
                            
                            # 这里做一个简化假设：如果句子有效且时间合理，就添加
                            # 为了避免重复，我们可以检查 results 中最后一个句子的开始时间
                            if text:
                                if not results or abs(results[-1]['start'] - start_time) > 0.01:
                                     results.append({
                                        "word": text,
                                        "start": start_time,
                                        "end": end_time
                                    })
                                elif results and abs(results[-1]['start'] - start_time) <= 0.01:
                                    # 如果开始时间相同，可能是更新，覆盖它
                                    results[-1] = {
                                        "word": text,
                                        "start": start_time,
                                        "end": end_time
                                    }

                else:
                    logger.error(f"识别流错误: {response.code} - {response.message}")
                    stream_error = True
            
            logger.info(f"识别完成，共获取 {len(results)} 条句子")
            normalized_results = self._normalize(results)

            if stream_error and not normalized_results:
                # 接口报错且没拿到任何句子：按失败处理，不写缓存，避免把空结果当成“无语音”固化
                self.last_status = "failed"
                return self._fallback_fake_transcribe(audio_path, video_md5)

            self.last_status = "ok" if normalized_results else "no_speech"
            
            # 保存到全局缓存
            if video_md5:
                try:
                    cache_dir = os.path.join(GLOBAL_CACHE_DIR, video_md5)
                    os.makedirs(cache_dir, exist_ok=True)
                    cache_file = os.path.join(cache_dir, "raw_trans.json")
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(normalized_results, f, ensure_ascii=False)
                    logger.info(f"已写入全局缓存: {cache_file}")
                except Exception as e:
                    logger.error(f"写入全局缓存失败: {e}")

            return normalized_results

        except Exception as e:
            logger.error(f"调用 Paraformer API 失败: {e}")
            print(f"调用 Paraformer API 失败: {e}")
            self.last_status = "failed"
            return self._fallback_fake_transcribe(audio_path, video_md5)

    def _fallback_fake_transcribe(self, audio_path, video_md5=None):
        """
        识别失败时的回退：只允许使用与当前视频 MD5 绑定的缓存，
        不再读取 transcripts 目录下与视频无关的通用文件，避免把别的视频的转录混进知识库。
        找不到可信缓存就返回空列表，由调用方根据 last_status 决定如何处理。
        """
        if video_md5:
            cache_file = os.path.join(GLOBAL_CACHE_DIR, video_md5, "raw_trans.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    logger.warning(f"识别失败，回退到同 MD5 缓存: {cache_file}")
                    return self._normalize(data)
                except Exception as e:
                    logger.warning(f"读取同 MD5 缓存失败: {e}")
        logger.error(
            "语音识别失败且无同 MD5 缓存可用（%s），返回空转录；不会加载通用回退文件",
            os.path.basename(audio_path),
        )
        return []

    def split_by_punctuation(self, items):
        """
        按标点将 ASR 句子切分成更短的片段（用于视频剪辑）
        """
        segments = []
        punctuation_pattern = r'[，。！？；,.!?;]'

        for item in items:
            text = item["word"]
            start = item["start"]
            end = item["end"]

            duration = end - start
            if duration <= 0 or not text:
                continue

            # 按标点切文本（保留顺序）
            parts = re.split(punctuation_pattern, text)
            parts = [p.strip() for p in parts if p.strip()]

            if len(parts) == 1:
                # 没标点，直接返回原句
                segments.append(item)
                continue

            total_chars = sum(len(p) for p in parts)
            current_time = start

            for i, part in enumerate(parts):
                ratio = len(part) / total_chars
                part_duration = duration * ratio

                part_start = current_time
                part_end = end if i == len(parts) - 1 else current_time + part_duration

                segments.append({
                    "word": part,
                    "start": round(part_start, 3),
                    "end": round(part_end, 3)
                })

                current_time = part_end

        return segments

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