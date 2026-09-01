import json
import os
import sys
import requests
import re
import asyncio

# 将项目根目录添加到系统路径，以便导入 config 和 src
# 假设当前文件在 src/ 目录下，根目录是上一级
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
except ImportError:
    # 尝试直接导入（针对不同运行环境的兼容处理）
    try:
        import config
        DEEPSEEK_API_KEY = config.DEEPSEEK_API_KEY
        DEEPSEEK_BASE_URL = config.DEEPSEEK_BASE_URL
    except ImportError as e:
        print(f"Warning: Import failed: {e}")
        DEEPSEEK_API_KEY = ""
        DEEPSEEK_BASE_URL = ""

from src.prompts import get_classify_segments_prompt

import httpx

class TextAnalyzer:
    """文本分析器，使用DeepSeek API分析转录文本"""
    
    def __init__(self):
        """
        初始化文本分析器
        设置DeepSeek API的认证信息
        """
        self.api_key = DEEPSEEK_API_KEY
        if not self.api_key:
            print("警告: 未找到 DEEPSEEK_API_KEY，请在 .env 文件中配置")

        self.base_url = DEEPSEEK_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 定义分类关键词库
        self.keywords = { 
            "instruction": ["注意", "切记", "必须", "不能", "确保", "要", "不要", "应该", "不应该", "一定要", "千万"], 
            "action": ["拧", "焊", "切", "装", "拆", "按", "转", "推", "拉", "接", "固定", "安装"], 
            "demonstration": ["展示", "演示", "看", "请看", "这样做", "这样操作", "示范", "给大家看"], 
            "explanation": ["因为", "所以", "因此", "原理", "原因是", "由于", "意味着", "也就是说"], 
            "question": ["吗", "什么", "为什么", "如何", "怎样", "是不是", "有没有", "？", "?", "谁", "哪里"], 
            "review": ["回顾", "总结", "前面讲了", "刚才说了", "综上所述", "总的来说", "我们讲了", "已经学"], 
            "transition": ["接下来", "下面", "然后", "接着", "之后", "下面我们", "接下来我们", "好", "那么"], 
            "noise": ["嗯", "啊", "那个", "就是", "然后", "呃", "这个", "那个", "呃呃", "啊啊"] 
        }
        
        # 定义处理方式映射
        self.action_map = {
            "instruction": {"action": "keep", "speed": 1.0, "desc": "重要知识点，完整保留"},
            "action": {"action": "keep", "speed": 1.0, "desc": "实操精华，原速保留"},
            "demonstration": {"action": "keep", "speed": 1.25, "desc": "展示过程，略快"},
            "explanation": {"action": "compress", "speed": 1.5, "desc": "原理解释，压缩"},
            "question": {"action": "keep", "speed": 1.0, "desc": "互动环节，保留"},
            "review": {"action": "compress", "speed": 2.0, "desc": "回顾内容，总结或快进"},
            "transition": {"action": "fast_forward", "speed": 3.0, "desc": "过渡内容，快进"},
            "noise": {"action": "delete", "speed": 0, "desc": "无效片段，删除"}
        }

    def _classify_rule_based(self, text):
        """基于规则的简单分类"""
        for label, keywords in self.keywords.items():
            for kw in keywords:
                if kw in text:
                    return label
        return None

    def analyze_transcript(self, words, user_instruction=None):
        """同步入口函数，调用异步实现"""
        try:
            return asyncio.run(self.analyze_transcript_async(words, user_instruction))
        except Exception as e:
            print(f"Error in analyze_transcript: {e}")
            return []

    async def analyze_transcript_async(self, words, user_instruction=None) -> list:
        """
        混合分类策略分析转录文本
        """
        if not self.api_key:
             print("Error: Missing DEEPSEEK_API_KEY. Cannot analyze transcript.")
             return []

        print("正在进行混合策略文本分析...")
        
        segments = []
        uncertain_indices = []
        
        # 1. 预处理和规则分类
        for i, item in enumerate(words):
            # 兼容新旧格式提取内容
            if "time_range" in item:
                start = item["time_range"][0]
                end = item["time_range"][1]
                content = item.get("content", "")
            else:
                start = item.get("start", 0)
                end = item.get("end", 0)
                content = item.get("text", item.get("word", ""))
            
            label = self._classify_rule_based(content)
            
            segments.append({
                "id": i,
                "start": start,
                "end": end,
                "text": content,
                "label": label
            })
            
            if not label:
                uncertain_indices.append(i)
        
        print(f"规则分类完成: {len(segments) - len(uncertain_indices)}/{len(segments)} 已分类")
        
        # 2. LLM 补充分类 (如果有未分类的)
        if uncertain_indices:
            print(f"正在调用 LLM 对剩余 {len(uncertain_indices)} 个片段进行分类...")
            
            items_to_classify = []
            for idx in uncertain_indices:
                items_to_classify.append({
                    "id": idx,
                    "text": segments[idx]["text"]
                })
            
            # 分批处理
            batch_size = 20  # 每批20个片段
            batches = [items_to_classify[i:i+batch_size] for i in range(0, len(items_to_classify), batch_size)]
            
            # 限制并发数，避免 API 限流
            sem = asyncio.Semaphore(10)
            
            async def process_batch(batch):
                async with sem:
                    prompt = get_classify_segments_prompt(batch)
                    return await self._call_llm_async(prompt)

            tasks = [process_batch(b) for b in batches]
            
            # 使用 tqdm 显示进度
            try:
                from tqdm.asyncio import tqdm as async_tqdm
                llm_results_list = await async_tqdm.gather(*tasks, desc="LLM分析进度", unit="batch")
            except ImportError:
                print("tqdm not found, running without progress bar")
                llm_results_list = await asyncio.gather(*tasks)
            
            for llm_results in llm_results_list:
                if llm_results and isinstance(llm_results, list):
                    for res in llm_results:
                        idx = res.get("id")
                        label = res.get("label")
                        if idx is not None and label in self.action_map and idx < len(segments):
                            segments[idx]["label"] = label

        # 3. 生成最终剪辑列表
        final_clips = []
        for seg in segments:
            label = seg.get("label")
            if not label:
                label = "explanation" # 默认兜底
            
            action_info = self.action_map.get(label, self.action_map["explanation"])
            
            if action_info["action"] == "delete":
                continue
            
            # 构造输出格式，兼容原有结构
            final_clips.append({
                "start_time": seg["start"],
                "end_time": seg["end"],
                "label": label,
                "reason": f"[{label}] {action_info['desc']}",
                "score": 10 if label in ["instruction", "action"] else 5, # 简单评分
                "speed": action_info["speed"],
                "text": seg["text"]
            })
            
        return final_clips

    async def _call_llm_async(self, prompt):
        """异步调用 LLM 的辅助函数"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a helpful JSON assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "stream": False,
            "response_format": {"type": "json_object"}
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=self.headers, json=payload, timeout=60)
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content']
                
                # 简单的 JSON 清洗和解析
                try:
                    start = content.find('[')
                    end = content.rfind(']') + 1
                    if start != -1 and end != -1:
                        json_str = content[start:end]
                        return json.loads(json_str)
                    else:
                        data = json.loads(content)
                        if isinstance(data, list): return data
                        for key in data:
                            if isinstance(data[key], list): return data[key]
                        return []
                except:
                    return []
            except httpx.ReadTimeout:
                print("LLM call timed out.")
                return []
            except Exception as e:
                print(f"LLM call failed: {e}")
                return []
'''
# 测试代码
if __name__ == "__main__":
    # 模拟数据加载路径
    # 这里的路径需要根据实际运行位置调整，这里假设从项目根目录或src目录运行
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "transcripts", "text_trans.json")
    
    def load_trans_data():
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    transcript = load_trans_data()
    if transcript:
        print(f"成功加载 {len(transcript)} 条字幕数据")
        
        analyzer = TextAnalyzer()
        # 假定用户输入
        user_instruction = "去除无意义的语气词和意外插曲，只保留核心介绍内容。"
        
        results = analyzer.analyze_transcript(transcript, user_instruction)
        
        print("\n--- 分析结果 ---")
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print("未找到测试数据，请检查 data/transcripts/text_trans.json")
'''
