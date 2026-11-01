import json
import os
import asyncio
import requests
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from src.prompts import get_summarize_visual_prompt

# 加载环境变量
load_dotenv()

# DeepSeek 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# --- 配置区 ---
CONCURRENCY_LIMIT = 50  # DeepSeek 并行可以开大点，建议 20-50
# --------------

class AsyncDataCleaner:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }

    async def summarize_visual_async(self, long_text):
        """
        单个任务的异步包装
        """
        if not long_text or len(long_text) < 50:
            return long_text

        prompt = get_summarize_visual_prompt(long_text)

        async with self.semaphore:  # 控制并发数
            payload = {
                "model": "deepseek-chat",
                "messages": [{
                    "role": "user", 
                    "content": prompt
                }],
                "temperature": 0.1,
                "max_tokens": 100
            }

            # 使用 to_thread 让同步的 requests 不卡住异步循环
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: requests.post(DEEPSEEK_URL, headers=self.headers, json=payload, timeout=20)
                )
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content'].strip()
                else:
                    print(f"API 错误: {response.status_code} - {response.text}")
                    return long_text
            except Exception as e:
                print(f"请求异常: {str(e)}")
                return long_text

    async def process_file_async(self, input_path, output_path, category_tag):
        print(f"📂 正在读取: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"🚀 开始异步清洗 {len(data)} 条数据...")
        
        # 1. 准备所有异步任务
        tasks = []
        # 保存原始 item 的引用，以便后续合并
        items_to_process = []
        
        for item in data:
            visual_context = item.get("visual_context", "")
            items_to_process.append(item)
            tasks.append(self.summarize_visual_async(visual_context))

        # 2. 并行执行所有任务，并显示进度条
        summaries = await tqdm.gather(*tasks, desc="API 请求进度")

        # 3. 将结果拼回原数据
        cleaned_data = []
        for item, summary in zip(items_to_process, summaries):
            new_item = {
                "id": str(item['id']),
                "start": item['time_range'][0],
                "end": item['time_range'][1],
                "type": item['type'],
                "content": item['content'],
                "category": category_tag,
                "visual_summary": summary,
                # 原始画面描述与关键帧不丢：承诺卡展示依据时要能回到原图原文
                "visual_raw": item.get("visual_raw", "") or item.get("visual_context", ""),
                "frames": item.get("frames", []),
            }

            # 构造 RAG 文本
            rag_text = ""
            if new_item["visual_summary"]:
                rag_text += f"[画面] {new_item['visual_summary']} "
            if new_item["content"]:
                rag_text += f"[语音] {new_item['content']}"
            
            new_item["rag_text"] = rag_text.strip()
            cleaned_data.append(new_item)

        # 4. 保存文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"💾 正在保存到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        print("✅ 清洗完成！")

def clean_json_data(input_path, output_path, category_tag="general"):
    """
    主处理函数 (兼容旧接口，内部调用异步实现)
    """
    if not os.path.exists(input_path):
        print(f"❌ 错误：找不到文件 {input_path}")
        return

    cleaner = AsyncDataCleaner()
    asyncio.run(cleaner.process_file_async(input_path, output_path, category_tag))

if __name__ == "__main__":
    # 配置你的路径
    INPUT_FILE = "data/transcripts/raw_video_analysis.json" 
    OUTPUT_FILE = "data/transcripts/rag_ready_data.json"
    CATEGORY = "cooking" 

    clean_json_data(INPUT_FILE, OUTPUT_FILE, CATEGORY)
