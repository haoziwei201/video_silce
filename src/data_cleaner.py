import json
import os
import requests  # 复用已有的库
from dotenv import load_dotenv
from tqdm import tqdm

# 加载环境变量
load_dotenv()

# DeepSeek 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

def summarize_visual(long_text):
    """
    使用 requests 直接调用 DeepSeek API 进行摘要
    """
    if not long_text or len(long_text) < 50:
        return long_text

    prompt = f"""
    你是一个数据清洗助手。请将以下这段冗长的视频画面描述，精简为【一句话摘要】。
    
    要求：
    1. 保留核心动作（如“切肉”、“拧螺丝”）。
    2. 保留关键物体（如“菜刀”、“万用表”）。
    3. 去除所有修饰性废话。
    4. 字数控制在 50 字以内。
    5. 直接输出摘要，不要包含任何解释。

    待处理文本：
    {long_text}
    """

    # 构造请求头和数据
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 100,
        "stream": False
    }

    try:
        # 核心修改：用 requests 发送 POST 请求
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            print(f"API 错误: {response.status_code} - {response.text}")
            return long_text
            
    except Exception as e:
        print(f"请求异常: {e}")
        return long_text

def clean_json_data(input_path, output_path, category_tag="general"):
    """
    主处理函数 (逻辑不变)
    """
    if not os.path.exists(input_path):
        print(f"❌ 错误：找不到文件 {input_path}")
        return

    print(f"📂 正在读取: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = []
    
    print("🚀 开始清洗数据...")
    # 如果没装 tqdm，可以把 tqdm(data) 改成 data
    for item in tqdm(data):
        new_item = {
            "id": str(item['id']),
            "start": item['time_range'][0],
            "end": item['time_range'][1],
            "type": item['type'],
            "content": item['content'],
            "category": category_tag,
        }

        # 处理视觉描述
        if "visual_context" in item and item["visual_context"]:
            summary = summarize_visual(item["visual_context"])
            new_item["visual_summary"] = summary
        else:
            new_item["visual_summary"] = ""

        # 构造 RAG 文本
        rag_text = ""
        if new_item["visual_summary"]:
            rag_text += f"[画面] {new_item['visual_summary']} "
        if new_item["content"]:
            rag_text += f"[语音] {new_item['content']}"
        
        new_item["rag_text"] = rag_text.strip()

        cleaned_data.append(new_item)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"💾 正在保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print("✅ 清洗完成！")

if __name__ == "__main__":
    # 配置你的路径
    INPUT_FILE = "data/transcripts/raw_video_analysis.json" 
    OUTPUT_FILE = "data/transcripts/rag_ready_data.json"
    CATEGORY = "cooking" 

    clean_json_data(INPUT_FILE, OUTPUT_FILE, CATEGORY)