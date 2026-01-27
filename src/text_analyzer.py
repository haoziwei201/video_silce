import json
import os
import requests

# API 配置
API_KEY = "sk-370cdf2f51a44cccb046967e6869f499"
API_URL = "https://api.deepseek.com/chat/completions"

# 假设文件路径
file_path = os.path.join("data", "transcripts", "text_trans.json")

def load_trans_data():
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def call_deepseek(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful video editing assistant. Always output valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"API 调用失败: {e}")
        return None

class TextAnalyzer:
    def analyze_transcript(self, transcript_list, user_instruction):
        """
        transcript_list: 就是 load_trans_data() 返回的那个列表
        user_instruction: 用户的命令，如“只要干货”
        """
        
        # 1. 准备发给 AI 的素材
        # 我们通常把整个文本拼起来给 AI 看，方便它理解上下文
        context_text = ""
        for item in transcript_list:
            # 拼成：[0.0-3.5] 各位评委好...
            context_text += f"[{item['start']}-{item['end']}] {item['word']}\n"

        # 2. 构造 Prompt
        # 告诉 AI：根据 context_text 和 user_instruction，给我返回 JSON
        prompt = f"""
        你是一个剪辑助手。
        用户需求：{user_instruction}
        
        以下是视频字幕：
        {context_text}
        
        请找出符合需求的片段，根据用户需求对每个片段打分，根据与用户需求的匹配程度从低到高打0-10分，返回 JSON 列表。
        **重要：请只返回纯 JSON 字符串，不要包含 markdown 代码块标记（如 ```json ... ```）。**
        
        返回格式示例：
        [
            {{"start": 0.0, "end": 3.5, "score": 10, "reason": "开场白"}},
            ...
        ]
        """

        # 3. 调用 DeepSeek API
        print(f"正在根据需求 '{user_instruction}' 分析字幕...")
        response = call_deepseek(prompt)
        
        if not response:
            return []
        
        # 4. 解析 AI 返回的 JSON 字符串
        try:
            # 清理可能的 markdown 标记
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("\n", 1)[1]
                if clean_response.endswith("```"):
                    clean_response = clean_response.rsplit("\n", 1)[0]
            
            result_list = json.loads(clean_response) 
            return result_list 
        except Exception as e:
            print(f"JSON 解析失败: {e}")
            print(f"原始返回: {response}")
            return []

# 测试打印
if __name__ == "__main__":
    transcript = load_trans_data()
    if not transcript:
        print("未加载到字幕数据，请检查文件路径。")
    else:
        print(f"成功加载 {len(transcript)} 条字幕数据")
        
        # 假定用户输入
        user_instruction = "去除无意义的语气词和意外插曲（如翻页笔没电），只保留核心介绍内容。"
        
        analyzer = TextAnalyzer()
        results = analyzer.analyze_transcript(transcript, user_instruction)
        
        print("\n--- 分析结果 ---")
        print(json.dumps(results, indent=2, ensure_ascii=False))