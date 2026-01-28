import json
import os
import sys
import requests

# 将项目根目录添加到系统路径，以便导入 config 和 src
# 假设当前文件在 src/ 目录下，根目录是上一级
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
    from src.prompts import MAIN_PROMPT_TEMPLATE
except ImportError:
    # 尝试直接导入（针对不同运行环境的兼容处理）
    try:
        import config
        DEEPSEEK_API_KEY = config.DEEPSEEK_API_KEY
        DEEPSEEK_BASE_URL = config.DEEPSEEK_BASE_URL
        from prompts import MAIN_PROMPT_TEMPLATE
    except ImportError as e:
        print(f"Warning: Import failed: {e}")
        DEEPSEEK_API_KEY = ""
        DEEPSEEK_BASE_URL = ""
        MAIN_PROMPT_TEMPLATE = ""

class TextAnalyzer:
    """文本分析器，使用DeepSeek API分析转录文本"""
    
    def __init__(self):
        """
        初始化文本分析器
        设置DeepSeek API的认证信息
        """
        self.api_key = DEEPSEEK_API_KEY
        self.base_url = DEEPSEEK_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def analyze_transcript(self, words, user_instruction) -> list:
        """
        分析转录文本，返回推荐的剪辑片段
        参数Args:
            words: 带时间戳的单词列表，也就是上一个python源文件的返回结果
            user_instruction: 用户指令（如"找出最精彩的部分"）
        返回Returns:
            列表list: 要剪辑片段列表
                每个元素字典格式: {
                    "start_time": float,
                    "end_time": float,
                    "reason": str,
                    "score": int
                }
        """
        
        # 1. 准备发给 AI 的素材
        context_text = ""
        for item in words:
            t = item.get("text", item.get("word", ""))
            context_text += f"[{item['start']}-{item['end']}] {t}\n"

        # 2. 构造 Prompt
        if not MAIN_PROMPT_TEMPLATE:
             print("Error: Prompt template not loaded.")
             return []

        prompt = MAIN_PROMPT_TEMPLATE.format(
            user_instruction=user_instruction,
            context_text=context_text
        )

        # 3. 调用 DeepSeek API
        # 注意：config.py 中 BASE_URL 通常不带 /chat/completions 后缀，需要拼接
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a helpful video editing assistant. Always output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "stream": False
        }
        
        print(f"正在调用 DeepSeek API 分析字幕...")
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # 4. 解析 AI 返回的 JSON 字符串
            clean_response = content.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("\n", 1)[1]
                if clean_response.endswith("```"):
                    clean_response = clean_response.rsplit("\n", 1)[0]
            
            result_list = json.loads(clean_response)
            return result_list
            
        except Exception as e:
            print(f"API 调用或解析失败: {e}")
            return []

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
