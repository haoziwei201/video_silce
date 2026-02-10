import json

# For visual_recognition.py
VISUAL_ANALYSIS_PROMPT = "请详细描述这张图片的内容，包括场景、人物、动作和关键视觉元素。"

# For data_cleaner.py
def get_summarize_visual_prompt(long_text):
    return f"""
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

# For text_analyzer.py
def get_classify_segments_prompt(items_to_classify):
    return f"""
你是一个专业的视频剪辑助手。请对以下视频字幕片段进行分类。
分类标签及其定义如下：
- instruction: 重要知识点，不能省略 (关键词: 注意, 切记, 必须...)
- action: 实操精华，需要看清每个步骤 (关键词: 拧, 焊, 装, 拆...)
- demonstration: 展示过程，不需要逐帧看 (关键词: 展示, 演示, 看...)
- explanation: 原理解释 (关键词: 因为, 所以, 原理...)
- question: 互动环节 (关键词: 为什么, 如何, 吗...)
- review: 回顾内容 (关键词: 回顾, 总结...)
- transition: 过渡内容 (关键词: 接下来, 然后...)
- noise: 无效片段 (关键词: 嗯, 啊, 呃...)

请根据文本内容判断最合适的标签。
返回格式必须是合法的 JSON 列表，每项包含 "id" 和 "label"。
例如: [{{"id": 1, "label": "instruction"}}, {{"id": 2, "label": "noise"}}]

待分类片段：
{json.dumps(items_to_classify, ensure_ascii=False)}
"""
