# video_silce

######  我们现阶段的目标是让程序能正常跑起来，尽量保证不会出bug，关于准确性、RAG检索之类的提升剪辑精确性的功能我个人建议后期完善。

### 文件基础结构

##### video_silce/
##### │
##### ├── main.py                    // 程序入口，通过该文件执行完整过程
##### ├── config.py                  // 配置文件（记录API密钥、视频路径等）
##### ├── .env                  // 以环境变量形式存放API信息，保护API （现阶段可以不要，直接在config.py中存储API信息）
##### ├── requirements.txt           // 依赖库列表及其版本
##### ├── README.md                  // 项目说明
##### │
##### ├── src/                       // 主要代码文件
##### │&emsp;   ├── __init__.py            // 空文件，把src文件当成一个python包
##### │&emsp;   ├── video_processor.py     // 视频处理模块，对视频进行分割、剪辑等视频操作，同时也包含讲视频转换为音频的功能
##### │&emsp;   ├── speech_to_text.py      // 音频转文字模块，利用模型讲音频转为带有时间戳的文本
##### │&emsp;   ├── text_analyzer.py       // 文本分析模块（或包含评分模块），利用ai把转录的文本进行处理、评分、筛选
##### │&emsp;   ├── prompts.py             // ai的提示词模板
##### │&emsp;   ├── utils.py               // 工具函数，如时间的格式化显式之类的小功能，暂时可以用不上
##### │&emsp;   └──………………                 // 后续改进新增的文件   
##### │ 
##### └── data/                      // 数据目录
#####  &emsp;   ├── input_videos/         // 存放输入的原始视频的文件夹
#####  &emsp;   ├── processed_audio/      // 存放视频转换成的音频文件的文件夹
#####  &emsp;   ├── output_videos/        // 存放生成的短视频的文件夹
#####  &emsp;   └──……………………………………        // 后续改进新增的文件

### 接口定义
#### config.py：各种路径、API、模型等参数
##### 全局变量：
#####    DEEPSEEK_API_KEY,      // DeepSeek API密钥
#####    DEEPSEEK_BASE_URL      //DeepSeek API地址
#####    INPUT_VIDEO_DIR,       // 输入视频目录路径
#####    OUTPUT_VIDEO_DIR,      // 输出视频目录路径
#####    PROCESSED_AUDIO_DIR,   // 音频文件路径
#####    TRANSCRIPTS_DIR,       // 音频转成的文本文件路径
#####
##### 使用示例：from config import DEEPSEEK_API_KEY(其他的全局变量同理使用INPUT_VIDEO_DIR,OUTPUT_VIDEO_DIR,PROCESSED_AUDIO_DIR,TRANSCRIPTS_DIR,PARAFORMER_MODEL)
##### video_path = os.path.join(INPUT_VIDEO_DIR, "test.mp4")
##### audio_path = os.path.join(PROCESSED_AUDIO_DIR, "test.wav")
#####
##### 注：音频转文本的操作要调用的模型或者API这里没有写，因为这种模型或API有很多种，负责这一块操作的人可以先决定要使用什么模型，后续再补充新的接口
##### 
### video_processor.py:视频相关操作
##### 类定义：
    class VideoProcessor:    
    def __init__(self): # 初始化的
        pass  
    
    def extract_audio(self, video_path, output_audio_path):
        """
        从视频中提取音频，也就是将视频转换为音频
        参数Args:
            video_path: 输入视频文件路径
            output_audio_path: 输出音频文件路径
        返回Returns:
            bool: 是否成功，成功返回True，失败返回False
        """
    
    def create_clip(self, video_path, start_time, end_time, output_path):
        """
        剪辑视频的片段，输入参数有起始时间和终止时间，要做的是把这两个时间区间的视频片段剪出来
        参数Args:
            video_path: 输入视频的路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            output_path: 输出剪辑视频的路径，这里不是最终的输出视频的路径，可以创建临时的位置，比如直接在跟目录创建并在程序运行最后删除
        Returns:
            bool: 是否成功，成功返回True，失败返回False
        """
    
    def combine_clips(self, clip_paths, output_path):
        """
        合并多个视频片段，把上个函数那些视频片段整合起来。
        参数Args:
            clip_paths: 视频片段路径列表，也就是上个函数剪出来的视频的地址的列表。
            output_path: 合并后的视频存放路径，也就是输出视频的存放路径
        Returns:
            bool: 是否成功
        """
#####
### speech_to_text.py：将音频转换为带有时间戳的文本
##### 类定义：
    class SpeechToText:
    """语音识别器，使用Paraformer进行语音转文字"""
    
    def __init__(self):
        """
        初始化语音识别模型或者API的调用设置
        """
    
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
#####
### text_analyzer.py：文本语句分析，衔接上一个python源文件
##### 类定义：
    class TextAnalyzer:
    """文本分析器，使用DeepSeek API分析转录文本"""
    
    def __init__(self):
        """
        初始化文本分析器
        设置DeepSeek API的认证信息
        """
        
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
        
        示例返回:
        [
            {
                "start_time": 15.5,
                "end_time": 32.0,
                "reason": "这部分介绍了核心概念，讲解清晰",
                "score": 9
            }
        ]
        """
#####
### prompts.py：提示词模板
##### 全局变量：
##### MAIN_PROMPT_TEMPLATE //提示词模板，调用AI的时候用
#####
### main.py:程序执行的集中部分。
##### 具体过程简述：
##### 1. 初始化
##### 2. 获取用户输入信息，确认目标视频是否存在
##### 3. 提取音频
##### 4. 语言转文本+时间戳
##### 5. 文本分析
##### 6. 根据分析结果剪辑视频，可以剪辑多个片段，或者把多个片段合并
##### 7. 输出结果到目标文件夹


