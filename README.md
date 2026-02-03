# Video Swagger: 垂直行业私有视频资产 AI 炼金术师

> **Harnessing the Power of LLMs for Non-Linear Editing and RAG-Based Question-Answering**

**Video Swagger** 是一个面向垂直行业（职业教育、非遗传承、技能培训）的智能视频资产管理与重构系统。

旨在解决长视频（如 2 小时的技能教学、手工艺录像）**“非结构化、难检索、难传播”**的痛点。通过多模态 AI 技术，我们将沉睡在硬盘里的“死视频”炼化为**“高传播力的短视频”**和**“可交互的知识库”**。

---

## 🌟 核心价值

*   **社会价值（Social Impact）**：响应“数字工匠”与“乡村振兴”号召。帮助非遗传承人和新农人将隐性的实操技艺，转化为显性的、易于传播的数字资产，降低技能学习门槛。
*   **商业价值（Commercial Value）**：为职业培训机构提供降本增效工具。将课程制作成本降低 90%，并提供“7x24小时 AI 助教”功能，实现从“卖视频”到“卖服务”的转型。

---

## 🏗️ 系统架构与功能模块

本项目采用 **“感知 - 决策 - 执行”** 的闭环架构，包含四大核心模块：

### 1. 多模态资产解析引擎 (Ingestion & Analysis Engine)
*负责将非结构化的视频流转化为结构化的语义数据。*
*   **高精度语音转写 (ASR)**：集成阿里 **Paraformer** 模型，实现毫秒级时间戳对齐，支持识别情绪、笑声与环境音。
*   **静默操作理解 (Visual Understanding)**：*(开发中)* 针对技能教学中“只做不说”的静默片段，引入 **VLM (如 Qwen-VL)** 进行视觉抽帧描述，确保关键动作不丢失。

### 2. 智能营销内容工厂 (Marketing Content Factory) —— *Current MVP*
*负责“对外传播”，解决获客难问题。*
*   **非线性语义剪辑**：利用 **DeepSeek** 大模型理解视频逻辑，打破原始时间轴，根据“信息密度”和“传播价值”重组片段。
*   **自动化切片**：自动提取金句、高难度操作瞬间，生成 <60s 的高光短视频。

### 3. 交互式视频知识库 (Interactive Video RAG) —— *In Progress*
*负责“对内赋能”，解决检索难问题。*
*   **语义检索 (Semantic Search)**：用户通过自然语言提问（如“发动机异响怎么排查？”），系统精准定位到视频中的具体操作步骤。
*   **精准空降播放**：回答不仅是文字，而是直接播放包含答案的 15 秒视频切片，实现“视频即知识”。

### 4. 企业级安全与管理 (Enterprise Security) —— *Roadmap*
*负责私有资产保护。*
*   支持私有化部署，确保核心技术视频不出域。

---

## 🚀 快速开始 (Getting Started)

### 1. 环境准备
```bash
# 建议使用 python 3.10+
pip install -r requirements.txt
```

### 2. 配置文件
在项目根目录下创建 `.env` 文件（参考 `.env.example`），配置 API 密钥：
```ini
DASHSCOPE_API_KEY=sk-xxxx  # 阿里云灵积平台 Key (用于语音识别)
DEEPSEEK_API_KEY=sk-xxxx   # DeepSeek Key (用于语义分析)
```
*注意：请勿将 `.env` 上传至版本控制系统。*

### 3. 运行主程序
将待处理视频放入 `data/input_videos/`，运行：
```bash
python main.py
```

---

## 📂 文件结构说明

```
video_silce/
├── main.py                    # [入口] 交互式命令行主程序
├── config.py                  # [配置] 环境参数与路径管理
├── requirements.txt           # [依赖] 项目依赖库
├── .gitignore                 # [规范] Git 忽略规则
├── src/                       # [核心代码]
│   ├── video_processor.py     # 视频处理层：基于 MoviePy 的物理剪辑与合并
│   ├── speech_to_text.py      # 感知层：调用 Paraformer 进行 ASR 转写
│   ├── text_analyzer.py       # 决策层：调用 DeepSeek 进行语义评分与切片决策
│   ├── prompts.py             # 提示工程：存储针对垂直行业的 Prompt 模板
│   └── utils.py               # 工具层：日志记录与文件操作
└── data/                      # [数据中心] (大文件已配置 gitignore)
    ├── input_videos/          # 原始素材输入
    ├── processed_audio/       # 中间态：提取的音频
    ├── transcripts/           # 中间态：结构化语义数据 (JSON)
    ├── analysis_results/      # 中间态：AI 剪辑决策表 (EDL)
    └── output_videos/         # 最终产物：营销短视频/知识切片
```

---

## 🗺️ 开发路线图 (Roadmap)

我们正致力于将 Video Swagger 打造成行业标准的基础设施，未来计划逐步实现以下模块：

- [ ] **多源兼容适配**：支持 Zoom/腾讯会议录制链接直接导入，支持流媒体格式。
- [ ] **知识点图谱化**：自动提取视频中的 PPT 翻页与关键术语，生成可视化的“视频目录树”。
- [ ] **视觉增强渲染**：自动生成动态字幕（AIGC Captions）与关键词高亮，提升完播率。
- [ ] **人工微调工作台 (HITL)**：提供 Web 端可视化界面，允许专家对 AI 剪辑结果进行微调。
- [ ] **企业级权限管理**：基于角色的访问控制 (RBAC) 与数据资产看板。

---

## 🛠️ 技术栈

*   **LLM (大脑)**: DeepSeek-V3 / R1
*   **ASR (听觉)**: Alibaba Paraformer / SenseVoice
*   **Video Engine (手脚)**: MoviePy / FFmpeg
*   **Architecture**: Python / RAG (Retrieval-Augmented Generation)

---

*Copyright © 2025 Video Swagger Team. All Rights Reserved.*