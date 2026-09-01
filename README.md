# Video Swagger: 垂直行业私有视频资产 AI 炼金术师

> **Harnessing the Power of LLMs for Non-Linear Editing and RAG-Based Question-Answering**

**Video Swagger** 是一个面向垂直行业（如职业教育、技能培训、非遗传承）的智能视频处理系统。它能将长达数小时的非结构化视频（如教学录像、操作演示），自动转化为**高传播力的短视频切片**和**可交互的视频知识库**。

---

## 🌟 核心功能 (Key Features)

### 1. 智能切片 (Smart Slicing)
解决“长视频没人看”的问题。
*   **自动去水**：识别并剔除废话、静音和无效片段。
*   **高光提取**：基于 DeepSeek 语义分析，自动提取“核心知识点”和“高难度操作”瞬间。
*   **自动化剪辑**：一键生成小于 60 秒的短视频，适合抖音/小红书传播。

### 2. 视频知识库 (Video RAG)
解决“视频无法检索”的问题。
*   **语义搜索**：用户用自然语言提问（如“怎么判断电机过热？”），系统直接定位到视频中的具体步骤。
*   **跨视频检索**：在一个知识库中同时搜索多个视频的内容。
*   **视觉理解**：不仅听懂语音 (ASR)，还能看懂画面 (VLM)，理解“只做不说”的操作细节。

---

## � 项目结构说明 (Project Structure)

为了方便理解，我们将项目文件分为**执行入口**、**核心逻辑 (src)** 和 **数据中心 (data)** 三部分。

### 1. 执行入口 (Root)
| 文件名 | 说明 |
| :--- | :--- |
| **`streamlit_app.py`** | **[Web UI 主入口]** 可视化管理界面。集成资产管理、知识库问答和切片功能。 |
| **`main.py`** | **[命令行工具]** 交互式 CLI。用于处理单个视频，生成短视频切片。 |
| **`batch_processor.py`** | **[命令行工具]** 批量处理脚本。用于将视频目录导入 RAG 知识库。 |
| `.env` | **[配置]** 存放 API Key (DeepSeek, 阿里云等) 的私密文件。 |
| `requirements.txt` | **[依赖]** 项目所需的 Python 库列表。 |
| `.gitignore` | **[规范]** 定义了哪些文件不应上传到 Git。 |

### 🧠 2. 核心逻辑 (`src/`)
这里是系统的“大脑”和“手脚”。

*   **感知层 (Perception)**
    *   `speech_to_text.py`: **耳朵**。调用 Paraformer 将视频声音转为文字。
    *   `visual_recognition.py`: **眼睛**。调用 VLM 模型识别视频画面中的操作动作。
*   **决策层 (Decision)**
    *   `text_analyzer.py`: **大脑**。调用 DeepSeek 分析文本语义，决定哪些片段该留，哪些该删。
    *   `rag_engine.py`: **记忆**。基于 ChromaDB 实现向量检索，负责知识库的搜索功能。
*   **管理层 (Management)**
    *   `asset_manager.py`: **仓库管理员**。管理 `video_pool` 和 `global_cache`，确保视频不重复处理。
    *   `library_manager.py`: **图书管理员**。管理不同的知识库 (`libraries`)。
*   **执行层 (Execution)**
    *   `video_processing.py`: **剪刀**。基于 MoviePy 进行物理层面的视频剪辑和合成。

### 💾 3. 数据中心 (`data/`)
为了节省空间和提升效率，我们采用了独特的数据存储结构：

```text
data/
├── input_videos/          # [输入] 待处理的原始视频存放处
├── video_pool/            # [存储] 视频资产池。所有视频按 MD5 哈希存储，避免重复文件。
├── global_cache/          # [缓存] 全局分析缓存。存储 ASR 和 VLM 结果，同一视频只需分析一次。
├── libraries/             # [知识库] 逻辑上的“视频文件夹”。
│   └── default_lib/       # 例如：一个名为 "default_lib" 的知识库
│       ├── lib_config.json # 库配置
│       └── chroma_db/      # 向量数据库索引 (二进制文件)
├── analysis_results/      # [结果] 剪辑决策表 (JSON)
├── slice_video/           # [产物] 智能切片后的短视频片段
└── output_videos/         # [产物] 最终合成的成片
```

---

## � 快速开始 (Getting Started)

### 1. 环境准备
确保已安装 Python 3.10+。
```bash
# 激活虚拟环境 (如果有)
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置密钥
在项目根目录创建 `.env` 文件，填入必要的 API Key：
```ini
# .env 文件内容示例
DASHSCOPE_API_KEY=sk-xxxxxx  # 阿里云 (用于语音识别)
DEEPSEEK_API_KEY=sk-xxxxxx   # DeepSeek (用于语义分析)
```

### 3. 运行项目

推荐使用 **Streamlit Web 界面** 进行交互。

#### 🖥️ 方式一：Web 可视化界面 (推荐)
启动 Web UI，一站式管理视频资产和知识库：
```bash
streamlit run streamlit_app.py
```
*   **资产中心**：查看 `data/input_videos` 和 `video_pool` 中的视频状态。
*   **知识库管理**：创建新库、导入视频、构建 RAG 索引。
*   **智能应用**：进行视频切片或知识库问答。

#### ⌨️ 方式二：命令行工具 (CLI)
如果你偏好命令行操作，也可以直接运行脚本：

**1. 生成短视频 (Slicing Mode)**
```bash
python main.py
```
*交互式选择视频，自动分析并生成切片。*

**2. 批量构建知识库 (Batch Processing)**
```bash
# 将 data/input_videos 下的所有视频导入名为 "my_course" 的知识库
python batch_processor.py -i data/input_videos -l my_course
```
*适合批量初始化数据。*

---

## ⚠️ 注意事项

*   **数据隐私**：`data/` 目录下的具体视频文件和数据库已被 `.gitignore` 忽略，不会上传到代码仓库，保护你的资产安全。
*   **缓存机制**：如果视频内容没有变化，系统会自动读取 `data/global_cache/` 中的分析结果，跳过昂贵的 AI 分析步骤。
