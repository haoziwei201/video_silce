# PromiseOS 演示运行说明（demo_readme）

**更新日期：** 2026-11-01　**位置：** `video_silce/`
**一句话：** 全程用界面完成——答辩演示推荐 `streamlit run streamlit_sales.py`（销售对话主界面：对话为主、准备收侧栏）；`streamlit run streamlit.py` 为全功能界面；命令行脚本 `scripts/prepare_demo.py` 仅作为自动化备选。

---

## 1. 真实运行 vs 未实现（答辩前先读）

| 环节 | 状态 | 证据 |
|---|---|---|
| 视频解析（转写/关键帧/视觉描述/合并/索引） | **真实运行**（2026-11-01 实测） | `data/global_cache/<md5>/` 各 JSON；构造输入测试 7 项 |
| 纯视觉信息进入索引（无语音/首部/尾部/短空隙） | **已修复并测试** | `python tests/run_all.py` |
| 文档资料包白名单接入（阿里云包 6 文档＋3 图） | **真实运行** | `data/libraries/promiseos_demo/`，`source_registry.json` |
| 图片视觉描述（界面状态文字原样保留） | **真实运行**（qwen3-vl-plus） | 登记表 `images.*.vlm_description` |
| 问题拆解 → 检索 → 提取 → 程序校验 → 待确认分流 → 草稿 | **真实运行**（qwen-plus） | `data/libraries/promiseos_demo/answer_log/answer_*.json` |
| 证据跳转（文档/图片/视频定位） | 代码已完成；**界面操作需现场演示核对** | streamlit “💬 销售答复”tab |
| 12 问对照实验（vs 认真调优的普通问答） | **未运行**，题集在资料包 `05_客户问题与参考答案.md` | 由 D 按修改包 04 §2—3 执行 |
| 留出题（4 问） | **未运行**，须系统冻结后一次跑完，不得用于调参 | 同上 |

## 2. 环境（四处与原配置不同，重要）

1. **`.env` 里的 `DEEPSEEK_API_KEY` 已失效（HTTP 401）**。文本模型会自动切到 DashScope 的 `qwen-plus`（compatible-mode），
   界面/日志里 `llm.provider` 显示 `dashscope` 即为备选在跑。补上新 DeepSeek key 后自动切回；也可在 `.env` 设 `LLM_PROVIDER=dashscope` 固定。
2. **视觉模型已改为 `qwen3-vl-plus`**：`qwen-vl-plus`/`qwen-vl-max` 在当前账号返回 "Model not exist"（旧配置已不可用）。
   要换模型在 `.env` 设 `QWEN_VL_MODEL=...`，不要改 `config.py`。
3. **原 `video_processor.py` 有两个会导致视觉证据丢失的既有 bug，已修复**：
   - `cv2.imwrite` 在 Windows 上遇到含中文的绝对路径（本项目路径含“大创”）会**静默失败**，关键帧一个都写不出来（旧代码不检查返回值，照样把路径加进列表）——已改为 `imencode + tofile` 并检查结果；
   - 扫描循环在首帧就先 `retrieve()` 后 `grab()`，ffmpeg 报 "Picture does not contain data" 直接退出——**PPT/文字检测和运动检测两个阶段此前对所有视频都从未真正运行过**，已修正顺序，并给 PPT 检测加了 2 秒限频（软件界面录屏会每帧命中，不限频会提出 200+ 帧）。
4. `DASHSCOPE_API_KEY` 正常（Embedding / Paraformer / qwen3-vl-plus / qwen-plus 共用）。
5. 主界面是 **`streamlit.py`（新版）**，启动必须用 `streamlit run streamlit.py`。
   不要用 `python streamlit.py` 直接运行——文件名与 streamlit 包同名会自我遮蔽、报循环导入（用 `streamlit run` 没有这个问题，CLI 会先加载真包）。
   旧界面 `streamlit_app.py` 仍保留（含同样的“销售答复”tab），如需避免混淆可归档。

## 3. 界面操作流程（全程无需命令行）

```bash
cd video_silce
streamlit run streamlit.py
```

| 步骤 | 在哪里做 | 做什么 |
|---|---|---|
| ① 建库 | 侧边栏 ⚙️ 库管理 | 新建 `promiseos_demo` 并点“选择”切到它 |
| ② 导入资料包 | **🔍 RAG构建 → 📥 资料包导入** | 目录已预填 `../第二步_国内公开资料包_阿里云`，保持勾选“视觉模型描述”，点“导入到当前知识库”（首次 1–2 分钟）。导入后可展开来源登记明细（状态/版本/审核人） |
| ③ 处理演示视频 | **📊 数据准备** | 拖拽上传 `第二步_国内公开资料包_阿里云/素材/CN-V01_长文档信息抽取接入视频.mp4`；**视频分类选 `product_demo`**；点“处理上传的视频”（首次约 3–6 分钟：转写＋关键帧＋视觉分析）。若语音识别失败会有明确警告，可重试 |
| ④ 关联视频 | **🔍 RAG构建 → 关联资产** | 勾选刚处理的资产 → “关联选中资产”（42 条片段入索引） |
| ⑤ 生成答复 | **💬 销售答复** | 演示主问题已预填；点“🧭 生成答复草稿”（约 30–60 秒）→ 状态统计 → 建议回复 → 逐张承诺卡（点开“依据”看文档原文/官方配图/视频定位）→ 待确认两栏 |
| ⑥ 留档 | 同上 | “💾 保存答复日志”（写入 `data/libraries/<库>/answer_log/`）或“⬇️ 导出待确认清单” |

视频依据默认跳到**证据片段**起播，勾选“展开上下文”改为播放前后片段；界面下方 caption 会如实显示文本模型提供方、被程序丢弃的幻觉引用与失效来源。

> 语义搜索（🔍 RAG构建 → 🔎 语义搜索）也能检索文档条目了：文档/图片结果显示标题与类型，视频结果显示证据片段与上下文双范围。

**等价命令行（仅供自动化/调试，界面操作已完全覆盖）：**

```bash
python scripts/prepare_demo.py                                   # ①+②
python scripts/prepare_demo.py --video "<CN-V01路径>"             # ③+④
python scripts/prepare_demo.py --ask-file scripts/demo_question.txt --save-log   # ⑤+⑥
```

改动 `data_merger`/`data_cleaner` 后重建：界面在“数据准备”勾选“强制重建合并/清洗”再处理；命令行加 `--force-rebuild`（语音/视觉缓存都会复用，不重复计费）。

## 4. 界面演示要点（答辩用）

入口：侧边栏 **💬 销售答复**（页面顶部有资料包导入入口，演示前先完成 ②③④）。

演示话术要点：**模型拆解/提取/起草，程序校验引用存在性、推导状态、门禁失效来源、保留冲突**——这是与普通问答的区别所在；
被丢弃的幻觉引用、失效来源都会在卡片上如实显示（黄色 caption）；文本模型提供方（deepseek/dashscope）在 caption 与答复日志里可查。

## 5. 测试

```bash
python tests/run_all.py   # 32 项构造输入测试，不调用任何外部 API
```

覆盖：E1 五种视觉遗漏场景／E2 帧路径元数据／E3 双时间范围／E4 回退只认同 MD5 缓存／承诺卡校验规则
（幻觉引用丢弃、失效降级、冲突保留、核验状态、问客户/问技术分流）／资料包白名单纪律（答案与人工观察不进索引）。

## 6. 文件改动清单（本次）

- 改：`src/data_merger.py`（E1 重写）、`src/asset_manager.py`（E2＋缓存版本＋ASR 状态＋按类别选提示词）、`src/speech_to_text.py`（E4）、
  `src/rag_engine.py`（E3＋kind＋`add_doc_segments`＋upsert＋Embedding 超长截断）、`src/data_cleaner.py`（透传 frames/visual_raw）、
  `src/prompts.py`（界面状态摘要＋UI 分析提示词）、`src/video_processor.py`（关键帧写盘与扫描循环两处既有 bug）、
  `config.py`（视觉模型名可配）、
  **`streamlit.py`（主界面：新增“💬 销售答复”页、RAG构建页“📥 资料包导入”tab、数据准备页分类/强制重建选项、语义搜索显示文档条目）**、
  **`server.py`（新增 `ingest_doc_pack / get_source_registry / prepare_sales_answer / save_sales_answer_log / get_video_pool_path`；`data_processing` 支持 `force_rebuild` 并返回 `asr_status`；`rag_search` 结果带证据字段）**、
  `streamlit_app.py`（旧界面，同样加了销售答复 tab）
- 新：`src/doc_ingest.py`、`src/commitment.py`、`src/prompt_texts.py`、`tests/`（5 个文件）、`scripts/prepare_demo.py`、
  `scripts/demo_question.txt`、本文件
- 未动：`server.py` 的既有接口、`library_manager.py`（`add_asset_to_library` 原样可用）、`video_processor.py`

## 7. 已知边界

- DeepSeek 失效期间，`data_cleaner` 的画面摘要会退化为“保留原始描述不压缩”（无信息损失，仅索引文本变长）；换 `AsyncDataCleaner` 走备选模型可后续再做。
- 来源登记中所有条目 `human_reviewer=null`，承诺卡一律显示“⚠️ 来源待核验”——**这是如实状态**，人工抽查登记后即变“已核验”。
- 12 问与留出题尚未运行；运行前不得据其调参（修改包 04 §2 规则）。
