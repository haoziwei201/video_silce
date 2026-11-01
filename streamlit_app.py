import streamlit as st
import os
import asyncio
import tempfile
from src.asset_manager import AssetManager
from src.library_manager import LibraryManager
from src.rag_engine import VideoKnowledgeBase
from src.video_processor import VideoProcessor
from src.text_analyzer import TextAnalyzer
from src import doc_ingest, commitment
import json
import time
from config import GLOBAL_CACHE_DIR

# 设置页面配置
st.set_page_config(layout="wide", page_title="Video Swagger 视频知识库 (SSOT)")

def main():
    st.title("Video Swagger：单一事实来源 (SSOT) 架构")

    # 管理器初始化
    am = AssetManager()
    lm = LibraryManager()

    # --- 侧边栏：知识库管理 ---
    st.sidebar.title("📚 知识库管理")
    
    # 新建知识库
    with st.sidebar.expander("➕ 新建知识库"):
        new_lib_name = st.text_input("知识库名称")
        if st.button("创建知识库"):
            if new_lib_name:
                success, msg = lm.create_library(new_lib_name)
                if success:
                    st.sidebar.success(msg)
                    st.rerun()
                else:
                    st.sidebar.error(msg)
    
    # 选择知识库
    libraries = lm.list_libraries()
    selected_lib = st.sidebar.selectbox("当前知识库", libraries, index=0 if libraries else None)

    # --- 主导航 ---
    page = st.radio("导航模式", ["🏭 资产中心 (全局池)", "🚀 应用中心 (业务应用)"], horizontal=True)
    st.divider()

    if page == "🏭 资产中心 (全局池)":
        render_asset_center(am)
    else:
        render_app_center(am, lm, selected_lib)

def render_asset_center(am):
    st.header("资产中心：炼金工厂")
    st.markdown("在此上传视频以处理并存入全局池。系统将计算哈希 (MD5) 并进行 AI 分析 (ASR/VLM)。")

    # 上传区域
    uploaded_file = st.file_uploader("上传视频", type=['mp4', 'mov', 'avi', 'mkv'])
    
    if uploaded_file:
        if st.button("🚀 处理并入库"):
            with st.spinner("正在存入全局池..."):
                # 保存临时文件
                # 获取原始文件扩展名，确保临时文件和最终池化文件带有正确的后缀
                file_ext = os.path.splitext(uploaded_file.name)[1]
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) 
                tfile.write(uploaded_file.read())
                tfile.close()
                
                try:
                    # 异步处理
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    md5 = loop.run_until_complete(am.process_video_asset(tfile.name, original_filename=uploaded_file.name))
                    loop.close()
                    
                    if md5:
                        st.success(f"入库成功！MD5: {md5}")
                    else:
                        st.error("处理失败。")
                except Exception as e:
                    st.error(f"错误: {e}")
                finally:
                    os.unlink(tfile.name)

    st.subheader("全局资产池")
    assets = am.list_all_assets()
    if assets:
        st.dataframe(assets)
    else:
        st.info("资产池为空。")

def render_app_center(am, lm, selected_lib):
    if not selected_lib:
        st.warning("请在侧边栏创建或选择一个知识库。")
        return

    st.header(f"应用中心：{selected_lib}")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔗 关联资产", "🔍 RAG 检索", "🎬 语义剪辑", "📱 竖屏生成", "💬 销售答复"])

    # --- Tab 1: 关联资产 ---
    with tab1:
        st.subheader("挂载全局资产到当前库")
        global_assets = am.list_all_assets()
        
        # 过滤已存在的资产
        current_lib_assets = lm.get_library_assets(selected_lib)
        current_md5s = current_lib_assets.keys()
        
        available_assets = [a for a in global_assets if a['md5'] not in current_md5s]
        
        if not available_assets:
            st.info("所有全局资产已关联到此库（或池为空）。")
        else:
            selected_assets = st.multiselect(
                "选择要关联的资产", 
                options=[a['md5'] for a in available_assets],
                format_func=lambda x: f"{x} ({next((a['filename'] for a in available_assets if a['md5']==x), '未知')})"
            )
            
            if st.button("关联选中资产"):
                progress_bar = st.progress(0)
                for i, md5 in enumerate(selected_assets):
                    success, msg = lm.add_asset_to_library(selected_lib, md5)
                    if success:
                        st.toast(f"已关联 {md5}: {msg}")
                    else:
                        st.error(f"关联失败 {md5}: {msg}")
                    progress_bar.progress((i + 1) / len(selected_assets))
                st.success("关联完成！")
                st.rerun()
        
        st.subheader("已关联资产")
        if current_lib_assets:
            st.json(current_lib_assets)
        else:
            st.info("暂无关联资产。")

    # --- Tab 2: RAG 检索 ---
    with tab2:
        vkb = VideoKnowledgeBase(lib_name=selected_lib)
        query = st.text_input("搜索视频内容", key="search_query")
        
        col1, col2 = st.columns(2)
        with col1: top_k = st.slider("返回数量", 1, 10, 3)
        with col2: expand = st.checkbox("扩展上下文", value=True)

        if query:
            with st.spinner("正在搜索..."):
                results = vkb.search(query, top_k=top_k, expand_context=expand)
            
            if results and results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    meta = results['metadatas'][0][i]
                    doc = results['documents'][0][i]
                    md5 = meta.get('source_video_md5')
                    start = meta.get('start', 0)
                    end = meta.get('end', 0)
                    
                    st.markdown(f"**结果 {i+1}** ({start:.1f}s - {end:.1f}s)")
                    st.caption(doc)
                    
                    video_path = am.get_video_path(md5)
                    if video_path:
                        st.video(video_path, start_time=int(start))
                        
                        # 剪辑生成 (带缓存)
                        with st.expander("✂️ 生成片段"):
                            if st.button(f"生成片段 {i}", key=f"clip_{i}"):
                                # 检查缓存
                                cached_path = am.get_cached_slice_path(md5, start, end)
                                if cached_path:
                                    st.success("命中缓存！⚡")
                                    st.video(cached_path)
                                else:
                                    st.info("正在渲染... ⏳")
                                    vp = VideoProcessor()
                                    temp_path = os.path.join("data", "output_videos", f"temp_{doc_id}.mp4")
                                    if vp.create_clip(video_path, start, end, temp_path):
                                        final_path = am.save_slice_to_cache(temp_path, md5, start, end)
                                        st.success("渲染并缓存完成！")
                                        st.video(final_path)
                                        if os.path.exists(temp_path): os.remove(temp_path)
                                    else:
                                        st.error("剪辑失败")
                    else:
                        st.error(f"源视频丢失: {md5}")
                    st.divider()
            else:
                st.info("未找到结果。")

    # --- Tab 3: 语义剪辑 ---
    with tab3:
        st.subheader("智能语义剪辑")
        
        # 复用 get_video_list 逻辑
        video_index = lm.get_library_assets(selected_lib)
        if not video_index:
             st.info("当前库中没有视频。")
        else:
            # 选择视频
            sel_md5 = st.selectbox("选择视频", options=list(video_index.keys()), format_func=lambda x: video_index[x]['filename'], key="sem_clip_vid")
            
            if sel_md5:
                v_info = video_index[sel_md5]
                st.video(v_info['path'])
                
                # 模式选择
                mode = st.radio("剪辑模式", ["🧠 智能剪辑 (Prompt)", "🛠️ 手动剪辑 (Manual)"], horizontal=True)
                
                if mode == "🧠 智能剪辑 (Prompt)":
                    col1, col2 = st.columns(2)
                    with col1:
                        user_instruction = st.text_area("剪辑指令 (Prompt)", placeholder="例如：提取所有关于焊接的步骤，去除废话", height=100)
                    with col2:
                        max_duration = st.number_input("目标最大时长 (秒)", min_value=5, max_value=300, value=60)
                    
                    if st.button("🎬 开始智能剪辑"):
                        if not user_instruction:
                            st.warning("请输入剪辑指令。")
                        else:
                            with st.spinner("正在分析文本与视觉内容..."):
                                # 1. 获取 Transcript
                                cache_dir = os.path.join(GLOBAL_CACHE_DIR, sel_md5)
                                # 优先尝试 raw_trans.json (ASR结果)
                                trans_path = os.path.join(cache_dir, "raw_trans.json")
                                # 如果没有，尝试 rag_ready.json
                                if not os.path.exists(trans_path):
                                     trans_path = os.path.join(cache_dir, "rag_ready.json")

                                if not os.path.exists(trans_path):
                                    st.error(f"未找到转录数据: {trans_path}")
                                else:
                                    try:
                                        with open(trans_path, 'r', encoding='utf-8') as f:
                                            transcript = json.load(f)
                                        
                                        # 2. 分析
                                        ta = TextAnalyzer()
                                        segments = ta.analyze_transcript(transcript, user_instruction)
                                        
                                        if not segments:
                                            st.warning("未找到符合要求的片段。")
                                        else:
                                            st.info(f"找到 {len(segments)} 个候选片段，正在筛选...")
                                            
                                            # 3. 筛选关键片段
                                            vp = VideoProcessor()
                                            selected_segments = vp.select_key_clips(segments, max_duration)
                                            
                                            if not selected_segments:
                                                st.warning("筛选后无有效片段。")
                                            else:
                                                # 显示计划
                                                st.write("📋 剪辑计划:")
                                                st.table([{"Start": f"{s['start_time']:.2f}s", "End": f"{s['end_time']:.2f}s", "Reason": s.get('reason', '')} for s in selected_segments])
                                                
                                                # 4. 执行剪辑
                                                clip_paths = []
                                                progress_bar = st.progress(0)
                                                
                                                temp_dir = os.path.join("data", "output_videos", "temp_clips")
                                                if not os.path.exists(temp_dir):
                                                    os.makedirs(temp_dir)
                                                    
                                                for i, seg in enumerate(selected_segments):
                                                    s_t, e_t = seg['start_time'], seg['end_time']
                                                    
                                                    # 检查缓存 (单个片段)
                                                    cached_clip = am.get_cached_slice_path(sel_md5, s_t, e_t)
                                                    if cached_clip:
                                                        clip_paths.append(cached_clip)
                                                    else:
                                                        # 渲染
                                                        temp_name = f"{sel_md5}_{i}_{s_t}_{e_t}.mp4"
                                                        temp_path = os.path.join(temp_dir, temp_name)
                                                        if vp.create_clip(v_info['path'], s_t, e_t, temp_path):
                                                            # 存入缓存
                                                            final_p = am.save_slice_to_cache(temp_path, sel_md5, s_t, e_t)
                                                            clip_paths.append(final_p)
                                                            if os.path.exists(temp_path): os.remove(temp_path)
                                                    
                                                    progress_bar.progress((i + 1) / len(selected_segments))
                                                
                                                # 5. 合并
                                                if clip_paths:
                                                    st.info("正在合并片段...")
                                                    final_output_name = f"edited_{sel_md5}_{int(time.time())}.mp4"
                                                    # 使用绝对路径，防止 combine_clips 内部重复拼接路径
                                                    final_output_path = os.path.abspath(os.path.join("data", "output_videos", final_output_name))
                                                    
                                                    if vp.combine_clips(clip_paths, final_output_path):
                                                        st.success("🎉 剪辑完成！")
                                                        st.video(final_output_path)
                                                    else:
                                                        st.error("合并失败。")
                                                else:
                                                    st.error("片段生成失败。")
                                                    
                                    except Exception as e:
                                        st.error(f"处理出错: {e}")
                
                else:
                    # 手动剪辑模式 (保留原有逻辑)
                    s_col, e_col = st.columns(2)
                    s_t = s_col.number_input("开始时间", 0.0, step=1.0, key="man_start")
                    e_t = e_col.number_input("结束时间", 0.0, step=1.0, value=10.0, key="man_end")
                    
                    if st.button("剪辑"):
                        cached = am.get_cached_slice_path(sel_md5, s_t, e_t)
                        if cached:
                            st.success("命中缓存！")
                            st.video(cached)
                        else:
                            vp = VideoProcessor()
                            temp_p = os.path.join("data", "output_videos", f"man_{sel_md5}.mp4")
                            if vp.create_clip(v_info['path'], s_t, e_t, temp_p):
                                final_p = am.save_slice_to_cache(temp_p, sel_md5, s_t, e_t)
                                st.success("完成！")
                                st.video(final_p)
                                os.remove(temp_p)
    
    # --- Tab 4: 竖屏生成 ---
    with tab4:
        st.subheader("一键竖屏生成")
        
        # 复用 get_library_assets 逻辑
        video_index_v = lm.get_library_assets(selected_lib)
        
        if not video_index_v:
             st.info("当前库中没有视频。")
        else:
            sel_md5_v = st.selectbox("选择视频", options=list(video_index_v.keys()), format_func=lambda x: video_index_v[x]['filename'], key="vert_gen_vid")
            
            if sel_md5_v:
                v_info = video_index_v[sel_md5_v]
                st.video(v_info['path'])
                
                # 转换选项
                method = st.selectbox("填充模式", ["solid", "blur"], format_func=lambda x: "纯色填充 (Solid)" if x == "solid" else "模糊背景 (Blur)", index=0)
                
                if st.button("📱 生成竖屏视频"):
                    with st.spinner("正在转换视频方向，请稍候..."):
                        vp = VideoProcessor()
                        
                        # 定义输出路径
                        output_filename = f"{os.path.splitext(v_info['filename'])[0]}_vertical_{method}.mp4"
                        output_path = os.path.join("data", "output_videos", output_filename)
                        
                        # 确保输出目录存在
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # 调用转换
                        if vp.convert_to_vertical(v_info['path'], output_path, method=method):
                            st.success("转换成功！")
                            st.video(output_path)
                        else:
                            st.error("转换失败，请检查日志。")

    # --- Tab 5: 销售答复（PromiseOS） ---
    with tab5:
        render_sales_reply_tab(am, selected_lib)


STATUS_ICONS = {
    commitment.STATUS_SUPPORTED: "🟢",
    commitment.STATUS_CONDITIONAL: "🟡",
    commitment.STATUS_NEEDS_TECH: "🔴",
    commitment.STATUS_CONFLICT: "🟠",
}

DEMO_QUESTION = "我们有一批中文扫描合同，要提取指定字段并接入现有系统。是不是开通就能上线，以后完全不用人工处理？"
DEFAULT_PACK_DIR = os.path.join("..", "第二步_国内公开资料包_阿里云")


def _render_evidence(ev, am, key_prefix):
    """按证据类型展示依据：文档原文 / 官方配图 / 视频定位（证据范围 ↔ 展开上下文）。"""
    head = f"`{ev.get('kind')}` **{ev.get('title') or ev.get('source_id')}**"
    if ev.get("section"):
        head += f" ｜ {ev['section']}"
    if ev.get("api_version"):
        head += f" ｜ API {ev['api_version']}"
    st.markdown(head)
    review = "✅ 已有人工审核记录" if ev.get("human_reviewer") else "⚠️ 来源待核验（未有人工审核记录）"
    st.caption(f"{review} ｜ 来源状态：{ev.get('status') or '未登记'}")

    if ev.get("kind") == "video":
        video_path = am.get_video_path(ev.get("video_md5", ""))
        e_s, e_e = ev.get("evidence_start"), ev.get("evidence_end")
        c_s, c_e = ev.get("context_start"), ev.get("context_end")
        st.caption(f"证据片段 {e_s}s–{e_e}s ｜ 上下文 {c_s}s–{c_e}s")
        if video_path:
            expand = st.checkbox("展开上下文（播放前后片段）", key=f"{key_prefix}_ctx")
            start_at = c_s if expand else e_s
            st.video(video_path, start_time=int(start_at or 0))
        else:
            st.error(f"源视频不在资产池：{ev.get('video_md5')}")
        frames = [p for p in (ev.get("frame_paths") or []) if p and os.path.exists(p)]
        if frames:
            cols = st.columns(min(len(frames), 4))
            for i, p in enumerate(frames[:4]):
                cols[i].image(p, caption=os.path.basename(p), width=220)
        if ev.get("visual_summary"):
            st.caption(f"画面摘要：{ev['visual_summary']}")
        st.markdown(f"> {ev.get('text', '')}")
    elif ev.get("kind") == "image":
        fp = ev.get("frame_path") or ev.get("local_path")
        if fp and os.path.exists(fp):
            st.image(fp, width=640)
        st.markdown(f"> {ev.get('text', '')}")
        if ev.get("url"):
            st.markdown(f"[官方原图]({ev['url']})")
    else:
        st.markdown(f"> {ev.get('text', '')}")
        if ev.get("boundary"):
            st.caption(f"整理边界（非原文）：{ev['boundary']}")
        if ev.get("url"):
            st.markdown(f"[打开原文]({ev['url']})")
        if ev.get("local_path"):
            st.caption(f"本地摘要：{ev['local_path']}")


def render_sales_reply_tab(am, selected_lib):
    st.subheader("销售答复：把客户的一句话变成有依据、带条件、知道该问谁的答复草稿")
    st.caption("模型负责拆解 / 提取 / 起草；引用是否真实存在、状态判定、来源门禁、冲突保留由程序完成。演示素材为公开资料整理，不代表企业合作。")

    # ---- 资料包导入 ----
    with st.expander("📥 导入资料包（文档摘要 + 官方配图 → 当前知识库）"):
        pack_dir = st.text_input("资料包目录（含来源登记 JSON）", value=DEFAULT_PACK_DIR, key="pack_dir")
        run_vlm = st.checkbox("用视觉模型为图片生成描述（调用 Qwen-VL，有缓存）", value=True, key="pack_vlm")
        if st.button("导入到当前库", key="pack_import"):
            if not os.path.isdir(pack_dir):
                st.error(f"目录不存在：{pack_dir}")
            else:
                with st.spinner("导入中（首次需要调用 Embedding / 视觉模型）..."):
                    try:
                        report = doc_ingest.ingest_pack(selected_lib, pack_dir, run_vlm=run_vlm)
                        st.success("导入完成")
                        st.json(report)
                    except Exception as e:
                        st.error(f"导入失败：{e}")
        reg = doc_ingest.load_registry(selected_lib)
        if reg.get("datasets"):
            st.caption(
                f"本库已登记资料集 {len(reg['datasets'])} 个 ｜ 来源 {len(reg.get('sources', {}))} 条 ｜ "
                f"图片 {len(reg.get('images', {}))} 张 ｜ 视频登记 {len(reg.get('videos', {}))} 条"
            )

    # ---- 输入 ----
    question = st.text_area("客户的问题（原话即可）", value=DEMO_QUESTION, height=90, key="q_text")
    known = st.text_input("销售已掌握的客户条件（可空）", placeholder="例如：客户用的是用友 U8，每天约 300 份扫描合同", key="q_known")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        top_k = st.slider("每个事项检索条数", 2, 8, 4, key="q_topk")
    with col_b:
        st.caption("建议先导入资料包并关联演示视频，再生成答复。")

    if st.button("🧭 生成答复草稿", type="primary", key="q_run"):
        if not question.strip():
            st.warning("请输入客户问题。")
        else:
            with st.spinner("拆解问题 → 检索资料 → 提取条件 → 程序校验 → 起草回复..."):
                try:
                    st.session_state["promise_answer"] = commitment.prepare_answer(
                        question.strip(), selected_lib, known_conditions=known.strip(), top_k=top_k
                    )
                except Exception as e:
                    st.error(f"生成失败：{e}")

    answer = st.session_state.get("promise_answer")
    if not answer or answer.get("lib_name") != selected_lib:
        return

    # ---- 结果 ----
    st.divider()
    llm = answer.get("llm", {})
    counts = answer.get("status_counts", {})
    mc = st.columns(5)
    mc[0].metric("事项数", len(answer.get("claims", [])))
    mc[1].metric("🟢 有依据", counts.get(commitment.STATUS_SUPPORTED, 0))
    mc[2].metric("🟡 需附条件", counts.get(commitment.STATUS_CONDITIONAL, 0))
    mc[3].metric("🔴 待技术确认", counts.get(commitment.STATUS_NEEDS_TECH, 0))
    mc[4].metric("🟠 来源冲突", counts.get(commitment.STATUS_CONFLICT, 0))
    st.caption(
        f"耗时 {answer.get('elapsed_sec')}s ｜ 模型拆解 {'✓' if llm.get('decomposed') else '✗(启发式)'} ｜ "
        f"模型提取 {'✓' if llm.get('extracted') else '✗(保守)'} ｜ 模型起草 {'✓' if llm.get('drafted') else '✗(模板)'}"
        + (f" ｜ 最近错误：{llm.get('last_error')}" if llm.get("last_error") else "")
    )

    st.markdown("### ✉️ 建议回复（销售核对后使用，系统不自动发送）")
    st.text_area("回复草稿", value=answer.get("reply", ""), height=170, key="reply_text")

    st.markdown("### 🪪 承诺卡")
    for idx, card in enumerate(answer.get("claims", [])):
        icon = STATUS_ICONS.get(card["status"], "⚪")
        with st.expander(f"{icon} {card['status']} ｜ {card['customer_ask']}", expanded=(idx == 0)):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**资料直接支持的内容**")
                st.write(card["supported_statement"] or "—（无可引用依据，不作承诺）")
                st.markdown("**适用前提 / 必要条件**")
                st.write("\n".join(f"- {x}" for x in card["conditions"]) or "—")
                if card.get("suggested_phrase"):
                    st.markdown("**建议表述**")
                    st.info(card["suggested_phrase"])
            with c2:
                st.markdown("**尚未确认的内容**")
                st.write("\n".join(f"- {x}" for x in card["unknowns"]) or "—")
                st.markdown(f"**来源核验：** {card['source_review']}")
                if card.get("given_conditions"):
                    st.caption("客户已说明：" + "；".join(card["given_conditions"]))
                if card.get("rationale"):
                    st.caption(f"提取说明：{card['rationale']}")
            if card.get("conflicts"):
                for cf in card["conflicts"]:
                    st.warning(f"来源冲突，待人工确认：{cf.get('note') or ''} ｜ 涉及 {', '.join(cf.get('evidence_ids', []))}")
            if card.get("dropped_evidence_ids"):
                st.caption(f"⚠️ 模型引用了不存在的证据 ID，已被程序丢弃：{', '.join(card['dropped_evidence_ids'])}")
            if card.get("stale_evidence_ids"):
                st.caption(f"⚠️ 以下来源标记为失效/历史，未计入依据：{', '.join(card['stale_evidence_ids'])}")

            st.markdown("**依据（点击查看）**")
            if not card["evidence"]:
                st.caption(f"无有效依据。候选 {len(card.get('candidate_evidence_ids', []))} 条均未被采用。")
            for j, ev in enumerate(card["evidence"]):
                with st.expander(f"依据 {j + 1}：{ev.get('title') or ev.get('source_id')}", expanded=(j == 0)):
                    _render_evidence(ev, am, key_prefix=f"ev_{idx}_{j}")

    st.markdown("### 📌 待确认清单")
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("**问客户（缺客户信息）**")
        if answer.get("ask_customer"):
            for q in answer["ask_customer"]:
                st.write(f"- {q['question']}")
        else:
            st.caption("—")
    with q2:
        st.markdown("**问技术 / 内部确认（缺产品依据）**")
        if answer.get("ask_tech"):
            for q in answer["ask_tech"]:
                st.write(f"- {q['question']} ｜ 建议确认：{q['suggested_role']}")
        else:
            st.caption("—")

    if answer.get("cited_sources"):
        with st.expander("📚 本次答复引用的资料与版本（答复日志会保存这份快照）"):
            st.json(answer["cited_sources"])

    b1, b2 = st.columns(2)
    with b1:
        if st.button("💾 保存答复日志", key="save_log"):
            path = commitment.save_answer_log(answer, selected_lib)
            st.success(f"已保存：{path}")
    with b2:
        st.download_button(
            "⬇️ 导出待确认清单 (JSON)",
            data=json.dumps({"question": answer["question"], "ask_customer": answer["ask_customer"], "ask_tech": answer["ask_tech"]}, ensure_ascii=False, indent=2),
            file_name="pending_confirmations.json",
            mime="application/json",
            key="dl_pending",
        )


if __name__ == "__main__":
    main()
