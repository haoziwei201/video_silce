import streamlit as st
import os
import asyncio
import tempfile
from src.asset_manager import AssetManager
from src.library_manager import LibraryManager
from src.rag_engine import VideoKnowledgeBase
from src.video_processor import VideoProcessor
from src.text_analyzer import TextAnalyzer
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
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔗 关联资产", "🔍 RAG 检索", "🎬 语义剪辑", "📱 竖屏生成"])

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

if __name__ == "__main__":
    main()
