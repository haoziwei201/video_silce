# streamlit_app.py
"""
streamlit_app.py - 视频智能剪辑工具Web界面
完整版，包含库管理UI
"""

import streamlit as st
import os
import sys
import json
import time
import urllib.parse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# 导入server模块
import server

# 设置页面配置
st.set_page_config(
    page_title="视频智能剪辑工具 - SSOT",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background-color: #e6f3ff;
        padding: 20px 10px;
    }
    
    /* 库管理卡片 */
    .lib-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .lib-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .lib-card.selected {
        border: 3px solid #ffd700;
    }
    .lib-name {
        font-size: 18px;
        font-weight: bold;
    }
    .lib-stats {
        font-size: 12px;
        opacity: 0.9;
    }
    
    /* 资产卡片 */
    .asset-card {
        background: white;
        border: 1px solid #e0e7ff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .asset-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .asset-md5 {
        font-family: monospace;
        background: #f3f4f6;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 12px;
    }
    .asset-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        margin-right: 5px;
    }
    .badge-success { background: #d4edda; color: #155724; }
    .badge-warning { background: #fff3cd; color: #856404; }
    .badge-info { background: #d1ecf1; color: #0c5460; }
    
    /* 主要内容卡片 */
    .content-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .section-title {
        color: #1e1e2f;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #667eea;
    }
    
    /* 结果卡片 */
    .result-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* 进度条美化 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 25px;
        border-radius: 25px;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* 文件列表样式 */
    .file-item {
        padding: 10px;
        margin: 5px 0;
        background: #f0f2f6;
        border-radius: 8px;
        border-left: 3px solid #667eea;
    }
    .file-name {
        font-weight: bold;
        color: #1e1e2f;
    }
    .file-info {
        font-size: 12px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
def init_session_state():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "资产中心"
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'video_files' not in st.session_state:
        st.session_state.video_files = []
    if 'transcript_files' not in st.session_state:
        st.session_state.transcript_files = []
    if 'rag_files' not in st.session_state:
        st.session_state.rag_files = []
    if 'analysis_files' not in st.session_state:
        st.session_state.analysis_files = []
    if 'libraries' not in st.session_state:
        st.session_state.libraries = []
    if 'selected_lib' not in st.session_state:
        st.session_state.selected_lib = "default_lib"
    if 'global_assets' not in st.session_state:
        st.session_state.global_assets = []
    if 'lib_info' not in st.session_state:
        st.session_state.lib_info = {}
    if 'show_lib_manager' not in st.session_state:
        st.session_state.show_lib_manager = False

def safe_display_filename(filename: str) -> str:
    """安全显示文件名（解码URL编码）"""
    try:
        return urllib.parse.unquote(filename)
    except:
        return filename

# 刷新数据
def refresh_data():
    with st.spinner("刷新数据..."):
        st.session_state.video_files = server.get_video_files()
        st.session_state.transcript_files = server.get_transcript_files()
        st.session_state.rag_files = server.get_rag_files()
        st.session_state.analysis_files = server.get_analysis_files()
        st.session_state.libraries = server.get_libraries()
        st.session_state.global_assets = server.get_global_assets()
        
        if st.session_state.selected_lib:
            st.session_state.lib_info = server.get_library_info(st.session_state.selected_lib)

# 侧边栏
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🎬 SSOT 视频知识库")
        st.markdown("---")
        
        # 库管理区域 - 修改后
        st.markdown("### 📚 知识库管理")
        
        # 库选择 - 条状可折叠
        if st.session_state.libraries:
            # 显示当前选中的库
            current_lib = st.session_state.selected_lib
            lib_info = server.get_library_info(current_lib)
            
            # 当前库条状显示
            with st.expander(f"📌 当前库: {current_lib} (资产: {lib_info.get('asset_count', 0)} | RAG: {lib_info.get('rag_count', 0)})", expanded=True):
                # 库列表
                for lib in st.session_state.libraries:
                    lib_info = server.get_library_info(lib)
                    is_selected = (lib == st.session_state.selected_lib)
                    
                    # 条状库卡片
                    col1, col2, col3 = st.columns([6, 2, 1])
                    
                    with col1:
                        if is_selected:
                            st.markdown(f"**🔵 {lib}**")
                        else:
                            st.markdown(f"📁 {lib}")
                    
                    with col2:
                        st.markdown(f"📊 {lib_info.get('asset_count', 0)}/{lib_info.get('rag_count', 0)}")
                    
                    with col3:
                        if not is_selected:
                            if st.button("选择", key=f"select_{lib}", help=f"切换到知识库 {lib}"):
                                st.session_state.selected_lib = lib
                                refresh_data()
                                st.rerun()
                        else:
                            st.markdown("✅")
                    
                    # 分隔线
                    if lib != st.session_state.libraries[-1]:
                        st.markdown("---")
            
            # 库管理面板（折叠在下方）
            with st.expander("⚙️ 库管理", expanded=False):
                # 新建库
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_lib = st.text_input("新库名称", key="new_lib_name", label_visibility="collapsed", placeholder="输入新知识库名称")
                with col2:
                    if st.button("➕ 创建", key="create_lib", use_container_width=True):
                        if new_lib:
                            result = server.create_library(new_lib)
                            if result["success"]:
                                st.success(f"✅ {result['message']}")
                                refresh_data()
                                st.rerun()
                            else:
                                st.error(f"❌ {result['message']}")
                
                # 删除库
                if st.session_state.libraries:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        lib_to_delete = st.selectbox(
                            "选择要删除的库",
                            [l for l in st.session_state.libraries if l != "default_lib"],  # 排除默认库
                            key="delete_lib_select",
                            label_visibility="collapsed",
                            placeholder="选择要删除的知识库"
                        )
                    with col2:
                        if st.button("🗑️ 删除", key="delete_lib", use_container_width=True):
                            if lib_to_delete:
                                result = server.delete_library(lib_to_delete)
                                if result["success"]:
                                    st.success(f"✅ {result['message']}")
                                    if st.session_state.selected_lib == lib_to_delete:
                                        st.session_state.selected_lib = "default_lib"
                                    refresh_data()
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result['message']}")
        else:
            st.info("暂无知识库，请先创建")
            
            # 无库时的创建面板
            with st.expander("⚙️ 创建第一个知识库", expanded=True):
                new_lib = st.text_input("新库名称", key="first_lib_name", placeholder="输入知识库名称")
                if st.button("➕ 创建", key="create_first_lib", use_container_width=True):
                    if new_lib:
                        result = server.create_library(new_lib)
                        if result["success"]:
                            st.success(f"✅ {result['message']}")
                            st.session_state.selected_lib = new_lib
                            refresh_data()
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
        
        st.markdown("---")
        
        # 导航
        pages = {
            "🏭 资产中心": "资产中心",
            "📊 数据准备": "数据准备",
            "🔍 RAG构建": "RAG构建",
            "💬 销售答复": "销售答复",
            "✂️ 智能剪辑": "智能剪辑",
            "📱 竖屏转换": "竖屏转换",
            "📝 添加字幕": "添加字幕"
        }
        
        for display, page_id in pages.items():
            if st.button(
                display,
                key=f"nav_{page_id}",
                use_container_width=True,
                type="primary" if st.session_state.current_page == page_id else "secondary"
            ):
                st.session_state.current_page = page_id
                st.rerun()
        
        st.markdown("---")
        
        # 当前库信息
        if st.session_state.selected_lib:
            lib_info = st.session_state.lib_info
            st.markdown(f"""
            ### 📊 当前库: {st.session_state.selected_lib}
            - 资产数: {lib_info.get('asset_count', 0)}
            - RAG条数: {lib_info.get('rag_count', 0)}
            """)
# 页面: 资产中心
def page_asset_center():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏭 资产中心 - 全局资产池</div>', unsafe_allow_html=True)
    st.markdown("全局资产池存储所有视频及其分析结果，是SSOT的核心")
    
    # 全局资产列表
    if not st.session_state.global_assets:
        st.info("全局资产池为空，请先上传视频")
    else:
        st.markdown(f"**共 {len(st.session_state.global_assets)} 个资产**")
        
        for asset in st.session_state.global_assets:
            with st.expander(f"📹 {asset.get('filename', '未知')}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"""
                    **MD5:** `{asset['md5']}`  
                    **大小:** {asset.get('size_formatted', '未知')}  
                    **创建时间:** {asset.get('created', '未知')}
                    """)
                
                with col2:
                    st.markdown("**分析状态:**")
                    if asset.get('has_asr'):
                        st.markdown("✅ ASR完成")
                    else:
                        st.markdown("⏳ ASR待处理")
                    
                    if asset.get('has_vlm'):
                        st.markdown("✅ VLM完成")
                    else:
                        st.markdown("⏳ VLM待处理")
                
                with col3:
                    if st.button("📋 详情", key=f"detail_{asset['md5']}"):
                        asset_info = server.get_asset_info(asset['md5'])
                        if asset_info['success']:
                            st.session_state['current_asset'] = asset_info['asset_info']
                            st.rerun()
                    
                    if st.button("🗑️ 删除", key=f"delete_{asset['md5']}"):
                        result = server.delete_asset(asset['md5'])
                        if result['success']:
                            st.success(f"✅ {result['message']}")
                            refresh_data()
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 显示当前选中的资产详情
    if 'current_asset' in st.session_state:
        asset = st.session_state['current_asset']
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown(f"### 资产详情: {asset['md5']}")
        
        tabs = st.tabs(["📄 转录", "🖼️ 关键帧", "✂️ 片段缓存"])
        
        with tabs[0]:
            if asset.get('has_transcript'):
                trans_path = os.path.join(asset['cache_dir'], "raw_trans.json")
                if os.path.exists(trans_path):
                    with open(trans_path, 'r', encoding='utf-8') as f:
                        trans_data = json.load(f)
                    st.json(trans_data[:5] if len(trans_data) > 5 else trans_data)
            else:
                st.info("暂无转录数据")
        
        with tabs[1]:
            keyframes = asset.get('keyframes', [])
            if keyframes:
                cols = st.columns(3)
                for i, kf in enumerate(keyframes[:6]):
                    with cols[i % 3]:
                        st.image(kf['path'], caption=f"帧 {kf['name']}")
            else:
                st.info("暂无关键帧")
        
        with tabs[2]:
            slices = asset.get('slices', [])
            if slices:
                for slice in slices:
                    st.markdown(f"""
                    **片段:** {slice['start']:.1f}s - {slice['end']:.1f}s  
                    大小: {server.format_size(slice['size'])}
                    """)
            else:
                st.info("暂无缓存片段")
        
        st.markdown('</div>', unsafe_allow_html=True)

# 页面: 数据准备
CATEGORY_OPTIONS = {
    "product_demo（产品/软件界面演示）": "product_demo",
    "general（通用视频）": "general",
}

def _show_processing_result(result):
    """统一的视频处理结果展示（上传路径与选择现有文件路径共用）。"""
    if result["success"]:
        st.success(f"✅ {result['message']}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("转录片段", result.get("transcript_count", 0))
        with col2:
            st.metric("关键帧", result.get("keyframes_count", 0))
        with col3:
            st.metric("视觉片段", result.get("visual_segments_count", 0))

        asr_status = result.get("asr_status", "unknown")
        if asr_status == "failed":
            st.warning("⚠️ 语音识别失败（本次未写入语音缓存，避免把失败当成“无语音”固化）；画面信息仍已入库，可重试。")
        elif asr_status == "no_speech":
            st.info("ℹ️ 本视频没有识别到语音；画面信息已按静默片段入库。")

        st.balloons()
        refresh_data()
    else:
        st.error(f"❌ {result['message']}")

def page_data_processing():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 数据准备</div>', unsafe_allow_html=True)
    st.markdown("上传视频并进行分析，结果存入全局资产池")

    # 处理选项（上传 与 选择现有文件 两条路径共用）
    opt1, opt2 = st.columns([2, 1])
    with opt1:
        category_label = st.selectbox(
            "视频分类",
            list(CATEGORY_OPTIONS.keys()),
            key="dp_category",
            help="product_demo：视觉模型按软件界面分析关键帧，界面状态文字（如“待复核”“待配置”）、按钮与人工环节会被原样保留；general：通用场景描述。"
        )
        category_tag = CATEGORY_OPTIONS[category_label]
    with opt2:
        force_rebuild = st.checkbox(
            "强制重建合并/清洗",
            key="dp_force_rebuild",
            help="合并/清洗逻辑升级后勾选；语音与视觉分析缓存仍会复用，不会重复调用付费接口"
        )
    
    # 拖拽上传区域
    st.markdown("### 📤 拖拽上传新视频")
    
    # 自定义拖拽上传样式
    st.markdown("""
    <style>
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background-color: #f8f9ff;
        margin: 10px 0 20px 0;
        transition: all 0.3s;
    }
    .upload-area:hover {
        background-color: #e8eaff;
        border-color: #764ba2;
    }
    .upload-icon {
        font-size: 48px;
        color: #667eea;
        margin-bottom: 10px;
    }
    .upload-text {
        color: #1e1e2f;
        font-weight: bold;
    }
    .upload-hint {
        color: #666;
        font-size: 12px;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 拖拽上传组件
    st.markdown("""
    <div class="upload-area">
        <div class="upload-icon">📁</div>
        <div class="upload-text">拖拽视频文件到此处</div>
        <div class="upload-hint">或点击下方按钮选择文件</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "选择视频文件",
        type=['mp4', 'mov', 'avi', 'mkv'],
        key="video_uploader",
        label_visibility="collapsed",
        accept_multiple_files=False
    )
    
    # 处理上传的文件
    if uploaded_file is not None:
        # 显示上传的文件信息
        file_size = uploaded_file.size / (1024 * 1024)  # 转换为 MB
        st.info(f"📹 已选择: {uploaded_file.name} ({file_size:.1f} MB)")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 处理上传的视频", use_container_width=True, type="primary"):
                with st.spinner("正在上传并处理视频..."):
                    # 保存上传的文件到 input 目录
                    file_path = os.path.join(server.INPUT_VIDEO_DIR, uploaded_file.name)
                    
                    # 避免文件名冲突
                    base, ext = os.path.splitext(uploaded_file.name)
                    counter = 1
                    while os.path.exists(file_path):
                        new_name = f"{base}_{counter}{ext}"
                        file_path = os.path.join(server.INPUT_VIDEO_DIR, new_name)
                        counter += 1
                    
                    # 保存文件
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # 处理视频
                    result = server.data_processing(os.path.basename(file_path), category=category_tag, force_rebuild=force_rebuild)
                    _show_processing_result(result)
    
    st.markdown("---")
    st.markdown("### 或者从现有文件选择")
    
    # 检查是否有现有视频文件
    if not st.session_state.video_files:
        st.warning("⚠️ 没有找到现有视频文件，请上传新视频")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # 现有文件选择
    video_options = {f['name']: f['name'] for f in st.session_state.video_files}
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 1. 选择视频文件")
        selected_video_display = st.selectbox(
            "选择要处理的视频",
            list(video_options.keys()),
            key="dp_video_select",
            label_visibility="collapsed",
            format_func=lambda x: safe_display_filename(x)
        )
        selected_video = video_options[selected_video_display]
    
    with col2:
        st.markdown("### 2. 处理选项")
        process_type = st.radio(
            "选择处理类型",
            ["完整处理", "仅语音转文字", "仅画面分析"],
            horizontal=True,
            key="process_type_radio"
        )
        st.caption(f"分类：`{category_tag}` ｜ 强制重建：{'是' if force_rebuild else '否'}（在页面顶部修改）")

    st.markdown("---")

    if st.button("🚀 开始处理选中的视频", use_container_width=True, type="primary"):
        if selected_video:
            with st.spinner("正在处理中，请稍候（首次处理需要转写+关键帧+视觉分析，可能需要几分钟）..."):
                result = server.data_processing(selected_video, category=category_tag, force_rebuild=force_rebuild)
                _show_processing_result(result)

    st.markdown('</div>', unsafe_allow_html=True)

# 页面: RAG构建
def page_rag_building():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 RAG知识库构建</div>', unsafe_allow_html=True)
    st.markdown(f"当前知识库: **{st.session_state.selected_lib}**")
    
    # 资产关联
    st.markdown("### 关联资产到知识库")
    if st.session_state.global_assets:
        # 获取当前库已有的资产
        lib_assets = st.session_state.lib_info.get('assets', {})
        current_md5s = list(lib_assets.keys())
        
        # 过滤出未关联的资产
        available_assets = [a for a in st.session_state.global_assets if a['md5'] not in current_md5s]
        
        if available_assets:
            asset_options = {a['md5']: a['filename'] for a in available_assets}
            selected_md5s = st.multiselect(
                "选择要关联的资产",
                options=list(asset_options.keys()),
                format_func=lambda x: f"{asset_options[x]} ({x[:8]}...)"
            )
            
            if st.button("🔗 关联选中资产"):
                progress_bar = st.progress(0)
                for i, md5 in enumerate(selected_md5s):
                    result = server.add_asset_to_library(st.session_state.selected_lib, md5)
                    if result['success']:
                        st.toast(f"✅ 已关联: {md5[:8]}...")
                    else:
                        st.error(f"❌ 关联失败: {result['message']}")
                    progress_bar.progress((i + 1) / len(selected_md5s))
                
                refresh_data()
                st.rerun()
        else:
            st.info("所有资产都已关联到此库")
    else:
        st.info("全局资产池为空，请先处理视频")
    
    st.markdown("---")
    
    tabs = st.tabs(["🧹 数据清洗", "🏗️ 构建知识库", "🔎 语义搜索", "📥 资料包导入"])
    
    # 数据清洗标签
    with tabs[0]:
        st.markdown("### 清洗转录数据为RAG格式")
        
        if not st.session_state.transcript_files:
            st.warning("没有找到转录文件，请先进行数据准备")
        else:
            transcript_options = {f['name']: f['name'] for f in st.session_state.transcript_files}
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_json_display = st.selectbox(
                    "选择要清洗的转录文件",
                    list(transcript_options.keys()),
                    key="rag_clean_select",
                    format_func=lambda x: safe_display_filename(x)
                )
                selected_json = transcript_options[selected_json_display]
            
            with col2:
                category = st.text_input("分类标签", value="general")
            
            if st.button("🧹 开始清洗", use_container_width=True):
                with st.spinner("正在清洗数据..."):
                    result = server.rag_building(
                        source_json=selected_json,
                        category=category,
                        lib_name=st.session_state.selected_lib
                    )
                    
                    if result["success"]:
                        st.success(f"✅ 清洗完成!")
                        st.info(f"生成文件: {result['rag_filename']}")
                        st.balloons()
                        refresh_data()
                    else:
                        st.error(f"❌ {result['message']}")
    
    # 构建知识库标签
    with tabs[1]:
        st.markdown("### 构建向量知识库")
        
        if not st.session_state.rag_files:
            st.warning("没有找到清洗后的RAG文件，请先进行数据清洗")
        else:
            rag_options = {f['name']: f['name'] for f in st.session_state.rag_files}
            
            selected_rag_display = st.selectbox(
                "选择RAG文件",
                list(rag_options.keys()),
                key="rag_build_select",
                format_func=lambda x: safe_display_filename(x)
            )
            selected_rag = rag_options[selected_rag_display]
            
            if st.button("🏗️ 开始构建", use_container_width=True, type="primary"):
                with st.spinner("正在构建知识库..."):
                    progress_bar = st.progress(0)
                    for i in range(101):
                        progress_bar.progress(i / 100)
                        time.sleep(0.02)
                    
                    result = server.rag_building(
                        rag_filename=selected_rag,
                        lib_name=st.session_state.selected_lib
                    )
                    
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.metric("总数据量", result.get("total_items", 0))
                        st.metric("向量库数量", result.get("collection_count", 0))
                        st.balloons()
                    else:
                        st.error(f"❌ {result['message']}")
    
    # 语义搜索标签
    with tabs[2]:
        st.markdown("### 语义搜索")
        
        query = st.text_input(
            "输入搜索内容",
            placeholder="例如: 找出切肉的画面、讲解关键技术的部分...",
            key="rag_query"
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            top_k = st.number_input("返回结果数", min_value=1, max_value=20, value=5)
        
        if query and st.button("🔎 搜索", use_container_width=True):
            with st.spinner("正在搜索..."):
                result = server.rag_search(
                    query=query,
                    top_k=top_k,
                    lib_name=st.session_state.selected_lib,
                    expand_context=True
                )
                
                if result["success"]:
                    st.success(f"找到 {len(result['results'])} 个相关结果")

                    for i, item in enumerate(result["results"]):
                        if item.get("kind") and item["kind"] != "video":
                            st.markdown(f"""
                            <div class="result-card">
                                <h4>结果 {i+1} ｜ {item.get('kind')} ｜ {item.get('title') or item.get('source_id') or ''}</h4>
                                <p><b>内容:</b> {item['content'][:200]}...</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-card">
                                <h4>结果 {i+1}</h4>
                                <p><b>时间:</b> {item['start']:.1f}s - {item['end']:.1f}s
                                    （证据片段 {item.get('evidence_start') or item['start']:.1f}s - {item.get('evidence_end') or item['end']:.1f}s）</p>
                                <p><b>类型:</b> {item['type']} ｜ <b>分类:</b> {item['category']}</p>
                                <p><b>内容:</b> {item['content'][:200]}...</p>
                                {'<p><b>✨ 已扩展上下文</b></p>' if item.get('is_expanded') else ''}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info(result["message"])

    # 资料包导入标签
    with tabs[3]:
        render_pack_import(lib_name=st.session_state.selected_lib, key_prefix="rag")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页面: 智能剪辑
def page_video_editing():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">✂️ 智能视频剪辑</div>', unsafe_allow_html=True)
    st.markdown(f"当前知识库: **{st.session_state.selected_lib}**")
    
    # 获取当前库的资产
    lib_assets = st.session_state.lib_info.get('assets', {})
    
    if not lib_assets:
        st.warning("当前知识库没有资产，请先关联资产")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    asset_options = {md5: info['filename'] for md5, info in lib_assets.items()}
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 1. 选择视频")
        selected_md5 = st.selectbox(
            "选择要剪辑的视频",
            options=list(asset_options.keys()),
            format_func=lambda x: asset_options[x],
            key="edit_asset_select"
        )
        
        if selected_md5:
            asset_path = lib_assets[selected_md5]['path']
            st.video(asset_path)
    
    with col2:
        st.markdown("### 2. 剪辑参数")
        max_duration = st.slider("最大时长（秒）", 10, 300, 60, 10)
    
    st.markdown("### 3. 剪辑要求")
    instruction = st.text_area(
        "输入剪辑要求",
        placeholder="例如: 找出切肉的画面、选择讲解关键技术的部分...",
        height=100
    )
    
    if st.button("🎬 开始智能剪辑", use_container_width=True, type="primary"):
        if not instruction:
            st.error("请输入剪辑要求")
        elif not selected_md5:
            st.error("请选择视频")
        else:
            with st.spinner("正在剪辑视频..."):
                filename = asset_options[selected_md5]
                result = server.video_editing(
                    video_filename=filename,
                    user_instruction=instruction,
                    max_duration=max_duration
                )
                
                if result["success"]:
                    st.success(f"✅ {result['message']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("原始片段", len(result.get("segments", [])))
                    with col2:
                        st.metric("选中片段", len(result.get("selected_segments", [])))
                    with col3:
                        st.metric("总时长", f"{result.get('total_duration', 0):.1f}秒")
                    
                    if result.get("output_path"):
                        st.video(result["output_path"])
                    
                    st.balloons()
                else:
                    st.error(f"❌ {result['message']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页面: 竖屏转换
def page_convert_to_vertical():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📱 横屏转竖屏</div>', unsafe_allow_html=True)
    
    if not st.session_state.video_files:
        st.warning("没有找到视频文件，请先上传视频")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    video_options = {f['name']: f['name'] for f in st.session_state.video_files}
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 1. 选择视频")
        selected_video_display = st.selectbox(
            "选择要转换的视频",
            list(video_options.keys()),
            key="vertical_video_select",
            format_func=lambda x: safe_display_filename(x)
        )
        selected_video = video_options[selected_video_display]
    
    with col2:
        st.markdown("### 2. 转换设置")
        conversion_method = st.radio(
            "转换方法",
            ["solid", "blur"],
            format_func=lambda x: "纯色填充" if x == "solid" else "模糊背景",
            horizontal=True
        )
    
    if st.button("🔄 开始转换", use_container_width=True, type="primary"):
        with st.spinner("正在转换..."):
            result = server.convert_to_vertical(
                video_filename=selected_video,
                method=conversion_method
            )
            
            if result["success"]:
                st.success(f"✅ {result['message']}")
                if result.get("output_path"):
                    st.video(result["output_path"])
                st.balloons()
            else:
                st.error(f"❌ {result['message']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页面: 添加字幕
def page_add_subtitles():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 添加字幕</div>', unsafe_allow_html=True)
    
    if not st.session_state.video_files:
        st.warning("没有找到视频文件，请先上传视频")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    if not st.session_state.transcript_files:
        st.warning("没有找到转录文件，请先进行数据准备")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    video_options = {f['name']: f['name'] for f in st.session_state.video_files}
    transcript_options = {f['name']: f['name'] for f in st.session_state.transcript_files}
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 1. 选择视频")
        selected_video_display = st.selectbox(
            "选择视频",
            list(video_options.keys()),
            key="subtitle_video_select",
            format_func=lambda x: safe_display_filename(x)
        )
        selected_video = video_options[selected_video_display]
    
    with col2:
        st.markdown("### 2. 选择字幕")
        selected_transcript_display = st.selectbox(
            "选择字幕文件",
            list(transcript_options.keys()),
            key="subtitle_transcript_select",
            format_func=lambda x: safe_display_filename(x)
        )
        selected_transcript = transcript_options[selected_transcript_display]
    
    st.markdown("### 3. 字幕样式")
    
    col1, col2 = st.columns(2)
    with col1:
        font_color = st.color_picker("字体颜色", "#FFFFFF")
        font_size = st.slider("字体大小", 12, 72, 36)
    
    with col2:
        bg_color = st.color_picker("背景颜色", "#000000")
        bg_opacity = st.slider("背景透明度", 0, 100, 50)
        position = st.select_slider("位置", options=["顶部", "中部", "底部"], value="底部")
    
    if st.button("✨ 生成字幕视频", use_container_width=True, type="primary"):
        with st.spinner("正在添加字幕..."):
            result = server.add_subtitles_to_video(
                video_filename=selected_video,
                transcript_filename=selected_transcript
            )
            
            if result["success"]:
                st.success(f"✅ {result['message']}")
                if result.get("output_path"):
                    st.video(result["output_path"])
                st.balloons()
            else:
                st.error(f"❌ {result['message']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== PromiseOS：资料包导入 与 销售答复 ====================

DEFAULT_PACK_DIR = os.path.join("..", "第二步_国内公开资料包_阿里云")
DEMO_QUESTION = "我们有一批中文扫描合同，要提取指定字段并接入现有系统。是不是开通就能上线，以后完全不用人工处理？"

STATUS_ICONS = {
    "有依据": "🟢",
    "需附条件": "🟡",
    "待技术确认": "🔴",
    "待确认（来源冲突）": "🟠",
}


def render_pack_import(lib_name: str, key_prefix: str = "pack"):
    """资料包导入（文档摘要 + 官方配图 → 当前知识库）。RAG构建页与销售答复页共用。"""
    st.markdown("### 📥 导入资料包")
    st.caption("按来源登记 JSON 的白名单导入：只有登记过的文档摘要和图片会进检索库；参考答案、人工画面观察不会混入。")

    pack_dir = st.text_input(
        "资料包目录（含 02_来源登记.json）",
        value=DEFAULT_PACK_DIR,
        key=f"{key_prefix}_dir",
        help="相对路径以 video_silce/ 为起点，例如 ../第二步_国内公开资料包_阿里云"
    )
    run_vlm = st.checkbox(
        "用视觉模型为图片生成界面状态描述（首次调用 Qwen-VL，有缓存）",
        value=True,
        key=f"{key_prefix}_vlm"
    )

    if st.button("📥 导入到当前知识库", key=f"{key_prefix}_btn", use_container_width=True):
        with st.spinner("导入中（首次需要调用 Embedding / 视觉模型）..."):
            result = server.ingest_doc_pack(lib_name, pack_dir, run_vlm=run_vlm)
        if result["success"]:
            st.success(f"✅ {result['message']}")
            st.json({k: v for k, v in result.items() if k not in ("success", "message", "lib_name", "pack_dir")})
            refresh_data()
        else:
            st.error(f"❌ {result['message']}")

    # 已登记来源概况
    reg = server.get_source_registry(lib_name)
    if reg.get("datasets"):
        st.markdown(
            f"**已登记**：资料集 {len(reg.get('datasets', {}))} 个 ｜ 来源 {len(reg.get('sources', {}))} 条 ｜ "
            f"图片 {len(reg.get('images', {}))} 张 ｜ 视频登记 {len(reg.get('videos', {}))} 条"
        )
        with st.expander("查看来源登记明细（状态 / 版本 / 审核人）"):
            rows = []
            for sid, s in reg.get("sources", {}).items():
                rows.append({
                    "来源ID": sid, "标题": s.get("title", ""), "API版本": s.get("api_version") or "",
                    "状态": s.get("status", ""), "审核人": s.get("human_reviewer") or "（待核验）",
                })
            for iid, im in reg.get("images", {}).items():
                rows.append({
                    "来源ID": iid, "标题": f"配图（{im.get('section', '')}）", "API版本": "",
                    "状态": ("视觉模型已描述" if im.get("vlm_description") else "未生成画面描述") + "；待核验",
                    "审核人": im.get("human_reviewer") or "（待核验）",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)


def render_evidence_item(ev: dict, key_prefix: str):
    """按证据类型展示依据：文档原文 / 官方配图 / 视频定位（证据片段 ↔ 展开上下文）。"""
    head = f"`{ev.get('kind')}` **{ev.get('title') or ev.get('source_id')}**"
    if ev.get("section"):
        head += f" ｜ {ev['section']}"
    if ev.get("api_version"):
        head += f" ｜ API {ev['api_version']}"
    st.markdown(head)
    review = "✅ 已有人工审核记录" if ev.get("human_reviewer") else "⚠️ 来源待核验（未有人工审核记录）"
    st.caption(f"{review} ｜ 来源状态：{ev.get('status') or '未登记'}")

    if ev.get("kind") == "video":
        video_path = server.get_video_pool_path(ev.get("video_md5", ""))
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


# 页面: 销售答复（PromiseOS）
def page_sales_reply():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💬 销售答复</div>', unsafe_allow_html=True)
    st.markdown(f"当前知识库: **{st.session_state.selected_lib}**")
    st.caption("把客户的一句话变成有依据、带条件、知道该问谁的答复草稿。模型负责拆解/提取/起草；"
               "引用是否真实存在、状态判定、失效来源门禁、冲突保留由程序完成。演示素材为公开资料整理，不代表企业合作。")

    with st.expander("📥 导入资料包（尚未导入时先做这步）"):
        render_pack_import(lib_name=st.session_state.selected_lib, key_prefix="sales")

    question = st.text_area("客户的问题（原话即可）", value=DEMO_QUESTION, height=90, key="q_text")
    known = st.text_input("销售已掌握的客户条件（可空）", placeholder="例如：客户用的是用友 U8，每天约 300 份扫描合同", key="q_known")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        top_k = st.slider("每个事项检索条数", 2, 8, 4, key="q_topk")
    with col_b:
        st.caption("生成前提：当前库已导入资料包；若要用视频做依据，先在“数据准备”处理视频（分类选 product_demo），再到“RAG构建”关联资产。")

    if st.button("🧭 生成答复草稿", key="q_run", use_container_width=True, type="primary"):
        if not question.strip():
            st.warning("请输入客户问题。")
        else:
            with st.spinner("拆解问题 → 检索资料 → 提取条件 → 程序校验 → 起草回复..."):
                result = server.prepare_sales_answer(question.strip(), st.session_state.selected_lib,
                                                     known_conditions=known.strip(), top_k=top_k)
            if result["success"]:
                st.session_state["promise_answer"] = result["answer"]
            else:
                st.error(f"❌ {result['message']}")

    answer = st.session_state.get("promise_answer")
    if not answer or answer.get("lib_name") != st.session_state.selected_lib:
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown("---")
    llm = answer.get("llm", {})
    counts = answer.get("status_counts", {})
    mc = st.columns(5)
    mc[0].metric("事项数", len(answer.get("claims", [])))
    mc[1].metric("🟢 有依据", counts.get("有依据", 0))
    mc[2].metric("🟡 需附条件", counts.get("需附条件", 0))
    mc[3].metric("🔴 待技术确认", counts.get("待技术确认", 0))
    mc[4].metric("🟠 来源冲突", counts.get("待确认（来源冲突）", 0))
    provider_note = f"文本模型提供方：{llm.get('provider') or '（未调用）'}"
    if llm.get("last_error"):
        provider_note += f" ｜ 备注：{llm['last_error']}"
    st.caption(
        f"耗时 {answer.get('elapsed_sec')}s ｜ 拆解 {'LLM' if llm.get('decomposed') else '启发式'} ｜ "
        f"提取 {'LLM' if llm.get('extracted') else '保守降级'} ｜ 起草 {'LLM' if llm.get('drafted') else '模板'} ｜ {provider_note}"
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
                    render_evidence_item(ev, key_prefix=f"ev_{idx}_{j}")

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
        if st.button("💾 保存答复日志", key="save_log", use_container_width=True):
            r = server.save_sales_answer_log(answer)
            if r["success"]:
                st.success(f"✅ {r['message']}")
            else:
                st.error(f"❌ {r['message']}")
    with b2:
        st.download_button(
            "⬇️ 导出待确认清单 (JSON)",
            data=json.dumps({"question": answer["question"], "ask_customer": answer["ask_customer"], "ask_tech": answer["ask_tech"]}, ensure_ascii=False, indent=2),
            file_name="pending_confirmations.json",
            mime="application/json",
            key="dl_pending",
            use_container_width=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


# 主函数
def main():
    init_session_state()
    refresh_data()
    
    # 渲染侧边栏
    render_sidebar()
    
    # 主要内容区域
    st.markdown(f'<h1 style="text-align: center; color: #1e1e2f;">🎬 {st.session_state.current_page}</h1>', 
                unsafe_allow_html=True)
    
    # 根据当前页面显示内容
    if st.session_state.current_page == "资产中心":
        page_asset_center()
    elif st.session_state.current_page == "数据准备":
        page_data_processing()
    elif st.session_state.current_page == "RAG构建":
        page_rag_building()
    elif st.session_state.current_page == "销售答复":
        page_sales_reply()
    elif st.session_state.current_page == "智能剪辑":
        page_video_editing()
    elif st.session_state.current_page == "竖屏转换":
        page_convert_to_vertical()
    elif st.session_state.current_page == "添加字幕":
        page_add_subtitles()

if __name__ == "__main__":
    main()