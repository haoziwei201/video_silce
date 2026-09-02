# streamlit_app.py
"""
streamlit_app.py - 视频智能剪辑工具Web界面
完整版，修复视频路径问题和搜索显示问题
"""

import streamlit as st
import os
import json
import urllib.parse

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
    [data-testid="stSidebar"] {
        background-color: #e0e5eb;
        padding: 20px 10px;
    }
    
    .badge-success { background: #d4edda; color: #155724; }
    .badge-warning { background: #fff3cd; color: #856404; }
    
    .content-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .section-title::after {
        content: '';
        flex: 1;
        height: 2px;
        background: linear-gradient(90deg, #667eea, transparent);
        margin-left: 20px;
    }
    
    .result-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* ===== 按钮样式 ===== */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    .stButton > button::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 2s, height 0.6s;
    }
    
    .stButton > button:not(:hover)::after {
    width: 0 !important;        /* 鼠标移开时立即消失 */
    height: 0 !important;
    transition: none;           /* 移开时无过渡动画 */
    }
    
    .stButton > button:hover::after {
        width: 2000px;
        height: 300px;
    }
    
    /* 次级按钮 */
    .stButton > button.secondary {
        background: white;
        color: #667eea;
        border: 2px solid #667eea;
        box-shadow: none;
    }
    
    .stButton > button.secondary:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
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

# 初始化session state
def init_session_state():
    if 'current_module' not in st.session_state:
        st.session_state.current_module = "视频资源"
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "资产中心"
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

# 顶栏导航
def render_top_nav():
    # 使用列来限制宽度并居中（按钮样式统一由页面顶部全局 CSS 控制）
    _, col_nav, _ = st.columns([1, 1, 1])
    
    with col_nav:
        # 创建内层按钮列
        c1, c2 = st.columns(2)
        
        with c1:
            is_kb = st.session_state.current_module == "知识库"
            if st.button(
                "📚 知识库", 
                key="btn_nav_kb", 
                use_container_width=True,
                type="primary" if is_kb else "secondary"
            ):
                if not is_kb:
                    st.session_state.current_module = "知识库"
                    st.session_state.current_page = "知识库概览"
                    st.rerun()
        
        with c2:
            is_vr = st.session_state.current_module == "视频资源"
            if st.button(
                "📹 视频资源", 
                key="btn_nav_vr", 
                use_container_width=True,
                type="primary" if is_vr else "secondary"
            ):
                if not is_vr:
                    st.session_state.current_module = "视频资源"
                    st.session_state.current_page = "资产中心"
                    st.rerun()

# 侧边栏
def render_sidebar():
    with st.sidebar:
        st.image("logo.psd", width=500)
        st.markdown("---")
        
        if st.session_state.current_module == "知识库":
            st.markdown("### 📚 知识库管理")
            
            kb_pages = {
                "🏠 知识库概览": "知识库概览",
                "📥 视频导入": "视频导入",
                "🔍 语义搜索": "语义搜索",
                "✂️ 智能剪辑": "智能剪辑"
            }
            
            for display, page_id in kb_pages.items():
                if st.button(
                    display,
                    key=f"nav_{page_id}",
                    use_container_width=True,
                    type="primary" if st.session_state.current_page == page_id else "secondary"
                ):
                    st.session_state.current_page = page_id
                    st.rerun()
            
            st.markdown("---")
            if st.session_state.selected_lib:
                lib_info = st.session_state.lib_info
                st.markdown(f"""
                ### 📊 当前库: {server.get_lib_display_name(st.session_state.selected_lib)}
                - 资产数: {lib_info.get('asset_count', 0)}
                - RAG条数: {lib_info.get('rag_count', 0)}
                """)
                
        else: # 视频资源
            st.markdown("### 📹 视频资源管理")
            
            vr_pages = {
                "📊 数据准备": "数据准备",
                "🏭 资产中心": "资产中心"
            }
            
            for display, page_id in vr_pages.items():
                if st.button(
                    display,
                    key=f"nav_{page_id}",
                    use_container_width=True,
                    type="primary" if st.session_state.current_page == page_id else "secondary"
                ):
                    st.session_state.current_page = page_id
                    st.rerun()
            
            st.markdown("---")
            # 在视频资源模块保留系统工具
            with st.expander("🛠️ 系统工具", expanded=False):
                if st.button("📦 迁移视频到池", use_container_width=True):
                    with st.spinner("正在迁移视频..."):
                        result = server.migrate_videos_to_pool()
                        st.success(f"迁移完成: {len(result['migrated'])} 成功, {len(result['skipped'])} 已存在, {len(result['failed'])} 失败")
                        refresh_data()
                
                if st.button("🔧 检查缺失视频", use_container_width=True):
                    if st.session_state.selected_lib:
                        with st.spinner("正在检查..."):
                            result = server.fix_missing_video_links(st.session_state.selected_lib)
                            if 'error' in result:
                                st.error(f"检查失败: {result['error']}")
                            else:
                                st.info(f"总计 {result.get('total', 0)} 条记录，缺失 {result.get('missing', 0)} 个视频")

# 页面: 知识库概览
def page_library_overview():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏠 知识库概览</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📚 知识库列表")
        if st.session_state.libraries:
            for lib in st.session_state.libraries:
                lib_info = server.get_library_info(lib)
                is_selected = (lib == st.session_state.selected_lib)
                
                with st.container():
                    c1, c2, c3 = st.columns([6, 2, 1])
                    with c1:
                        if is_selected:
                            st.markdown(f"**🔵 {server.get_lib_display_name(lib)}** (当前已选)")
                        else:
                            st.markdown(f"📁 {server.get_lib_display_name(lib)}")
                    with c2:
                        st.markdown(f"📊 资产: {lib_info.get('asset_count', 0)} | RAG: {lib_info.get('rag_count', 0)}")
                    with c3:
                        if not is_selected:
                            if st.button("选择", key=f"select_page_{lib}"):
                                st.session_state.selected_lib = lib
                                refresh_data()
                                st.rerun()
                        else:
                            st.markdown("✅")
                    st.markdown("---")
        else:
            st.info("暂无知识库，请先创建")

    with col2:
        st.markdown("### ⚙️ 管理操作")
        with st.expander("➕ 新建知识库", expanded=True):
            new_lib = st.text_input("新库名称", key="overview_new_lib", placeholder="输入名称")
            if st.button("创建", key="overview_create_btn", use_container_width=True):
                if new_lib:
                    result = server.create_library(new_lib)
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.session_state.selected_lib = new_lib
                        refresh_data()
                        st.rerun()
                    else:
                        st.error(f"❌ {result['message']}")
        
        if st.session_state.libraries:
            with st.expander("🗑️ 删除知识库", expanded=False):
                lib_to_delete = st.selectbox(
                    "选择要删除的库",
                    [l for l in st.session_state.libraries if l != "default_lib"],
                    format_func=server.get_lib_display_name,
                    key="overview_delete_select"
                )
                if st.button("确认删除", key="overview_delete_btn", use_container_width=True):
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
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页面: 资产中心
def page_asset_center():
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    
    # 标题
    st.markdown("""
    <div class="section-title">
        <span>🏭 资产中心</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("全局资产池存储所有视频及其分析结果，是SSOT的核心")
    
    if not st.session_state.global_assets:
        st.info("✨ 全局资产池为空，请先上传视频")
    else:
        # 统计卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon">📹</div>
                <div class="metric-value">{}</div>
                <div class="metric-label">总资产</div>
            </div>
            """.format(len(st.session_state.global_assets)), unsafe_allow_html=True)
        
        with col2:
            asr_count = sum(1 for a in st.session_state.global_assets if a.get('has_asr'))
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon">🎤</div>
                <div class="metric-value">{}</div>
                <div class="metric-label">ASR完成</div>
            </div>
            """.format(asr_count), unsafe_allow_html=True)
        
        with col3:
            cleaned_count = sum(1 for a in st.session_state.global_assets if a.get('has_cleaned'))
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon">✨</div>
                <div class="metric-value">{}</div>
                <div class="metric-label">清洗完成</div>
            </div>
            """.format(cleaned_count), unsafe_allow_html=True)
        
        with col4:
            total_size = sum(os.path.getsize(a['path']) for a in st.session_state.global_assets if os.path.exists(a['path'])) / (1024**3)
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon">💾</div>
                <div class="metric-value">{:.2f}</div>
                <div class="metric-label">总大小(GB)</div>
            </div>
            """.format(total_size), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 资产列表
        for asset in st.session_state.global_assets:
            with st.expander(f"📹 {asset.get('display_name', asset.get('filename', '未知'))}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    **MD5:** `{asset['md5'][:8]}...{asset['md5'][-8:]}`  
                    **原名:** {asset.get('display_name', '未知')}  
                    **大小:** {asset.get('size_formatted', '未知')}
                    """)
                
                with col2:
                    # 状态徽章
                    status_html = ""
                    if asset.get('has_asr'):
                        status_html += '<span class="badge badge-success">✓ ASR完成</span> '
                    else:
                        status_html += '<span class="badge badge-warning">⏳ ASR待处理</span> '
                    
                    if asset.get('has_cleaned'):
                        status_html += '<span class="badge badge-success">✓ 清洗完成</span>'
                    else:
                        status_html += '<span class="badge badge-warning">⏳ 清洗待处理</span>'
                    
                    st.markdown(f"**状态:** {status_html}", unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("📋 详情", key=f"detail_{asset['md5']}", use_container_width=True):
                            asset_info = server.get_asset_info(asset['md5'])
                            if asset_info['success']:
                                st.session_state['current_asset'] = asset_info['asset_info']
                                st.rerun()
                    
                    with col_b:
                        if st.button("🗑️ 删除", key=f"delete_{asset['md5']}", use_container_width=True):
                            result = server.delete_asset(asset['md5'])
                            if result['success']:
                                st.success(f"✅ {result['message']}")
                                refresh_data()
                                st.rerun()
                            else:
                                st.error(f"❌ {result['message']}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 资产详情
    if 'current_asset' in st.session_state:
        asset = st.session_state['current_asset']
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown(f"### 📋 资产详情: {asset.get('original_name', asset['md5'])}")
        
        if not asset.get('video_exists', False):
            st.warning("⚠️ 视频文件不存在，请运行迁移工具")
        
        tabs = st.tabs(["📄 元数据", "📝 转录", "🖼️ 关键帧", "✂️ 片段缓存"])
        
        with tabs[0]:
            if asset.get('metadata'):
                st.json(asset['metadata'])
            else:
                st.info("暂无元数据")
        
        with tabs[1]:
            if asset.get('has_raw_trans'):
                trans_path = os.path.join(asset['cache_dir'], "raw_trans.json")
                if os.path.exists(trans_path):
                    with open(trans_path, 'r', encoding='utf-8') as f:
                        trans_data = json.load(f)
                    st.json(trans_data[:5] if len(trans_data) > 5 else trans_data)
            else:
                st.info("暂无转录数据")
        
        with tabs[2]:
            keyframes = asset.get('keyframes', [])
            if keyframes:
                cols = st.columns(3)
                for i, kf in enumerate(keyframes[:6]):
                    with cols[i % 3]:
                        st.image(kf['path'], caption=f"帧 {kf['name']}")
            else:
                st.info("暂无关键帧")
        
        with tabs[3]:
            slices = asset.get('slices', [])
            if slices:
                for slice_item in slices[:5]:
                    st.markdown(f"""
                    **片段:** {slice_item['start']:.1f}s - {slice_item['end']:.1f}s  
                    大小: {server.format_size(slice_item['size'])}
                    """)
            else:
                st.info("暂无缓存片段")
        
        if st.button("关闭详情", use_container_width=True):
            del st.session_state['current_asset']
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)


# 页面: 数据准备
def page_data_processing():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 数据准备</div>', unsafe_allow_html=True)
    st.markdown("上传视频并进行分析，结果存入全局资产池")
    
    st.caption(f"📌 当前上传限制: {st.get_option('server.maxUploadSize')}MB（可在 .streamlit/config.toml 中调整）")
    
    st.markdown("### 📤 拖拽上传新视频")
    
    st.markdown("""
    <div class="upload-area">
        <div class="upload-icon">📁</div>
        <div class="upload-text">拖拽视频文件到此处</div>
        <div class="upload-hint">或点击下方按钮选择文件（支持最大2GB）</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "选择视频文件",
        type=['mp4', 'mov', 'avi', 'mkv'],
        key="video_uploader",
        label_visibility="collapsed",
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        file_size_gb = file_size_mb / 1024
        size_display = f"{file_size_mb:.1f} MB" if file_size_mb < 1024 else f"{file_size_gb:.2f} GB"
        
        st.info(f"📹 已选择: {uploaded_file.name} ({size_display})")
        
        max_size = st.get_option('server.maxUploadSize')
        if file_size_mb > max_size * 0.9:
            st.warning(f"⚠️ 文件较大，接近上传限制 {max_size}MB，建议调整 config.toml 增加限制")
        
        category = st.text_input("分类标签", value="general", key="upload_category")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 处理上传的视频", use_container_width=True, type="primary"):
                with st.spinner("正在上传并处理视频（大文件可能需要较长时间）..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    file_path = os.path.join(server.INPUT_VIDEO_DIR, uploaded_file.name)
                    
                    base, ext = os.path.splitext(uploaded_file.name)
                    counter = 1
                    while os.path.exists(file_path):
                        new_name = f"{base}_{counter}{ext}"
                        file_path = os.path.join(server.INPUT_VIDEO_DIR, new_name)
                        counter += 1
                    
                    chunk_size = 32 * 1024 * 1024
                    bytes_written = 0
                    total_bytes = uploaded_file.size
                    
                    with open(file_path, 'wb') as f:
                        while True:
                            chunk = uploaded_file.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            bytes_written += len(chunk)
                            progress = bytes_written / total_bytes
                            progress_bar.progress(progress)
                            status_text.text(f"上传进度: {progress:.1%} ({bytes_written / (1024*1024):.1f}MB / {file_size_mb:.1f}MB)")
                    
                    status_text.text("上传完成，开始处理视频...")
                    
                    result = server.data_processing(os.path.basename(file_path), category)
                    
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("转录片段", result.get("transcript_count", 0))
                        with col2:
                            st.metric("关键帧", result.get("keyframes_count", 0))
                        with col3:
                            st.metric("视觉片段", result.get("visual_segments_count", 0))
                        
                        st.balloons()
                        refresh_data()
                    else:
                        st.error(f"❌ {result['message']}")
    
    st.markdown("---")
    st.markdown("### 或者从现有文件选择")
    
    if not st.session_state.video_files:
        st.warning("⚠️ 没有找到现有视频文件，请上传新视频")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
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
        category = st.text_input("分类标签", value="general", key="existing_category")
    
    st.markdown("---")
    
    if st.button("🚀 开始处理选中的视频", use_container_width=True, type="primary"):
        if selected_video:
            with st.spinner("正在处理中，请稍候..."):
                result = server.data_processing(selected_video, category)
                
                if result["success"]:
                    st.success(f"✅ {result['message']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("转录片段", result.get("transcript_count", 0))
                    with col2:
                        st.metric("关键帧", result.get("keyframes_count", 0))
                    with col3:
                        st.metric("视觉片段", result.get("visual_segments_count", 0))
                    
                    st.balloons()
                    refresh_data()
                else:
                    st.error(f"❌ {result['message']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页面: 视频导入
def page_video_import():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📥 视频导入</div>', unsafe_allow_html=True)
    st.markdown(f"当前知识库: **{server.get_lib_display_name(st.session_state.selected_lib)}**")
    
    st.markdown("### 关联资产到知识库")
    if st.session_state.global_assets:
        lib_assets = st.session_state.lib_info.get('assets', {})
        current_md5s = list(lib_assets.keys())
        
        available_assets = []
        for asset in st.session_state.global_assets:
            if asset['md5'] not in current_md5s:
                available_assets.append({
                    "md5": asset['md5'],
                    "display_name": asset.get('display_name', asset['filename']),
                    "filename": asset['filename']
                })
        
        if available_assets:
            asset_options = {a['display_name']: a['md5'] for a in available_assets}
            
            selected_display_names = st.multiselect(
                "选择要关联的资产",
                options=sorted(asset_options.keys()),
                format_func=lambda x: x
            )
            
            selected_md5s = [asset_options[name] for name in selected_display_names]
            
            if st.button("🔗 关联选中资产"):
                status_text = st.empty()
                status_text.text(f"正在关联资产... 0/{len(selected_md5s)}")
                progress_bar = st.progress(0)
                
                for i, md5 in enumerate(selected_md5s):
                    status_text.text(f"正在关联资产: {selected_display_names[i]} ({i+1}/{len(selected_md5s)})")
                    
                    result = server.add_asset_to_library(st.session_state.selected_lib, md5)
                    if result['success']:
                        st.toast(f"✅ 已关联: {selected_display_names[i]}")
                    else:
                        st.error(f"❌ 关联失败: {result['message']}")
                    
                    progress_bar.progress((i + 1) / len(selected_md5s))
                
                status_text.text("关联完成！正在刷新数据...")
                refresh_data()
                st.rerun()
        else:
            st.info("所有资产都已关联到此库")
    else:
        st.info("全局资产池为空，请先处理视频")
    
    st.markdown("---")
    
    tabs = st.tabs(["🧹 数据清洗", "🏗️ 构建知识库"])
    
    with tabs[0]:
        st.markdown("### 清洗转录数据为RAG格式")
        
        if not st.session_state.transcript_files and not st.session_state.global_assets:
            st.warning("没有找到转录文件，请先进行数据准备")
        else:
            options = []
            if st.session_state.transcript_files:
                for f in st.session_state.transcript_files:
                    options.append(f['name'])
            
            for asset in st.session_state.global_assets:
                if asset.get('has_raw_trans'):
                    options.append(f"[资产] {asset.get('display_name', asset['filename'])}")
            
            if options:
                selected_option = st.selectbox(
                    "选择要清洗的数据源",
                    options,
                    key="rag_clean_select"
                )
                
                category = st.text_input("分类标签", value="general", key="clean_category")
                
                if st.button("🧹 开始清洗", use_container_width=True):
                    with st.spinner("正在清洗数据..."):
                        if selected_option.startswith("[资产]"):
                            md5 = next((a['md5'] for a in st.session_state.global_assets if a.get('display_name', a['filename']) in selected_option), None)
                            if md5:
                                merged_path = os.path.join(server.GLOBAL_CACHE_DIR, md5, "merged_raw.json")
                                result = server.rag_building(
                                    source_json=merged_path,
                                    category=category,
                                    lib_name=st.session_state.selected_lib
                                )
                        else:
                            result = server.rag_building(
                                source_json=selected_option,
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
            else:
                st.warning("没有可用的数据源")
    
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
                    result = server.rag_building(
                        rag_filename=selected_rag,
                        lib_name=st.session_state.selected_lib
                    )
                    
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.metric("总数据量", result.get("total_items", 0))
                        st.metric("向量库数量", result.get("collection_count", 0))
                        st.balloons()
                        refresh_data()
                    else:
                        st.error(f"❌ {result['message']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页面: 语义搜索
def page_semantic_search():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 语义搜索</div>', unsafe_allow_html=True)
    st.markdown(f"当前知识库: **{server.get_lib_display_name(st.session_state.selected_lib)}**")
    
    query = st.text_input(
        "输入搜索内容",
        placeholder="例如: 找出切肉的画面、讲解关键技术的部分...",
        key="rag_query"
    )
    
    col1, col2 = st.columns([3, 1])
    with col2:
        top_k = st.number_input("返回结果数", min_value=1, max_value=20, value=5)
    
    with st.expander("高级选项"):
        expand_context = st.checkbox("扩展上下文", value=True)
    
    if query and st.button("🔎 搜索", use_container_width=True):
        with st.spinner("正在搜索..."):
            result = server.rag_search(
                query=query,
                top_k=top_k,
                lib_name=st.session_state.selected_lib,
                expand_context=expand_context
            )
            
            if result["success"]:
                st.success(f"找到 {len(result['results'])} 个相关结果")
                
                for i, item in enumerate(result["results"]):
                    display_name = item.get('original_name', item.get('video_name', '未知视频'))
                    if not item.get('video_exists', False):
                        display_name = f"⚠️ {display_name} (视频缺失)"
                    
                    with st.expander(f"结果 {i+1} - {display_name}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**时间范围:** {item['start']:.1f}s - {item['end']:.1f}s")
                            st.markdown(f"**类型:** {item['type']}")
                            st.markdown(f"**视频MD5:** `{item['video_md5']}`")
                        with col2:
                            st.markdown(f"**分类:** {item['category']}")
                            if item.get('is_expanded'):
                                st.markdown("**✨ 已扩展上下文**")
                        
                        st.markdown("**内容:**")
                        st.markdown(f">{item['content']}")
                        
                        st.markdown("**📹 相关视频:**")
                        
                        video_path = item.get('video_path')
                        
                        if video_path and os.path.exists(video_path):
                            st.success(f"✅ 找到视频: {os.path.basename(video_path)}")
                            
                            vid_col1, vid_col2 = st.columns([3, 1])
                            
                            with vid_col1:
                                st.video(
                                    video_path,
                                    start_time=int(item['start'])
                                )
                            
                            with vid_col2:
                                st.markdown(f"**视频名称:**")
                                st.caption(item.get('original_name', item.get('video_name', '未知')))
                                st.markdown(f"**文件大小:** {os.path.getsize(video_path)/1024/1024:.1f}MB")
                        else:
                            st.error(f"❌ 视频文件不存在")
            else:
                st.info(result["message"])
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页面: 智能剪辑 (整合功能)
def page_video_editing():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">✂️ 智能剪辑</div>', unsafe_allow_html=True)
    st.markdown(f"当前知识库: **{server.get_lib_display_name(st.session_state.selected_lib)}**")
    
    tabs = st.tabs(["🎬 视频剪辑", "📱 竖屏转换", "📝 添加字幕"])
    
    with tabs[0]:
        _render_clip_tool()
    
    with tabs[1]:
        _render_vertical_tool()
    
    with tabs[2]:
        _render_subtitle_tool()
        
    st.markdown('</div>', unsafe_allow_html=True)

# 工具: 剪辑
def _render_clip_tool():
    st.markdown("### 🎬 智能视频剪辑")
    
    lib_info = server.get_library_info(st.session_state.selected_lib)
    assets = lib_info.get('assets', {})
    
    if not assets:
        st.warning("当前知识库没有资产，请先关联资产")
        return
    
    asset_options = {}
    for md5, info in assets.items():
        display_name = info.get('display_name', info.get('filename', md5))
        if not info.get('exists', True):
            display_name = f"⚠️ {display_name} (视频缺失)"
        asset_options[display_name] = {
            'md5': md5,
            'path': info.get('path'),
            'filename': info.get('filename'),
            'display_name': info.get('display_name', info.get('filename', md5)),
            'exists': info.get('exists', False)
        }
    
    sorted_display_names = sorted(asset_options.keys())
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 1. 选择视频（可多选）")
        selected_displays = st.multiselect(
            "选择要剪辑的视频",
            options=sorted_display_names,
            key="edit_asset_select",
            label_visibility="collapsed",
            format_func=lambda x: x
        )
        
        if selected_displays:
            first_display = selected_displays[0]
            selected_asset = asset_options[first_display]
            asset_path = selected_asset['path']
            if asset_path and os.path.exists(asset_path):
                st.video(asset_path)
            else:
                st.warning("视频文件缺失，无法预览")
        else:
            st.info("👆 请从上方选择至少一个视频")
    
    with col2:
        st.markdown("#### 2. 剪辑参数")
        max_duration = st.slider("最大时长（秒）", 10, 300, 60, 10)
    
    st.markdown("#### 3. 剪辑要求")
    instruction = st.text_area(
        "输入剪辑要求",
        placeholder="例如: 找出切肉的画面、选择讲解关键技术的部分...",
        height=100,
        key="clip_instruction"
    )
    
    btn_label = "🎬 开始智能剪辑"
    if selected_displays:
        btn_label = f"🎬 批量剪辑 {len(selected_displays)} 个视频"
    
    if st.button(btn_label, use_container_width=True, type="primary"):
        if not instruction:
            st.error("请输入剪辑要求")
        elif not selected_displays:
            st.error("请至少选择一个视频")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            total = len(selected_displays)
            for i, display_name in enumerate(selected_displays):
                status_text.text(f"正在处理 ({i+1}/{total}): {display_name}")
                asset = asset_options[display_name]
                md5 = asset['md5']
                original_filename = asset['display_name']
                
                input_video_path = os.path.join(server.INPUT_VIDEO_DIR, original_filename)
                if not os.path.exists(input_video_path):
                    results.append({
                        "video": display_name,
                        "success": False,
                        "message": f"原始视频文件不存在: {input_video_path}",
                        "output_path": None,
                        "segments": [],
                        "selected_segments": [],
                        "total_duration": 0
                    })
                    progress_bar.progress((i + 1) / total)
                    continue
                
                result = server.video_editing(
                    video_filename=original_filename,
                    user_instruction=instruction,
                    max_duration=max_duration
                )
                
                results.append({
                    "video": display_name,
                    "success": result.get("success", False),
                    "message": result.get("message", ""),
                    "output_path": result.get("output_path"),
                    "segments": result.get("segments", []),
                    "selected_segments": result.get("selected_segments", []),
                    "total_duration": result.get("total_duration", 0)
                })
                
                progress_bar.progress((i + 1) / total)
            
            status_text.text("批量处理完成！")
            success_count = sum(1 for r in results if r["success"])
            st.success(f"✅ 处理完成，成功 {success_count} 个，失败 {total - success_count} 个")
            
            import pandas as pd
            df = pd.DataFrame(results)
            df['output_file'] = df['output_path'].apply(lambda x: os.path.basename(x) if x else '')
            df_display = df[['video', 'success', 'message', 'output_file']]
            st.dataframe(df_display, use_container_width=True)
            
            with st.expander("查看成功生成的视频预览"):
                for res in results:
                    if res["success"] and res.get("output_path") and os.path.exists(res["output_path"]):
                        st.markdown(f"**{res['video']}**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("原始片段", len(res.get("segments", [])))
                        with col2:
                            st.metric("选中片段", len(res.get("selected_segments", [])))
                        with col3:
                            st.metric("总时长", f"{res.get('total_duration', 0):.1f}秒")
                        st.video(res["output_path"])
                        st.markdown("---")
            
            if success_count > 0:
                st.balloons()

# 工具: 竖屏转换
def _render_vertical_tool():
    st.markdown("### 📱 横屏转竖屏")
    
    if not st.session_state.video_files:
        st.warning("没有找到视频文件，请先上传视频")
        return
    
    video_options = {f['name']: f['name'] for f in st.session_state.video_files}
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 1. 选择视频")
        selected_video_display = st.selectbox(
            "选择要转换的视频",
            list(video_options.keys()),
            key="vertical_video_select",
            format_func=lambda x: safe_display_filename(x)
        )
        selected_video = video_options[selected_video_display]
    
    with col2:
        st.markdown("#### 2. 转换设置")
        conversion_method = st.radio(
            "转换方法",
            ["solid", "blur", "static"],
            format_func=lambda x: {
                "solid": "纯色填充",
                "blur": "模糊背景",
                "static": "静态背景"
            }[x],
            horizontal=True,
            key="vertical_method"
        )
    
    if st.button("🔄 开始转换", use_container_width=True, type="primary", key="vertical_btn"):
        with st.spinner("正在转换..."):
            result = server.convert_to_vertical(
                video_filename=selected_video,
                method=conversion_method
            )
            
            if result["success"]:
                st.success(f"✅ {result['message']}")
                if result.get("output_path") and os.path.exists(result["output_path"]):
                    st.video(result["output_path"])
                st.balloons()
            else:
                st.error(f"❌ {result['message']}")

# 工具: 添加字幕
def _render_subtitle_tool():
    st.markdown("### 📝 添加字幕")
    
    if not st.session_state.video_files:
        st.warning("没有找到视频文件，请先上传视频")
        return
    
    video_options = {f['name']: f['name'] for f in st.session_state.video_files}
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 1. 选择视频")
        selected_video_display = st.selectbox(
            "选择视频",
            list(video_options.keys()),
            key="subtitle_video_select",
            format_func=lambda x: safe_display_filename(x)
        )
        selected_video = video_options[selected_video_display]
        
        video_path = os.path.join(server.INPUT_VIDEO_DIR, selected_video)
        if os.path.exists(video_path):
            st.video(video_path)
    
    with col2:
        st.markdown("#### 2. 选择字幕")
        
        transcript_options = []
        if st.session_state.transcript_files:
            for f in st.session_state.transcript_files:
                transcript_options.append(f['name'])
        
        video_md5 = server.get_file_hash(os.path.join(server.INPUT_VIDEO_DIR, selected_video))
        if video_md5:
            cache_path = os.path.join(server.GLOBAL_CACHE_DIR, video_md5, "raw_trans.json")
            if os.path.exists(cache_path):
                transcript_options.append(f"[缓存] {video_md5[:8]}_raw_trans.json")
        
        if transcript_options:
            selected_transcript = st.selectbox(
                "选择字幕文件",
                transcript_options,
                key="subtitle_transcript_select"
            )
        else:
            st.warning("没有找到字幕文件")
            selected_transcript = None
    
    st.markdown("#### 3. 字幕样式")
    
    col1, col2 = st.columns(2)
    with col1:
        font_color = st.color_picker("字体颜色", "#FFFFFF", key="sub_font_color")
        font_size = st.slider("字体大小", 12, 72, 36, key="sub_font_size")
    
    with col2:
        bg_color = st.color_picker("背景颜色", "#000000", key="sub_bg_color")
        bg_opacity = st.slider("背景透明度", 0, 100, 50, key="sub_bg_opacity")
        position = st.select_slider("位置", options=["顶部", "中部", "底部"], value="底部", key="sub_pos")
    
    position_map = {"顶部": "top", "中部": "middle", "底部": "bottom"}

    if st.button("✨ 生成字幕视频", use_container_width=True, type="primary", key="sub_btn"):
        if selected_video and selected_transcript:
            with st.spinner("正在添加字幕..."):
                result = server.add_subtitles_to_video(
                    video_filename=selected_video,
                    transcript_filename=selected_transcript if not selected_transcript.startswith("[缓存]") else None,
                    font_size=font_size,
                    font_color=font_color,
                    bg_color=bg_color,
                    bg_opacity=bg_opacity,
                    position=position_map[position],
                )
                
                if result["success"]:
                    st.success(f"✅ {result['message']}")
                    if result.get("output_path") and os.path.exists(result["output_path"]):
                        st.video(result["output_path"])
                    st.balloons()
                else:
                    st.error(f"❌ {result['message']}")

# 主函数
def main():
    init_session_state()
    refresh_data()
    
    render_top_nav()
    render_sidebar()
    
    st.markdown(f'<h1 style="text-align: center; color: #1e1e2f;">🎬 {st.session_state.current_page}</h1>', 
                unsafe_allow_html=True)
    
    # 知识库模块
    if st.session_state.current_page == "知识库概览":
        page_library_overview()
    elif st.session_state.current_page == "视频导入":
        page_video_import()
    elif st.session_state.current_page == "语义搜索":
        page_semantic_search()
    elif st.session_state.current_page == "智能剪辑":
        page_video_editing()
    
    # 视频资源模块
    elif st.session_state.current_page == "数据准备":
        page_data_processing()
    elif st.session_state.current_page == "资产中心":
        page_asset_center()

if __name__ == "__main__":
    main()
