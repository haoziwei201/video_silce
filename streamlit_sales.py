# streamlit_sales.py
"""
诺界 · PromiseOS ｜ 以销售对话为主入口的界面
====================================================
主区域是一段对话：销售贴入客户原话 → 系统给出建议回复 + 承诺卡 + 待确认清单；
支持多轮追问，会话历史保留在页面里，可整段导出。
准备工作（建库 / 导资料包 / 传视频 / 关联）收进侧栏与"🧰 准备工作"页。

启动方式（二选一）：
    streamlit run streamlit_sales.py        ← 本文件（销售对话主界面）
    streamlit run streamlit.py              ← 原全功能界面（含剪辑/竖屏/字幕等）

注意：本文件不要改名为 streamlit.py（会遮蔽 streamlit 包）；也不要用 python 直接运行。
"""

import streamlit as st
import os
import json
import time
import urllib.parse
from datetime import datetime

import server  # 所有后端调用统一走 server 层（与 streamlit.py 相同约定）

# 设置页面配置
st.set_page_config(
    page_title="诺界 PromiseOS · 销售对话",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- 样式 ----------------------------
st.markdown("""
<style>
    .chat-user, .chat-assistant {
        border-radius: 14px;
        padding: 14px 18px;
        margin: 10px 0;
    }
    .chat-user {
        background: linear-gradient(135deg, #e8f0fe 0%, #dce7fd 100%);
        border-left: 4px solid #4a6cf7;
    }
    .chat-assistant {
        background: #ffffff;
        border: 1px solid #e5e9f2;
        border-left: 4px solid #7a5af8;
    }
    .chat-name { font-weight: bold; font-size: 13px; color: #555; margin-bottom: 6px; }
    .status-chip {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 12px; font-weight: bold; margin-right: 6px;
    }
    .chip-green  { background: #d4edda; color: #155724; }
    .chip-yellow { background: #fff3cd; color: #856404; }
    .chip-red    { background: #f8d7da; color: #721c24; }
    .chip-orange { background: #ffe5cc; color: #8a4b00; }
    .ready-ok  { color: #155724; }
    .ready-todo{ color: #856404; }
    .section-title {
        color: #1e1e2f; font-size: 22px; font-weight: bold;
        margin: 8px 0 14px 0; padding-bottom: 8px; border-bottom: 2px solid #7a5af8;
    }
    .stButton > button { border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------- 常量 ----------------------------
DEFAULT_PACK_DIR = os.path.join("..", "第二步_国内公开资料包_阿里云")
CATEGORY_OPTIONS = {
    "product_demo（产品/软件界面演示）": "product_demo",
    "general（通用视频）": "general",
}
QUICK_QUESTIONS = [
    "我们有一批中文扫描合同，要提取指定字段并接入现有系统。是不是开通就能上线，以后完全不用人工处理？",
    "开通之后是不是今天就能上线，扫描件能不能完全不人工处理？",
    "你们有 API 吗？能直接接进我们现在的业务系统吗？",
    "异步接口提交成功之后，是不是马上就能拿到识别结果？",
]
STATUS_ICONS = {
    "有依据": "🟢",
    "需附条件": "🟡",
    "待技术确认": "🔴",
    "待确认（来源冲突）": "🟠",
}
STATUS_CHIPS = {
    "有依据": "chip-green",
    "需附条件": "chip-yellow",
    "待技术确认": "chip-red",
    "待确认（来源冲突）": "chip-orange",
}

# ---------------------------- 会话状态 ----------------------------
def init_state():
    defaults = {
        "selected_lib": None,
        "sales_chat": [],        # [{ts, question, known, answer}]
        "page": "对话",
        "libs": [],
        "lib_info": {},
        "global_assets": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def refresh_data():
    """每次 rerun 刷新只读数据（与 streamlit.py 相同模式）。"""
    st.session_state.libs = server.get_libraries()
    if not st.session_state.selected_lib and st.session_state.libs:
        st.session_state.selected_lib = st.session_state.libs[0]
    if st.session_state.selected_lib:
        st.session_state.lib_info = server.get_library_info(st.session_state.selected_lib)
        st.session_state.global_assets = server.get_global_assets()


def readiness():
    """准备工作就绪清单：库 / 资料包 / 视频。返回 (项列表, 是否全部就绪)。"""
    lib = st.session_state.selected_lib
    reg = server.get_source_registry(lib) if lib else {"datasets": {}}
    assets = (st.session_state.lib_info or {}).get("assets", {})
    items = [
        ("📚 知识库", bool(lib), f"{lib or '未选择'}"),
        ("📥 资料包", len(reg.get("datasets", {})) > 0,
         f"{len(reg.get('datasets', {}))} 个资料集 / {len(reg.get('sources', {}))} 条来源"),
        ("🎬 演示视频", len(assets) > 0, f"{len(assets)} 个已关联视频资产"),
    ]
    return items, all(ok for _, ok, _ in items)


# ---------------------------- 通用小组件 ----------------------------
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
            st.video(video_path, start_time=int((c_s if expand else e_s) or 0))
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


def render_card(card: dict, turn_idx: int, card_idx: int, expanded: bool = False):
    """一张承诺卡（对话内嵌版）。"""
    icon = STATUS_ICONS.get(card["status"], "⚪")
    chip = STATUS_CHIPS.get(card["status"], "chip-yellow")
    title = f"{card['customer_ask']}"
    with st.expander(f"{icon} {title}", expanded=expanded):
        review_chip = "chip-green" if card["source_review"] == "已核验" else "chip-yellow"
        st.markdown(
            f'<span class="status-chip {chip}">{card["status"]}</span>'
            f'<span class="status-chip {review_chip}">来源{card["source_review"]}</span>',
            unsafe_allow_html=True,
        )
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
            if card.get("given_conditions"):
                st.caption("客户已说明：" + "；".join(card["given_conditions"]))
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
            with st.expander(f"依据 {j + 1}：{ev.get('title') or ev.get('source_id')}（{ev.get('kind')}）", expanded=(j == 0 and expanded)):
                render_evidence_item(ev, key_prefix=f"ev_{turn_idx}_{card_idx}_{j}")


def render_turn(turn: dict, idx: int, is_latest: bool):
    """渲染一轮对话。"""
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(f"客户原话（销售转述）：")
        st.markdown(f"> {turn['question']}")
        if turn.get("known"):
            st.caption(f"已知客户条件：{turn['known']}")

    with st.chat_message("assistant", avatar="🤖"):
        answer = turn["answer"]
        llm = answer.get("llm", {})
        counts = answer.get("status_counts", {})
        chips = "".join(
            f'<span class="status-chip {STATUS_CHIPS.get(k, "chip-yellow")}">{STATUS_ICONS.get(k, "")} {k} ×{v}</span>'
            for k, v in counts.items()
        )
        st.markdown(chips, unsafe_allow_html=True)
        st.caption(
            f"{turn['ts']} ｜ 耗时 {answer.get('elapsed_sec')}s ｜ 文本模型：{llm.get('provider') or '（未调用）'}"
            + (f" ｜ {llm['last_error']}" if llm.get("last_error") else "")
        )

        st.markdown("**✉️ 建议回复（核对后使用，系统不自动发送）**")
        st.text_area("回复草稿", value=answer.get("reply", ""), height=150,
                     key=f"reply_{idx}", label_visibility="collapsed")

        for ci, card in enumerate(answer.get("claims", [])):
            render_card(card, idx, ci, expanded=(is_latest and ci == 0))

        q1, q2 = st.columns(2)
        with q1:
            st.markdown("**📌 问客户（缺客户信息）**")
            if answer.get("ask_customer"):
                for q in answer["ask_customer"]:
                    st.write(f"- {q['question']}")
            else:
                st.caption("—")
        with q2:
            st.markdown("**📌 问技术 / 内部确认（缺产品依据）**")
            if answer.get("ask_tech"):
                for q in answer["ask_tech"]:
                    st.write(f"- {q['question']} ｜ 建议：{q['suggested_role']}")
            else:
                st.caption("—")

        if is_latest:
            b1, b2 = st.columns(2)
            with b1:
                if st.button("💾 保存本轮答复日志", key=f"save_{idx}", use_container_width=True):
                    r = server.save_sales_answer_log(answer)
                    (st.success if r["success"] else st.error)(r["message"])
            with b2:
                st.download_button(
                    "⬇️ 导出本轮待确认清单",
                    data=json.dumps({"question": answer["question"],
                                     "ask_customer": answer["ask_customer"],
                                     "ask_tech": answer["ask_tech"]}, ensure_ascii=False, indent=2),
                    file_name=f"pending_{idx + 1}.json", mime="application/json",
                    key=f"dl_{idx}", use_container_width=True,
                )


# ---------------------------- 页面：对话（主页面） ----------------------------
def page_chat():
    items, ready = readiness()
    if not ready:
        with st.container(border=True):
            st.markdown("#### ⚙️ 准备工作尚未完成")
            for name, ok, note in items:
                st.markdown(f"- {'✅' if ok else '⬜'} **{name}**：{note}")
            st.info("对话功能本身可用；但依据会不完整。请到左侧「🧰 准备工作」完成建库、导入资料包并关联演示视频。")

    st.markdown('<div class="section-title">💬 销售对话</div>', unsafe_allow_html=True)
    st.caption("把客户原话贴进来，系统给出有依据、带条件、知道该问谁的答复草稿。"
               "模型负责拆解/提取/起草；引用真实性、状态判定、失效来源门禁、冲突保留由程序完成。素材为公开资料整理，不代表企业合作。")

    # 快捷问题（直接写入带 key 的组件状态，再 rerun；带 key 的组件不会理会 value= 参数）
    st.markdown("**快速开始（点击填入常见客户问法，鼠标悬停可预览）**")
    chip_cols = st.columns(len(QUICK_QUESTIONS))
    for i, q in enumerate(QUICK_QUESTIONS):
        if chip_cols[i].button(f"问法 {i + 1}", key=f"quick_{i}", use_container_width=True, help=q):
            st.session_state["q_text_input"] = q
            st.rerun()

    # 输入区
    if "q_text_input" not in st.session_state:
        st.session_state["q_text_input"] = QUICK_QUESTIONS[0]
    with st.container(border=True):
        question = st.text_area("客户的问题（原话即可，可多轮追问）",
                                height=90, key="q_text_input", label_visibility="collapsed")
        col_known, col_run = st.columns([3, 1])
        with col_known:
            known = st.text_input("销售已掌握的客户条件（可空）", key="q_known_input",
                                  placeholder="例如：客户用的是用友 U8，每天约 300 份扫描合同")
        with col_run:
            top_k = st.slider("检索条数/事项", 2, 8, 4, key="q_topk")
        if st.button("🧭 生成答复草稿", key="q_run", use_container_width=True, type="primary"):
            if not question.strip():
                st.warning("请输入客户问题。")
            elif not st.session_state.selected_lib:
                st.warning("请先在左侧选择或创建知识库。")
            else:
                with st.spinner("拆解问题 → 检索资料 → 提取条件 → 程序校验 → 起草回复..."):
                    result = server.prepare_sales_answer(
                        question.strip(), st.session_state.selected_lib,
                        known_conditions=known.strip(), top_k=top_k)
                if result["success"]:
                    st.session_state.sales_chat.append({
                        "ts": datetime.now().strftime("%H:%M:%S"),
                        "question": question.strip(),
                        "known": known.strip(),
                        "answer": result["answer"],
                    })
                    st.rerun()   # 立即渲染新回合
                else:
                    st.error(f"❌ {result['message']}")

    # 会话历史
    st.markdown("---")
    if not st.session_state.sales_chat:
        st.info("尚无对话。贴入客户问题，点「生成答复草稿」开始第一轮。")
        return

    for idx, turn in enumerate(st.session_state.sales_chat):
        render_turn(turn, idx, is_latest=(idx == len(st.session_state.sales_chat) - 1))

    # 会话级操作
    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "⬇️ 导出整个会话 (JSON)",
            data=json.dumps(st.session_state.sales_chat, ensure_ascii=False, indent=2, default=str),
            file_name=f"sales_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json", key="dl_chat", use_container_width=True,
        )
    with e2:
        if st.button("🗑️ 清空会话", key="clear_chat", use_container_width=True):
            st.session_state.sales_chat = []
            st.rerun()


# ---------------------------- 页面：准备工作 ----------------------------
def page_setup():
    st.markdown('<div class="section-title">🧰 准备工作</div>', unsafe_allow_html=True)
    items, ready = readiness()
    for name, ok, note in items:
        st.markdown(f"- {'✅' if ok else '⬜'} **{name}**：{note}")
    if ready:
        st.success("准备工作全部就绪，去「💬 对话」开始使用。")

    st.markdown("### 1️⃣ 导入资料包（文档摘要＋官方配图）")
    pack_dir = st.text_input("资料包目录（含 02_来源登记.json）", value=DEFAULT_PACK_DIR, key="pack_dir")
    run_vlm = st.checkbox("用视觉模型为图片生成界面状态描述（首次调用 Qwen-VL，有缓存）", value=True, key="pack_vlm")
    if st.button("📥 导入到当前知识库", key="pack_btn", use_container_width=True):
        with st.spinner("导入中..."):
            result = server.ingest_doc_pack(st.session_state.selected_lib, pack_dir, run_vlm=run_vlm)
        (st.success if result["success"] else st.error)(result["message"])
        if result["success"]:
            st.json({k: v for k, v in result.items() if k not in ("success", "message", "lib_name", "pack_dir")})
            refresh_data()

    reg = server.get_source_registry(st.session_state.selected_lib) if st.session_state.selected_lib else {"sources": {}, "images": {}}
    if reg.get("datasets"):
        with st.expander("来源登记明细（状态 / 版本 / 审核人）"):
            rows = []
            for sid, s in reg.get("sources", {}).items():
                rows.append({"来源ID": sid, "标题": s.get("title", ""), "API版本": s.get("api_version") or "",
                             "状态": s.get("status", ""), "审核人": s.get("human_reviewer") or "（待核验）"})
            for iid, im in reg.get("images", {}).items():
                rows.append({"来源ID": iid, "标题": f"配图（{im.get('section', '')}）", "API版本": "",
                             "状态": ("视觉模型已描述" if im.get("vlm_description") else "未生成画面描述") + "；待核验",
                             "审核人": im.get("human_reviewer") or "（待核验）"})
            st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("### 2️⃣ 上传并处理演示视频")
    st.caption("产品/软件界面演示请选 product_demo 分类——视觉模型会抄写界面状态文字而不是描述场景。")
    cat_label = st.selectbox("视频分类", list(CATEGORY_OPTIONS.keys()), key="cat_sel")
    category_tag = CATEGORY_OPTIONS[cat_label]
    force_rebuild = st.checkbox("强制重建合并/清洗缓存（逻辑升级后再勾）", key="force_rb")
    uploaded = st.file_uploader("选择视频文件", type=['mp4', 'mov', 'avi', 'mkv'], key="vid_up")
    if uploaded is not None:
        st.info(f"📹 已选择: {uploaded.name}（{uploaded.size / 1024 / 1024:.1f} MB）")
        if st.button("🚀 处理上传的视频", key="proc_btn", type="primary", use_container_width=True):
            file_path = os.path.join(server.INPUT_VIDEO_DIR, uploaded.name)
            base, ext = os.path.splitext(uploaded.name)
            counter = 1
            while os.path.exists(file_path):
                file_path = os.path.join(server.INPUT_VIDEO_DIR, f"{base}_{counter}{ext}")
                counter += 1
            with open(file_path, "wb") as f:
                f.write(uploaded.getbuffer())
            with st.spinner("处理中（首次需转写＋关键帧＋视觉分析，约 3–6 分钟）..."):
                result = server.data_processing(os.path.basename(file_path), category=category_tag, force_rebuild=force_rebuild)
            if result["success"]:
                st.success(f"✅ {result['message']}")
                c1, c2, c3 = st.columns(3)
                c1.metric("转录片段", result.get("transcript_count", 0))
                c2.metric("关键帧", result.get("keyframes_count", 0))
                c3.metric("视觉片段", result.get("visual_segments_count", 0))
                if result.get("asr_status") == "failed":
                    st.warning("⚠️ 语音识别失败（未写语音缓存，可重试）；画面信息仍已入库。")
                refresh_data()
            else:
                st.error(f"❌ {result['message']}")

    st.markdown("### 3️⃣ 把视频关联进当前知识库")
    assets = (st.session_state.lib_info or {}).get("assets", {})
    linked = set(assets.keys())
    available = [a for a in st.session_state.global_assets if a["md5"] not in linked]
    if available:
        opts = {a["md5"]: a.get("filename", a["md5"]) for a in available}
        chosen = st.multiselect("选择要关联的资产", list(opts.keys()),
                                format_func=lambda m: f"{opts[m]}（{m[:8]}…）", key="link_sel")
        if st.button("🔗 关联选中资产", key="link_btn", use_container_width=True):
            prog = st.progress(0.0)
            for i, md5 in enumerate(chosen):
                r = server.add_asset_to_library(st.session_state.selected_lib, md5)
                (st.toast if r["success"] else st.error)(r["message"])
                prog.progress((i + 1) / len(chosen))
            refresh_data()
            st.rerun()
    else:
        st.caption("没有待关联的资产（全部已关联，或资产池为空——先完成第 2 步）。")


# ---------------------------- 侧边栏 ----------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## 💬 诺界 · PromiseOS")
        st.caption("销售对话助手：让每一句对客户的承诺都有依据")
        st.markdown("---")

        # 知识库选择与新建
        st.markdown("### 📚 知识库")
        libs = st.session_state.libs
        if libs:
            current = st.session_state.selected_lib if st.session_state.selected_lib in libs else libs[0]
            selected = st.selectbox("当前知识库", libs, index=libs.index(current), key="lib_sel")
            st.session_state.selected_lib = selected
        else:
            st.info("暂无知识库")
        new_lib = st.text_input("新建知识库", key="new_lib", placeholder="例如 promiseos_demo")
        if st.button("➕ 创建并切换", key="create_lib", use_container_width=True) and new_lib:
            r = server.create_library(new_lib)
            if r["success"]:
                st.session_state.selected_lib = new_lib
                refresh_data()
                st.rerun()
            else:
                st.error(r["message"])

        st.markdown("---")

        # 就绪清单
        st.markdown("### ✅ 就绪状态")
        items, ready = readiness()
        for name, ok, note in items:
            st.markdown(f"<div class='{'ready-ok' if ok else 'ready-todo'}'>{'✅' if ok else '⬜'} {name}<br>"
                        f"<small>{note}</small></div>", unsafe_allow_html=True)
        lib_info = st.session_state.lib_info or {}
        st.caption(f"库内索引 {lib_info.get('rag_count', 0)} 条 ｜ 视频资产 {lib_info.get('asset_count', 0)} 个")

        st.markdown("---")

        # 页面切换
        page = st.radio("导航", ["💬 对话", "🧰 准备工作"], key="page_sel", label_visibility="collapsed")
        st.session_state.page = "对话" if page.startswith("💬") else "准备"

        st.markdown("---")
        st.caption("演示素材为公开资料整理（阿里云文档自学习），不代表企业合作。\n"
                   "被销售的产品是第三方示例；本系统是 PromiseOS 本身。")


# ---------------------------- 主函数 ----------------------------
def main():
    init_state()
    refresh_data()
    render_sidebar()

    if st.session_state.page == "准备":
        page_setup()
    else:
        page_chat()


if __name__ == "__main__":
    main()
