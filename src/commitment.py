"""
承诺卡引擎（R1 / R2 / R3 / R4a）

流程：客户问题 → 拆分事项 → 逐项检索（文档+视频） → 模型提取支持/条件/未知
     → 程序校验 → 待确认分流（问客户/问技术） → 答复草稿 → 答复日志

分工原则：模型负责理解与提取，程序负责校验与判定：
- 引用的证据 ID 必须在本次检索结果中真实存在，不存在的一律丢弃并记录；
- 状态由规则推导（见 judge_status），不由模型自报；
- 来源状态含失效/冲突标记的不得作为当前依据；
- 来源冲突保留给人确认，不让模型二选一；
- 没有有效证据时不允许保留“资料支持”的表述。
"""
import os
import re
import json
import time
import logging
import datetime

import requests

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL, LIBRARIES_DIR, DASHSCOPE_API_KEY
from src.rag_engine import VideoKnowledgeBase
from src.doc_ingest import load_registry
from src.prompt_texts import DECOMPOSE_SYSTEM, EXTRACT_SYSTEM, DRAFT_SYSTEM

logger = logging.getLogger(__name__)

# 文本模型提供方：auto = DeepSeek 为主、鉴权失败自动切到 DashScope 兼容模式的 qwen-plus
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
DASHSCOPE_COMPAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_TEXT_MODEL = os.getenv("DASHSCOPE_TEXT_MODEL", "qwen-plus")
_AUTH_FAIL_CODES = (401, 402, 403)

STATUS_SUPPORTED = "有依据"
STATUS_CONDITIONAL = "需附条件"
STATUS_NEEDS_TECH = "待技术确认"
STATUS_CONFLICT = "待确认（来源冲突）"

REVIEW_VERIFIED = "已核验"
REVIEW_PENDING = "待核验"

# 出现这些词，说明缺的是客户那边的信息，应当追问客户而不是转给工程师
CUSTOMER_INFO_KEYWORDS = [
    "客户系统", "现有系统", "业务系统", "贵方系统", "贵司系统", "对方系统", "旧系统", "ERP", "OA", "CRM",
    "接口文档", "接口协议", "接口规格", "API规格", "数据格式", "字段要求", "字段清单", "样本", "样例",
    "单据类型", "文件类型", "文档类型", "扫描件质量", "图像质量", "分辨率", "文件大小", "页数",
    "业务范围", "上线范围", "期望时间", "预期时间", "排期", "历史数据", "数据迁移", "权限", "管理员账号",
    "预算", "处理量", "每天", "每月", "并发", "部署环境", "网络环境", "客户环境", "客户的", "贵方", "贵司",
]

# 转技术时建议找谁；按关键词先到先得
TECH_ROLE_HINTS = [
    ("实施", "售前/实施"), ("周期", "售前/实施"), ("上线", "售前/实施"), ("部署", "交付/实施"),
    ("计费", "商务/产品"), ("费用", "商务/产品"), ("价格", "商务/产品"), ("免费", "商务/产品"),
    ("版本", "产品"), ("接口", "售前工程师"), ("兼容", "售前工程师"), ("对接", "售前工程师"),
    ("准确率", "产品/算法"), ("效果", "产品/算法"), ("训练", "产品/算法"), ("模型", "产品/算法"),
    ("安全", "安全/合规"), ("合规", "安全/合规"),
]

STALE_MARKERS = ("失效", "历史资料", "已废弃", "过期", "已下线")

_CLAUSE_SPLIT_RE = re.compile(r"[？?！!；;。]|，(?=\s*(?:能|是|有|多|以后|还|可|需要|要|马上|立刻|直接|今天|明天|什么时候))")


# ---------------------------------------------------------------- LLM
class LLMClient:
    """
    OpenAI 兼容 JSON 模式的最小封装。
    - available() 为 False 或主提供方鉴权失败（401/402/403）时，自动委托给 fallback；
    - 鉴权失败只判定一次，之后不再反复重试主提供方；
    - used_provider 记录最近一次真正产出结果的提供方，便于答辩时如实说明。
    """

    def __init__(self, api_key=None, base_url=None, model=None, timeout=60, max_retries=2,
                 provider="deepseek", fallback=None):
        self.api_key = api_key if api_key is not None else DEEPSEEK_API_KEY
        self.base_url = (base_url if base_url is not None else DEEPSEEK_BASE_URL or "").rstrip("/")
        self.model = model or LLM_MODEL or "deepseek-chat"
        self.timeout = timeout
        self.max_retries = max_retries
        self.provider = provider
        self.fallback = fallback
        self.last_error = None
        self.used_provider = None
        self._primary_dead = False

    def available(self):
        if self.api_key and self.base_url and not self._primary_dead:
            return True
        return bool(self.fallback and self.fallback.available())

    @staticmethod
    def _parse_json(text):
        if text is None:
            return None
        s = text.strip()
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE).strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

    def _delegate(self, system, user, temperature):
        if not self.fallback:
            return None
        result = self.fallback.json(system, user, temperature)
        self.used_provider = self.fallback.used_provider
        if result is None:
            self.last_error = f"{self.last_error}; fallback({self.fallback.provider}): {self.fallback.last_error}"
        return result

    def json(self, system, user, temperature=0.1):
        if self._primary_dead or not (self.api_key and self.base_url):
            if not self._primary_dead:
                self.last_error = f"{self.provider} 未配置"
            return self._delegate(system, user, temperature)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code in _AUTH_FAIL_CODES:
                    self._primary_dead = True
                    self.last_error = f"{self.provider} 鉴权失败 HTTP {resp.status_code}（key 无效/欠费），已切换备选"
                    logger.error(self.last_error)
                    break
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = self._parse_json(content)
                if parsed is not None:
                    self.used_provider = self.provider
                    return parsed
                self.last_error = f"返回内容不是合法 JSON: {content[:200]}"
            except Exception as e:
                self.last_error = str(e)
                logger.warning(f"LLM({self.provider}) 调用失败（第 {attempt + 1} 次）: {e}")
                time.sleep(1.5 * (attempt + 1))
        return self._delegate(system, user, temperature)


def default_llm():
    """按 LLM_PROVIDER 组装客户端：deepseek / dashscope / auto（DeepSeek 主 + qwen-plus 备）。"""
    dashscope_client = LLMClient(api_key=DASHSCOPE_API_KEY or "", base_url=DASHSCOPE_COMPAT_URL,
                                 model=DASHSCOPE_TEXT_MODEL, provider="dashscope")
    deepseek_client = LLMClient(provider="deepseek")
    if LLM_PROVIDER == "dashscope":
        return dashscope_client
    if LLM_PROVIDER == "deepseek":
        return deepseek_client
    if DASHSCOPE_API_KEY:
        deepseek_client.fallback = dashscope_client
    return deepseek_client


# ---------------------------------------------------------------- 拆解


def _heuristic_decompose(question):
    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(question or "") if p and p.strip()]
    parts = [p for p in parts if len(p) >= 2] or [question.strip()]
    claims = []
    for i, p in enumerate(parts, 1):
        claims.append({
            "claim_id": f"C{i}",
            "customer_ask": p,
            "topic": "其他",
            "given_conditions": [],
            "search_query": p,
        })
    return claims


def decompose_question(question, known_conditions="", llm=None):
    """返回事项列表；LLM 不可用或输出异常时退化为按标点切分。"""
    llm = llm or default_llm()
    user = f"客户原话：{question}\n销售已知的客户条件：{known_conditions or '（无）'}"
    data = llm.json(DECOMPOSE_SYSTEM, user) if llm.available() else None
    claims = (data or {}).get("claims") if isinstance(data, dict) else None
    if not claims or not isinstance(claims, list):
        return _heuristic_decompose(question), False
    cleaned = []
    for i, c in enumerate(claims, 1):
        if not isinstance(c, dict) or not c.get("customer_ask"):
            continue
        cleaned.append({
            "claim_id": c.get("claim_id") or f"C{i}",
            "customer_ask": str(c["customer_ask"]).strip(),
            "topic": str(c.get("topic") or "其他"),
            "given_conditions": [str(x) for x in (c.get("given_conditions") or [])],
            "search_query": str(c.get("search_query") or c["customer_ask"]),
        })
    return (cleaned or _heuristic_decompose(question)), True


# ---------------------------------------------------------------- 检索与证据归一化
def _json_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        try:
            return json.loads(v)
        except Exception:
            return []
    return []


def normalize_evidence(search_results, registry=None):
    """把 VideoKnowledgeBase.search 的返回统一成证据字典列表。"""
    registry = registry or {}
    out = []
    if not search_results or not search_results.get("ids") or not search_results["ids"][0]:
        return out
    ids = search_results["ids"][0]
    docs = search_results["documents"][0]
    metas = search_results["metadatas"][0]
    dists = (search_results.get("distances") or [[]])[0] or [None] * len(ids)
    for i, eid in enumerate(ids):
        meta = metas[i] or {}
        kind = meta.get("kind") or "video"
        ev = {
            "evidence_id": eid,
            "kind": kind,
            "text": docs[i],
            "distance": dists[i] if i < len(dists) else None,
            "title": meta.get("title", ""),
            "section": meta.get("section", ""),
            "url": meta.get("url", ""),
            "local_path": meta.get("local_path", ""),
            "status": meta.get("status", ""),
            "human_reviewer": meta.get("human_reviewer", "") or "",
            "api_version": meta.get("api_version", ""),
            "product_version": meta.get("product_version", ""),
            "boundary": meta.get("boundary", ""),
            "raw_content": meta.get("raw_content", ""),
        }
        if kind == "video":
            md5 = meta.get("source_video_md5", "")
            ev.update({
                "source_id": f"VIDEO:{md5}#{meta.get('seq_id', '')}",
                "video_md5": md5,
                "evidence_start": meta.get("evidence_start", meta.get("start")),
                "evidence_end": meta.get("evidence_end", meta.get("end")),
                "context_start": meta.get("context_start", meta.get("start")),
                "context_end": meta.get("context_end", meta.get("end")),
                "frame_paths": _json_list(meta.get("frame_paths_json")),
                "visual_summary": meta.get("visual_summary", ""),
                "segment_type": meta.get("type", ""),
            })
            if not ev["status"]:
                ev["status"] = "视频片段；时间定位由系统处理生成，画面是否支持表述需人工复核"
        else:
            ev.update({
                "source_id": meta.get("source_id", ""),
                "frame_path": meta.get("frame_path", ""),
            })
            reg_src = (registry.get("sources") or {}).get(ev["source_id"]) or (registry.get("images") or {}).get(ev["source_id"])
            if reg_src:
                ev.setdefault("url", reg_src.get("url") or reg_src.get("source_url", ""))
                if not ev["human_reviewer"]:
                    ev["human_reviewer"] = reg_src.get("human_reviewer") or ""
                if not ev["status"]:
                    ev["status"] = reg_src.get("status", "")
        out.append(ev)
    return out


def retrieve_for_claim(vkb, claim, top_k=4, registry=None):
    query = claim.get("search_query") or claim.get("customer_ask")
    res = vkb.search(query, top_k=top_k, expand_context=True)
    return normalize_evidence(res, registry)


# ---------------------------------------------------------------- 提取


def _format_evidence_for_prompt(evidences, max_chars=600):
    lines = []
    for i, ev in enumerate(evidences, 1):
        head = f"[{i}] evidence_id={ev['evidence_id']} | 类型={ev['kind']} | 来源={ev.get('title') or ev.get('source_id')}"
        if ev.get("section"):
            head += f" | 章节={ev['section']}"
        if ev.get("api_version"):
            head += f" | API版本={ev['api_version']}"
        if ev.get("status"):
            head += f" | 来源状态={ev['status']}"
        if ev["kind"] == "video":
            head += f" | 片段={ev.get('evidence_start')}s–{ev.get('evidence_end')}s"
        body = (ev.get("text") or "")[:max_chars]
        lines.append(head + "\n    内容：" + body.replace("\n", " "))
        if ev.get("boundary"):
            lines.append("    【整理边界，非原文】" + str(ev["boundary"])[:200])
    return "\n".join(lines)


def extract_support(claim, evidences, llm=None):
    """返回 (extraction_dict, used_llm)。无证据或 LLM 不可用时返回保守结果。"""
    base = {"supported_statement": "", "conditions": [], "unknowns": [claim.get("customer_ask", "")],
            "cited_evidence_ids": [], "conflicts": [], "rationale": ""}
    if not evidences:
        base["rationale"] = "未检索到相关资料"
        return base, False
    llm = llm or default_llm()
    if not llm.available():
        base["rationale"] = "LLM 未配置，无法提取；按待确认处理"
        return base, False
    user = (
        f"客户事项：{claim.get('customer_ask')}\n"
        f"客户已说明的条件：{'；'.join(claim.get('given_conditions') or []) or '（无）'}\n\n"
        f"候选资料：\n{_format_evidence_for_prompt(evidences)}"
    )
    data = llm.json(EXTRACT_SYSTEM, user)
    if not isinstance(data, dict):
        base["rationale"] = f"LLM 提取失败：{llm.last_error}"
        return base, False
    out = {
        "supported_statement": str(data.get("supported_statement") or "").strip(),
        "conditions": [str(x).strip() for x in (data.get("conditions") or []) if str(x).strip()],
        "unknowns": [str(x).strip() for x in (data.get("unknowns") or []) if str(x).strip()],
        "cited_evidence_ids": [str(x).strip() for x in (data.get("cited_evidence_ids") or [])],
        "conflicts": [c for c in (data.get("conflicts") or []) if isinstance(c, dict) and c.get("evidence_ids")],
        "rationale": str(data.get("rationale") or ""),
    }
    return out, True


# ---------------------------------------------------------------- 程序校验（纯函数，可单测）
def is_stale(evidence):
    status = str(evidence.get("status") or "")
    return any(m in status for m in STALE_MARKERS)


def judge_status(valid_count, conditions, conflicts):
    if conflicts:
        return STATUS_CONFLICT
    if valid_count == 0:
        return STATUS_NEEDS_TECH
    if conditions:
        return STATUS_CONDITIONAL
    return STATUS_SUPPORTED


def verify_and_judge(extraction, evidences):
    """
    对模型提取结果做程序校验：
    - 引用 ID 必须存在于候选证据；不存在 → dropped；来源失效 → stale；
    - 冲突只保留双方 ID 都有效的；
    - 无有效证据则清空 supported_statement；
    - 状态由规则推导；来源核验状态看是否有人工审核记录。
    """
    cand = {ev["evidence_id"]: ev for ev in evidences}
    valid, dropped, stale = [], [], []
    for eid in extraction.get("cited_evidence_ids") or []:
        if eid not in cand:
            dropped.append(eid)
        elif is_stale(cand[eid]):
            stale.append(eid)
        else:
            valid.append(eid)
    # 去重保持顺序
    valid = list(dict.fromkeys(valid))

    conflicts = []
    for c in extraction.get("conflicts") or []:
        ids = [str(x) for x in (c.get("evidence_ids") or [])]
        if len(ids) >= 2 and all(i in cand for i in ids):
            conflicts.append({"evidence_ids": ids, "note": str(c.get("note") or "")})

    conditions = list(extraction.get("conditions") or [])
    unknowns = list(extraction.get("unknowns") or [])
    supported = extraction.get("supported_statement") or ""
    if not valid:
        if supported:
            unknowns.insert(0, f"资料中未找到可引用的依据来支持：{supported}")
        supported = ""
    if stale and not valid:
        unknowns.append("检索到的相关资料已标记失效/历史，需要确认当前版本口径")

    status = judge_status(len(valid), conditions, conflicts)
    source_review = REVIEW_VERIFIED if any(cand[i].get("human_reviewer") for i in valid) else REVIEW_PENDING
    return {
        "status": status,
        "source_review": source_review,
        "supported_statement": supported,
        "conditions": conditions,
        "unknowns": list(dict.fromkeys([u for u in unknowns if u])),
        "valid_evidence_ids": valid,
        "dropped_evidence_ids": dropped,
        "stale_evidence_ids": stale,
        "conflicts": conflicts,
        "rationale": extraction.get("rationale", ""),
    }


# ---------------------------------------------------------------- 待确认分流（R2）
def classify_unknown(text):
    t = str(text or "")
    if any(k in t for k in CUSTOMER_INFO_KEYWORDS):
        return "customer"
    return "tech"


def suggest_role(text):
    t = str(text or "")
    for kw, role in TECH_ROLE_HINTS:
        if kw in t:
            return role
    return "售前/产品"


def route_unknowns(unknowns):
    ask_customer, ask_tech = [], []
    for u in unknowns or []:
        if classify_unknown(u) == "customer":
            q = u if u.endswith(("？", "?")) else f"请贵方补充：{u}"
            ask_customer.append({"item": u, "question": q})
        else:
            ask_tech.append({"item": u, "suggested_role": suggest_role(u), "question": f"需内部确认：{u}"})
    return {"ask_customer": ask_customer, "ask_tech": ask_tech}


# ---------------------------------------------------------------- 承诺卡
def build_card(claim, judged, routed, evidences):
    by_id = {ev["evidence_id"]: ev for ev in evidences}
    return {
        "claim_id": claim.get("claim_id"),
        "customer_ask": claim.get("customer_ask"),
        "topic": claim.get("topic", "其他"),
        "given_conditions": claim.get("given_conditions") or [],
        "status": judged["status"],
        "source_review": judged["source_review"],
        "supported_statement": judged["supported_statement"],   # 资料直接支持的内容
        "conditions": judged["conditions"],                     # 适用前提 / 必要条件
        "unknowns": judged["unknowns"],                         # 尚未确认的内容
        "ask_customer": routed["ask_customer"],
        "ask_tech": routed["ask_tech"],
        "evidence": [by_id[i] for i in judged["valid_evidence_ids"] if i in by_id],
        "candidate_evidence_ids": [ev["evidence_id"] for ev in evidences],
        "dropped_evidence_ids": judged["dropped_evidence_ids"],
        "stale_evidence_ids": judged["stale_evidence_ids"],
        "conflicts": judged["conflicts"],
        "rationale": judged.get("rationale", ""),
    }


# ---------------------------------------------------------------- 草稿


def _fallback_reply(cards):
    parts = []
    for c in cards:
        if c["supported_statement"]:
            s = c["supported_statement"]
            if c["conditions"]:
                s += "，前提是" + "、".join(c["conditions"])
            parts.append(s + "。")
        else:
            parts.append(f"关于“{c['customer_ask']}”，目前资料不足以直接承诺，需要先确认相关条件。")
    nxt = []
    for c in cards:
        for q in c["ask_customer"]:
            nxt.append(q["question"])
    if nxt:
        parts.append("为了给出准确方案，建议先补充：" + "；".join(dict.fromkeys(nxt)) + "。")
    return "".join(parts)


def draft_reply(question, cards, llm=None):
    llm = llm or default_llm()
    fallback = _fallback_reply(cards)
    if not llm.available():
        return fallback, {}, False
    slim = []
    for c in cards:
        slim.append({
            "claim_id": c["claim_id"], "customer_ask": c["customer_ask"], "status": c["status"],
            "supported_statement": c["supported_statement"], "conditions": c["conditions"],
            "unknowns": c["unknowns"],
            "ask_customer": [q["question"] for q in c["ask_customer"]],
            "ask_tech": [q["item"] for q in c["ask_tech"]],
        })
    user = f"客户原话：{question}\n\n已校验的承诺卡：\n{json.dumps(slim, ensure_ascii=False, indent=1)}"
    data = llm.json(DRAFT_SYSTEM, user, temperature=0.3)
    if isinstance(data, dict) and data.get("reply"):
        return str(data["reply"]).strip(), dict(data.get("per_claim_phrases") or {}), True
    return fallback, {}, False


# ---------------------------------------------------------------- 编排 + 日志（R4a）
def prepare_answer(question, lib_name, known_conditions="", top_k=4, llm=None, vkb=None):
    """主入口：返回完整答复对象（草稿 + 承诺卡 + 待确认 + 引用资料快照）。"""
    t0 = time.time()
    llm = llm or default_llm()
    vkb = vkb or VideoKnowledgeBase(lib_name=lib_name)
    registry = load_registry(lib_name)

    claims, llm_decomposed = decompose_question(question, known_conditions, llm)
    cards = []
    llm_extracted_any = False
    for claim in claims:
        evidences = retrieve_for_claim(vkb, claim, top_k=top_k, registry=registry)
        extraction, used = extract_support(claim, evidences, llm)
        llm_extracted_any = llm_extracted_any or used
        judged = verify_and_judge(extraction, evidences)
        routed = route_unknowns(judged["unknowns"])
        cards.append(build_card(claim, judged, routed, evidences))

    reply, phrases, llm_drafted = draft_reply(question, cards, llm)
    for c in cards:
        c["suggested_phrase"] = phrases.get(c["claim_id"], "")

    cited = {}
    for c in cards:
        for ev in c["evidence"]:
            cited[ev["evidence_id"]] = {
                "source_id": ev.get("source_id"), "kind": ev.get("kind"), "title": ev.get("title"),
                "section": ev.get("section"), "url": ev.get("url"), "api_version": ev.get("api_version"),
                "product_version": ev.get("product_version"), "status": ev.get("status"),
                "human_reviewer": ev.get("human_reviewer"),
                "video_md5": ev.get("video_md5"), "evidence_start": ev.get("evidence_start"), "evidence_end": ev.get("evidence_end"),
            }
    status_counts = {}
    for c in cards:
        status_counts[c["status"]] = status_counts.get(c["status"], 0) + 1

    return {
        "question": question,
        "known_conditions": known_conditions,
        "lib_name": lib_name,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 2),
        "llm": {"available": llm.available(), "provider": llm.used_provider, "decomposed": llm_decomposed,
                "extracted": llm_extracted_any, "drafted": llm_drafted, "last_error": llm.last_error},
        "reply": reply,
        "claims": cards,
        "ask_customer": [q for c in cards for q in c["ask_customer"]],
        "ask_tech": [q for c in cards for q in c["ask_tech"]],
        "cited_sources": cited,
        "status_counts": status_counts,
        "pipeline_note": "模型负责拆解/提取/起草；引用存在性、状态判定、来源门禁、冲突保留由程序完成。",
    }


def save_answer_log(answer, lib_name=None):
    lib_name = lib_name or answer.get("lib_name") or "default_lib"
    log_dir = os.path.join(LIBRARIES_DIR, lib_name, "answer_log")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"answer_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(answer, f, ensure_ascii=False, indent=2)
    return path
