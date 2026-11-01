"""
资料包接入（WP11）

读取素材包里的来源登记 JSON（如 第二步_国内公开资料包_阿里云/02_来源登记.json），
只按 ingestion 白名单导入文档摘要与图片，写入知识库并在库目录下留一份 source_registry.json。

纪律：
- 问题参考答案、人工“画面观察”、能力总结都不进检索索引（登记里 rule 字段写明了）。
- 文档条目没有视频时间戳，用 kind=doc 区分，不伪造 start/end。
- “理解边界（整理判断）”是团队判断，不是原文，只放元数据 boundary，不进检索文本。
- 图片的检索文本来自视觉模型描述；人工 observation 只保留在登记表供复核。
"""
import os
import re
import glob
import json
import asyncio
import logging
import datetime

from config import LIBRARIES_DIR
from src.rag_engine import VideoKnowledgeBase

logger = logging.getLogger(__name__)

_REGISTRY_PATTERNS = ("*来源登记*.json", "*来源与能力登记*.json", "*registry*.json")
_SKIP_LINE_PREFIXES = ("来源：", "来源:", "采集日期：", "采集日期:", "理解边界", "边界：", "边界:")
MAX_CHUNK_CHARS = 600


def find_registry(pack_dir):
    for pat in _REGISTRY_PATTERNS:
        hits = sorted(glob.glob(os.path.join(pack_dir, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"在 {pack_dir} 下找不到来源登记 JSON（{'/'.join(_REGISTRY_PATTERNS)}）")


def load_pack(pack_dir):
    reg_path = find_registry(pack_dir)
    with open(reg_path, "r", encoding="utf-8") as f:
        pack = json.load(f)
    pack["_pack_dir"] = os.path.abspath(pack_dir)
    pack["_registry_path"] = reg_path
    pack.setdefault("dataset_id", os.path.basename(os.path.normpath(pack_dir)))
    pack.setdefault("sources", [])
    pack.setdefault("images", [])
    pack.setdefault("videos", [])
    pack.setdefault("ingestion", {})
    return pack


def _norm(path):
    return os.path.normpath(path).replace("\\", "/")


def parse_entry_markdown(text):
    """
    把一篇 资料条目/*.md 拆成 (title, body_paragraphs, boundary_text)。
    来源/采集日期/理解边界 这些行不进正文。
    """
    title = ""
    body = []
    boundary = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if not title:
                title = line.lstrip("#").strip()
            continue
        if line.startswith(("理解边界", "边界：", "边界:")):
            boundary = re.sub(r"^[^：:]*[：:]\s*", "", line, count=1)
            continue
        if line.startswith(_SKIP_LINE_PREFIXES):
            continue
        body.append(line)
    return title, body, boundary


def chunk_paragraphs(paragraphs, max_chars=MAX_CHUNK_CHARS):
    chunks, cur = [], ""
    for p in paragraphs:
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_chars:
            cur = f"{cur}\n{p}"
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def build_text_segments(pack):
    """按 ingestion.text_paths 白名单生成文档段落。"""
    pack_dir = pack["_pack_dir"]
    by_path = {_norm(s.get("local_path", "")): s for s in pack["sources"] if s.get("local_path")}
    product = pack.get("primary_product", "")
    segments = []
    for rel in pack["ingestion"].get("text_paths", []):
        abs_path = os.path.join(pack_dir, rel)
        if not os.path.exists(abs_path):
            logger.warning(f"白名单文件不存在，跳过: {abs_path}")
            continue
        src = by_path.get(_norm(rel), {})
        with open(abs_path, "r", encoding="utf-8") as f:
            title, body, boundary_in_md = parse_entry_markdown(f.read())
        source_id = src.get("id") or os.path.splitext(os.path.basename(rel))[0].split("_")[0]
        for idx, chunk in enumerate(chunk_paragraphs(body)):
            segments.append({
                "kind": "doc",
                "source_id": source_id,
                "chunk_index": idx,
                "text": chunk,
                "title": src.get("title") or title,
                "section": src.get("section", ""),
                "url": src.get("url", ""),
                "local_path": abs_path,
                "product_id": product,
                "product_version": src.get("product_version"),
                "api_version": src.get("api_version"),
                "status": src.get("status", ""),
                "human_reviewer": src.get("human_reviewer"),
                "boundary": src.get("boundary") or boundary_in_md,
            })
    return segments


def build_image_segments(pack, descriptions=None):
    """
    按 ingestion.image_paths 白名单生成图片条目。
    descriptions: {image_id: 视觉模型描述}；没有描述的图片只带占位文本，并标记未分析。
    """
    pack_dir = pack["_pack_dir"]
    allowed = {_norm(p) for p in pack["ingestion"].get("image_paths", [])}
    by_id = {s.get("id"): s for s in pack["sources"]}
    product = pack.get("primary_product", "")
    descriptions = descriptions or {}
    segments = []
    for img in pack["images"]:
        rel = img.get("local_path", "")
        if allowed and _norm(rel) not in allowed:
            continue
        abs_path = os.path.join(pack_dir, rel)
        parent = by_id.get(img.get("parent_source_id"), {})
        desc = descriptions.get(img["id"])
        if desc:
            text = f"{parent.get('title', '')}官方配图（{img.get('section', '')}）画面内容：{desc}"
            analyzed = True
        else:
            text = f"{parent.get('title', '')}官方配图（{img.get('section', '')}）；画面描述尚未生成。"
            analyzed = False
        segments.append({
            "kind": "image",
            "source_id": img["id"],
            "chunk_index": 0,
            "text": text,
            "title": f"{parent.get('title', '')} 配图 {img['id']}",
            "section": img.get("section", ""),
            "url": img.get("source_url", ""),
            "local_path": abs_path,
            "frame_path": abs_path,
            "product_id": product,
            "product_version": None,
            "api_version": parent.get("api_version"),
            "status": ("视觉模型已描述" if analyzed else "未生成画面描述") + "；" + str(img.get("rights_status", "")),
            "human_reviewer": img.get("human_reviewer"),
            "boundary": img.get("boundary", ""),
        })
    return segments


async def describe_images_async(pack, prompt=None):
    """用视觉模型为白名单图片生成描述（真实调用 Qwen-VL，有缓存）。"""
    from src.visual_recognition import VisualRecognition
    from src.prompts import VISUAL_UI_ANALYSIS_PROMPT
    prompt = prompt or VISUAL_UI_ANALYSIS_PROMPT
    allowed = {_norm(p) for p in pack["ingestion"].get("image_paths", [])}
    vr = VisualRecognition()
    out = {}
    for img in pack["images"]:
        rel = img.get("local_path", "")
        if allowed and _norm(rel) not in allowed:
            continue
        abs_path = os.path.join(pack["_pack_dir"], rel)
        if not os.path.exists(abs_path):
            logger.warning(f"图片不存在: {abs_path}")
            continue
        desc = await vr.analyze_image_async(abs_path, prompt=prompt, auto_save=False)
        if desc:
            out[img["id"]] = desc.strip()
    await vr.save_cache()
    return out


def registry_path(lib_name):
    return os.path.join(LIBRARIES_DIR, lib_name, "source_registry.json")


def load_registry(lib_name):
    path = registry_path(lib_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"datasets": {}, "sources": {}, "images": {}, "videos": {}}


def save_registry(lib_name, registry):
    os.makedirs(os.path.dirname(registry_path(lib_name)), exist_ok=True)
    with open(registry_path(lib_name), "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def update_registry(lib_name, pack, image_descriptions=None, ingested_counts=None):
    """把资料包的来源/图片/视频登记合并进库级 source_registry.json。"""
    reg = load_registry(lib_name)
    ds = pack["dataset_id"]
    reg["datasets"][ds] = {
        "primary_product": pack.get("primary_product"),
        "collected_at": pack.get("collected_at"),
        "registry_path": pack.get("_registry_path"),
        "ingested_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ingested_counts": ingested_counts or {},
        "default_source_preference": pack.get("default_source_preference"),
    }
    for s in pack["sources"]:
        reg["sources"][s["id"]] = dict(s, dataset_id=ds)
    for img in pack["images"]:
        entry = dict(img, dataset_id=ds)
        if image_descriptions and img["id"] in image_descriptions:
            entry["vlm_description"] = image_descriptions[img["id"]]
        reg["images"][img["id"]] = entry
    for v in pack["videos"]:
        reg["videos"][v["id"]] = dict(v, dataset_id=ds)
    save_registry(lib_name, reg)
    return reg


def ingest_pack(lib_name, pack_dir, run_vlm=True):
    """
    一键接入：读登记 → 白名单文档段落 → （可选）VLM 描述图片 → 写索引 → 更新登记。
    返回报告 dict。
    """
    pack = load_pack(pack_dir)
    text_segments = build_text_segments(pack)
    descriptions = {}
    if run_vlm and pack["ingestion"].get("image_paths"):
        try:
            descriptions = asyncio.run(describe_images_async(pack))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            descriptions = loop.run_until_complete(describe_images_async(pack))
            loop.close()
        except Exception as e:
            logger.error(f"图片视觉描述失败，图片将以占位文本入库: {e}")
    image_segments = build_image_segments(pack, descriptions)

    vkb = VideoKnowledgeBase(lib_name=lib_name)
    n_doc = vkb.add_doc_segments(pack["dataset_id"], text_segments)
    n_img = vkb.add_doc_segments(pack["dataset_id"], image_segments)
    counts = {"doc_segments": n_doc, "image_segments": n_img, "images_described": len(descriptions)}
    update_registry(lib_name, pack, descriptions, counts)
    return {
        "dataset_id": pack["dataset_id"],
        "primary_product": pack.get("primary_product"),
        "lib_name": lib_name,
        **counts,
        "videos_registered": [v["id"] for v in pack["videos"]],
        "note": "文档/图片已入索引；视频文件需另行通过资产中心处理后关联到本库。",
    }
