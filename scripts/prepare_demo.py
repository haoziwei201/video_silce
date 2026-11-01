"""
一键准备演示环境（可重复运行，缓存命中会很快）：

  python scripts/prepare_demo.py                       # 建库 + 导入阿里云资料包（文档+图片）
  python scripts/prepare_demo.py --video "../第二步_国内公开资料包_阿里云/素材/CN-V01_长文档信息抽取接入视频.mp4"
                                                        # 另外处理演示视频（ASR+关键帧+VLM+清洗）并关联到库
  python scripts/prepare_demo.py --ask "我们有一批中文扫描合同……"   # 跑一题并打印承诺卡
  python scripts/prepare_demo.py --ask "..." --save-log            # 同时写 answer_log

说明：导入资料包会调用 Embedding（文档/图片各一次）和 Qwen-VL（每张图一次，有缓存）；
处理视频会调用 Paraformer、Qwen-VL（每个关键帧）、DeepSeek（画面摘要）；提问会调用 DeepSeek 与 Embedding。
"""
import os
import sys
import json
import argparse
import asyncio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.library_manager import LibraryManager
from src.asset_manager import AssetManager
from src import doc_ingest, commitment

DEFAULT_PACK = os.path.join("..", "第二步_国内公开资料包_阿里云")


def ensure_library(lib_name):
    lm = LibraryManager()
    if lib_name not in lm.list_libraries():
        ok, msg = lm.create_library(lib_name)
        print(f"[库] {msg}")
    else:
        print(f"[库] 已存在：{lib_name}")
    return lm


def ingest_pack(lib_name, pack_dir, run_vlm):
    print(f"[资料包] 导入 {pack_dir} → {lib_name}（VLM 描述图片：{'是' if run_vlm else '否'}）")
    report = doc_ingest.ingest_pack(lib_name, pack_dir, run_vlm=run_vlm)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def process_video(lib_name, video_path, category, force_rebuild):
    am = AssetManager()
    lm = LibraryManager()
    print(f"[视频] 处理 {video_path}（force_rebuild={force_rebuild}）")
    loop = asyncio.new_event_loop()
    try:
        md5 = loop.run_until_complete(
            am.process_video_asset(video_path, category=category,
                                   original_filename=os.path.basename(video_path), force_rebuild=force_rebuild)
        )
    finally:
        loop.close()
    if not md5:
        print("[视频] 处理失败")
        return None
    meta = am.get_metadata(md5)
    print(f"[视频] MD5={md5} ｜ ASR 状态={meta.get('asr_status')} ｜ 流水线版本={meta.get('pipeline_version')}")
    ok, msg = lm.add_asset_to_library(lib_name, md5)
    print(f"[视频] 关联到库：{msg}")
    return md5


def ask(lib_name, question, known, top_k, save_log):
    print(f"\n[提问] {question}")
    answer = commitment.prepare_answer(question, lib_name, known_conditions=known, top_k=top_k)
    print(f"\n=== 建议回复（{answer['elapsed_sec']}s，LLM: {answer['llm']}）===\n{answer['reply']}\n")
    for c in answer["claims"]:
        print(f"--- [{c['status']}｜{c['source_review']}] {c['customer_ask']}")
        print(f"    资料直接支持：{c['supported_statement'] or '—'}")
        print(f"    适用前提：{'；'.join(c['conditions']) or '—'}")
        print(f"    尚未确认：{'；'.join(c['unknowns']) or '—'}")
        print(f"    依据：{[ (e['kind'], e.get('source_id')) for e in c['evidence'] ]}")
        if c["dropped_evidence_ids"]:
            print(f"    ⚠ 已丢弃不存在的引用：{c['dropped_evidence_ids']}")
        if c["conflicts"]:
            print(f"    ⚠ 来源冲突：{c['conflicts']}")
    print("\n问客户：", [q["question"] for q in answer["ask_customer"]])
    print("问技术：", [(q["item"], q["suggested_role"]) for q in answer["ask_tech"]])
    if save_log:
        path = commitment.save_answer_log(answer, lib_name)
        print(f"\n[日志] 已保存 {path}")
    return answer


def main():
    ap = argparse.ArgumentParser(description="PromiseOS 演示环境准备")
    ap.add_argument("--lib", default="promiseos_demo")
    ap.add_argument("--pack", default=DEFAULT_PACK, help="资料包目录；传 none 跳过导入")
    ap.add_argument("--no-vlm", action="store_true", help="导入资料包时不用视觉模型描述图片")
    ap.add_argument("--video", default=None, help="要处理并关联的视频路径")
    ap.add_argument("--category", default="product_demo")
    ap.add_argument("--force-rebuild", action="store_true", help="视频合并/清洗缓存强制重建")
    ap.add_argument("--ask", default=None, help="跑一题")
    ap.add_argument("--ask-file", default=None, help="从 UTF-8 文本文件读取问题（避免命令行中文编码问题）")
    ap.add_argument("--known", default="", help="销售已知的客户条件")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--save-log", action="store_true")
    args = ap.parse_args()

    ensure_library(args.lib)
    if args.pack and args.pack.lower() != "none":
        ingest_pack(args.lib, args.pack, run_vlm=not args.no_vlm)
    if args.video:
        process_video(args.lib, args.video, args.category, args.force_rebuild)
    question = args.ask
    if args.ask_file:
        with open(args.ask_file, "r", encoding="utf-8") as f:
            question = next((ln.strip() for ln in f if ln.strip()), None)
    if question:
        ask(args.lib, question, args.known, args.top_k, args.save_log)


if __name__ == "__main__":
    main()
