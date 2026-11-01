"""E2 / E3 / E4：证据链字段、双时间范围、语音回退、旧缓存帧路径重建。"""
import os
import json
import tempfile

from _common import ROOT
from src.rag_engine import build_video_record, build_doc_record, expand_video_hit, sanitize_metadata
from src.asset_manager import reconstruct_frame_paths, PIPELINE_VERSION
import src.speech_to_text as stt_mod


def test_video_record_carries_frames_and_kind():
    item = {"id": "7", "start": 2.0, "end": 4.0, "type": "silent_action", "content": "[静默操作]",
            "category": "demo", "visual_summary": "界面显示待人工复核",
            "frames": [{"frame_id": "frame_002500", "frame_path": "k/frame_002500.jpg", "time": 2.5}]}
    uid, doc, meta = build_video_record(item, "abc")
    assert uid == "abc_7" and meta["kind"] == "video"
    assert json.loads(meta["frame_paths_json"]) == ["k/frame_002500.jpg"] and meta["frame_count"] == 1
    assert all(isinstance(v, (str, int, float, bool)) for v in meta.values()), "Chroma 元数据必须是标量"


def test_doc_record_has_no_fake_timestamps():
    seg = {"source_id": "CN-D02", "text": "PredictModel 需要 ModelId", "title": "模型预测接口",
           "api_version": "2022-12-29", "status": "已读公开页面并整理摘要", "human_reviewer": None, "boundary": "仅此接口"}
    uid, doc, meta = build_doc_record("DS", seg)
    assert uid == "DS__CN-D02__0" and meta["kind"] == "doc"
    assert "start" not in meta and "end" not in meta, "文档条目不得伪造视频时间戳"
    assert meta["human_reviewer"] == "" and meta["api_version"] == "2022-12-29"


def test_sanitize_metadata_types():
    m = sanitize_metadata({"a": None, "b": [1, 2], "c": {"x": 1}, "d": 1.5, "e": True})
    assert m["a"] == "" and m["b"] == "[1, 2]" and json.loads(m["c"]) == {"x": 1} and m["d"] == 1.5 and m["e"] is True


def test_expand_keeps_evidence_and_context_ranges():
    store = {"abc_6": {"document": "前文", "metadata": {"start": 0.0, "end": 2.0}},
             "abc_8": {"document": "后文", "metadata": {"start": 4.0, "end": 6.0}}}
    fetch = lambda ids: {i: store[i] for i in ids if i in store}
    doc, meta = expand_video_hit("abc_7", "命中", {"source_video_md5": "abc", "seq_id": 7, "start": 2.0, "end": 4.0}, fetch)
    assert (meta["evidence_start"], meta["evidence_end"]) == (2.0, 4.0)
    assert (meta["context_start"], meta["context_end"]) == (0.0, 6.0)
    assert (meta["start"], meta["end"]) == (0.0, 6.0), "start/end 需与旧调用方兼容（扩展范围）"
    assert "【检索命中】命中" in doc and meta["is_expanded"] is True


def test_expand_without_neighbours():
    doc, meta = expand_video_hit("abc_1", "只有自己", {"source_video_md5": "abc", "seq_id": 1, "start": 0.0, "end": 1.0}, lambda ids: {})
    assert (meta["evidence_start"], meta["context_start"]) == (0.0, 0.0) and (meta["evidence_end"], meta["context_end"]) == (1.0, 1.0)


def test_reconstruct_frame_paths_from_keyframe_dir():
    tmp = tempfile.mkdtemp()
    for name in ("frame_002500.jpg", "fill_067114.jpg", "ppt_010000.jpg"):
        open(os.path.join(tmp, name), "wb").close()
    segs = [{"start": 2.5, "end": 4.5}, {"start": 67.1, "end": 69.1}, {"start": 10.0, "end": 12.0}, {"start": 500.0, "end": 502.0}]
    filled = reconstruct_frame_paths(segs, tmp)
    assert filled == 3, filled
    assert segs[0]["frame_id"] == "frame_002500" and segs[1]["frame_id"] == "fill_067114" and segs[2]["frame_id"] == "ppt_010000"
    assert "frame_path" not in segs[3], "找不到匹配帧时不得乱填"


def test_speech_fallback_only_uses_same_md5_cache():
    tmp = tempfile.mkdtemp()
    old = stt_mod.GLOBAL_CACHE_DIR
    stt_mod.GLOBAL_CACHE_DIR = tmp
    try:
        stt = stt_mod.SpeechToText()
        stt.api_key = ""
        assert stt.transcribe("x.wav", video_md5="none") == [] and stt.last_status == "failed"
        os.makedirs(os.path.join(tmp, "m1"))
        with open(os.path.join(tmp, "m1", "raw_trans.json"), "w", encoding="utf-8") as f:
            json.dump([{"word": "你好", "start": 0, "end": 1}], f)
        res = stt.transcribe("x.wav", video_md5="m1")
        assert res and res[0]["word"] == "你好" and stt.last_status == "cache"
        # 回退函数不再读取 transcripts/text_trans.json 这类与视频无关的文件
        assert stt._fallback_fake_transcribe("text_trans.wav", video_md5="nobody") == []
    finally:
        stt_mod.GLOBAL_CACHE_DIR = old


def test_pipeline_version_is_positive_int():
    assert isinstance(PIPELINE_VERSION, int) and PIPELINE_VERSION >= 2


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")


if __name__ == "__main__":
    run()
