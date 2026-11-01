"""WP11：资料包白名单接入（用真实的阿里云包做数据，但不调用 Embedding / VLM）。"""
import os

from _common import ALIYUN_PACK
from src import doc_ingest as di


def _pack():
    if not os.path.isdir(ALIYUN_PACK):
        print(f"  （跳过：资料包不存在 {ALIYUN_PACK}）")
        return None
    return di.load_pack(ALIYUN_PACK)


def test_registry_loads_and_has_whitelist():
    pack = _pack()
    if not pack:
        return
    assert pack["dataset_id"] == "PROMISEOS-CN-ALIYUN-01"
    assert len(pack["ingestion"]["text_paths"]) == 6 and len(pack["ingestion"]["image_paths"]) == 3


def test_text_segments_follow_whitelist_and_exclude_judgement_lines():
    pack = _pack()
    if not pack:
        return
    segs = di.build_text_segments(pack)
    assert len(segs) == 6 and {s["source_id"] for s in segs} == {f"CN-D0{i}" for i in range(1, 7)}
    for s in segs:
        assert s["kind"] == "doc" and s["text"]
        assert "理解边界" not in s["text"] and "来源：" not in s["text"] and "采集日期" not in s["text"]
        assert os.path.basename(s["local_path"]).startswith(s["source_id"])
    d02 = next(s for s in segs if s["source_id"] == "CN-D02")
    assert d02["api_version"] == "2022-12-29" and d02["boundary"]


def test_answers_and_observations_never_enter_index_text():
    pack = _pack()
    if not pack:
        return
    segs = di.build_text_segments(pack)
    assert not any("05_客户问题" in s["local_path"] for s in segs)
    imgs = di.build_image_segments(pack)   # 不传视觉描述
    assert len(imgs) == 3
    for img in imgs:
        assert img["kind"] == "image" and os.path.exists(img["frame_path"])
        # 登记表里的人工 observation（如“准确率100%”“驳回”）不得出现在检索文本里
        assert "准确率100%" not in img["text"] and "驳回" not in img["text"]
        assert "尚未生成" in img["text"] and "未生成画面描述" in img["status"]


def test_image_segments_use_vlm_description_when_given():
    pack = _pack()
    if not pack:
        return
    imgs = di.build_image_segments(pack, {"CN-I02": "页面显示“检查中”，右上角有驳回、跳过、提交任务按钮"})
    i02 = next(i for i in imgs if i["source_id"] == "CN-I02")
    assert "检查中" in i02["text"] and i02["status"].startswith("视觉模型已描述")


def test_parse_entry_markdown():
    title, body, boundary = di.parse_entry_markdown(
        "# CN-D09：示例\n\n来源：x。定位：y。\n\n采集日期：2026-11-01。\n\n正文一。\n\n理解边界（整理判断）：只描述此路线。\n"
    )
    assert title == "CN-D09：示例" and body == ["正文一。"] and boundary == "只描述此路线。"


def test_chunking_respects_limit():
    chunks = di.chunk_paragraphs(["a" * 400, "b" * 400, "c" * 100], max_chars=600)
    assert chunks == ["a" * 400, "b" * 400 + "\n" + "c" * 100]


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")


if __name__ == "__main__":
    run()
