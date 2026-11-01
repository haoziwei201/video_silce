"""E1：视听合并不再丢视觉信息（04 §1 E1 验收）。"""
from _common import V, S
from src.data_merger import merge_audio_visual_data as merge


def _types(out):
    return [(x["type"], x["time_range"]) for x in out]


def test_no_speech_keeps_visual():
    out = merge([], [V("待复核", 6, 8, "f/frame_006000.jpg")])
    assert len(out) == 1 and out[0]["type"] == "silent_action", _types(out)
    assert "待复核" in out[0]["visual_raw"]
    assert out[0]["frames"] and out[0]["frames"][0]["frame_path"] == "f/frame_006000.jpg"


def test_tail_visual_after_last_speech():
    out = merge([S("上传文件", 0, 2)], [V("待复核", 6, 8)])
    assert [x["type"] for x in out] == ["speech", "silent_action"], _types(out)
    assert out[1]["time_range"] == [6.0, 8.0]


def test_short_gap_visual():
    out = merge([S("a", 0, 2), S("b", 4, 5)], [V("待复核", 3, 3.5)])
    assert [x["type"] for x in out] == ["speech", "silent_action", "speech"], _types(out)


def test_exact_three_second_boundary():
    out = merge([S("a", 0, 2), S("b", 5, 6)], [V("待复核", 3.0, 5.0)])
    assert any(x["type"] == "silent_action" and "待复核" in x["visual_raw"] for x in out), _types(out)


def test_head_visual_and_attach_inside_speech():
    out = merge([S("a", 5, 8)], [V("首帧", 0, 2), V("内帧", 6, 8, "f/frame_006000.jpg")])
    assert out[0]["type"] == "silent_action" and "首帧" in out[0]["visual_raw"]
    assert out[1]["type"] == "speech" and "内帧" in out[1]["visual_raw"] and len(out[1]["frames"]) == 1


def test_each_visual_used_once_and_ids_sequential():
    out = merge([S("a", 0, 2), S("b", 2, 4)], [V("x", 2.0, 4.0)])
    total = sum(len(x["visual_raw"].split()) for x in out if x["visual_raw"])
    assert total == 1, "同一视觉条目被重复使用"
    assert [x["id"] for x in out] == list(range(1, len(out) + 1))


def test_cluster_groups_neighbouring_leftovers():
    out = merge([], [V("p1", 0, 2), V("p2", 2, 4), V("p3", 20, 22)])
    silent = [x for x in out if x["type"] == "silent_action"]
    assert len(silent) == 2, _types(out)
    assert silent[0]["time_range"] == [0.0, 4.0] and silent[1]["time_range"] == [20.0, 22.0]


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")


if __name__ == "__main__":
    run()
