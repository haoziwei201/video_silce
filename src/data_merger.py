import logging
import re

logger = logging.getLogger(__name__)

_VISUAL_PREFIX_RE = re.compile(r"^\s*\[(?:视觉画面|Visual)\s*[:：]\s*")


def _clean_visual_text(raw):
    """去掉 '[视觉画面: ...]' / '[Visual: ...]' 包装，只留描述正文。"""
    if not raw:
        return ""
    text = _VISUAL_PREFIX_RE.sub("", str(raw))
    text = text.rstrip()
    if text.endswith("]"):
        text = text[:-1]
    return text.strip()


def _normalize_visuals(visual_segments):
    """给每条视觉条目补齐 id / 时间 / 帧字段，并按开始时间排序。"""
    normalized = []
    for idx, v in enumerate(visual_segments or []):
        start = float(v.get("start", 0) or 0)
        end = float(v.get("end", start) or start)
        if end < start:
            end = start
        desc = _clean_visual_text(v.get("text") or v.get("word") or "")
        frame_path = v.get("frame_path") or None
        frame_id = v.get("frame_id") or None
        normalized.append({
            "vid": idx,
            "start": start,
            "end": end,
            "desc": desc,
            "frame_path": frame_path,
            "frame_id": frame_id,
        })
    normalized.sort(key=lambda x: x["start"])
    return normalized


def _frames_of(visuals):
    frames = []
    for v in visuals:
        if v["frame_path"] or v["frame_id"]:
            frames.append({
                "frame_id": v["frame_id"],
                "frame_path": v["frame_path"],
                "time": v["start"],
            })
    return frames


def _cluster(visuals, cluster_gap):
    """把时间上相邻（间隔 <= cluster_gap）的视觉条目聚成一组。"""
    clusters = []
    current = []
    for v in visuals:
        if not current or v["start"] - current[-1]["start"] <= cluster_gap:
            current.append(v)
        else:
            clusters.append(current)
            current = [v]
    if current:
        clusters.append(current)
    return clusters


def merge_audio_visual_data(transcript, visual_segments, cluster_gap=4.0):
    """
    融合语音转录和视觉识别结果。

    每条视觉条目只会被使用一次：落在某句语音时间范围内的挂到该语音；
    其余（无语音、首部、尾部、语音之间任意长度的空隙）按时间邻近聚成 silent_action，
    不会因为空隙长度不满某个阈值而被丢弃。

    Args:
        transcript: [{'word': str, 'start': float, 'end': float}]
        visual_segments: [{'text': str, 'start': float, 'end': float,
                           'frame_path'?: str, 'frame_id'?: str}]
        cluster_gap: 相邻未关联视觉条目聚合为同一静默片段的最大间隔（秒）

    Returns:
        list[dict]: 每项含 id / time_range / type / content / visual_context /
                    visual_raw / frames
    """
    speeches = sorted(
        (s for s in (transcript or []) if s is not None),
        key=lambda x: float(x.get("start", 0) or 0),
    )
    visuals = _normalize_visuals(visual_segments)
    used = set()
    merged = []

    for speech in speeches:
        s_start = float(speech.get("start", 0) or 0)
        s_end = float(speech.get("end", s_start) or s_start)
        attached = [
            v for v in visuals
            if v["vid"] not in used and s_start <= v["start"] <= s_end
        ]
        for v in attached:
            used.add(v["vid"])
        descs = [v["desc"] for v in attached if v["desc"]]
        merged.append({
            "time_range": [s_start, s_end],
            "type": "speech",
            "content": speech.get("word", "") or "",
            "visual_context": ("画面展示：" + " ".join(descs)) if descs else "",
            "visual_raw": " ".join(descs),
            "frames": _frames_of(attached),
        })

    leftovers = [v for v in visuals if v["vid"] not in used and (v["desc"] or v["frame_path"])]
    for group in _cluster(leftovers, cluster_gap):
        descs = [v["desc"] for v in group if v["desc"]]
        merged.append({
            "time_range": [group[0]["start"], max(v["end"] for v in group)],
            "type": "silent_action",
            "content": "[静默操作]",
            "visual_context": ("画面显示：" + " ".join(descs)) if descs else "",
            "visual_raw": " ".join(descs),
            "frames": _frames_of(group),
        })

    merged.sort(key=lambda x: (x["time_range"][0], 0 if x["type"] == "speech" else 1))
    for i, item in enumerate(merged, start=1):
        item["id"] = i

    logger.info(
        "合并完成：语音 %d 条，静默片段 %d 条，视觉条目 %d/%d 已使用",
        len(speeches), len(merged) - len(speeches), len(used), len(visuals),
    )
    return merged
