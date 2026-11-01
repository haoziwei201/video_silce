import os
import re
import json
import shutil
import asyncio
import logging
from tqdm.asyncio import tqdm as async_tqdm

from config import (
    VIDEO_POOL_DIR, GLOBAL_CACHE_DIR, PROCESSED_AUDIO_DIR, 
    KEYFRAMES_DIR, SLICE_CACHE_DIR
)
from src.utils import ensure_in_video_pool, get_file_hash
from src.video_processor import VideoProcessor
from src.speech_to_text import SpeechToText
from src.visual_recognition import VisualRecognition
from src.data_merger import merge_audio_visual_data
from src.data_cleaner import AsyncDataCleaner

logger = logging.getLogger(__name__)

# 合并/清洗逻辑的版本号。改动 data_merger / data_cleaner 的输出结构时必须递增，
# 否则旧的 cleaned_data.json 会被当作缓存命中而跳过重建。
PIPELINE_VERSION = 2

# 这些类别的视频按“软件界面/产品演示”处理：视觉模型抄写界面状态文字而不是描述场景氛围
UI_CATEGORIES = {"product_demo", "software_ui", "saas_demo", "ui"}

_FRAME_NAME_RE = re.compile(r"^(?:frame|ppt|fill)_(\d{6,})\.jpg$", re.IGNORECASE)


def _read_json(path, default=None):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"读取 {path} 失败: {e}")
    return default


def _write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def reconstruct_frame_paths(visual_segments, kf_dir):
    """
    为缺少 frame_path 的旧视觉缓存按时间戳反查关键帧文件。
    关键帧文件名形如 frame_002500.jpg（毫秒），允许 ±0.3s 的误差。
    返回补齐的条目数。
    """
    if not os.path.isdir(kf_dir):
        return 0
    by_ms = {}
    for name in os.listdir(kf_dir):
        m = _FRAME_NAME_RE.match(name)
        if m:
            by_ms[int(m.group(1))] = name
    if not by_ms:
        return 0
    keys = sorted(by_ms)
    filled = 0
    for seg in visual_segments:
        if seg.get("frame_path"):
            continue
        target = int(round(float(seg.get("start", 0)) * 1000))
        best = min(keys, key=lambda k: abs(k - target))
        if abs(best - target) <= 300:
            seg["frame_path"] = os.path.join(kf_dir, by_ms[best])
            seg["frame_id"] = os.path.splitext(by_ms[best])[0]
            filled += 1
    return filled


class AssetManager:
    """
    Manages physical assets (videos) and their derived global caches (ASR, VLM, RAG data).
    Implements Single Source of Truth (SSOT).
    """

    def __init__(self):
        pass

    def get_video_path(self, md5):
        """Find video file in pool by MD5"""
        for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            path = os.path.join(VIDEO_POOL_DIR, f"{md5}{ext}")
            if os.path.exists(path):
                return path
        return None

    def get_metadata(self, md5):
        return _read_json(os.path.join(GLOBAL_CACHE_DIR, md5, "metadata.json"), {}) or {}

    def list_all_assets(self):
        """List all assets in the global video pool."""
        assets = []
        if not os.path.exists(VIDEO_POOL_DIR):
            return []
            
        for f in os.listdir(VIDEO_POOL_DIR):
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
                md5 = os.path.splitext(f)[0]
                meta = self.get_metadata(md5)
                assets.append({
                    "md5": md5,
                    "filename": meta.get("original_filename", f),
                    "path": os.path.join(VIDEO_POOL_DIR, f),
                    "asr_status": meta.get("asr_status", "unknown"),
                    "pipeline_version": meta.get("pipeline_version"),
                })
        return assets

    async def process_video_asset(self, file_path, category="general", original_filename=None, force_rebuild=False):
        """
        Ingest a video into the asset system:
        1. Calculate MD5 & Move to Video Pool.
        2. Generate Global Cache (ASR, VLM, RAG Ready JSON).
        Returns: md5 (str) if successful, None otherwise.

        force_rebuild: 忽略已有的合并/清洗缓存重新生成（ASR 与 VLM 缓存仍复用）。
        """
        print(f"Processing asset: {os.path.basename(file_path)}")
        md5, pool_path = ensure_in_video_pool(file_path, VIDEO_POOL_DIR)
        if not md5:
            logger.error(f"Failed to process video: {file_path}")
            return None
        
        print(f"Asset MD5: {md5}")
        
        cache_dir = os.path.join(GLOBAL_CACHE_DIR, md5)
        os.makedirs(cache_dir, exist_ok=True)
        meta_path = os.path.join(cache_dir, "metadata.json")
        meta = _read_json(meta_path, {}) or {}
        if original_filename:
            meta["original_filename"] = original_filename
        _write_json(meta_path, meta)
        
        # 2. ASR (Speech)
        raw_trans_path = os.path.join(cache_dir, "raw_trans.json")
        transcript = []
        
        if os.path.exists(raw_trans_path):
            print("✓ ASR Cache Hit")
            transcript = _read_json(raw_trans_path, []) or []
            meta.setdefault("asr_status", "ok" if transcript else "no_speech")
        else:
            print("Running ASR...")
            audio_path = os.path.join(PROCESSED_AUDIO_DIR, f"{md5}.wav")
            vp = VideoProcessor()
            if not os.path.exists(audio_path):
                vp.extract_audio(pool_path, audio_path)
            
            stt = SpeechToText()
            raw_res = stt.transcribe(audio_path, video_md5=md5)
            asr_status = stt.last_status or ("ok" if raw_res else "failed")
            meta["asr_status"] = asr_status
            transcript = stt.split_by_punctuation(raw_res) if raw_res else []
            if asr_status in ("ok", "no_speech", "cache"):
                _write_json(raw_trans_path, transcript)
            else:
                # 识别失败不落缓存，否则下次会把空结果当成“无语音”直接命中
                logger.error(f"[{md5}] 语音识别失败，本次不写入 raw_trans.json；视觉信息仍会继续处理")
        _write_json(meta_path, meta)

        # 3. VLM (Visual)
        visual_analysis_path = os.path.join(cache_dir, "visual_analysis.json")
        visual_segments = []
        kf_dir = os.path.join(KEYFRAMES_DIR, md5)
        
        if os.path.exists(visual_analysis_path):
            print("✓ VLM Cache Hit")
            visual_segments = _read_json(visual_analysis_path, []) or []
            if any(not s.get("frame_path") for s in visual_segments):
                filled = reconstruct_frame_paths(visual_segments, kf_dir)
                if filled:
                    print(f"  为旧缓存补齐关键帧路径 {filled} 条")
                    _write_json(visual_analysis_path, visual_segments)
        else:
            print("Running VLM...")
            vp = VideoProcessor()
            keyframes = vp.extract_keyframes(pool_path, kf_dir, interval=2.0)
            
            vr = VisualRecognition()
            sem = asyncio.Semaphore(5)
            # 产品演示/软件界面类视频用界面状态提示词：抄写状态标签、按钮、人工环节；其余保持通用描述
            from src.prompts import VISUAL_ANALYSIS_PROMPT, VISUAL_UI_ANALYSIS_PROMPT
            vlm_prompt = VISUAL_UI_ANALYSIS_PROMPT if category in UI_CATEGORIES else VISUAL_ANALYSIS_PROMPT

            async def analyze_wrapper(kf):
                async with sem:
                    desc = await vr.analyze_image_async(kf['path'], prompt=vlm_prompt, auto_save=False)
                    return kf['time'], kf['path'], desc

            tasks = [analyze_wrapper(kf) for kf in keyframes]
            results = await async_tqdm.gather(*tasks, desc="VLM Analysis")
            
            await vr.save_cache()
            
            for timestamp, frame_path, desc in results:
                if desc:
                    visual_segments.append({
                        "word": f"[Visual: {desc}]",
                        "text": f"[Visual: {desc}]",
                        "start": timestamp,
                        "end": timestamp + 2.0,
                        "frame_path": frame_path,
                        "frame_id": os.path.splitext(os.path.basename(frame_path))[0],
                    })
            
            _write_json(visual_analysis_path, visual_segments)

        # 4. Data Merge & Cleaning (RAG Ready)
        cleaned_data_path = os.path.join(cache_dir, "cleaned_data.json")
        cache_ok = (
            os.path.exists(cleaned_data_path)
            and meta.get("pipeline_version") == PIPELINE_VERSION
            and not force_rebuild
        )
        if cache_ok:
            print("✓ Cleaned Data Cache Hit")
        else:
            if os.path.exists(cleaned_data_path):
                print(f"合并/清洗缓存版本过旧或要求重建（当前 v{PIPELINE_VERSION}），重新生成...")
            else:
                print("Running Data Cleaning...")
            merged_data = merge_audio_visual_data(transcript, visual_segments)
            merged_path = os.path.join(cache_dir, "merged_raw.json")
            _write_json(merged_path, merged_data)
            
            cleaner = AsyncDataCleaner()
            await cleaner.process_file_async(merged_path, cleaned_data_path, category)
            meta["pipeline_version"] = PIPELINE_VERSION
            _write_json(meta_path, meta)
            
        return md5

    def get_cached_slice_path(self, md5, start_time, end_time, params_hash="default"):
        """
        Check if a slice already exists in cache.
        Slice Cache Naming: {md5}_{start}_{end}_{params}.mp4
        """
        filename = f"{md5}_{int(start_time*100)}_{int(end_time*100)}_{params_hash}.mp4"
        path = os.path.join(SLICE_CACHE_DIR, filename)
        if os.path.exists(path):
            return path
        return None

    def save_slice_to_cache(self, temp_path, md5, start_time, end_time, params_hash="default"):
        """
        Save a generated slice to cache.
        """
        filename = f"{md5}_{int(start_time*100)}_{int(end_time*100)}_{params_hash}.mp4"
        target_path = os.path.join(SLICE_CACHE_DIR, filename)
        shutil.copy2(temp_path, target_path)
        return target_path
