import os
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

    def list_all_assets(self):
        """List all assets in the global video pool."""
        assets = []
        if not os.path.exists(VIDEO_POOL_DIR):
            return []
            
        for f in os.listdir(VIDEO_POOL_DIR):
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
                # Format: {md5}.ext
                md5 = os.path.splitext(f)[0]
                
                # Try to retrieve original filename from metadata
                display_name = f
                meta_path = os.path.join(GLOBAL_CACHE_DIR, md5, "metadata.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as mf:
                            meta = json.load(mf)
                            if "original_filename" in meta:
                                display_name = meta["original_filename"]
                    except Exception as e:
                        logger.warning(f"Failed to read metadata for {md5}: {e}")

                assets.append({"md5": md5, "filename": display_name, "path": os.path.join(VIDEO_POOL_DIR, f)})
        return assets

    async def process_video_asset(self, file_path, category="general", original_filename=None):
        """
        Ingest a video into the asset system:
        1. Calculate MD5 & Move to Video Pool.
        2. Generate Global Cache (ASR, VLM, RAG Ready JSON).
        Returns: md5 (str) if successful, None otherwise.
        """
        # 1. Ingest to Pool
        print(f"Processing asset: {os.path.basename(file_path)}")
        md5, pool_path = ensure_in_video_pool(file_path, VIDEO_POOL_DIR)
        if not md5:
            logger.error(f"Failed to process video: {file_path}")
            return None
        
        print(f"Asset MD5: {md5}")
        
        # Prepare Cache Directory
        cache_dir = os.path.join(GLOBAL_CACHE_DIR, md5)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Save Metadata (Original Filename)
        if original_filename:
            meta_path = os.path.join(cache_dir, "metadata.json")
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                except:
                    pass
            
            meta["original_filename"] = original_filename
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        
        # 2. Check/Generate ASR (Speech)
        raw_trans_path = os.path.join(cache_dir, "raw_trans.json")
        transcript = []
        
        if os.path.exists(raw_trans_path):
            print("✓ ASR Cache Hit")
            with open(raw_trans_path, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
        else:
            print("Running ASR...")
            audio_path = os.path.join(PROCESSED_AUDIO_DIR, f"{md5}.wav")
            vp = VideoProcessor()
            if not os.path.exists(audio_path):
                vp.extract_audio(pool_path, audio_path)
            
            stt = SpeechToText()
            # Assuming stt.transcribe returns normalized text list or dict
            raw_res = stt.transcribe(audio_path, video_md5=md5)
            transcript = stt.split_by_punctuation(raw_res)
            
            with open(raw_trans_path, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, ensure_ascii=False)

        # 3. Check/Generate VLM (Visual)
        visual_analysis_path = os.path.join(cache_dir, "visual_analysis.json")
        visual_segments = []
        
        if os.path.exists(visual_analysis_path):
            print("✓ VLM Cache Hit")
            with open(visual_analysis_path, 'r', encoding='utf-8') as f:
                visual_segments = json.load(f)
        else:
            print("Running VLM...")
            kf_dir = os.path.join(KEYFRAMES_DIR, md5)
            vp = VideoProcessor()
            # Re-use existing keyframe extraction logic
            keyframes = vp.extract_keyframes(pool_path, kf_dir, interval=2.0)
            
            vr = VisualRecognition()
            sem = asyncio.Semaphore(5) # Limit concurrency
            
            async def analyze_wrapper(kf):
                async with sem:
                    desc = await vr.analyze_image_async(kf['path'], auto_save=False)
                    return kf['time'], desc

            tasks = [analyze_wrapper(kf) for kf in keyframes]
            results = await async_tqdm.gather(*tasks, desc="VLM Analysis")
            
            await vr.save_cache() # Save internal cache of VR module if any
            
            for timestamp, desc in results:
                if desc:
                    visual_segments.append({
                        "word": f"[Visual: {desc}]",
                        "text": f"[Visual: {desc}]",
                        "start": timestamp,
                        "end": timestamp + 2.0
                    })
            
            with open(visual_analysis_path, 'w', encoding='utf-8') as f:
                json.dump(visual_segments, f, ensure_ascii=False)

        # 4. Data Merge & Cleaning (RAG Ready)
        rag_ready_path = os.path.join(cache_dir, "rag_ready.json") # cleaned_data.json
        # User mentioned cleaned_data.json, but code uses rag_ready.json. 
        # I will use cleaned_data.json as per user instruction, but for compatibility 
        # with existing batch_processor patterns I'll check what's best.
        # User said: "global_cache/{MD5}/... cleaned_data.json"
        # I will use `cleaned_data.json` to be compliant with new instructions.
        cleaned_data_path = os.path.join(cache_dir, "cleaned_data.json")
        
        if os.path.exists(cleaned_data_path):
            print("✓ Cleaned Data Cache Hit")
        else:
            print("Running Data Cleaning...")
            merged_data = merge_audio_visual_data(transcript, visual_segments)
            merged_path = os.path.join(cache_dir, "merged_raw.json")
            with open(merged_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False)
            
            cleaner = AsyncDataCleaner()
            await cleaner.process_file_async(merged_path, cleaned_data_path, category)
            
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

