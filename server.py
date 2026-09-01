# server.py
"""
server.py - 视频智能剪辑工具后端服务
完整版，修复视频路径问题和搜索显示问题
"""

import os
import sys
import json
import logging
import glob
import asyncio
import hashlib
import shutil
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入配置
from config import (
    INPUT_VIDEO_DIR, OUTPUT_VIDEO_DIR, PROCESSED_AUDIO_DIR,
    TRANSCRIPTS_DIR, ANALYSIS_RESULTS_DIR, SLICE_VIDEO_DIR, 
    KEYFRAMES_DIR, RAGSCRIPTS_DIR, VERTICAL_VIDEO_DIR, 
    VIDEO_POOL_DIR, GLOBAL_CACHE_DIR, LIBRARIES_DIR,
    SLICE_CACHE_DIR
)

# 导入各模块
from src.asset_manager import AssetManager
from src.library_manager import LibraryManager
from src.rag_engine import VideoKnowledgeBase
from src.video_processor import VideoProcessor
from src.speech_to_text import SpeechToText
from src.text_analyzer import TextAnalyzer
from src.visual_recognition import VisualRecognition
from src.data_merger import merge_audio_visual_data
from src.data_cleaner import clean_json_data, AsyncDataCleaner
from src.utils import get_file_hash, ensure_in_video_pool

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 辅助函数 ====================

def ensure_directories():
    """确保所有必要的目录都存在"""
    directories = [
        INPUT_VIDEO_DIR, OUTPUT_VIDEO_DIR, PROCESSED_AUDIO_DIR,
        TRANSCRIPTS_DIR, ANALYSIS_RESULTS_DIR, SLICE_VIDEO_DIR,
        KEYFRAMES_DIR, VERTICAL_VIDEO_DIR, VIDEO_POOL_DIR,
        GLOBAL_CACHE_DIR, LIBRARIES_DIR, SLICE_CACHE_DIR,
        RAGSCRIPTS_DIR
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"确保目录存在: {directory}")
    
    return True

def safe_filename(filename: str) -> str:
    """安全处理文件名，保留中文字符但移除危险字符"""
    dangerous_chars = '<>:"/\\|?*'
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    return filename

def encode_filename(filename: str) -> str:
    """URL编码文件名（用于传输）"""
    return urllib.parse.quote(filename)

def decode_filename(encoded: str) -> str:
    """URL解码文件名"""
    return urllib.parse.unquote(encoded)

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

# ==================== 文件列表获取函数 ====================

def get_video_files() -> List[Dict[str, Any]]:
    """获取输入目录中的所有视频文件"""
    video_files = []
    if os.path.exists(INPUT_VIDEO_DIR):
        for file in os.listdir(INPUT_VIDEO_DIR):
            if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                file_path = os.path.join(INPUT_VIDEO_DIR, file)
                stat = os.stat(file_path)
                video_files.append({
                    "name": file,
                    "name_encoded": encode_filename(file),
                    "path": file_path,
                    "size": stat.st_size,
                    "size_formatted": format_size(stat.st_size),
                    "modified": stat.st_mtime,
                    "modified_str": datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
    return sorted(video_files, key=lambda x: x['name'])

def get_transcript_files() -> List[Dict[str, Any]]:
    """获取转录目录中的所有JSON文件"""
    json_files = []
    if os.path.exists(TRANSCRIPTS_DIR):
        for file in os.listdir(TRANSCRIPTS_DIR):
            if file.lower().endswith('.json'):
                file_path = os.path.join(TRANSCRIPTS_DIR, file)
                stat = os.stat(file_path)
                json_files.append({
                    "name": file,
                    "name_encoded": encode_filename(file),
                    "path": file_path,
                    "size": stat.st_size,
                    "size_formatted": format_size(stat.st_size),
                    "modified": stat.st_mtime,
                    "modified_str": datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
    return sorted(json_files, key=lambda x: x['name'])

def get_rag_files() -> List[Dict[str, Any]]:
    """获取RAG目录中的RAG文件"""
    rag_files = []
    if os.path.exists(RAGSCRIPTS_DIR):
        for file in os.listdir(RAGSCRIPTS_DIR):
            if file.lower().endswith('_rag.json') or file.lower().endswith('_cleaned.json'):
                file_path = os.path.join(RAGSCRIPTS_DIR, file)
                stat = os.stat(file_path)
                rag_files.append({
                    "name": file,
                    "name_encoded": encode_filename(file),
                    "path": file_path,
                    "size": stat.st_size,
                    "size_formatted": format_size(stat.st_size),
                    "modified": stat.st_mtime,
                    "modified_str": datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
    return sorted(rag_files, key=lambda x: x['name'])

def get_analysis_files() -> List[Dict[str, Any]]:
    """获取分析结果目录中的文件"""
    analysis_files = []
    if os.path.exists(ANALYSIS_RESULTS_DIR):
        for file in os.listdir(ANALYSIS_RESULTS_DIR):
            if file.lower().endswith('_analysis.json'):
                file_path = os.path.join(ANALYSIS_RESULTS_DIR, file)
                stat = os.stat(file_path)
                analysis_files.append({
                    "name": file,
                    "name_encoded": encode_filename(file),
                    "path": file_path,
                    "size": stat.st_size,
                    "size_formatted": format_size(stat.st_size),
                    "modified": stat.st_mtime,
                    "modified_str": datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
    return sorted(analysis_files, key=lambda x: x['name'])

def get_video_path_by_md5(md5: str) -> Optional[Dict[str, Any]]:
    """通过MD5查找视频文件路径"""
    if not md5:
        return None
    
    result = {
        "md5": md5,
        "path": None,
        "filename": None,
        "original_name": None,
        "exists": False
    }
    
    # 1. 从视频池查找（主要存储位置）
    for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
        pool_path = os.path.join(VIDEO_POOL_DIR, f"{md5}{ext}")
        if os.path.exists(pool_path):
            result["path"] = pool_path
            result["filename"] = f"{md5}{ext}"
            result["exists"] = True
            break
    
    # 2. 如果视频池没有，从input目录查找（兼容旧数据）
    if not result["exists"]:
        input_files = get_video_files()
        for f in input_files:
            if md5 in f['name'] or md5 == os.path.splitext(f['name'])[0]:
                result["path"] = f['path']
                result["filename"] = f['name']
                result["exists"] = True
                break
    
    # 3. 获取原名
    if result["exists"]:
        meta_path = os.path.join(GLOBAL_CACHE_DIR, md5, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                    result["original_name"] = meta_data.get("original_filename", result["filename"])
            except Exception as e:
                logger.error(f"读取元数据失败 {md5}: {e}")
                result["original_name"] = result["filename"]
        else:
            result["original_name"] = result["filename"]
    
    return result

# ==================== 库管理函数 ====================

def create_library(lib_name: str) -> Dict[str, Any]:
    """创建新的知识库"""
    result = {
        "success": False,
        "message": "",
        "lib_name": lib_name
    }
    
    try:
        lm = LibraryManager()
        success, message = lm.create_library(lib_name)
        
        result["success"] = success
        result["message"] = message
        
        if success:
            logger.info(f"创建知识库成功: {lib_name}")
        else:
            logger.warning(f"创建知识库失败: {message}")
            
    except Exception as e:
        logger.error(f"创建知识库出错: {e}")
        result["message"] = f"创建知识库出错: {e}"
    
    return result

def delete_library(lib_name: str) -> Dict[str, Any]:
    """删除知识库"""
    result = {
        "success": False,
        "message": "",
        "lib_name": lib_name
    }
    
    try:
        if lib_name == "default_lib":
            result["message"] = "不能删除默认库"
            return result
            
        lm = LibraryManager()
        libraries = lm.list_libraries()
        
        if lib_name not in libraries:
            result["message"] = f"知识库不存在: {lib_name}"
            return result
        
        # 删除库目录
        lib_path = os.path.join(LIBRARIES_DIR, lib_name)
        if os.path.exists(lib_path):
            shutil.rmtree(lib_path)
        
        result["success"] = True
        result["message"] = f"知识库已删除: {lib_name}"
        logger.info(f"删除知识库成功: {lib_name}")
            
    except Exception as e:
        logger.error(f"删除知识库出错: {e}")
        result["message"] = f"删除知识库出错: {e}"
    
    return result

def get_libraries() -> List[str]:
    """获取所有可用的逻辑库名称"""
    try:
        lm = LibraryManager()
        return lm.list_libraries()
    except Exception as e:
        logger.error(f"获取知识库列表失败: {e}")
        return ["default_lib"]

def get_library_info(lib_name: str) -> Dict[str, Any]:
    """获取知识库详细信息，返回带原名的资产信息"""
    result = {
        "success": False,
        "message": "",
        "lib_name": lib_name,
        "assets": {},
        "asset_count": 0,
        "rag_count": 0,
        "created_at": None
    }
    
    try:
        lm = LibraryManager()
        assets = lm.get_library_assets(lib_name)
        
        # 增强资产信息，添加原名
        enhanced_assets = {}
        for md5, asset_info in assets.items():
            # 获取视频信息
            video_info = get_video_path_by_md5(md5)
            
            enhanced_assets[md5] = {
                "md5": md5,
                "filename": asset_info.get('filename', f"{md5}.mp4"),
                "path": video_info['path'] if video_info else asset_info.get('path'),
                "display_name": video_info['original_name'] if video_info and video_info['original_name'] else asset_info.get('filename', md5),
                "exists": video_info['exists'] if video_info else False
            }
        
        # 获取RAG信息
        try:
            vkb = VideoKnowledgeBase(lib_name=lib_name)
            rag_count = vkb.collection.count()
        except:
            rag_count = 0
        
        # 获取创建时间
        lib_path = os.path.join(LIBRARIES_DIR, lib_name)
        if os.path.exists(lib_path):
            created_at = datetime.fromtimestamp(os.path.getctime(lib_path)).isoformat()
        else:
            created_at = None
        
        result["success"] = True
        result["assets"] = enhanced_assets
        result["asset_count"] = len(enhanced_assets)
        result["rag_count"] = rag_count
        result["created_at"] = created_at
        result["message"] = f"获取知识库信息成功"
        
    except Exception as e:
        logger.error(f"获取知识库信息出错: {e}")
        result["message"] = f"获取知识库信息出错: {e}"
    
    return result

def add_asset_to_library(lib_name: str, asset_md5: str) -> Dict[str, Any]:
    """添加资产到知识库"""
    result = {
        "success": False,
        "message": "",
        "lib_name": lib_name,
        "asset_md5": asset_md5
    }
    
    try:
        lm = LibraryManager()
        success, message = lm.add_asset_to_library(lib_name, asset_md5)
        
        result["success"] = success
        result["message"] = message
        
        if success:
            logger.info(f"添加资产到知识库成功: {lib_name} - {asset_md5}")
        
    except Exception as e:
        logger.error(f"添加资产到知识库出错: {e}")
        result["message"] = f"添加资产到知识库出错: {e}"
    
    return result

def remove_asset_from_library(lib_name: str, asset_md5: str) -> Dict[str, Any]:
    """从知识库移除资产"""
    result = {
        "success": False,
        "message": "",
        "lib_name": lib_name,
        "asset_md5": asset_md5
    }
    
    try:
        lib_config_path = os.path.join(LIBRARIES_DIR, lib_name, "lib_config.json")
        if os.path.exists(lib_config_path):
            with open(lib_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if asset_md5 in config.get("videos", []):
                config["videos"].remove(asset_md5)
                
                with open(lib_config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                
                # 从向量库中删除
                try:
                    vkb = VideoKnowledgeBase(lib_name=lib_name)
                    ids_to_delete = vkb.collection.get(where={"source_video_md5": asset_md5})['ids']
                    if ids_to_delete:
                        vkb.collection.delete(ids=ids_to_delete)
                except Exception as e:
                    logger.warning(f"从向量库删除失败: {e}")
                
                result["success"] = True
                result["message"] = f"资产已从知识库移除: {asset_md5}"
                logger.info(f"从知识库移除资产成功: {lib_name} - {asset_md5}")
            else:
                result["message"] = f"资产不在知识库中: {asset_md5}"
        else:
            result["message"] = f"知识库配置文件不存在: {lib_config_path}"
        
    except Exception as e:
        logger.error(f"从知识库移除资产出错: {e}")
        result["message"] = f"从知识库移除资产出错: {e}"
    
    return result

# ==================== 资产管理函数 ====================

def get_global_assets() -> List[Dict[str, Any]]:
    """获取全局资产列表"""
    try:
        am = AssetManager()
        assets = am.list_all_assets()
        
        # 增强资产信息
        enhanced_assets = []
        for asset in assets:
            md5 = asset['md5']
            cache_dir = os.path.join(GLOBAL_CACHE_DIR, md5)
            
            # 检查是否有分析结果
            has_asr = os.path.exists(os.path.join(cache_dir, "raw_trans.json"))
            has_cleaned = os.path.exists(os.path.join(cache_dir, "cleaned_data.json"))
            has_visual = os.path.exists(os.path.join(cache_dir, "visual_analysis.json"))
            has_keyframes = os.path.exists(os.path.join(KEYFRAMES_DIR, md5))
            
            # 获取原名
            original_name = asset['filename']
            meta_path = os.path.join(cache_dir, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        if "original_filename" in meta:
                            original_name = meta["original_filename"]
                except:
                    pass
            
            enhanced_assets.append({
                "md5": md5,
                "filename": asset['filename'],
                "display_name": original_name,
                "path": asset['path'],
                "has_asr": has_asr,
                "has_cleaned": has_cleaned,
                "has_visual": has_visual,
                "has_keyframes": has_keyframes,
                "cache_dir": cache_dir if os.path.exists(cache_dir) else None
            })
        
        return enhanced_assets
    except Exception as e:
        logger.error(f"获取全局资产失败: {e}")
        return []

def get_asset_info(asset_md5: str) -> Dict[str, Any]:
    """获取资产详细信息"""
    result = {
        "success": False,
        "message": "",
        "asset_md5": asset_md5,
        "exists": False
    }
    
    try:
        cache_dir = os.path.join(GLOBAL_CACHE_DIR, asset_md5)
        
        if not os.path.exists(cache_dir):
            result["message"] = f"资产不存在: {asset_md5}"
            return result
        
        # 获取视频信息
        video_info = get_video_path_by_md5(asset_md5)
        
        # 收集资产信息
        asset_info = {
            "md5": asset_md5,
            "cache_dir": cache_dir,
            "video_path": video_info['path'] if video_info else None,
            "video_exists": video_info['exists'] if video_info else False,
            "original_name": video_info['original_name'] if video_info else None,
            "files": os.listdir(cache_dir) if os.path.exists(cache_dir) else [],
            "has_raw_trans": os.path.exists(os.path.join(cache_dir, "raw_trans.json")),
            "has_merged_raw": os.path.exists(os.path.join(cache_dir, "merged_raw.json")),
            "has_cleaned_data": os.path.exists(os.path.join(cache_dir, "cleaned_data.json")),
            "has_visual_analysis": os.path.exists(os.path.join(cache_dir, "visual_analysis.json")),
            "has_metadata": os.path.exists(os.path.join(cache_dir, "metadata.json")),
            "keyframes": []
        }
        
        # 获取关键帧列表
        keyframes_dir = os.path.join(KEYFRAMES_DIR, asset_md5)
        if os.path.exists(keyframes_dir):
            keyframes = []
            for f in os.listdir(keyframes_dir):
                if f.lower().endswith(('.jpg', '.png')):
                    keyframes.append({
                        "name": f,
                        "path": os.path.join(keyframes_dir, f),
                        "size": os.path.getsize(os.path.join(keyframes_dir, f))
                    })
            asset_info["keyframes"] = sorted(keyframes, key=lambda x: x['name'])
        
        # 获取切片列表
        slices_dir = SLICE_CACHE_DIR
        if os.path.exists(slices_dir):
            slices = []
            for f in os.listdir(slices_dir):
                if f.startswith(asset_md5) and f.endswith('.mp4'):
                    try:
                        parts = f.replace('.mp4', '').split('_')
                        if len(parts) >= 3:
                            start = float(parts[1]) / 100
                            end = float(parts[2]) / 100
                        else:
                            start = end = 0
                    except:
                        start = end = 0
                    
                    slices.append({
                        "name": f,
                        "path": os.path.join(slices_dir, f),
                        "start": start,
                        "end": end,
                        "size": os.path.getsize(os.path.join(slices_dir, f))
                    })
            asset_info["slices"] = sorted(slices, key=lambda x: x['start'])
        
        # 读取元数据
        meta_path = os.path.join(cache_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                asset_info["metadata"] = json.load(f)
        
        result["success"] = True
        result["asset_info"] = asset_info
        result["exists"] = True
        result["message"] = f"获取资产信息成功"
        
    except Exception as e:
        logger.error(f"获取资产信息出错: {e}")
        result["message"] = f"获取资产信息出错: {e}"
    
    return result

def delete_asset(asset_md5: str) -> Dict[str, Any]:
    """从全局池删除资产"""
    result = {
        "success": False,
        "message": "",
        "asset_md5": asset_md5
    }
    
    try:
        cache_dir = os.path.join(GLOBAL_CACHE_DIR, asset_md5)
        keyframes_dir = os.path.join(KEYFRAMES_DIR, asset_md5)
        
        # 删除缓存
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        
        # 删除关键帧
        if os.path.exists(keyframes_dir):
            shutil.rmtree(keyframes_dir)
        
        # 删除视频池中的文件
        for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            pool_path = os.path.join(VIDEO_POOL_DIR, f"{asset_md5}{ext}")
            if os.path.exists(pool_path):
                os.remove(pool_path)
        
        # 删除切片缓存
        slices_dir = SLICE_CACHE_DIR
        if os.path.exists(slices_dir):
            for f in os.listdir(slices_dir):
                if f.startswith(asset_md5):
                    os.remove(os.path.join(slices_dir, f))
        
        result["success"] = True
        result["message"] = f"资产已删除: {asset_md5}"
        logger.info(f"删除资产成功: {asset_md5}")
        
    except Exception as e:
        logger.error(f"删除资产出错: {e}")
        result["message"] = f"删除资产出错: {e}"
    
    return result

# ==================== 视频处理函数 ====================

async def process_video_async(video_path: str, original_filename: str, category: str = "general") -> Dict[str, Any]:
    """异步处理视频"""
    result = {
        "success": False,
        "message": "",
        "md5": None
    }
    
    try:
        am = AssetManager()
        md5 = await am.process_video_asset(video_path, category, original_filename)
        
        if md5:
            result["success"] = True
            result["md5"] = md5
            result["message"] = f"视频处理成功，MD5: {md5}"
        else:
            result["message"] = "视频处理失败"
            
    except Exception as e:
        logger.error(f"处理视频出错: {e}")
        result["message"] = f"处理视频出错: {e}"
    
    return result

def process_video_sync(video_path: str, original_filename: str, category: str = "general") -> Dict[str, Any]:
    """同步处理视频（包装异步函数）"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_video_async(video_path, original_filename, category))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"同步处理视频出错: {e}")
        return {
            "success": False,
            "message": f"处理视频出错: {e}",
            "md5": None
        }

def data_processing(video_filename: str, category: str = "general") -> Dict[str, Any]:
    """
    数据准备功能：处理视频文件
    """
    result = {
        "success": False,
        "message": "",
        "video_name": "",
        "video_md5": "",
        "transcript_count": 0,
        "keyframes_count": 0,
        "visual_segments_count": 0
    }
    
    try:
        logger.info(f"开始进行数据处理: {video_filename}")
        
        video_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
        
        if not os.path.exists(video_path):
            error_msg = f"视频文件不存在: {video_path}"
            logger.error(error_msg)
            result["message"] = error_msg
            return result
        
        # 使用AssetManager处理
        am = AssetManager()
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            md5 = loop.run_until_complete(
                am.process_video_asset(video_path, category, video_filename)
            )
            loop.close()
        except RuntimeError:
            md5 = asyncio.run(am.process_video_asset(video_path, category, video_filename))
        
        if not md5:
            result["message"] = "视频处理失败"
            return result
        
        video_name = os.path.splitext(video_filename)[0]
        result["video_name"] = video_name
        result["video_md5"] = md5
        
        # 统计转录数量
        cache_dir = os.path.join(GLOBAL_CACHE_DIR, md5)
        trans_path = os.path.join(cache_dir, "raw_trans.json")
        if os.path.exists(trans_path):
            with open(trans_path, 'r', encoding='utf-8') as f:
                trans_data = json.load(f)
                result["transcript_count"] = len(trans_data)
        
        # 统计关键帧数量
        kf_dir = os.path.join(KEYFRAMES_DIR, md5)
        if os.path.exists(kf_dir):
            result["keyframes_count"] = len([f for f in os.listdir(kf_dir) if f.endswith(('.jpg', '.png'))])
        
        # 统计视觉片段
        visual_path = os.path.join(cache_dir, "visual_analysis.json")
        if os.path.exists(visual_path):
            with open(visual_path, 'r', encoding='utf-8') as f:
                visual_data = json.load(f)
                result["visual_segments_count"] = len(visual_data)
        
        result["success"] = True
        result["message"] = f"数据处理完成，MD5: {md5}"
        
    except Exception as e:
        logger.exception(f"数据处理出错: {str(e)}")
        result["message"] = f"数据处理出错: {str(e)}"
    
    return result

# ==================== RAG构建函数 ====================

def rag_building(rag_filename: Optional[str] = None, 
                 source_json: Optional[str] = None,
                 category: str = "general",
                 lib_name: str = "default_lib") -> Dict[str, Any]:
    """
    RAG构建功能：清洗数据并构建知识库
    """
    result = {
        "success": False,
        "message": "",
        "rag_filename": None,
        "total_items": 0,
        "collection_count": 0,
        "lib_name": lib_name,
        "video_md5": None
    }
    
    try:
        logger.info("开始RAG数据准备与测试")
        
        # 如果提供了源JSON，先清洗
        if source_json:
            json_path = os.path.join(TRANSCRIPTS_DIR, source_json)
            if not os.path.exists(json_path):
                for md5_dir in os.listdir(GLOBAL_CACHE_DIR):
                    merged_path = os.path.join(GLOBAL_CACHE_DIR, md5_dir, "merged_raw.json")
                    if os.path.exists(merged_path):
                        json_path = merged_path
                        break
            
            if not os.path.exists(json_path):
                result["message"] = f"JSON文件不存在: {json_path}"
                return result
            
            rag_filename = source_json.replace(".json", "_rag.json")
            rag_path = os.path.join(RAGSCRIPTS_DIR, rag_filename)
            
            logger.info(f"清洗数据: {source_json} -> {rag_filename}")
            clean_json_data(json_path, rag_path, category_tag=category)
            result["rag_filename"] = rag_filename
        
        # 如果没有指定RAG文件名，使用最新的
        if not rag_filename:
            rag_files = [f['name'] for f in get_rag_files()]
            if not rag_files:
                result["message"] = "没有找到RAG文件"
                return result
            rag_filename = rag_files[-1]
        
        rag_path = os.path.join(RAGSCRIPTS_DIR, rag_filename)
        if not os.path.exists(rag_path):
            for md5_dir in os.listdir(GLOBAL_CACHE_DIR):
                cleaned_path = os.path.join(GLOBAL_CACHE_DIR, md5_dir, "cleaned_data.json")
                if os.path.exists(cleaned_path):
                    rag_path = os.path.join(RAGSCRIPTS_DIR, f"{md5_dir}_rag.json")
                    shutil.copy2(cleaned_path, rag_path)
                    rag_filename = os.path.basename(rag_path)
                    break
        
        if not os.path.exists(rag_path):
            result["message"] = f"RAG文件不存在: {rag_path}"
            return result
        
        # 加载清洗后的数据
        with open(rag_path, 'r', encoding='utf-8') as f:
            rag_data = json.load(f)
        
        logger.info(f"RAG数据加载完成，共 {len(rag_data)} 条")
        result["total_items"] = len(rag_data)
        
        # 获取 Video MD5
        base_name = rag_filename.replace("_rag.json", "").replace("_cleaned.json", "").replace(".json", "")
        video_md5 = None
        
        for ext in ['.mp4', '.mov', '.avi', '.mkv']:
            pool_path = os.path.join(VIDEO_POOL_DIR, f"{base_name}{ext}")
            if os.path.exists(pool_path):
                video_md5 = base_name
                logger.info(f"找到对应视频MD5: {video_md5}")
                break
        
        if not video_md5:
            if os.path.exists(os.path.join(GLOBAL_CACHE_DIR, base_name)):
                video_md5 = base_name
            else:
                logger.warning("未找到原始视频文件，使用文件名生成的伪MD5")
                video_md5 = hashlib.md5(base_name.encode()).hexdigest()
        
        result["video_md5"] = video_md5
        
        try:
            vkb = VideoKnowledgeBase(lib_name=lib_name)
            existing_count = vkb.collection.count()
            result["collection_count"] = existing_count
            logger.info(f"逻辑库 '{lib_name}' 中已有 {existing_count} 条数据")
            
            BATCH_SIZE = 20
            total_items = len(rag_data)
            logger.info(f"准备入库 {total_items} 条数据，分批处理中...")
            
            for i in range(0, total_items, BATCH_SIZE):
                batch_data = rag_data[i : i + BATCH_SIZE]
                vkb.add_data(batch_data, video_md5)
                logger.info(f"进度: {min(i + BATCH_SIZE, total_items)}/{total_items} 已处理")
            
            new_count = vkb.collection.count()
            result["collection_count"] = new_count
            result["success"] = True
            result["message"] = f"RAG知识库构建完成，逻辑库 '{lib_name}' 中共 {new_count} 条数据"
            
        except Exception as e:
            logger.error(f"RAG Error: {e}")
            result["message"] = f"RAG构建失败: {e}"
            return result
        
    except Exception as e:
        logger.exception(f"RAG处理出错: {e}")
        result["message"] = f"RAG处理出错: {e}"
    
    return result

def rag_search(query: str, 
               top_k: int = 3, 
               lib_name: str = "default_lib",
               expand_context: bool = True) -> Dict[str, Any]:
    """
    RAG搜索功能，返回带视频路径的结果
    """
    result = {
        "success": False,
        "message": "",
        "query": query,
        "results": []
    }
    
    try:
        vkb = VideoKnowledgeBase(lib_name=lib_name)
        
        logger.info(f"正在检索: '{query}'...")
        search_results = vkb.search(query, top_k=top_k, expand_context=expand_context)
        
        if search_results and 'documents' in search_results and search_results['documents']:
            for i, doc in enumerate(search_results['documents'][0]):
                if i < len(search_results['metadatas'][0]):
                    meta = search_results['metadatas'][0][i]
                    is_expanded = meta.get('is_expanded', False)
                    
                    video_md5 = meta.get('source_video_md5', '')
                    
                    # 获取视频信息
                    video_info = get_video_path_by_md5(video_md5) if video_md5 else None
                    
                    result["results"].append({
                        "content": doc,
                        "start": meta.get('start', 0),
                        "end": meta.get('end', 0),
                        "type": meta.get('type', 'unknown'),
                        "category": meta.get('category', 'general'),
                        "video_md5": video_md5,
                        "video_path": video_info['path'] if video_info else None,
                        "video_name": video_info['filename'] if video_info else None,
                        "original_name": video_info['original_name'] if video_info else None,
                        "video_exists": video_info['exists'] if video_info else False,
                        "is_expanded": is_expanded,
                        "raw_content": meta.get('raw_content', '')
                    })
            
            result["success"] = True
            result["message"] = f"找到 {len(result['results'])} 个结果"
        else:
            result["message"] = "未找到相关结果"
        
    except Exception as e:
        logger.error(f"搜索出错: {e}")
        import traceback
        traceback.print_exc()
        result["message"] = f"搜索出错: {e}"
    
    return result

# ==================== 视频剪辑函数 ====================

def save_analysis_results(segments, video_name, user_instruction):
    """保存分析结果到文件"""
    try:
        safe_name = safe_filename(video_name)
        results = {
            "video_name": video_name,
            "user_instruction": user_instruction,
            "segments": segments,
            "total_segments": len(segments),
            "total_duration": sum(seg["end_time"] - seg["start_time"] for seg in segments if "start_time" in seg and "end_time" in seg)
        }
        
        results_path = os.path.join(ANALYSIS_RESULTS_DIR, f"{safe_name}_analysis.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析结果已保存到: {results_path}")
        return results_path
    except Exception as e:
        logger.error(f"保存分析结果失败: {str(e)}")
        return None

def video_editing(video_filename: str, 
                  user_instruction: str, 
                  max_duration: int) -> Dict[str, Any]:
    """
    智能视频剪辑
    """
    result = {
        "success": False,
        "message": "",
        "video_name": "",
        "segments": [],
        "selected_segments": [],
        "clip_paths": [],
        "output_path": None,
        "total_duration": 0
    }
    
    try:
        logger.info("开始执行视频剪辑及文本分析功能")
        
        video_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
        if not os.path.exists(video_path):
            result["message"] = f"视频文件不存在: {video_path}"
            return result
        
        video_name = os.path.splitext(video_filename)[0]
        result["video_name"] = video_name
        safe_name = safe_filename(video_name)
        
        video_processor = VideoProcessor()
        text_analyzer = TextAnalyzer()
        
        video_md5 = get_file_hash(video_path)
        cache_dir = os.path.join(GLOBAL_CACHE_DIR, video_md5) if video_md5 else None
        
        transcript = None
        if cache_dir and os.path.exists(cache_dir):
            cleaned_path = os.path.join(cache_dir, "cleaned_data.json")
            if os.path.exists(cleaned_path):
                with open(cleaned_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    transcript = []
                    for item in data:
                        transcript.append({
                            "start": item.get('start', 0),
                            "end": item.get('end', 0),
                            "word": item.get('content', ''),
                            "text": item.get('content', '')
                        })
            else:
                trans_path = os.path.join(cache_dir, "raw_trans.json")
                if os.path.exists(trans_path):
                    with open(trans_path, 'r', encoding='utf-8') as f:
                        transcript = json.load(f)
        
        if not transcript:
            transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{safe_name}_transcript.json")
            if os.path.exists(transcript_path):
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = json.load(f)
        
        if not transcript:
            result["message"] = f"未找到转录数据，请先进行数据准备"
            return result
        
        logger.info("读取转录数据完成")
        
        logger.info("分析文本内容")
        segments = text_analyzer.analyze_transcript(transcript, user_instruction)
        result["segments"] = segments
        
        if not segments:
            logger.warning("未找到匹配的剪辑片段，使用默认剪辑")
            segments = [{
                "start_time": 0.0,
                "end_time": min(30.0, max_duration),
                "reason": "默认剪辑：视频开头部分",
                "score": 5
            }]
        
        logger.info(f"文本分析完成，找到 {len(segments)} 个剪辑片段")
        
        logger.info("选择关键片段")
        selected_segments = video_processor.select_key_clips(segments, max_duration)
        result["selected_segments"] = selected_segments
        
        if not selected_segments:
            result["message"] = "未选择到有效的关键片段"
            return result
        
        logger.info(f"已选择 {len(selected_segments)} 个关键片段，总时长约 {max_duration} 秒")
        
        for i, segment in enumerate(selected_segments):
            segment["clip_index"] = i + 1
        
        save_analysis_results(selected_segments, video_name, user_instruction)
        
        logger.info("剪辑视频片段")
        clip_paths = []
        
        for segment in selected_segments:
            if "start_time" not in segment or "end_time" not in segment:
                continue
            
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            
            if end_time <= start_time:
                continue
            
            cached_path = None
            if video_md5:
                am = AssetManager()
                cached_path = am.get_cached_slice_path(video_md5, start_time, end_time)
            
            if cached_path and os.path.exists(cached_path):
                clip_paths.append(cached_path)
                logger.info(f"使用缓存片段: {cached_path}")
            else:
                clip_filename = f"{safe_name}_clip_{segment['clip_index']}.mp4"
                clip_path = os.path.join(SLICE_VIDEO_DIR, clip_filename)
                
                success = video_processor.create_clip(video_path, start_time, end_time, clip_path)
                if success:
                    if video_md5:
                        am = AssetManager()
                        am.save_slice_to_cache(clip_path, video_md5, start_time, end_time)
                    clip_paths.append(clip_path)
                    logger.info(f"片段 {segment['clip_index']}: {start_time:.1f}s - {end_time:.1f}s")
        
        result["clip_paths"] = clip_paths
        
        if not clip_paths:
            result["message"] = "所有视频片段剪辑都失败"
            return result
        
        logger.info(f"共成功剪辑 {len(clip_paths)} 个片段")
        
        logger.info("合并剪辑片段")
        output_filename = f"{safe_name}_edited.mp4"
        output_path = os.path.join(OUTPUT_VIDEO_DIR, output_filename)
        
        success = video_processor.combine_clips(clip_paths, output_path)
        if not success:
            result["message"] = "合并视频片段失败"
            return result
        
        total_duration = 0
        for segment in selected_segments:
            if "start_time" in segment and "end_time" in segment:
                total_duration += segment["end_time"] - segment["start_time"]
        
        result["output_path"] = output_path
        result["total_duration"] = total_duration
        result["success"] = True
        result["message"] = f"视频剪辑完成，输出: {output_filename}"
        
        logger.info(f"视频处理完成，输出文件: {output_path}")
        
    except Exception as e:
        logger.exception(f"视频剪辑出错: {str(e)}")
        result["message"] = f"视频剪辑出错: {str(e)}"
    
    return result

def convert_to_vertical(video_filename: str, method: str = "solid") -> Dict[str, Any]:
    """
    横屏转竖屏功能 - 使用原名作为输出文件名
    支持传入MD5格式或原始文件名
    """
    result = {
        "success": False,
        "message": "",
        "output_path": None
    }
    
    try:
        logger.info("开始执行横屏转竖屏功能")
        
        # 解析文件名
        base_name = os.path.splitext(video_filename)[0]
        ext = os.path.splitext(video_filename)[1]
        
        # 判断是否为MD5格式
        is_md5 = len(base_name) == 32 and all(c in '0123456789abcdef' for c in base_name.lower())
        
        video_path = None
        original_name = None
        
        if is_md5:
            # 情况1：传入的是MD5格式
            video_info = get_video_path_by_md5(base_name)
            if not video_info or not video_info['exists']:
                result["message"] = f"视频池中未找到视频: {base_name}"
                return result
            video_path = video_info['path']
            original_name = video_info.get('original_name', base_name + ext)
        else:
            # 情况2：传入的是原始文件名
            input_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
            if not os.path.exists(input_path):
                result["message"] = f"输入目录中未找到视频文件: {input_path}"
                return result
            video_path = input_path
            original_name = video_filename
        
        # 获取原名（不含扩展名）
        original_name_without_ext = os.path.splitext(original_name)[0]
        safe_name = safe_filename(original_name_without_ext)
        
        # 生成输出文件名 - 使用原名
        output_filename = f"{safe_name}_vertical.mp4"
        output_path = os.path.join(VERTICAL_VIDEO_DIR, output_filename)
        
        logger.info(f"转换视频: {original_name}, 方法: {method}, 输出: {output_filename}")
        
        video_processor = VideoProcessor()
        success = video_processor.convert_to_vertical(video_path, output_path, method=method)
        
        if success:
            result["success"] = True
            result["output_path"] = output_path
            result["message"] = f"转换完成: {output_filename}"
            logger.info(f"横屏转竖屏完成，输出文件: {output_path}")
        else:
            result["message"] = "转换失败"
            logger.error("横屏转竖屏失败")
        
    except Exception as e:
        logger.exception(f"横屏转竖屏出错: {str(e)}")
        result["message"] = f"横屏转竖屏出错: {str(e)}"
    
    return result

def add_subtitles_to_video(video_filename: str, 
                          transcript_filename: Optional[str] = None,
                          font_size: int = 36,
                          font_color: str = "#FFFFFF",
                          bg_color: str = "#000000",
                          bg_opacity: int = 50,
                          position: str = "bottom") -> Dict[str, Any]:
    """
    为视频添加字幕功能 - 使用原名作为输出文件名
    支持传入MD5格式或原始文件名
    """
    result = {
        "success": False,
        "message": "",
        "output_path": None
    }
    
    try:
        logger.info("开始执行为视频添加字幕功能")
        
        video_path = None
        original_name = None
        video_md5 = None
        
        
        input_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
        if not os.path.exists(input_path):
            result["message"] = f"输入目录中未找到视频文件: {input_path}"
            return result
        video_path = input_path
        original_name = video_filename
        # 计算MD5以便查找资产缓存中的字幕
        video_md5 = get_file_hash(video_path)
        
        # 获取原名（不含扩展名）
        original_name_without_ext = os.path.splitext(original_name)[0]
        safe_name = safe_filename(original_name_without_ext)
        
        # 确定字幕文件路径
        transcript_path = None
        if transcript_filename:
            # 如果指定了字幕文件名，直接拼接路径
            candidate = os.path.join(TRANSCRIPTS_DIR, transcript_filename)
            if os.path.exists(candidate):
                transcript_path = candidate
            else:
                result["message"] = f"指定的字幕文件不存在: {candidate}"
                return result
        else:
            # 未指定字幕文件，尝试从资产缓存中查找
            if video_md5:
                cache_dir = os.path.join(GLOBAL_CACHE_DIR, video_md5)
                cache_path = os.path.join(cache_dir, "raw_trans.json")
                if os.path.exists(cache_path):
                    transcript_path = cache_path
            # 如果缓存中找不到，尝试从 transcripts 目录查找
            if transcript_path is None:
                candidate = os.path.join(TRANSCRIPTS_DIR, f"{safe_name}_transcript.json")
                if os.path.exists(candidate):
                    transcript_path = candidate
            
            if transcript_path is None:
                result["message"] = f"未找到对应的字幕文件，请先进行数据准备或指定字幕文件"
                return result
        
        # 调用 VideoProcessor 添加字幕
        output_filename = f"{safe_name}_with_subtitles.mp4"
        output_path = os.path.join(OUTPUT_VIDEO_DIR, output_filename)
        
        video_processor = VideoProcessor()
        success = video_processor.add_subtitles(
            video_path, transcript_path, output_path,
            font_size=font_size, font_color=font_color,
            bg_color=bg_color, bg_opacity=bg_opacity, position=position
        )
        
        if success:
            result["success"] = True
            result["output_path"] = output_path
            result["message"] = f"字幕添加完成: {output_filename}"
            logger.info(f"字幕添加完成，输出文件: {output_path}")
        else:
            result["message"] = "添加字幕失败"
            logger.error("字幕添加失败")
        
    except Exception as e:
        logger.exception(f"添加字幕出错: {str(e)}")
        result["message"] = f"添加字幕出错: {str(e)}"
    
    return result

# ==================== 工具函数 ====================

def check_video_exists(md5: str) -> Dict[str, Any]:
    """检查视频文件是否存在，返回详细信息"""
    result = {
        "md5": md5,
        "exists": False,
        "paths_checked": [],
        "found_path": None,
        "file_size": 0,
        "original_name": None
    }
    
    if not md5:
        return result
    
    for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
        path = os.path.join(VIDEO_POOL_DIR, f"{md5}{ext}")
        result["paths_checked"].append(path)
        if os.path.exists(path):
            result["exists"] = True
            result["found_path"] = path
            result["file_size"] = os.path.getsize(path)
            
            meta_path = os.path.join(GLOBAL_CACHE_DIR, md5, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        result["original_name"] = meta.get("original_filename")
                except:
                    pass
            return result
    
    input_files = get_video_files()
    for f in input_files:
        if md5 in f['name'] or md5 == os.path.splitext(f['name'])[0]:
            result["exists"] = True
            result["found_path"] = f['path']
            result["file_size"] = f['size']
            result["paths_checked"].append(f['path'])
            return result
    
    return result

def migrate_videos_to_pool() -> Dict[str, Any]:
    """将input目录的视频迁移到视频池"""
    migrated = []
    failed = []
    skipped = []
    
    input_files = get_video_files()
    
    for file_info in input_files:
        file_path = file_info['path']
        file_name = file_info['name']
        
        md5 = get_file_hash(file_path)
        if not md5:
            failed.append({"file": file_name, "reason": "MD5计算失败"})
            continue
        
        ext = os.path.splitext(file_name)[1]
        pool_path = os.path.join(VIDEO_POOL_DIR, f"{md5}{ext}")
        
        if os.path.exists(pool_path):
            skipped.append({
                "file": file_name,
                "md5": md5,
                "reason": "already_exists",
                "path": pool_path
            })
            continue
        
        try:
            shutil.copy2(file_path, pool_path)
            
            cache_dir = os.path.join(GLOBAL_CACHE_DIR, md5)
            os.makedirs(cache_dir, exist_ok=True)
            
            meta_path = os.path.join(cache_dir, "metadata.json")
            meta = {
                "original_filename": file_name,
                "md5": md5,
                "migrated_at": datetime.now().isoformat(),
                "size": file_info['size']
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            
            migrated.append({
                "file": file_name,
                "md5": md5,
                "status": "migrated",
                "path": pool_path
            })
            
        except Exception as e:
            failed.append({"file": file_name, "reason": str(e)})
    
    return {
        "total": len(input_files),
        "migrated": migrated,
        "skipped": skipped,
        "failed": failed
    }

def fix_missing_video_links(lib_name: str = "default_lib") -> Dict[str, Any]:
    """修复缺失的视频链接"""
    try:
        vkb = VideoKnowledgeBase(lib_name=lib_name)
        
        all_data = vkb.collection.get()
        
        fixed_count = 0
        missing_count = 0
        
        if all_data and all_data['metadatas']:
            for i, meta in enumerate(all_data['metadatas']):
                video_md5 = meta.get('source_video_md5')
                if video_md5:
                    check = check_video_exists(video_md5)
                    if not check['exists']:
                        missing_count += 1
                        logger.warning(f"视频不存在: {video_md5}")
        
        return {
            "total": len(all_data['metadatas']) if all_data and all_data['metadatas'] else 0,
            "missing": missing_count,
            "fixed": fixed_count
        }
    except Exception as e:
        logger.error(f"修复失败: {e}")
        return {"error": str(e)}

# ==================== 初始化 ====================

ensure_directories()
logger.info("视频智能剪辑工具后端服务初始化完成")

# 导出函数列表
__all__ = [
    'ensure_directories',
    'get_video_files',
    'get_transcript_files',
    'get_rag_files',
    'get_analysis_files',
    'get_video_path_by_md5',
    'create_library',
    'delete_library',
    'get_libraries',
    'get_library_info',
    'add_asset_to_library',
    'remove_asset_from_library',
    'get_global_assets',
    'get_asset_info',
    'delete_asset',
    'process_video_sync',
    'data_processing',
    'rag_building',
    'rag_search',
    'video_editing',
    'convert_to_vertical',
    'add_subtitles_to_video',
    'check_video_exists',
    'migrate_videos_to_pool',
    'fix_missing_video_links',
    'safe_filename',
    'encode_filename',
    'decode_filename',
    'format_size',
    'INPUT_VIDEO_DIR',
    'OUTPUT_VIDEO_DIR',
    'TRANSCRIPTS_DIR',
    'RAGSCRIPTS_DIR',
    'ANALYSIS_RESULTS_DIR',
    'SLICE_VIDEO_DIR',
    'KEYFRAMES_DIR',
    'VERTICAL_VIDEO_DIR',
    'VIDEO_POOL_DIR',
    'GLOBAL_CACHE_DIR',
    'LIBRARIES_DIR',
    'SLICE_CACHE_DIR'

]
