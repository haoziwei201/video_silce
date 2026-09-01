import os
import json
import shutil
from config import LIBRARIES_DIR, GLOBAL_CACHE_DIR
from src.rag_engine import VideoKnowledgeBase

class LibraryManager:
    """
    Manages logical libraries and their indices.
    Decoupled from physical asset storage.
    """

    def __init__(self):
        pass

    def list_libraries(self):
        """List all available libraries."""
        if not os.path.exists(LIBRARIES_DIR):
            return []
        return [d for d in os.listdir(LIBRARIES_DIR) if os.path.isdir(os.path.join(LIBRARIES_DIR, d))]

    def create_library(self, name):
        """Create a new library."""
        lib_path = os.path.join(LIBRARIES_DIR, name)
        if os.path.exists(lib_path):
            return False, "知识库已存在"
        os.makedirs(lib_path)
        os.makedirs(os.path.join(lib_path, "chroma_db"))
        
        # Init config
        config_path = os.path.join(lib_path, "lib_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({"videos": []}, f, ensure_ascii=False)
            
        return True, f"知识库 {name} 创建成功"

    def add_asset_to_library(self, lib_name, asset_md5):
        """
        Link an asset (MD5) to a library.
        1. Read cleaned_data.json from Global Cache.
        2. Insert into Library's ChromaDB.
        3. Update Library's metadata (lib_config.json).
        """
        # 1. Get Cleaned Data
        cache_dir = os.path.join(GLOBAL_CACHE_DIR, asset_md5)
        cleaned_data_path = os.path.join(cache_dir, "cleaned_data.json")
        
        # Fallback to rag_ready.json if cleaned_data doesn't exist (compatibility)
        if not os.path.exists(cleaned_data_path):
             cleaned_data_path = os.path.join(cache_dir, "rag_ready.json")

        if not os.path.exists(cleaned_data_path):
            return False, "资产数据未就绪 (缺少 cleaned_data.json)"
            
        with open(cleaned_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 2. Insert into ChromaDB
        try:
            vkb = VideoKnowledgeBase(lib_name=lib_name)
            # Check if already exists in index? vkb.add_data doesn't check, it appends/upserts.
            # Ideally we should delete old entries for this video if updating, but for now we assume append.
            # But wait, if we add same video twice, IDs might conflict or duplicate.
            # IDs in add_data are f"{video_md5}_{item['id']}". So it will overwrite (upsert) if ID exists.
            # That's good.
            
            vkb.add_data(data, asset_md5)
        except Exception as e:
            return False, f"索引构建失败: {e}"
            
        # 3. Update lib_config.json
        lib_config_path = os.path.join(LIBRARIES_DIR, lib_name, "lib_config.json")
        lib_config = {"videos": []}
        if os.path.exists(lib_config_path):
            with open(lib_config_path, 'r', encoding='utf-8') as f:
                lib_config = json.load(f)
        
        if asset_md5 not in lib_config["videos"]:
            lib_config["videos"].append(asset_md5)
            with open(lib_config_path, 'w', encoding='utf-8') as f:
                json.dump(lib_config, f, ensure_ascii=False, indent=2)
                
        return True, "资产关联成功"

    def get_library_assets(self, lib_name):
        """Get list of assets in a library."""
        vkb = VideoKnowledgeBase(lib_name=lib_name)
        return vkb.get_video_list()
