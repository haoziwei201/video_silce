import hashlib
import os
import shutil
import logging

logger = logging.getLogger(__name__)

def get_file_hash(file_path):
    """
    计算文件的MD5哈希值
    """
    if not os.path.exists(file_path):
        return None
        
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        # 分块读取，避免大文件占用过多内存
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def ensure_in_video_pool(file_path, pool_dir):
    """
    确保视频在物理资源池中
    返回: (md5, pool_path)
    """
    if not os.path.exists(pool_dir):
        os.makedirs(pool_dir, exist_ok=True)
        
    file_hash = get_file_hash(file_path)
    if not file_hash:
        return None, None
        
    ext = os.path.splitext(file_path)[1]
    pool_filename = f"{file_hash}{ext}"
    pool_path = os.path.join(pool_dir, pool_filename)
    
    if os.path.exists(pool_path):
        logger.info(f"视频已存在于资源池: {pool_path}")
    else:
        logger.info(f"复制视频到资源池: {pool_path}")
        shutil.copy2(file_path, pool_path)
        
    return file_hash, pool_path
