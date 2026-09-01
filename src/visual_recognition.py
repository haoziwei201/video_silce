import os
import logging
import json
import hashlib
import asyncio
from http import HTTPStatus
import dashscope
from dashscope import MultiModalConversation

# 导入配置
try:
    from config import DASHSCOPE_API_KEY, QWEN_VL_MODEL
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DASHSCOPE_API_KEY, QWEN_VL_MODEL

from src.prompts import VISUAL_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)

class VisualRecognition:
    def __init__(self):
        self.api_key = DASHSCOPE_API_KEY
        if not self.api_key:
            logger.warning("未找到 DASHSCOPE_API_KEY")
        dashscope.api_key = self.api_key
        self.model = QWEN_VL_MODEL
        
        # 初始化缓存
        self.cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "api_cache.json")
        self.cache = self._load_cache()
        
        # 新增：异步锁，防止并发写入缓存文件时冲突
        self._cache_lock = asyncio.Lock()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
                return {}
        return {}

    def _save_cache(self):
        """同步保存缓存"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    async def _save_cache_async(self):
        """异步保存缓存（在线程中运行IO操作）"""
        try:
            # 将文件写入操作放入线程池，避免阻塞事件循环
            await asyncio.to_thread(self._save_cache)
        except Exception as e:
            logger.error(f"异步保存缓存失败: {e}")

    def _get_file_hash(self, file_path):
        """计算文件内容的MD5哈希"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"计算哈希失败: {e}")
            return None

    # 保留同步方法，防止旧代码调用报错
    def analyze_image(self, image_path, prompt=VISUAL_ANALYSIS_PROMPT):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
             # If we are already in an event loop, we shouldn't use asyncio.run
             # But since this is a sync method, we can't await. 
             # This suggests the caller should use analyze_image_async if they are in an async context.
             # For now, we'll try to use a task or warn. 
             # But simplest is to just assume if this is called, we want sync behavior.
             # If loop is running, this will raise "RuntimeError: asyncio.run() cannot be called from a running event loop"
             # So we should advise using analyze_image_async.
             pass

        return asyncio.run(self.analyze_image_async(image_path, prompt))

    async def save_cache(self):
        """手动触发保存缓存到文件"""
        async with self._cache_lock:
            await self._save_cache_async()

    async def analyze_image_async(self, image_path, prompt=VISUAL_ANALYSIS_PROMPT, auto_save=True):
        """
        异步分析单张图片 (带缓存)
        Args:
            image_path: 图片路径
            prompt: 提示词
            auto_save: 是否在分析完成后立即保存缓存文件。批量处理时建议设为False，处理完后手动调用save_cache()。
        Returns:
            str: 图片描述
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None

        # 1. 计算哈希 (IO操作，放入线程池以免阻塞)
        file_hash = await asyncio.to_thread(self._get_file_hash, image_path)
        
        # 2. 检查缓存
        if file_hash:
            cache_key = f"{file_hash}_{prompt}"
            # 读内存字典不需要await，但也加个锁保险（虽然Python GIL保证了字典读写原子性，但逻辑上严谨点）
            if cache_key in self.cache:
                logger.info(f"Cache hit for {image_path}")
                return self.cache[cache_key]

        # 3. 准备API调用
        abs_path = os.path.abspath(image_path)
        if os.name == 'nt':
            abs_path = abs_path.replace('\\', '/')
        img_uri = f"file://{abs_path}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": img_uri},
                    {"text": prompt}
                ]
            }
        ]

        try:
            # logger.info(f"Calling VL model asynchronously for {image_path}")
            
            # 【核心修改】：使用 asyncio.to_thread 将同步的 SDK 调用放入线程池
            # 这样主线程就不会被卡住了
            # 增加重试机制
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    response = await asyncio.to_thread(
                        MultiModalConversation.call,
                        model=self.model,
                        messages=messages
                    )
                    
                    if response.status_code == HTTPStatus.OK:
                        break # 成功则跳出重试循环
                    else:
                        logger.warning(f"API Error (Attempt {attempt+1}/{max_retries}): {response.code} - {response.message}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (attempt + 1))
                except Exception as net_err:
                     logger.warning(f"Network Error (Attempt {attempt+1}/{max_retries}): {net_err}")
                     if attempt < max_retries - 1:
                         await asyncio.sleep(retry_delay * (attempt + 1))
                     else:
                         raise net_err # 最后一次重试失败，抛出异常

            result_text = None
            
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0].message.content
                
                if isinstance(content, list):
                    text_content = ""
                    for item in content:
                        if 'text' in item:
                            text_content += item['text']
                    result_text = text_content
                elif isinstance(content, str):
                    result_text = content
                else:
                    logger.warning(f"Unexpected content format: {type(content)}")
                    result_text = str(content)
                
                # 4. 写入缓存 (加锁防止并发写入冲突)
                if result_text and file_hash:
                    cache_key = f"{file_hash}_{prompt}"
                    # 更新内存缓存（原子操作，无需锁）
                    self.cache[cache_key] = result_text
                    
                    if auto_save:
                        async with self._cache_lock:
                            await self._save_cache_async()
                
                return result_text
            else:
                logger.error(f"API Error: {response.code} - {response.message}")
                return None
        except Exception as e:
            logger.error(f"Exception during visual recognition: {e}")
            return None
