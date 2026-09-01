import chromadb
import dashscope
from chromadb import Documents, EmbeddingFunction, Embeddings
import json
import os
from config import LIBRARIES_DIR

# 1. 定义阿里的 Embedding 函数（让 Chroma 知道怎么把字变成向量）
class DashScopeEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # 调用阿里 text-embedding-v2
        # API 限制 batch size <= 25，这里进行分批处理
        batch_size = 25
        all_embeddings = []
        
        for i in range(0, len(input), batch_size):
            batch_input = input[i : i + batch_size]
            try:
                resp = dashscope.TextEmbedding.call(
                    model=dashscope.TextEmbedding.Models.text_embedding_v2,
                    input=batch_input,
                    api_key=os.getenv("DASHSCOPE_API_KEY")
                )
                if resp.status_code == 200:
                    batch_embeddings = [item['embedding'] for item in resp.output['embeddings']]
                    all_embeddings.extend(batch_embeddings)
                else:
                    raise Exception(f"Embedding Error: {resp.message}")
            except Exception as e:
                # 简单重试或直接抛出，这里直接抛出让上层感知
                raise e
                
        return all_embeddings

class VideoKnowledgeBase:
    def __init__(self, lib_name="default_lib"):
        """
        初始化 RAG 引擎
        lib_name: 逻辑库名称
        """
        self.lib_name = lib_name
        # 路径指向 libraries/{lib_name}/chroma_db
        self.db_path = os.path.join(LIBRARIES_DIR, lib_name, "chroma_db")
        os.makedirs(self.db_path, exist_ok=True)
        
        # 视频索引文件路径
        self.index_path = os.path.join(LIBRARIES_DIR, lib_name, "video_index.json")
        
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # 使用阿里 Embedding
        self.embedding_fn = DashScopeEmbeddingFunction()
        
        # 创建集合（名称与 lib_name 一致）
        self.collection = self.client.get_or_create_collection(
            name=lib_name,
            embedding_function=self.embedding_fn
        )

    def add_data(self, json_data, video_md5):
        """
        把你的 JSON 数据存进去
        video_md5: 视频的 MD5，用于唯一标识和定位
        """
        ids = []
        documents = []
        metadatas = []

        for item in json_data:
            # 1. 构造 ID (唯一标识: {md5}_{seq_id})
            # 假设 item['id'] 是序号
            unique_id = f"{video_md5}_{item['id']}"
            ids.append(unique_id)
            
            # 2. 构造向量化内容 (Document)
            # 技巧：把【语音】和【视觉摘要】拼在一起，这样搜画面或搜语音都能搜到！
            # 注意：这里要用清洗后的 visual_summary，别用那个几百字的 visual_context
            mixed_text = f"语音内容：{item['content']}。画面内容：{item.get('visual_summary', '')}"
            documents.append(mixed_text)
            
            # 3. 构造元数据 (Metadata) -> 用于混合检索（过滤）
            # 注意：data_cleaner 已经把 time_range 拆成了 start 和 end
            start_time = item.get('start', 0)
            end_time = item.get('end', 0)
            
            # 如果是旧数据格式，尝试兼容 time_range
            if 'time_range' in item:
                start_time = item['time_range'][0]
                end_time = item['time_range'][1]
            
            metadatas.append({
                "start": start_time,
                "end": end_time,
                "type": item['type'],
                "category": item.get("category", "general"), # 动态获取
                "raw_content": item['content'], # 存原始文本方便展示
                "source_video_md5": video_md5, # 【关键】关联物理文件
                "seq_id": item['id'] # 方便上下文扩展
            })

        # 批量写入
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"✅ 成功存入 {len(ids)} 条数据到库 {self.lib_name}！")
            
            # 更新视频索引
            self._update_video_index(video_md5, len(ids))

    def _update_video_index(self, video_md5, segment_count):
        """更新视频索引文件"""
        import datetime
        
        index_data = {}
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load video index: {e}")
        
        # 即使已存在也更新，因为片段数量可能变化
        index_data[video_md5] = {
            "added_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "segment_count": segment_count
            # 可以在这里扩展更多信息，如 filename，但目前 add_data 里没有 filename 参数
            # filename 可以通过 video_pool 反查，或者在 add_data 增加参数
        }
        
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    def get_video_list(self):
        """
        从 lib_config.json 获取视频列表，并关联 video_pool 中的物理文件
        """
        config_path = os.path.join(LIBRARIES_DIR, self.lib_name, "lib_config.json")
        if not os.path.exists(config_path):
            return {}
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                video_md5s = config.get("videos", [])
        except Exception as e:
            print(f"Error reading lib_config.json: {e}")
            return {}
            
        video_list = {}
        # 常见视频扩展名
        exts = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
        
        from config import VIDEO_POOL_DIR
        
        for md5 in video_md5s:
            # 尝试查找物理文件
            video_path = None
            filename = f"{md5} (文件丢失)"
            
            for ext in exts:
                potential_path = os.path.join(VIDEO_POOL_DIR, f"{md5}{ext}")
                if os.path.exists(potential_path):
                    video_path = potential_path
                    filename = f"{md5}{ext}"
                    break
            
            # 简单统计一下片段数（可选，如果太慢可以去掉）
            # count = self.collection.count(where={"source_video_md5": md5})
            
            video_list[md5] = {
                "filename": filename,
                "path": video_path,
                "md5": md5
                # "segment_count": count
            }
            
        return video_list

    def search(self, query, category=None, top_k=3, expand_context=True):
        """
        【混合检索核心】
        query: 用户问的问题
        category: 过滤器（比如只搜“cooking”类视频）
        expand_context: 是否自动扩展上下文
        """
        # 构造过滤条件 (Where Clause)
        where_filter = {}
        if category:
            where_filter["category"] = category
            
        # 还可以过滤掉静音片段，或者只搜静音片段
        # where_filter["type"] = "speech" 

        results = self.collection.query(
            query_texts=[query], # Chroma 会自动调用上面的 Embedding 函数转向量
            n_results=top_k,
            where=where_filter if where_filter else None # <--- 只有在有过滤条件时才传字典，否则传 None
        )
        
        if not expand_context or not results['ids']:
            return results

        # 上下文扩展逻辑
        expanded_results = {
            'ids': [],
            'documents': [],
            'metadatas': [],
            'distances': [] # 扩展的片段没有距离，或者设为 0
        }
        
        # 遍历每个检索结果
        for i, doc_id in enumerate(results['ids'][0]):
            meta = results['metadatas'][0][i]
            md5 = meta['source_video_md5']
            try:
                seq_id = int(meta['seq_id'])
            except:
                seq_id = int(doc_id.split('_')[-1]) # Fallback

            # 扩展前后 2 个片段
            ids_to_fetch = []
            for offset in range(-2, 3): # -2, -1, 0, 1, 2
                if offset == 0: continue
                target_id = f"{md5}_{seq_id + offset}"
                ids_to_fetch.append(target_id)
            
            # 获取扩展片段
            context_docs = self.collection.get(ids=ids_to_fetch)
            
            # 简单的合并逻辑：将当前片段和扩展片段合并成一个大段落返回
            # 或者返回列表。这里为了前端展示方便，我们合并内容。
            
            # 整理 fetch 到的结果
            fetched_map = {}
            if context_docs['ids']:
                for j, fid in enumerate(context_docs['ids']):
                    fetched_map[fid] = {
                        'document': context_docs['documents'][j],
                        'metadata': context_docs['metadatas'][j]
                    }
            
            # 按顺序合并
            merged_doc = ""
            merged_start = meta['start']
            merged_end = meta['end']
            
            # 前文
            for offset in range(-2, 0):
                target_id = f"{md5}_{seq_id + offset}"
                if target_id in fetched_map:
                    item = fetched_map[target_id]
                    merged_doc += f"{item['document']} "
                    merged_start = min(merged_start, item['metadata']['start'])
            
            # 当前文
            merged_doc += f"【检索命中】{results['documents'][0][i]} "
            
            # 后文
            for offset in range(1, 3):
                target_id = f"{md5}_{seq_id + offset}"
                if target_id in fetched_map:
                    item = fetched_map[target_id]
                    merged_doc += f"{item['document']} "
                    merged_end = max(merged_end, item['metadata']['end'])
            
            # 更新结果
            expanded_results['ids'].append(doc_id)
            expanded_results['documents'].append(merged_doc.strip())
            
            # 更新 metadata
            new_meta = meta.copy()
            new_meta['start'] = merged_start
            new_meta['end'] = merged_end
            new_meta['is_expanded'] = True
            expanded_results['metadatas'].append(new_meta)
            
        # 包装成类似 chroma 的返回格式 (List[List])
        return {
            'ids': [expanded_results['ids']],
            'documents': [expanded_results['documents']],
            'metadatas': [expanded_results['metadatas']],
            'distances': results['distances'] # 保持原距离
        }
