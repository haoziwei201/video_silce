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


# Chroma 的 metadata 只接受 str/int/float/bool，这里统一做一次收口
def sanitize_metadata(meta):
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, (list, dict)):
            clean[k] = json.dumps(v, ensure_ascii=False)
        else:
            clean[k] = str(v)
    return clean


# 阿里 text-embedding-v2 单条输入上限 2048；画面摘要未压缩时静默片段可能拼出几千字，入索引前截断。
# 完整原文仍保留在 global_cache/cleaned_data.json 与元数据中，只影响检索向量。
EMBED_MAX_CHARS = 1900


def clip_for_embedding(text, limit=EMBED_MAX_CHARS):
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def build_video_record(item, video_md5):
    """把一条 cleaned_data 条目转成 (id, document, metadata)。"""
    unique_id = f"{video_md5}_{item['id']}"
    mixed_text = clip_for_embedding(f"语音内容：{item['content']}。画面内容：{item.get('visual_summary', '')}")

    start_time = item.get('start', 0)
    end_time = item.get('end', 0)
    if 'time_range' in item:
        start_time = item['time_range'][0]
        end_time = item['time_range'][1]

    frames = item.get("frames", []) or []
    metadata = sanitize_metadata({
        "kind": "video",
        "start": start_time,
        "end": end_time,
        "type": item['type'],
        "category": item.get("category", "general"),
        "raw_content": item['content'],
        "visual_summary": (item.get("visual_summary") or "")[:500],
        "source_video_md5": video_md5,
        "seq_id": item['id'],
        "frame_ids_json": [f.get("frame_id") for f in frames if f.get("frame_id")],
        "frame_paths_json": [f.get("frame_path") for f in frames if f.get("frame_path")],
        "frame_count": len(frames),
    })
    return unique_id, mixed_text, metadata


def build_doc_record(dataset_id, segment):
    """
    把一条资料段落（文档摘要或图片描述）转成 (id, document, metadata)。
    文档条目没有视频时间戳，用 kind 区分，不伪造 start/end。
    """
    source_id = segment["source_id"]
    chunk_index = int(segment.get("chunk_index", 0))
    unique_id = f"{dataset_id}__{source_id}__{chunk_index}"
    metadata = sanitize_metadata({
        "kind": segment.get("kind", "doc"),
        "dataset_id": dataset_id,
        "source_id": source_id,
        "chunk_index": chunk_index,
        "title": segment.get("title", ""),
        "section": segment.get("section", ""),
        "url": segment.get("url", ""),
        "local_path": segment.get("local_path", ""),
        "product_id": segment.get("product_id", ""),
        "product_version": segment.get("product_version", ""),
        "api_version": segment.get("api_version", ""),
        "status": segment.get("status", ""),
        "human_reviewer": segment.get("human_reviewer", ""),
        "boundary": (segment.get("boundary") or "")[:500],
        "frame_path": segment.get("frame_path", ""),
        "raw_content": (segment.get("text") or "")[:1000],
    })
    return unique_id, clip_for_embedding(segment["text"]), metadata


def expand_video_hit(doc_id, document, meta, fetch_fn, window=2):
    """
    为一条视频命中补前后 window 个片段的上下文。
    返回 (merged_document, new_meta)。new_meta 同时保留：
      evidence_start/evidence_end —— 原始命中片段的范围（承诺卡定位用）
      context_start/context_end   —— 扩展后的范围（“展开上下文”用）
      start/end                   —— 与旧调用方兼容，等于扩展后的范围
    fetch_fn(ids) -> {id: {"document": str, "metadata": dict}}
    """
    md5 = meta.get('source_video_md5')
    try:
        seq_id = int(meta.get('seq_id'))
    except (TypeError, ValueError):
        seq_id = int(str(doc_id).split('_')[-1])

    ids_to_fetch = [f"{md5}_{seq_id + off}" for off in range(-window, window + 1) if off != 0]
    fetched = fetch_fn(ids_to_fetch) or {}

    merged_doc = ""
    merged_start = meta['start']
    merged_end = meta['end']
    for off in range(-window, 0):
        item = fetched.get(f"{md5}_{seq_id + off}")
        if item:
            merged_doc += f"{item['document']} "
            merged_start = min(merged_start, item['metadata']['start'])
    merged_doc += f"【检索命中】{document} "
    for off in range(1, window + 1):
        item = fetched.get(f"{md5}_{seq_id + off}")
        if item:
            merged_doc += f"{item['document']} "
            merged_end = max(merged_end, item['metadata']['end'])

    new_meta = dict(meta)
    new_meta['evidence_start'] = meta['start']
    new_meta['evidence_end'] = meta['end']
    new_meta['context_start'] = merged_start
    new_meta['context_end'] = merged_end
    new_meta['start'] = merged_start
    new_meta['end'] = merged_end
    new_meta['is_expanded'] = True
    return merged_doc.strip(), new_meta


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
        把 cleaned_data.json 的内容存进去
        video_md5: 视频的 MD5，用于唯一标识和定位
        """
        ids, documents, metadatas = [], [], []
        for item in json_data:
            uid, doc, meta = build_video_record(item, video_md5)
            ids.append(uid)
            documents.append(doc)
            metadatas.append(meta)

        if ids:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            print(f"✅ 成功存入 {len(ids)} 条视频数据到库 {self.lib_name}！")
            self._update_video_index(video_md5, len(ids))

    def add_doc_segments(self, dataset_id, segments):
        """
        存入资料段落（kind=doc 的文档摘要 / kind=image 的图片描述）。
        segments 每项至少含 source_id 与 text，其余字段见 build_doc_record。
        """
        ids, documents, metadatas = [], [], []
        for seg in segments:
            if not seg.get("text"):
                continue
            uid, doc, meta = build_doc_record(dataset_id, seg)
            ids.append(uid)
            documents.append(doc)
            metadatas.append(meta)
        if ids:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            print(f"✅ 成功存入 {len(ids)} 条资料段落（{dataset_id}）到库 {self.lib_name}！")
        return len(ids)

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
        
        index_data[video_md5] = {
            "added_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "segment_count": segment_count
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
        exts = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
        
        from config import VIDEO_POOL_DIR
        
        for md5 in video_md5s:
            video_path = None
            filename = f"{md5} (文件丢失)"
            
            for ext in exts:
                potential_path = os.path.join(VIDEO_POOL_DIR, f"{md5}{ext}")
                if os.path.exists(potential_path):
                    video_path = potential_path
                    filename = f"{md5}{ext}"
                    break
            
            video_list[md5] = {
                "filename": filename,
                "path": video_path,
                "md5": md5
            }
            
        return video_list

    def _fetch_by_ids(self, ids):
        got = self.collection.get(ids=ids)
        out = {}
        if got and got.get('ids'):
            for j, fid in enumerate(got['ids']):
                out[fid] = {'document': got['documents'][j], 'metadata': got['metadatas'][j]}
        return out

    def search(self, query, category=None, top_k=3, expand_context=True, kinds=None):
        """
        【混合检索核心】
        query: 用户问的问题
        category: 过滤器（比如只搜“cooking”类视频）
        expand_context: 是否为视频命中自动扩展上下文（文档/图片条目不扩展）
        kinds: 只保留这些类型的条目，如 ["video"] / ["doc", "image"]；
               旧索引里没有 kind 字段的条目一律视为 video。
        """
        where_filter = {}
        if category:
            where_filter["category"] = category

        # 按类型筛选在 Python 侧完成（旧数据没有 kind 字段，无法用 where 过滤），因此多取一些再筛
        n_results = top_k * 3 if kinds else top_k
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter if where_filter else None
        )

        if not results or not results.get('ids') or not results['ids'][0]:
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

        hits = []
        for i, doc_id in enumerate(results['ids'][0]):
            meta = results['metadatas'][0][i] or {}
            kind = meta.get('kind') or 'video'
            if kinds and kind not in kinds:
                continue
            dist = results['distances'][0][i] if results.get('distances') and results['distances'][0] else None
            hits.append((doc_id, results['documents'][0][i], meta, dist, kind))
            if len(hits) >= top_k:
                break

        out = {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}
        for doc_id, document, meta, dist, kind in hits:
            if expand_context and kind == 'video' and meta.get('source_video_md5'):
                document, meta = expand_video_hit(doc_id, document, meta, self._fetch_by_ids)
            else:
                meta = dict(meta)
                if kind == 'video':
                    meta.setdefault('evidence_start', meta.get('start'))
                    meta.setdefault('evidence_end', meta.get('end'))
                    meta.setdefault('context_start', meta.get('start'))
                    meta.setdefault('context_end', meta.get('end'))
                meta.setdefault('is_expanded', False)
            out['ids'].append(doc_id)
            out['documents'].append(document)
            out['metadatas'].append(meta)
            out['distances'].append(dist)

        return {
            'ids': [out['ids']],
            'documents': [out['documents']],
            'metadatas': [out['metadatas']],
            'distances': [out['distances']],
        }
