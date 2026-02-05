import chromadb
import dashscope
from chromadb import Documents, EmbeddingFunction, Embeddings
import json
import os

# 1. 定义阿里的 Embedding 函数（让 Chroma 知道怎么把字变成向量）
class DashScopeEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # 调用阿里 text-embedding-v2
        resp = dashscope.TextEmbedding.call(
            model=dashscope.TextEmbedding.Models.text_embedding_v2,
            input=input,
            api_key=os.getenv("DASHSCOPE_API_KEY") # 记得配 .env
        )
        if resp.status_code == 200:
            # 阿里返回的格式里 embeddings 是个列表，我们需要提取 embedding 字段
            return [item['embedding'] for item in resp.output['embeddings']]
        else:
            raise Exception(f"Embedding Error: {resp.message}")

class VideoKnowledgeBase:
    def __init__(self, db_path="data/chroma_db"):
        # 【关键点】这里用 PersistentClient，不需要 Docker！
        # 它会把数据存在本地的 data/chroma_db 文件夹里
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 使用阿里 Embedding
        self.embedding_fn = DashScopeEmbeddingFunction()
        
        # 创建集合（类似 SQL 的表）
        self.collection = self.client.get_or_create_collection(
            name="video_clips",
            embedding_function=self.embedding_fn
        )

    def add_data(self, json_data):
        """
        把你的 JSON 数据存进去
        """
        ids = []
        documents = []
        metadatas = []

        for item in json_data:
            # 1. 构造 ID (唯一标识)
            ids.append(str(item['id']))
            
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
                "category": "cooking", # 假设这是烹饪视频，你可以动态传参
                "raw_content": item['content'] # 存原始文本方便展示
            })

        # 批量写入
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"✅ 成功存入 {len(ids)} 条数据！")

    def search(self, query, category=None, top_k=3):
        """
        【混合检索核心】
        query: 用户问的问题
        category: 过滤器（比如只搜“cooking”类视频）
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
        
        return results