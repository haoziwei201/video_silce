"""
RAG 功能独立测试脚本
直接读取 transcripts 目录下的 JSON 文件进行清洗、入库和检索测试
"""

import os
import sys
import json
import logging

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_cleaner import clean_json_data
from src.rag_engine import VideoKnowledgeBase

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置路径
TRANSCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "transcripts")

def list_transcript_files():
    """列出可用的转录文件"""
    if not os.path.exists(TRANSCRIPTS_DIR):
        print(f"目录不存在: {TRANSCRIPTS_DIR}")
        return []
    
    files = [f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".json") and not f.endswith("_rag.json")]
    return files

def test_rag_pipeline(transcript_filename):
    """运行 RAG 测试流程"""
    transcript_path = os.path.join(TRANSCRIPTS_DIR, transcript_filename)
    rag_ready_path = transcript_path.replace(".json", "_rag.json")
    
    print(f"\n{'='*50}")
    print(f"开始测试文件: {transcript_filename}")
    print(f"{'='*50}")

    # 1. 数据清洗
    print("\n[Step 1] 清洗数据 (生成 RAG 专用格式)...")
    
    skip_cleaning = False
    if os.path.exists(rag_ready_path):
        print(f"✅ 检测到本地已存在清洗后的数据: {rag_ready_path}")
        print("⏭️  自动跳过清洗步骤，直接使用现有文件。(如需重新清洗，请手动删除该文件)")
        skip_cleaning = True
    
    if not skip_cleaning:
        clean_json_data(transcript_path, rag_ready_path, category_tag="general")
    
    if not os.path.exists(rag_ready_path):
        logger.error("数据清洗失败，未生成文件")
        return

    # 2. 构建向量库
    print("\n[Step 2] 构建/更新向量知识库...")
    try:
        with open(rag_ready_path, 'r', encoding='utf-8') as f:
            rag_data = json.load(f)
        
        vkb = VideoKnowledgeBase()
        
        # 检查向量库是否已有数据，避免重复 Embedding 消耗
        existing_count = vkb.collection.count()
        print(f"当前向量库中已有 {existing_count} 条数据。")
        
        do_upsert = True
        if existing_count > 0:
            user_choice = input("是否跳过入库步骤直接进行检索？(y/n) [y]: ").strip().lower()
            if user_choice in ['', 'y', 'yes']:
                do_upsert = False
                print("⏭️  已跳过入库步骤。")

        if do_upsert:
            # 分批处理以避免 Embedding API 限制 (通常为 25)
            BATCH_SIZE = 20
            total_items = len(rag_data)
            print(f"准备入库 {total_items} 条数据，分批处理中 (Batch Size: {BATCH_SIZE})...")

            for i in range(0, total_items, BATCH_SIZE):
                batch_data = rag_data[i : i + BATCH_SIZE]
                
                batch_ids = [str(item['id']) for item in batch_data]
                batch_documents = [item['rag_text'] for item in batch_data]
                batch_metadatas = [{
                    "start": item['start'],
                    "end": item['end'],
                    "type": item['type'],
                    "category": item['category'],
                    "raw_content": item['content']
                } for item in batch_data]

                vkb.collection.upsert(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                print(f"  - 进度: {min(i + BATCH_SIZE, total_items)}/{total_items} 已处理")

            print(f"向量库更新完成")
    except Exception as e:
        logger.error(f"向量库构建失败: {e}")
        return

    # 3. 交互式检索测试
    print(f"\n{'='*50}")
    print("RAG 检索测试 (输入 'q' 退出)")
    print(f"{'='*50}")
    
    while True:
        query = input("\n请输入查询指令 (例如: '找出切肉的画面'): ").strip()
        if query.lower() == 'q':
            break
        
        if not query:
            continue
            
        try:
            print(f"正在检索: '{query}'...")
            results = vkb.search(query, top_k=3)
            
            print("\n检索结果:")
            if results and 'documents' in results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    if i < len(results['metadatas'][0]):
                        meta = results['metadatas'][0][i]
                        print(f"  {i+1}. [{meta['start']:.1f}s - {meta['end']:.1f}s] {doc}")
                    else:
                        print(f"  {i+1}. {doc}")
            else:
                print("  未找到相关结果")
                
        except Exception as e:
            print(f"检索出错: {e}")

def main():
    print("RAG 独立测试工具")
    
    files = list_transcript_files()
    if not files:
        print("未找到任何转录文件 (.json)")
        return

    print("\n可用的转录文件:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")
    
    while True:
        try:
            choice = input(f"\n请选择要测试的文件 (1-{len(files)}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    selected_file = files[idx]
                    break
            print("输入无效，请重新选择")
        except ValueError:
            pass
            
    test_rag_pipeline(selected_file)

if __name__ == "__main__":
    main()
