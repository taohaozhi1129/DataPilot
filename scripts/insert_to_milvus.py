import json
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

MILVUS_URI = "http://localhost:19530"
EMBEDDING_MODEL = r"C:\\embedding\\bge-large-zh-v1.5"

TABLE_COLLECTION = "metadata_table"
COLUMN_COLLECTION = "metadata_column"

client = MilvusClient(uri=MILVUS_URI)
model = SentenceTransformer(EMBEDDING_MODEL)


def embed_text(text: str):
    return model.encode(
        text,
        normalize_embeddings=True
    ).tolist()


import os

# 获取项目根目录 (假设 scripts 在根目录下的一级目录)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "tests", "data")

def insert_tables():
    file_path = os.path.join(DATA_DIR, "tables.json")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        tables = json.load(f)

    batch = []
    for t in tables:
        vector = embed_text(t["text"])

        batch.append({
            "table_name": t["table_name"],
            "schema_name": t["schema_name"],
            "table_comment": t["table_comment"],
            "business_domain": t["business_domain"],
            "data_source": t["data_source"],
            "is_fact_table": t["is_fact_table"],
            "text": t["text"],
            "vector": vector
        })

    client.insert(TABLE_COLLECTION, batch)
    print(f"插入表数据 {len(batch)} 条")


def insert_columns():
    file_path = os.path.join(DATA_DIR, "columns.json")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        columns = json.load(f)

    batch = []
    for c in columns:
        vector = embed_text(c["text"])

        batch.append({
            "table_name": c["table_name"],
            "column_name": c["column_name"],
            "column_comment": c["column_comment"],
            "data_type": c["data_type"],
            "is_primary_key": c["is_primary_key"],
            "is_foreign_key": c["is_foreign_key"],
            "is_nullable": c["is_nullable"],
            "business_meaning": c["business_meaning"],
            "metric_type": c["metric_type"],
            "enum_values": c["enum_values"],
            "text": c["text"],
            "vector": vector
        })

    client.insert(COLUMN_COLLECTION, batch)
    print(f"插入字段数据 {len(batch)} 条")


if __name__ == "__main__":
    insert_tables()
    insert_columns()