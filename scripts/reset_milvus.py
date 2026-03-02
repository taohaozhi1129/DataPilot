import sys
import os

# 将项目根目录添加到 pythonpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import MilvusClient
from config import settings

def reset_milvus():
    print(f"Connecting to Milvus at {settings.MILVUS_URI}...")
    try:
        client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    collections = ["metadata_collection", "template_collection"]
    for col in collections:
        if client.has_collection(col):
            print(f"Dropping collection: {col}")
            client.drop_collection(col)
            print(f"Successfully dropped: {col}")
        else:
            print(f"Collection {col} does not exist.")

    print("\nDone. Milvus collections dropped. Restart the application to recreate them with new dimension (1024).")

if __name__ == "__main__":
    reset_milvus()
