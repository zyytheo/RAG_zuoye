# config.py

# 文本分割参数
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

SPLIT_SIZE=512

# 知识库目录
# KNOWLEDGE_BASE_DIR = "D:\\KB-files\\"
KNOWLEDGE_BASE_DIR = "./KB-files"

# Ollama 模型名称
OLLAMA_MODEL_NAME = "deepseek-r1:8b"

# 嵌入模型名称
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
# 重排序模型名称
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"


#HuggingFace 参数
ACCESS_TOKEN = "your HuggingFace access token"

# 索引与文件状态存储路径
VECTOR_INDEX_DIR = 'index/faiss_bge_large_en'
FILE_STATE_FILE = 'data/file_states.pkl'
