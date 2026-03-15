"""
全局配置：Ollama / Qdrant 连接参数
修改这里即可切换模型或地址，不需要改其他文件
"""

# ── Ollama ────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"

# 推理（答案生成）用的大模型
# qwen3:14b           → 推理质量更强，每题约87秒
# qwen2.5-coder:7b-instruct-q4_K_M → 速度快3-4倍，适合测试阶段
OLLAMA_LLM_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"

# Embedding 模型（中文支持更好）
OLLAMA_EMBED_MODEL = "nomic-embed-text"
# 备选（更大更强，内存够时用）：
# OLLAMA_EMBED_MODEL = "mxbai-embed-large"

# Ollama 请求超时（秒）
OLLAMA_TIMEOUT = 60

# ── Qdrant ────────────────────────────────────────────────────
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# 存储节点的 collection 名称
QDRANT_COLLECTION = "memory_nodes"

# 向量维度（nomic-embed-text 输出 768 维，mxbai-embed-large 输出 1024 维）
# nomic-embed-text → 768
# mxbai-embed-large → 1024
QDRANT_VECTOR_SIZE = 768

# 检索时返回候选数上限
QDRANT_TOP_K_DEFAULT = 10
