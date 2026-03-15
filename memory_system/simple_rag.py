"""
普通RAG对照组
只做向量相似度检索，不做联想推理
"""

import time
from typing import List, Dict
from memory_network import MemoryNetwork


class SimpleRAG:
    """
    标准RAG：向量检索 top-K，直接拼接内容作为答案
    没有图谱导航、没有多轮联想、没有缺口检测、没有验证
    """

    def __init__(self, memory_network: MemoryNetwork, top_k: int = 5):
        self.net = memory_network
        self.top_k = top_k

    def query(self, question: str, verbose: bool = True) -> Dict:
        start = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"[普通RAG] 问题: {question}")
            print(f"{'='*60}")

        # 一次向量检索
        results = self.net.vector_search(question, top_k=self.top_k)

        retrieved = []
        for node_id, score in results:
            node = self.net.get_node(node_id)
            if node:
                retrieved.append((node.content, score))
                if verbose:
                    print(f"  · [{score:.3f}] {node.content[:60]}")

        # 直接拼接检索结果作为答案
        if retrieved:
            answer_parts = ["基于相似度检索，找到以下相关内容："]
            for i, (content, score) in enumerate(retrieved, 1):
                answer_parts.append(f"  {i}. {content} (相似度:{score:.2f})")
            answer = "\n".join(answer_parts)
        else:
            answer = "未找到相关内容。"

        elapsed = (time.time() - start) * 1000

        if verbose:
            print(f"[普通RAG] 完成，耗时 {elapsed:.1f}ms")

        return {
            "answer": answer,
            "retrieved_count": len(retrieved),
            "top_score": retrieved[0][1] if retrieved else 0.0,
            "elapsed_ms": elapsed,
            "retrieved": retrieved,
        }
