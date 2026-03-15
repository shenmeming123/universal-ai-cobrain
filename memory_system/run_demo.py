"""
主验证程序
对比：联想推理引擎 vs 普通RAG
后端：Ollama qwen2.5:7b（推理）+ nomic-embed-text（向量）+ Qdrant（持久化检索）

启动前确认：
  1. docker run -p 6333:6333 qdrant/qdrant
  2. ollama serve（已后台运行）
  3. ollama pull qwen2.5:7b && ollama pull nomic-embed-text
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_base import build_knowledge_base
from associative_engine import AssociativeReasoningEngine
from simple_rag import SimpleRAG

GRAPH_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_state.json")

QUESTIONS = [
    {
        "id": 1,
        "question": "为什么男人在陌生环境里会本能地观察出口？",
        "note": "需要推理链：男人→人类祖先→草原生存压力→危险感知→自然选择→本能行为→空间警觉性",
    },
    {
        "id": 2,
        "question": "男人观察出口的行为和猴子的行为有什么关联？",
        "note": "需要跨域联想：猴子行为 → 灵长类共同祖先 → 进化溯源",
    },
    {
        "id": 3,
        "question": "为什么有些行为特征不需要学习就会有？",
        "note": "需要推理：本能行为 ← 自然选择 ← 生存压力，跨越多个抽象层级",
    },
]


def init_network(force_rebuild: bool = False):
    """
    初始化记忆网络：
    - 优先从 graph_state.json 加载（持久化，跳过重建）
    - 若不存在或 force_rebuild=True，则重新构建并保存
    向量始终存储在 Qdrant，graph_state.json 只存图结构
    """
    from memory_network import MemoryNetwork
    net = MemoryNetwork()

    if not force_rebuild and net.load_graph(GRAPH_STATE_PATH):
        print("[初始化] 从持久化文件加载图谱成功，跳过重建")
        # 检查 Qdrant 里是否有向量（增量补全缺失向量）
        print("[初始化] 同步向量索引到 Qdrant...")
        net.build_vectors()
    else:
        print("[初始化] 构建新知识库...")
        net = build_knowledge_base()
        net.summary()
        print("[初始化] 构建 Qdrant 向量索引...")
        net.build_vectors()
        print("[初始化] 保存图谱结构...")
        net.save_graph(GRAPH_STATE_PATH)

    return net


def run_comparison(question_data: dict, net, verbose: bool = True):
    """对一个问题运行两种方式并对比"""
    q = question_data["question"]
    print(f"\n{'#'*70}")
    print(f"# 问题 {question_data['id']}: {q}")
    print(f"# 说明: {question_data['note']}")
    print(f"{'#'*70}")

    # 普通RAG
    rag = SimpleRAG(net, top_k=5)
    rag_result = rag.query(q, verbose=verbose)

    # 联想推理引擎
    engine = AssociativeReasoningEngine(net, max_depth=4)
    assoc_result = engine.reason(q, verbose=verbose)

    print(f"\n{'─'*60}")
    print(f"【对比结果】")
    print(f"{'─'*60}")

    print(f"\n[普通RAG]:")
    print(f"  检索到节点数: {rag_result['retrieved_count']}")
    print(f"  最高相似度:   {rag_result['top_score']:.3f}")
    print(f"  耗时:         {rag_result['elapsed_ms']:.1f}ms")
    print(f"  答案:\n{rag_result['answer']}")

    print(f"\n[联想推理引擎]:")
    print(f"  激活节点数:   {len(assoc_result.activated_nodes)}")
    print(f"  推理步骤数:   {len(assoc_result.reasoning_chain)}")
    print(f"  置信度:       {assoc_result.confidence:.3f}")
    print(f"  验证通过:     {'是' if assoc_result.validation_passed else '否'}")
    print(f"  耗时:         {assoc_result.elapsed_ms:.1f}ms")
    print(f"  激活路径:     {' -> '.join(assoc_result.activated_nodes[:8])}")
    if assoc_result.gaps_found:
        print(f"  发现缺口:     {assoc_result.gaps_found}")
    print(f"  答案:\n{assoc_result.answer}")

    rag_nodes = set(r[0] for r in rag_result["retrieved"])
    assoc_nodes = set(assoc_result.activated_nodes)
    extra_nodes = assoc_nodes - rag_nodes

    print(f"\n[额外发现] 联想引擎比RAG多激活了 {len(extra_nodes)} 个节点:")
    for nid in extra_nodes:
        node = net.get_node(nid)
        if node:
            print(f"  + {node.content[:60]}")

    return rag_result, assoc_result


def main():
    print("=" * 70)
    print("  联想推理引擎 vs 普通RAG — 对比验证")
    print("  后端: Ollama qwen2.5:7b + nomic-embed-text + Qdrant")
    print("=" * 70)

    # 连接检查
    print("\n[连接检查] 验证 Ollama 和 Qdrant 是否可用...")
    try:
        import ollama
        from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, OLLAMA_LLM_MODEL
        client = ollama.Client(host=OLLAMA_BASE_URL)
        models = [m.model for m in client.list().models]
        print(f"  Ollama 可用，已加载模型: {models}")
        if OLLAMA_EMBED_MODEL not in models and not any(OLLAMA_EMBED_MODEL in m for m in models):
            print(f"  [警告] 未找到 embedding 模型 {OLLAMA_EMBED_MODEL}，请先运行: ollama pull {OLLAMA_EMBED_MODEL}")
        if OLLAMA_LLM_MODEL not in models and not any(OLLAMA_LLM_MODEL in m for m in models):
            print(f"  [警告] 未找到 LLM 模型 {OLLAMA_LLM_MODEL}，请先运行: ollama pull {OLLAMA_LLM_MODEL}")
    except Exception as e:
        print(f"  [错误] Ollama 连接失败: {e}")
        print("  请确认 ollama serve 正在运行")
        return

    try:
        from qdrant_client import QdrantClient
        from config import QDRANT_HOST, QDRANT_PORT
        qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = qc.get_collections()
        print(f"  Qdrant 可用，现有 collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"  [错误] Qdrant 连接失败: {e}")
        print("  请确认 Docker Qdrant 容器正在运行: docker ps")
        return

    print("\n[连接检查] 通过 ✓\n")

    # 初始化记忆网络
    net = init_network(force_rebuild=False)
    net.summary()

    # 逐题对比
    all_results = []
    for q_data in QUESTIONS:
        rag_r, assoc_r = run_comparison(q_data, net, verbose=True)
        all_results.append({
            "question": q_data["question"],
            "rag_nodes": rag_r["retrieved_count"],
            "assoc_nodes": len(assoc_r.activated_nodes),
            "rag_score": rag_r["top_score"],
            "assoc_confidence": assoc_r.confidence,
            "validation_passed": assoc_r.validation_passed,
        })

    # 汇总
    print(f"\n{'='*70}")
    print("  汇总对比")
    print(f"{'='*70}")
    print(f"{'问题':<8} {'RAG节点':>8} {'联想节点':>8} {'RAG得分':>8} {'联想置信':>8} {'验证':>6}")
    print(f"{'─'*50}")
    for i, r in enumerate(all_results, 1):
        print(f"问题{i:<4} {r['rag_nodes']:>8} {r['assoc_nodes']:>8} "
              f"{r['rag_score']:>8.3f} {r['assoc_confidence']:>8.3f} "
              f"{'OK' if r['validation_passed'] else 'NG':>6}")

    # 保存更新后的图谱（含推理结论节点）
    net.save_graph(GRAPH_STATE_PATH)
    print(f"\n[完成] 图谱已更新保存到 {GRAPH_STATE_PATH}")


if __name__ == "__main__":
    main()
