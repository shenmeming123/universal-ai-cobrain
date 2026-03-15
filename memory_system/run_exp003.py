"""
EXP-003：规模化测试
对比 19节点知识库 vs 50节点+噪音知识库，使用相同问题集
测量指标：
  - Recall@K       : 期望关键节点中被检索到的比例
  - Precision@K    : 检索到的节点中真正相关的比例
  - Chain Completeness : 推理链完整度
  - Confidence Score   : 置信度
  - Gap Detection Rate : 缺口检测触发+补全率
  - 耗时 (ms)
  - 抗噪音率        : 噪音节点混入检索结果的比例（越低越好）
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from memory_network import MemoryNetwork
from memory_node import MemoryNode
from knowledge_base import build_knowledge_base
from knowledge_base_large import build_large_knowledge_base
from associative_engine import AssociativeReasoningEngine


# ═══════════════════════════════════════════════════════════════
# 测试问题集（固定，两个知识库用同一套）
# ═══════════════════════════════════════════════════════════════

TEST_QUESTIONS = [
    {
        "id": "Q-001",
        "query": "为什么猫在陌生环境会观察出口？",
        "key_nodes": ["本能行为", "空间警觉性", "危险环境感知", "自然选择", "早期人类祖先"],
        "expected_chain": ["本能行为", "空间警觉性", "危险环境感知"],
        "difficulty": "中等",
        "type": "因果推理",
    },
    {
        "id": "Q-002",
        "query": "猫的行为和猴子有什么进化上的关联？",
        "key_nodes": ["哺乳动物", "本能行为", "自然选择", "猴子群体行为", "空间警觉性"],
        "expected_chain": ["哺乳动物", "本能行为", "猴子群体行为"],
        "difficulty": "中等",
        "type": "跨域类比",
    },
    {
        "id": "Q-003",
        "query": "为什么人类对蛇有本能的恐惧？",
        "key_nodes": ["蛇类恐惧", "本能行为", "进化心理学", "杏仁核", "灵长类"],
        "expected_chain": ["蛇类恐惧", "本能行为", "杏仁核"],
        "difficulty": "高（依赖新增节点）",
        "type": "进化心理推理",
    },
    {
        "id": "Q-004",
        "query": "大脑的哪个区域负责感知危险并触发应激反应？",
        "key_nodes": ["杏仁核", "战斗逃跑反应", "皮质醇", "注意力系统"],
        "expected_chain": ["杏仁核", "战斗逃跑反应", "皮质醇"],
        "difficulty": "高（纯新增节点）",
        "type": "神经科学推理",
    },
    {
        "id": "Q-005",
        "query": "狩猎采集者如何在开放环境中生存？",
        "key_nodes": ["狩猎采集者", "空间警觉性", "危险环境感知", "工具制造", "认知地图"],
        "expected_chain": ["狩猎采集者", "空间警觉性", "工具制造"],
        "difficulty": "中等",
        "type": "多维度推理",
    },
    {
        "id": "Q-006",
        "query": "什么是量子纠缠现象？",  # 噪音问题：知识库弱相关
        "key_nodes": ["量子纠缠"],
        "expected_chain": [],
        "difficulty": "噪音（知识库无相关内容）",
        "type": "抗噪音测试",
    },
]


# ═══════════════════════════════════════════════════════════════
# 评估函数
# ═══════════════════════════════════════════════════════════════

def evaluate_result(result, question, net, verbose=False):
    """计算单个问题的各项指标"""
    activated = set(result.activated_nodes)
    key_nodes = set(question["key_nodes"])
    expected_chain = set(question["expected_chain"])

    # Recall@K: 期望关键节点被命中的比例
    hit_key = key_nodes & activated
    recall = len(hit_key) / len(key_nodes) if key_nodes else 0.0

    # Precision@K: 激活节点中，非噪音节点的比例
    noise_hits = sum(1 for nid in activated
                     if net.get_node(nid) and "NOISE" in net.get_node(nid).tags)
    total_activated = len(activated)
    precision = (total_activated - noise_hits) / total_activated if total_activated > 0 else 0.0

    # Chain Completeness
    chain_hit = expected_chain & activated
    chain_completeness = len(chain_hit) / len(expected_chain) if expected_chain else 1.0

    # 噪音混入率（越低越好）
    noise_ratio = noise_hits / total_activated if total_activated > 0 else 0.0

    # Gap detection
    gap_triggered = any("缺口" in str(step) or "gap" in str(step).lower()
                        for step in result.reasoning_chain)

    metrics = {
        "recall": recall,
        "precision": precision,
        "chain_completeness": chain_completeness,
        "confidence": result.confidence,
        "elapsed_ms": result.elapsed_ms,
        "activated_count": total_activated,
        "noise_hits": noise_hits,
        "noise_ratio": noise_ratio,
        "hit_key_nodes": list(hit_key),
        "missed_key_nodes": list(key_nodes - activated),
        "gap_triggered": gap_triggered,
        "validation_passed": result.validation_passed,
    }

    if verbose:
        print(f"  Recall:      {recall:.3f}  ({len(hit_key)}/{len(key_nodes)} 关键节点命中)")
        print(f"  Precision:   {precision:.3f}  (噪音混入: {noise_hits}/{total_activated})")
        print(f"  Chain Comp:  {chain_completeness:.3f}")
        print(f"  Confidence:  {result.confidence:.3f}")
        print(f"  Activated:   {total_activated} 节点")
        print(f"  耗时:        {result.elapsed_ms:.1f}ms")
        if metrics["missed_key_nodes"]:
            print(f"  遗漏关键节点: {metrics['missed_key_nodes']}")

    return metrics


def run_rag_evaluation(rag, question, net):
    """评估RAG在单问题上的表现（直接用向量检索，top_k=5）"""
    start = time.time()
    results = net.vector_search(question["query"], top_k=5)
    elapsed = (time.time() - start) * 1000

    retrieved_ids = [r[0] for r in results]
    key_nodes = set(question["key_nodes"])
    hit_key = key_nodes & set(retrieved_ids)
    recall = len(hit_key) / len(key_nodes) if key_nodes else 0.0

    noise_hits = sum(1 for nid in retrieved_ids
                     if net.get_node(nid) and "NOISE" in net.get_node(nid).tags)
    precision = (len(retrieved_ids) - noise_hits) / len(retrieved_ids) if retrieved_ids else 0.0
    noise_ratio = noise_hits / len(retrieved_ids) if retrieved_ids else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "noise_ratio": noise_ratio,
        "elapsed_ms": elapsed,
        "retrieved_count": len(retrieved_ids),
        "hit_key_nodes": list(hit_key),
        "missed_key_nodes": list(key_nodes - set(retrieved_ids)),
    }


# ═══════════════════════════════════════════════════════════════
# 主测试流程
# ═══════════════════════════════════════════════════════════════

def run_scale_test(net_label, net, questions, verbose=False):
    """在指定知识库上跑完整问题集"""
    print(f"\n{'='*65}")
    print(f"  知识库: {net_label}")
    print(f"  节点数: {len(net.nodes)}  关系数: {net.graph.number_of_edges()}")
    print(f"{'='*65}")

    # 构建向量索引
    print("\n[初始化] 构建向量索引...")
    net.build_vectors()

    engine = AssociativeReasoningEngine(net, max_depth=4, max_nodes_per_step=5,
                                        auto_store_threshold=0.85)

    all_engine_metrics = []
    all_rag_metrics = []

    for q in questions:
        print(f"\n{'─'*60}")
        print(f"[{q['id']}] {q['query']}")
        print(f"  难度: {q['difficulty']}  类型: {q['type']}")

        # 联想引擎
        result = engine.reason(q["query"], verbose=False)
        eng_m = evaluate_result(result, q, net, verbose=True)

        # RAG
        rag_m = run_rag_evaluation(None, q, net)
        print(f"\n  [RAG对比]")
        print(f"  Recall: {rag_m['recall']:.3f}  Precision: {rag_m['precision']:.3f}  "
              f"噪音混入: {rag_m['noise_ratio']:.3f}  耗时: {rag_m['elapsed_ms']:.1f}ms")

        all_engine_metrics.append(eng_m)
        all_rag_metrics.append(rag_m)

    # 汇总
    print(f"\n{'='*65}")
    print(f"  {net_label} — 汇总统计（{len(questions)}题）")
    print(f"{'='*65}")

    def avg(lst, key):
        vals = [x[key] for x in lst]
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n  联想推理引擎:")
    print(f"    平均 Recall:         {avg(all_engine_metrics,'recall'):.3f}")
    print(f"    平均 Precision:      {avg(all_engine_metrics,'precision'):.3f}")
    print(f"    平均 Chain Comp:     {avg(all_engine_metrics,'chain_completeness'):.3f}")
    print(f"    平均 Confidence:     {avg(all_engine_metrics,'confidence'):.3f}")
    print(f"    平均 噪音混入率:     {avg(all_engine_metrics,'noise_ratio'):.3f}")
    print(f"    平均激活节点数:      {avg(all_engine_metrics,'activated_count'):.1f}")
    print(f"    平均耗时:            {avg(all_engine_metrics,'elapsed_ms'):.1f}ms")

    print(f"\n  普通RAG:")
    print(f"    平均 Recall:         {avg(all_rag_metrics,'recall'):.3f}")
    print(f"    平均 Precision:      {avg(all_rag_metrics,'precision'):.3f}")
    print(f"    平均 噪音混入率:     {avg(all_rag_metrics,'noise_ratio'):.3f}")
    print(f"    平均耗时:            {avg(all_rag_metrics,'elapsed_ms'):.1f}ms")

    return {
        "label": net_label,
        "node_count": len(net.nodes),
        "relation_count": net.graph.number_of_edges(),
        "engine": {k: avg(all_engine_metrics, k) for k in
                   ["recall","precision","chain_completeness","confidence","noise_ratio","elapsed_ms","activated_count"]},
        "rag": {k: avg(all_rag_metrics, k) for k in
                ["recall","precision","noise_ratio","elapsed_ms"]},
        "per_question": list(zip(all_engine_metrics, all_rag_metrics)),
    }


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  EXP-003: 规模化测试 — 19节点 vs 50节点+噪音")
    print("=" * 65)

    # ── 构建两个知识库 ────────────────────────────
    print("\n[初始化] 构建 19节点 知识库...")
    net_small = build_knowledge_base()

    print("\n[初始化] 构建 50节点+噪音 知识库...")
    net_large = build_large_knowledge_base()

    # ── 分别测试 ──────────────────────────────────
    result_small = run_scale_test("19节点（基线）", net_small, TEST_QUESTIONS)
    result_large = run_scale_test("50节点+10噪音（扩展）", net_large, TEST_QUESTIONS)

    # ── 对比汇总 ──────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  规模化对比总结")
    print(f"{'='*65}")
    print(f"\n  {'指标':<22} {'19节点引擎':>12} {'50节点引擎':>12} {'变化':>10}")
    print(f"  {'-'*58}")

    metrics_to_compare = [
        ("Recall",        "recall"),
        ("Precision",     "precision"),
        ("Chain Comp",    "chain_completeness"),
        ("Confidence",    "confidence"),
        ("噪音混入率",    "noise_ratio"),
        ("激活节点数",    "activated_count"),
        ("耗时(ms)",      "elapsed_ms"),
    ]

    for label, key in metrics_to_compare:
        v_small = result_small["engine"][key]
        v_large = result_large["engine"][key]
        delta = v_large - v_small
        sign = "+" if delta > 0 else ""
        print(f"  {label:<22} {v_small:>12.3f} {v_large:>12.3f} {sign}{delta:>9.3f}")

    print(f"\n  {'RAG Recall':<22} {result_small['rag']['recall']:>12.3f} "
          f"{result_large['rag']['recall']:>12.3f} "
          f"{result_large['rag']['recall']-result_small['rag']['recall']:>+10.3f}")
    print(f"  {'RAG 噪音混入率':<22} {result_small['rag']['noise_ratio']:>12.3f} "
          f"{result_large['rag']['noise_ratio']:>12.3f} "
          f"{result_large['rag']['noise_ratio']-result_small['rag']['noise_ratio']:>+10.3f}")

    print(f"\n  引擎 vs RAG（50节点库）:")
    print(f"    引擎 Recall:    {result_large['engine']['recall']:.3f}  "
          f"RAG Recall:    {result_large['rag']['recall']:.3f}  "
          f"差值: {result_large['engine']['recall']-result_large['rag']['recall']:+.3f}")
    print(f"    引擎 噪音混入: {result_large['engine']['noise_ratio']:.3f}  "
          f"RAG 噪音混入: {result_large['rag']['noise_ratio']:.3f}  "
          f"差值: {result_large['engine']['noise_ratio']-result_large['rag']['noise_ratio']:+.3f}")

    print(f"\n{'='*65}")
    print("  测试完成，数据已输出，请记录到 experiment_log.txt")
    print(f"{'='*65}\n")
