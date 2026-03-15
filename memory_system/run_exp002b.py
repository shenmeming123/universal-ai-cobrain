"""
EXP-002b：消融实验
目的：去掉各个模块，测量每个模块对性能的独立贡献
消融方案：
  FULL    : 完整系统（基准）
  NO_GAP  : 去掉缺口检测（步骤5）
  NO_VALID: 去掉四重验证（步骤6，全部通过）
  NO_VERT : 去掉纵向追溯（步骤2）
  NO_HORIZ: 去掉横向扩展（步骤3）
  RAG_ONLY: 仅向量RAG（对照组）

知识库：使用50节点扩展版（更能体现差异）
问题集：使用EXP-003的前5题（排除噪音题Q-006）
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from memory_network import MemoryNetwork
from knowledge_base_large import build_large_knowledge_base
from associative_engine import AssociativeReasoningEngine, ReasoningResult
from context_layer_mapper import ContextLayerMapper, ContextProfile
from relation_types import RelationType, Relation
from memory_node import MemoryNode
from typing import List, Dict, Optional, Tuple
import uuid


# ═══════════════════════════════════════════════════════════════
# 消融版推理引擎（继承原引擎，通过开关禁用各步骤）
# ═══════════════════════════════════════════════════════════════

class AblationEngine(AssociativeReasoningEngine):
    """
    消融实验用引擎：通过标志位禁用各模块
    """
    def __init__(self, net, ablation: str = "FULL"):
        super().__init__(net, max_depth=4, max_nodes_per_step=5,
                         auto_store_threshold=0.99)  # 设高阈值，避免自动存储干扰
        self.ablation = ablation  # FULL / NO_GAP / NO_VALID / NO_VERT / NO_HORIZ

    def reason(self, query: str, verbose: bool = False,
               explicit_context: Optional[str] = None) -> ReasoningResult:
        import time as _time
        start = _time.time()
        steps = []
        all_activated = []

        context_profile = self.context_mapper.identify_context(
            query, explicit_context=explicit_context)

        # 步骤1：初始激活（所有方案都保留）
        step1 = self._step_initial_activation(query, False, context_profile)
        steps.append(step1)
        all_activated.extend(step1.activated_nodes)

        if not step1.activated_nodes:
            return ReasoningResult(
                answer="未找到相关信息",
                confidence=0.0,
                reasoning_chain=steps,
                activated_nodes=[],
                gaps_found=[],
                validation_passed=False,
                elapsed_ms=(_time.time() - start) * 1000,
                context_profile=context_profile,
            )

        # 步骤2：纵向追溯（消融 NO_VERT 时跳过）
        if self.ablation != "NO_VERT":
            step2 = self._step_vertical_traversal(step1.activated_nodes, query, False)
            steps.append(step2)
            all_activated.extend([n for n in step2.activated_nodes if n not in all_activated])

        # 步骤3：横向扩展（消融 NO_HORIZ 时跳过）
        if self.ablation != "NO_HORIZ":
            step3 = self._step_horizontal_expansion(all_activated, query, False)
            steps.append(step3)
            all_activated.extend([n for n in step3.activated_nodes if n not in all_activated])

        # 步骤4：信息组织
        step4, gaps = self._step_organize(all_activated, query, False)
        steps.append(step4)

        # 步骤5：缺口检测（消融 NO_GAP 时跳过）
        if gaps and self.ablation not in ("NO_GAP",):
            step5 = self._step_gap_filling(gaps, all_activated, query, False)
            steps.append(step5)
            all_activated.extend([n for n in step5.activated_nodes if n not in all_activated])

        # 步骤6：四重验证（消融 NO_VALID 时直接通过，置信度固定0.7）
        if self.ablation == "NO_VALID":
            answer = self._compose_answer(
                query, [self.net.get_node(n).content for n in all_activated
                        if self.net.get_node(n)][:6])
            confidence, passed = 0.7, True
        else:
            answer, confidence, passed, _negated = self._step_validate(all_activated, query, False)

        elapsed = (_time.time() - start) * 1000

        return ReasoningResult(
            answer=answer,
            confidence=confidence,
            reasoning_chain=steps,
            activated_nodes=all_activated,
            gaps_found=gaps,
            validation_passed=passed,
            elapsed_ms=elapsed,
            context_profile=context_profile,
        )


# ═══════════════════════════════════════════════════════════════
# 问题集（Q-001~005，排除噪音题）
# ═══════════════════════════════════════════════════════════════

TEST_QUESTIONS = [
    {
        "id": "Q-001",
        "query": "为什么猫在陌生环境会观察出口？",
        "key_nodes": ["本能行为", "空间警觉性", "危险环境感知", "自然选择", "早期人类祖先"],
        "expected_chain": ["本能行为", "空间警觉性", "危险环境感知"],
    },
    {
        "id": "Q-002",
        "query": "猫的行为和猴子有什么进化上的关联？",
        "key_nodes": ["哺乳动物", "本能行为", "自然选择", "猴子群体行为", "空间警觉性"],
        "expected_chain": ["哺乳动物", "本能行为", "猴子群体行为"],
    },
    {
        "id": "Q-003",
        "query": "为什么人类对蛇有本能的恐惧？",
        "key_nodes": ["蛇类恐惧", "本能行为", "进化心理学", "杏仁核", "灵长类"],
        "expected_chain": ["蛇类恐惧", "本能行为", "杏仁核"],
    },
    {
        "id": "Q-004",
        "query": "大脑的哪个区域负责感知危险并触发应激反应？",
        "key_nodes": ["杏仁核", "战斗逃跑反应", "皮质醇", "注意力系统"],
        "expected_chain": ["杏仁核", "战斗逃跑反应", "皮质醇"],
    },
    {
        "id": "Q-005",
        "query": "狩猎采集者如何在开放环境中生存？",
        "key_nodes": ["狩猎采集者", "空间警觉性", "危险环境感知", "工具制造", "认知地图"],
        "expected_chain": ["狩猎采集者", "空间警觉性", "工具制造"],
    },
]


# ═══════════════════════════════════════════════════════════════
# 评估函数
# ═══════════════════════════════════════════════════════════════

def evaluate(result: ReasoningResult, question: dict, net) -> dict:
    activated = set(result.activated_nodes)
    key_nodes = set(question["key_nodes"])
    expected_chain = set(question["expected_chain"])

    hit_key = key_nodes & activated
    recall = len(hit_key) / len(key_nodes) if key_nodes else 0.0

    noise_hits = sum(1 for nid in activated
                     if net.get_node(nid) and "NOISE" in net.get_node(nid).tags)
    precision = (len(activated) - noise_hits) / len(activated) if activated else 0.0

    chain_hit = expected_chain & activated
    chain_comp = len(chain_hit) / len(expected_chain) if expected_chain else 1.0

    return {
        "recall": recall,
        "precision": precision,
        "chain_completeness": chain_comp,
        "confidence": result.confidence,
        "elapsed_ms": result.elapsed_ms,
        "activated_count": len(activated),
        "noise_ratio": noise_hits / len(activated) if activated else 0.0,
    }


def avg(lst, key):
    vals = [x[key] for x in lst]
    return sum(vals) / len(vals) if vals else 0.0


# ═══════════════════════════════════════════════════════════════
# RAG基线
# ═══════════════════════════════════════════════════════════════

def run_rag(question, net):
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
    chain_comp = len(set(question["expected_chain"]) & set(retrieved_ids)) / \
                 len(question["expected_chain"]) if question["expected_chain"] else 1.0
    return {
        "recall": recall,
        "precision": precision,
        "chain_completeness": chain_comp,
        "confidence": 0.0,
        "elapsed_ms": elapsed,
        "activated_count": len(retrieved_ids),
        "noise_ratio": noise_hits / len(retrieved_ids) if retrieved_ids else 0.0,
    }


# ═══════════════════════════════════════════════════════════════
# 主测试
# ═══════════════════════════════════════════════════════════════

ABLATION_CONFIGS = [
    ("FULL",     "完整系统（基准）"),
    ("NO_GAP",   "去掉缺口检测"),
    ("NO_VALID", "去掉四重验证"),
    ("NO_VERT",  "去掉纵向追溯"),
    ("NO_HORIZ", "去掉横向扩展"),
    ("RAG_ONLY", "仅向量RAG"),
]

if __name__ == "__main__":
    print("=" * 65)
    print("  EXP-002b: 消融实验（50节点扩展知识库）")
    print("=" * 65)

    print("\n[初始化] 构建50节点扩展知识库...")
    net = build_large_knowledge_base()
    print("[初始化] 构建向量索引...")
    net.build_vectors()

    all_results = {}  # {config_key: [per_question_metrics]}

    for config_key, config_name in ABLATION_CONFIGS:
        print(f"\n{'─'*65}")
        print(f"  消融配置: [{config_key}] {config_name}")
        print(f"{'─'*65}")

        metrics_list = []
        for q in TEST_QUESTIONS:
            if config_key == "RAG_ONLY":
                m = run_rag(q, net)
            else:
                engine = AblationEngine(net, ablation=config_key)
                result = engine.reason(q["query"], verbose=False)
                m = evaluate(result, q, net)
            metrics_list.append(m)

            print(f"  {q['id']}: Recall={m['recall']:.3f}  "
                  f"Precision={m['precision']:.3f}  "
                  f"Chain={m['chain_completeness']:.3f}  "
                  f"噪音={m['noise_ratio']:.3f}  "
                  f"耗时={m['elapsed_ms']:.0f}ms  "
                  f"激活={m['activated_count']}")

        all_results[config_key] = metrics_list

        print(f"\n  平均: Recall={avg(metrics_list,'recall'):.3f}  "
              f"Precision={avg(metrics_list,'precision'):.3f}  "
              f"Chain={avg(metrics_list,'chain_completeness'):.3f}  "
              f"噪音={avg(metrics_list,'noise_ratio'):.3f}  "
              f"耗时={avg(metrics_list,'elapsed_ms'):.0f}ms")

    # 汇总对比表
    print(f"\n\n{'='*65}")
    print("  消融实验汇总对比（5题平均，50节点知识库）")
    print(f"{'='*65}")
    print(f"\n  {'配置':<18} {'Recall':>8} {'Precision':>10} {'Chain':>8} "
          f"{'噪音率':>8} {'激活数':>8} {'耗时ms':>8}")
    print(f"  {'-'*68}")

    full_recall = avg(all_results["FULL"], "recall")

    for config_key, config_name in ABLATION_CONFIGS:
        m = all_results[config_key]
        r = avg(m, "recall")
        delta = r - full_recall
        sign = f"({delta:+.3f})" if config_key != "FULL" else "       "
        print(f"  {config_name:<18} {r:>8.3f}{sign:>10} "
              f"{avg(m,'precision'):>10.3f} "
              f"{avg(m,'chain_completeness'):>8.3f} "
              f"{avg(m,'noise_ratio'):>8.3f} "
              f"{avg(m,'activated_count'):>8.1f} "
              f"{avg(m,'elapsed_ms'):>8.0f}")

    print(f"\n  各模块独立贡献（Recall delta，相对完整系统）:")
    baselines = {
        "NO_GAP":   "缺口检测贡献",
        "NO_VALID": "四重验证贡献",
        "NO_VERT":  "纵向追溯贡献",
        "NO_HORIZ": "横向扩展贡献",
    }
    for key, label in baselines.items():
        delta = full_recall - avg(all_results[key], "recall")
        bar = "█" * max(0, int(abs(delta) * 40))
        print(f"    {label:<14}: {delta:+.3f}  {bar}")

    print(f"\n{'='*65}")
    print("  测试完成，请记录到 experiment_log.txt")
    print(f"{'='*65}\n")
