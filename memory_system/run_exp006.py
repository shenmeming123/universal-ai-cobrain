"""
EXP-006：冲突/噪音攻击测试
目的：验证四重验证（步骤6）和纵向追溯（步骤2）在面对错误知识时的拦截能力
测试方法：
  1. 基线（干净知识库）：正常推理，记录置信度
  2. 注入矛盾知识：加入与已有知识直接矛盾的节点（有 OPPOSITE_TO 关系）
  3. 注入隐性错误：加入表述模糊但语义矛盾的节点（无显式对立关系）
  4. 注入权威干扰：加入高权重但错误的节点（weight=2.0）
  对比三种攻击下置信度的变化 + 四重验证拦截率
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from memory_network import MemoryNetwork
from memory_node import MemoryNode
from knowledge_base_large import build_large_knowledge_base
from associative_engine import AssociativeReasoningEngine
from relation_types import Relation, RelationType


# ═══════════════════════════════════════════════════════════════
# 攻击节点定义
# ═══════════════════════════════════════════════════════════════

def inject_explicit_contradictions(net: MemoryNetwork):
    """
    攻击类型1：显性矛盾注入
    加入与已有知识直接对立的节点，并建立 OPPOSITE_TO 关系
    四重验证的内部矛盾检测（验证1）应能检测到并降低置信度
    """
    attack_nodes = [
        MemoryNode("本能行为否定",
                   "动物所有行为都是后天习得的，不存在遗传决定的固定行为模式",
                   abstract_level=6, domain=["行为学"],
                   coverage=0.7, essence_features=["后天学习", "无本能"],
                   weight=1.0, tags=["攻击节点", "矛盾", "ATTACK"]),

        MemoryNode("自然选择否定",
                   "自然选择不存在，物种特征是随机变化的，与生存适应性无关",
                   abstract_level=8, domain=["进化论"],
                   coverage=0.8, essence_features=["随机", "无选择"],
                   weight=1.0, tags=["攻击节点", "矛盾", "ATTACK"]),

        MemoryNode("空间感知无用论",
                   "空间警觉性对动物生存没有帮助，动物在陌生环境中不需要感知出口位置",
                   abstract_level=4, domain=["心理学"],
                   coverage=0.5, essence_features=["无空间感知", "无威胁"],
                   weight=1.0, tags=["攻击节点", "矛盾", "ATTACK"]),
    ]
    for node in attack_nodes:
        net.add_node(node)

    # 建立显式 OPPOSITE_TO 关系（四重验证能检测到这些）
    contradictions = [
        Relation("本能行为否定",  "本能行为",    RelationType.OPPOSITE_TO, weight=1.0),
        Relation("自然选择否定",  "自然选择",    RelationType.OPPOSITE_TO, weight=1.0),
        Relation("空间感知无用论","空间警觉性",   RelationType.OPPOSITE_TO, weight=1.0),
    ]
    for rel in contradictions:
        if rel.source_id in net.nodes and rel.target_id in net.nodes:
            net.add_relation(rel)

    return attack_nodes


def inject_implicit_contradictions(net: MemoryNetwork):
    """
    攻击类型2：隐性矛盾注入
    加入语义上与已有知识矛盾的节点，但无显式 OPPOSITE_TO 关系
    这类攻击四重验证较难拦截，用于测试系统的边界
    """
    attack_nodes = [
        MemoryNode("习得优先论",
                   "哺乳动物的所有恐惧反应都是通过童年经历习得的，与基因无关",
                   abstract_level=5, domain=["心理学", "行为学"],
                   coverage=0.6, essence_features=["习得", "童年", "基因无关"],
                   weight=1.0, tags=["攻击节点", "隐性矛盾", "ATTACK"]),

        MemoryNode("杏仁核无关论",
                   "杏仁核与情绪和恐惧处理无关，其功能仅限于调节睡眠节律",
                   abstract_level=5, domain=["神经科学"],
                   coverage=0.5, essence_features=["睡眠", "无情绪"],
                   weight=1.0, tags=["攻击节点", "隐性矛盾", "ATTACK"]),
    ]
    for node in attack_nodes:
        net.add_node(node)
    # 故意不建立 OPPOSITE_TO 关系，测试系统能否通过语义检测发现矛盾
    return attack_nodes


def inject_high_weight_attacks(net: MemoryNetwork):
    """
    攻击类型3：高权重错误节点注入
    加入 weight=2.0 的权威性错误节点，测试高权重是否会绕过验证逻辑
    """
    attack_nodes = [
        MemoryNode("权威错误节点",
                   "战斗逃跑反应不依赖杏仁核，完全由前额叶皮层的理性判断触发",
                   abstract_level=6, domain=["神经科学"],
                   coverage=0.75, essence_features=["前额叶", "理性", "无杏仁核"],
                   weight=2.0, tags=["攻击节点", "高权重", "ATTACK"]),
    ]
    for node in attack_nodes:
        net.add_node(node)
    # 建立与已有知识的对立关系
    rel = Relation("权威错误节点", "战斗逃跑反应", RelationType.OPPOSITE_TO, weight=1.0)
    if rel.source_id in net.nodes and rel.target_id in net.nodes:
        net.add_relation(rel)
    return attack_nodes


# ═══════════════════════════════════════════════════════════════
# 测试问题集（针对被攻击知识的问题）
# ═══════════════════════════════════════════════════════════════

ATTACK_QUESTIONS = [
    {
        "id": "AQ-001",
        "query": "猫在陌生环境中观察出口的行为是本能的吗？",
        "target_knowledge": ["本能行为", "空间警觉性"],
        "attack_type": "显性矛盾（本能行为否定）",
        "expected_confidence_drop": True,  # 期望置信度降低
    },
    {
        "id": "AQ-002",
        "query": "自然选择如何塑造动物的本能行为？",
        "target_knowledge": ["自然选择", "本能行为"],
        "attack_type": "显性矛盾（自然选择否定）",
        "expected_confidence_drop": True,
    },
    {
        "id": "AQ-003",
        "query": "为什么人类对蛇有本能的恐惧？",
        "target_knowledge": ["蛇类恐惧", "杏仁核", "本能行为"],
        "attack_type": "隐性矛盾（习得优先论+杏仁核无关论）",
        "expected_confidence_drop": False,  # 隐性矛盾期望系统无法拦截
    },
    {
        "id": "AQ-004",
        "query": "大脑哪个区域触发应激反应？",
        "target_knowledge": ["杏仁核", "战斗逃跑反应"],
        "attack_type": "高权重错误节点（权威错误节点否定杏仁核→战斗逃跑）",
        "expected_confidence_drop": True,
    },
]


# ═══════════════════════════════════════════════════════════════
# 单问题评估
# ═══════════════════════════════════════════════════════════════

def evaluate_attack(engine, question, verbose=False):
    result = engine.reason(question["query"], verbose=verbose)
    return {
        "id": question["id"],
        "confidence": result.confidence,
        "validation_passed": result.validation_passed,
        "activated_count": len(result.activated_nodes),
        "elapsed_ms": result.elapsed_ms,
        "attack_nodes_activated": sum(
            1 for nid in result.activated_nodes
            if engine.net.get_node(nid) and "ATTACK" in engine.net.get_node(nid).tags
        ),
    }


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  EXP-006: 冲突/噪音攻击测试")
    print("  验证四重验证和纵向追溯对抗错误知识的能力")
    print("=" * 65)

    # ── 阶段0：基线（干净知识库）──────────────────────────
    print("\n" + "─" * 65)
    print("  阶段0：基线（干净50节点知识库，无攻击）")
    print("─" * 65)

    net_clean = build_large_knowledge_base()
    net_clean.build_vectors()
    engine_clean = AssociativeReasoningEngine(
        net_clean, max_depth=4, max_nodes_per_step=5, auto_store_threshold=1.1)  # 禁用自动存储

    baseline_results = []
    for q in ATTACK_QUESTIONS:
        r = evaluate_attack(engine_clean, q, verbose=False)
        baseline_results.append(r)
        print(f"  [{q['id']}] 置信度={r['confidence']:.3f}  验证={'通过' if r['validation_passed'] else '未通过'}  "
              f"耗时={r['elapsed_ms']:.0f}ms")

    avg_baseline_conf = sum(r["confidence"] for r in baseline_results) / len(baseline_results)
    print(f"\n  基线平均置信度: {avg_baseline_conf:.3f}")

    # ── 阶段1：显性矛盾攻击 ────────────────────────────────
    print("\n" + "-" * 65)
    print("  Stage1: explicit contradiction attack (3 nodes with OPPOSITE_TO)")
    print("-" * 65)

    net_attack1 = build_large_knowledge_base()
    injected1 = inject_explicit_contradictions(net_attack1)
    net_attack1.build_vectors()
    engine1 = AssociativeReasoningEngine(
        net_attack1, max_depth=4, max_nodes_per_step=5, auto_store_threshold=1.1)

    attack1_results = []
    for q in ATTACK_QUESTIONS:
        r = evaluate_attack(engine1, q, verbose=False)
        attack1_results.append(r)
        baseline_conf = baseline_results[ATTACK_QUESTIONS.index(q)]["confidence"]
        conf_drop = baseline_conf - r["confidence"]
        blocked = "[BLOCKED]" if not r["validation_passed"] else ("[DROP]" if conf_drop > 0.05 else "[MISS]")
        print(f"  [{q['id']}] conf={r['confidence']:.3f} (base={baseline_conf:.3f}, "
              f"drop={conf_drop:+.3f})  "
              f"valid={'OK' if r['validation_passed'] else 'FAIL'}  {blocked}")
        if r["attack_nodes_activated"] > 0:
            print(f"         attack_nodes_activated={r['attack_nodes_activated']}")

    avg_attack1_conf = sum(r["confidence"] for r in attack1_results) / len(attack1_results)
    blocked_count1 = sum(1 for r in attack1_results if not r["validation_passed"])
    print(f"\n  avg_conf_after_attack1: {avg_attack1_conf:.3f}  drop: {avg_baseline_conf - avg_attack1_conf:+.3f}")
    print(f"  blocked_by_validation: {blocked_count1}/{len(attack1_results)}")

    # ── 阶段2：隐性矛盾攻击 ────────────────────────────────
    print("\n" + "-" * 65)
    print("  Stage2: implicit contradiction attack (no OPPOSITE_TO relation)")
    print("-" * 65)

    net_attack2 = build_large_knowledge_base()
    inject_explicit_contradictions(net_attack2)   # 保留显性矛盾节点
    injected2 = inject_implicit_contradictions(net_attack2)
    net_attack2.build_vectors()
    engine2 = AssociativeReasoningEngine(
        net_attack2, max_depth=4, max_nodes_per_step=5, auto_store_threshold=1.1)

    attack2_results = []
    for q in ATTACK_QUESTIONS:
        r = evaluate_attack(engine2, q, verbose=False)
        attack2_results.append(r)
        baseline_conf = baseline_results[ATTACK_QUESTIONS.index(q)]["confidence"]
        conf_drop = baseline_conf - r["confidence"]
        blocked = "[BLOCKED]" if not r["validation_passed"] else ("[DROP]" if conf_drop > 0.05 else "[MISS]")
        print(f"  [{q['id']}] conf={r['confidence']:.3f} (base={baseline_conf:.3f}, "
              f"drop={conf_drop:+.3f})  "
              f"valid={'OK' if r['validation_passed'] else 'FAIL'}  {blocked}")

    avg_attack2_conf = sum(r["confidence"] for r in attack2_results) / len(attack2_results)
    blocked_count2 = sum(1 for r in attack2_results if not r["validation_passed"])
    print(f"\n  avg_conf_after_attack2: {avg_attack2_conf:.3f}  drop: {avg_baseline_conf - avg_attack2_conf:+.3f}")
    print(f"  implicit_blocked: {blocked_count2}/{len(attack2_results)}")

    # ── 阶段3：高权重节点攻击 ──────────────────────────────
    print("\n" + "-" * 65)
    print("  Stage3: high-weight error node (weight=2.0, has OPPOSITE_TO)")
    print("-" * 65)

    net_attack3 = build_large_knowledge_base()
    inject_explicit_contradictions(net_attack3)
    inject_implicit_contradictions(net_attack3)
    inject_high_weight_attacks(net_attack3)
    net_attack3.build_vectors()
    engine3 = AssociativeReasoningEngine(
        net_attack3, max_depth=4, max_nodes_per_step=5, auto_store_threshold=1.1)

    attack3_results = []
    for q in ATTACK_QUESTIONS:
        r = evaluate_attack(engine3, q, verbose=False)
        attack3_results.append(r)
        baseline_conf = baseline_results[ATTACK_QUESTIONS.index(q)]["confidence"]
        conf_drop = baseline_conf - r["confidence"]
        blocked = "[BLOCKED]" if not r["validation_passed"] else ("[DROP]" if conf_drop > 0.05 else "[MISS]")
        print(f"  [{q['id']}] conf={r['confidence']:.3f} (base={baseline_conf:.3f}, "
              f"drop={conf_drop:+.3f})  "
              f"valid={'OK' if r['validation_passed'] else 'FAIL'}  {blocked}")

    avg_attack3_conf = sum(r["confidence"] for r in attack3_results) / len(attack3_results)
    blocked_count3 = sum(1 for r in attack3_results if not r["validation_passed"])
    print(f"\n  avg_conf_after_attack3: {avg_attack3_conf:.3f}  drop: {avg_baseline_conf - avg_attack3_conf:+.3f}")
    print(f"  blocked_by_validation: {blocked_count3}/{len(attack3_results)}")

    # ── 汇总 ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EXP-006 Summary")
    print("=" * 65)

    print(f"\n  Scenario            avg_conf  vs_base  blocked/{len(ATTACK_QUESTIONS)}")
    print(f"  -----------------------------------------------------------")
    print(f"  Baseline (clean)    {avg_baseline_conf:.3f}     -        -")
    print(f"  Explicit attack     {avg_attack1_conf:.3f}     "
          f"{avg_baseline_conf - avg_attack1_conf:+.3f}    {blocked_count1}/{len(ATTACK_QUESTIONS)}")
    print(f"  Expl+Impl attack    {avg_attack2_conf:.3f}     "
          f"{avg_baseline_conf - avg_attack2_conf:+.3f}    {blocked_count2}/{len(ATTACK_QUESTIONS)}")
    print(f"  Full attack         {avg_attack3_conf:.3f}     "
          f"{avg_baseline_conf - avg_attack3_conf:+.3f}    {blocked_count3}/{len(ATTACK_QUESTIONS)}")

    expl_verdict = "EFFECTIVE" if blocked_count1 >= 2 else ("PARTIAL" if blocked_count1 >= 1 else "INEFFECTIVE")
    impl_extra = blocked_count2 - blocked_count1
    impl_verdict = "EFFECTIVE" if impl_extra >= 1 else "NO_EFFECT(expected)"
    hw_verdict = "NOT_BYPASSED" if avg_attack3_conf <= avg_attack1_conf else "BYPASSED"

    print(f"\n  Validation effectiveness:")
    print(f"    Explicit contradiction (OPPOSITE_TO): {blocked_count1}/{len(ATTACK_QUESTIONS)} blocked -> {expl_verdict}")
    print(f"    Implicit semantic contradiction:      {impl_extra}/{len(ATTACK_QUESTIONS)} extra blocked -> {impl_verdict}")
    print(f"    High-weight node bypass:              {hw_verdict}")

    print(f"\n  Paper significance:")
    print(f"    Validation blocks explicit contradictions: {blocked_count1}/{len(ATTACK_QUESTIONS)}")
    print(f"    Explains EXP-002b zero contribution: validation only helps under attack")

    print("\n" + "=" * 65)
    print("  Test complete. Please record to experiment_log.txt")
    print("=" * 65)

    return {
        "baseline": avg_baseline_conf,
        "attack1": avg_attack1_conf,
        "attack2": avg_attack2_conf,
        "attack3": avg_attack3_conf,
        "blocked1": blocked_count1,
        "blocked2": blocked_count2,
        "blocked3": blocked_count3,
        "detailed": {
            "baseline": baseline_results,
            "attack1": attack1_results,
            "attack2": attack2_results,
            "attack3": attack3_results,
        }
    }


if __name__ == "__main__":
    main()
