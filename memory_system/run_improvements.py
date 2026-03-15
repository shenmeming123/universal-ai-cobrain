"""
集成测试：验证四个改进模块
  - 改进1: 向量方向关系检测 (RelationDetector)
  - 改进2: 语境-层级映射 (ContextLayerMapper)
  - 改进3: 冲突分级处理 (ConflictResolver)
  - 改进4: 推理结论自动存储 (AssociativeReasoningEngine.auto_store)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_base import build_knowledge_base
from associative_engine import AssociativeReasoningEngine
from context_layer_mapper import ContextLayerMapper
from conflict_resolver import ConflictResolver, NewInformation
from relation_detector import RelationDetector


def separator(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ─────────────────────────────────────────────────────────────────
# 测试 改进2：语境-层级映射
# ─────────────────────────────────────────────────────────────────

def test_context_mapper():
    separator("改进2 测试：语境-层级映射")
    mapper = ContextLayerMapper()

    test_cases = [
        ("为什么猫会观察出口？",              None),        # 日常对话
        ("分析本能行为的神经机制和进化原理",   None),        # 科学分析
        ("如何实现一个行为识别系统？",         None),        # 工程
        ("请举例解释什么是本能行为",           None),        # 教育
        ("本能行为的本质是什么",               "philosophy"), # 显式指定哲学
    ]

    for query, explicit in test_cases:
        profile = mapper.identify_context(query, explicit_context=explicit)
        print(f"\n查询: {query[:40]}")
        print(f"  {mapper.describe_context(profile)}")

    print("\n[改进2] 语境识别完成")


# ─────────────────────────────────────────────────────────────────
# 测试 改进3：冲突分级处理
# ─────────────────────────────────────────────────────────────────

def test_conflict_resolver(net):
    separator("改进3 测试：冲突分级处理")
    resolver = ConflictResolver(net)

    # Case 1: 无冲突 - 新知识直接补充
    new1 = NewInformation(
        node_id="new_node_001",
        content="章鱼能感知偏振光，这是一种特殊的视觉能力",
        abstract_level=3,
        domain=["生物学", "感知系统"],
        coverage=0.2,
        essence_features=["感知能力", "视觉"],
        tags=["章鱼", "感知"],
        evidence_strength=0.85,
        source="外部知识库",
    )
    print("\n--- Case 1: 无冲突新知识 ---")
    resolver.process(new1, verbose=True)

    # Case 2: 局部冲突 - 例外情况
    new2 = NewInformation(
        node_id="new_node_002",
        content="企鹅虽然是鸟类，但不会飞，本能行为以游泳为主",
        abstract_level=2,
        domain=["生物学", "动物行为"],
        coverage=0.15,
        essence_features=["后天", "游泳"],  # "后天"会与"先天"产生局部冲突检测
        tags=["企鹅", "例外", "鸟类"],
        evidence_strength=0.9,
        source="用户输入",
    )
    print("\n--- Case 2: 局部冲突（例外情况）---")
    resolver.process(new2, verbose=True)

    # Case 3: 悖论 - 高层次矛盾
    new3 = NewInformation(
        node_id="new_node_003",
        content="所有动物的行为都不是本能的，完全由后天学习决定",
        abstract_level=9,  # 高抽象层
        domain=["生物学", "动物行为"],
        coverage=0.8,
        essence_features=["后天", "学习"],
        tags=["争议"],
        evidence_strength=0.3,  # 低证据强度 + 高层矛盾 = 悖论
        source="未知来源",
    )
    print("\n--- Case 3: 悖论（根本矛盾）---")
    resolver.process(new3, verbose=True)

    # 汇总
    stats = resolver.summary()
    print(f"\n[冲突处理器] 统计: {stats}")
    pending = resolver.get_pending_reviews()
    if pending:
        print(f"[需人工审核] {len(pending)} 条记录待处理")


# ─────────────────────────────────────────────────────────────────
# 测试 改进1：向量方向关系检测
# ─────────────────────────────────────────────────────────────────

def test_relation_detector(net):
    separator("改进1 测试：向量方向关系检测")

    detector = RelationDetector(net)

    # 从已有关系学习方向
    stats = detector.train_from_existing_relations(verbose=True)

    # 对几个已有节点对测试预测准确性（已知关系的预测）
    print("\n--- 对已知关系对进行方向预测（验证准确性）---")
    known_pairs = []
    count = 0
    for src, tgt, data in net.graph.edges(data=True):
        rel_obj = data.get("relation_obj")
        if rel_obj and not rel_obj.is_shortcut:
            known_pairs.append((src, tgt, rel_obj.relation_type))
            count += 1
            if count >= 5:
                break

    correct = 0
    for src, tgt, true_type in known_pairs:
        candidates = detector.detect_relation(src, tgt, top_k=3, verbose=False)
        predicted = candidates[0].relation_type if candidates else None
        is_correct = (predicted == true_type)
        if is_correct:
            correct += 1
        src_node = net.get_node(src)
        tgt_node = net.get_node(tgt)
        sn = src_node.content[:20] if src_node else src
        tn = tgt_node.content[:20] if tgt_node else tgt
        status = "OK" if is_correct else "NG"
        pred_name = predicted.value if predicted else "None"
        print(f"  [{status}] {sn} -> {tn}")
        print(f"       真实: {true_type.value} | 预测: {pred_name}")

    print(f"\n已知关系预测准确率: {correct}/{len(known_pairs)} = {correct/max(len(known_pairs),1):.1%}")

    # 扫描未链接节点对，发现潜在关系
    print("\n--- 扫描未链接节点对，发现潜在关系 ---")
    candidates = detector.scan_all_unlinked_pairs(
        min_vector_similarity=0.4, verbose=True)

    if candidates:
        print(f"\n置信度最高的 5 条候选关系：")
        for c in candidates[:5]:
            src_node = net.get_node(c.source_id)
            tgt_node = net.get_node(c.target_id)
            sn = src_node.content[:25] if src_node else c.source_id
            tn = tgt_node.content[:25] if tgt_node else c.target_id
            flag = "AUTO" if c.is_verified else "PENDING"
            print(f"  [{flag}] {sn} -[{c.relation_type.value}]-> {tn}  conf={c.confidence:.3f}")

    print("\n[改进1] 关系检测完成")


# ─────────────────────────────────────────────────────────────────
# 测试 改进4：推理结论自动存储（含语境映射集成）
# ─────────────────────────────────────────────────────────────────

def test_auto_store(net):
    separator("改进4 测试：推理结论自动存储 + 语境映射集成")

    engine = AssociativeReasoningEngine(
        net, max_depth=4, auto_store_threshold=0.8)

    nodes_before = len(net.nodes)
    print(f"推理前知识库节点数: {nodes_before}")

    # 在不同语境下测试同一知识库
    test_cases = [
        ("为什么猫在陌生环境会观察出口？", None),
        ("分析本能行为的进化机制原理", None),       # 科学语境，应激活高抽象层
        ("如何举例解释什么是本能行为", None),        # 教育语境，应激活中低层
    ]

    for query, ctx in test_cases:
        print(f"\n{'─'*55}")
        result = engine.reason(query, verbose=True, explicit_context=ctx)
        print(f"\n激活节点: {len(result.activated_nodes)} | 置信度: {result.confidence:.3f}")
        if result.context_profile:
            print(f"语境: {result.context_profile.context_type} "
                  f"层级范围: {result.context_profile.level_range}")
        if result.stored_as_node:
            print(f"[自动存储] 结论节点: {result.stored_as_node}")

    nodes_after = len(net.nodes)
    new_count = nodes_after - nodes_before
    print(f"\n推理后知识库节点数: {nodes_after}（新增 {new_count} 个推理结论节点）")

    if engine.stored_conclusions:
        print("\n自动存储记录：")
        for rec in engine.stored_conclusions:
            print(f"  · {rec['query'][:40]} -> {rec['node_id']} (conf={rec['confidence']:.2f})")

    print("\n[改进4] 自动存储测试完成")


# ─────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  四大改进模块集成测试")
    print("=" * 65)

    print("\n[初始化] 构建记忆知识库...")
    net = build_knowledge_base()
    net.summary()

    print("\n[初始化] 构建向量索引...")
    net.build_vectors()

    # 按顺序测试（改进2最简单，先跑；改进1依赖向量，放中间；改进4综合性最强，放最后）
    test_context_mapper()
    test_conflict_resolver(net)
    test_relation_detector(net)
    test_auto_store(net)

    print(f"\n{'='*65}")
    print("  全部测试完成")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
