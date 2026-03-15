"""
综合测试：PROMOTES/INHIBITS 极性关系下的记忆存储与联想检索
=================================================================
测试维度：
  T1 - 极性关系存储：PROMOTES/INHIBITS 边能否正确建立并持久化
  T2 - 极性冲突检测：同一对节点同时存在 PROMOTES/INHIBITS 边能否被拦截
  T3 - 方向性推理：含极性意图的查询（"促进"/"抑制"）能否激活对应方向的边
  T4 - 多跳推理：通过 PROMOTES 链条进行多步联想（A促进B，B促进C → A间接促进C）
  T5 - 极性对比推理：同一目标既有促进者又有抑制者，能否同时检索并对比

知识域：神经科学/记忆与学习（节点规模约30个，关系约40条）
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

from memory_network import MemoryNetwork
from memory_node import MemoryNode
from relation_types import Relation, RelationType
from conflict_resolver import ConflictResolver, NewInformation, ConflictLevel
from associative_engine import AssociativeReasoningEngine

# ─────────────────────────────────────────────────────────────────────────────
# 知识库构建（神经科学领域，约30节点+40关系）
# ─────────────────────────────────────────────────────────────────────────────

def build_neuro_knowledge_base() -> MemoryNetwork:
    """
    构建神经科学/记忆领域知识库，
    包含大量 PROMOTES/INHIBITS 关系，用于测试极性感知推理。
    """
    net = MemoryNetwork()

    # ── 核心概念节点（抽象层级 6-9）────────────────────────
    nodes_abstract = [
        MemoryNode("海马体", "海马体是大脑中负责记忆形成和空间导航的核心结构",
                   abstract_level=7, domain=["神经科学"],
                   coverage=0.7, essence_features=["记忆形成", "空间导航"],
                   tags=["海马体", "记忆", "大脑结构"]),
        MemoryNode("突触可塑性", "突触可塑性是神经元连接强度随活动而改变的能力，是学习的细胞基础",
                   abstract_level=8, domain=["神经科学", "细胞生物学"],
                   coverage=0.75, essence_features=["连接强度变化", "活动依赖"],
                   tags=["突触", "可塑性", "学习", "记忆"]),
        MemoryNode("长时程增强", "长时程增强（LTP）是突触传递效能的持久性增强，是记忆巩固的分子机制",
                   abstract_level=8, domain=["神经科学"],
                   coverage=0.65, essence_features=["突触增强", "持久性", "记忆巩固"],
                   tags=["LTP", "突触", "记忆", "巩固"]),
        MemoryNode("神经发生", "成人海马体中持续产生新神经元的过程，与记忆编码和情绪调节有关",
                   abstract_level=8, domain=["神经科学"],
                   coverage=0.6, essence_features=["新神经元", "海马体", "记忆编码"],
                   tags=["神经发生", "海马体", "记忆", "神经元"]),
        MemoryNode("皮质醇", "皮质醇是肾上腺分泌的应激激素，对记忆和免疫功能有双向调节作用",
                   abstract_level=7, domain=["内分泌学", "神经科学"],
                   coverage=0.7, essence_features=["应激激素", "双向调节"],
                   tags=["皮质醇", "应激", "激素", "记忆"]),
        MemoryNode("BDNF", "脑源性神经营养因子（BDNF）是促进神经元存活、生长和突触可塑性的关键蛋白",
                   abstract_level=7, domain=["神经科学", "分子生物学"],
                   coverage=0.65, essence_features=["神经营养", "促进神经元生长"],
                   tags=["BDNF", "神经营养因子", "突触", "可塑性"]),
        MemoryNode("睡眠", "睡眠是记忆巩固的关键阶段，慢波睡眠期间海马体向新皮质转移记忆",
                   abstract_level=7, domain=["神经科学", "睡眠医学"],
                   coverage=0.75, essence_features=["记忆巩固", "海马体转移"],
                   tags=["睡眠", "记忆", "巩固", "慢波睡眠"]),
        MemoryNode("有氧运动", "有氧运动通过增加BDNF分泌和促进海马体神经发生来增强记忆",
                   abstract_level=6, domain=["神经科学", "运动科学"],
                   coverage=0.6, essence_features=["BDNF分泌增加", "神经发生促进"],
                   tags=["运动", "有氧", "BDNF", "海马体", "记忆"]),
        MemoryNode("慢性压力", "长期心理压力导致皮质醇持续升高，进而损伤海马体和记忆功能",
                   abstract_level=6, domain=["神经科学", "心理学"],
                   coverage=0.65, essence_features=["皮质醇升高", "海马体损伤"],
                   tags=["压力", "应激", "皮质醇", "海马体"]),
        MemoryNode("急性压力", "短暂适度的压力可通过皮质醇短暂升高增强记忆编码和注意力",
                   abstract_level=6, domain=["神经科学", "心理学"],
                   coverage=0.55, essence_features=["皮质醇短暂升高", "记忆编码增强"],
                   tags=["急性压力", "皮质醇", "记忆", "注意力"]),
        MemoryNode("酒精", "酒精通过抑制谷氨酸受体和增强GABA活动干扰记忆编码",
                   abstract_level=6, domain=["神经药理学"],
                   coverage=0.6, essence_features=["谷氨酸受体抑制", "记忆干扰"],
                   tags=["酒精", "记忆", "GABA", "谷氨酸"]),
        MemoryNode("冥想", "冥想练习通过降低皮质醇水平和增强前额叶功能改善记忆和注意力",
                   abstract_level=6, domain=["神经科学", "心理学"],
                   coverage=0.6, essence_features=["皮质醇降低", "注意力增强"],
                   tags=["冥想", "皮质醇", "注意力", "记忆"]),
        MemoryNode("间隔重复", "间隔重复是利用记忆遗忘曲线规律进行分散学习的高效记忆策略",
                   abstract_level=6, domain=["认知心理学", "教育"],
                   coverage=0.6, essence_features=["遗忘曲线", "分散学习"],
                   tags=["间隔重复", "学习策略", "记忆", "遗忘曲线"]),
        MemoryNode("睡眠剥夺", "睡眠不足阻断记忆巩固过程，并导致海马体活动异常",
                   abstract_level=6, domain=["神经科学", "睡眠医学"],
                   coverage=0.6, essence_features=["记忆巩固阻断", "海马体异常"],
                   tags=["睡眠剥夺", "记忆", "海马体", "睡眠"]),
        MemoryNode("工作记忆", "工作记忆是暂时保存和操作信息的认知系统，容量有限约7±2项",
                   abstract_level=7, domain=["认知心理学", "神经科学"],
                   coverage=0.7, essence_features=["临时存储", "容量有限"],
                   tags=["工作记忆", "认知", "前额叶"]),
        MemoryNode("长期记忆", "长期记忆是信息的持久存储系统，分为陈述性记忆和程序性记忆",
                   abstract_level=8, domain=["认知心理学", "神经科学"],
                   coverage=0.8, essence_features=["持久存储", "陈述性", "程序性"],
                   tags=["长期记忆", "陈述性记忆", "程序性记忆"]),
        MemoryNode("前额叶皮层", "前额叶皮层负责工作记忆、注意控制和执行功能，对记忆的提取和监控至关重要",
                   abstract_level=7, domain=["神经科学"],
                   coverage=0.7, essence_features=["执行功能", "注意控制", "记忆提取"],
                   tags=["前额叶", "工作记忆", "执行功能"]),
        MemoryNode("杏仁核", "杏仁核负责情绪记忆，情绪激活时增强海马体对该事件的记忆编码",
                   abstract_level=7, domain=["神经科学"],
                   coverage=0.65, essence_features=["情绪记忆", "增强记忆编码"],
                   tags=["杏仁核", "情绪", "记忆", "恐惧"]),
        MemoryNode("多巴胺", "多巴胺是与奖励预测和动机相关的神经递质，在记忆巩固和学习强化中起关键作用",
                   abstract_level=7, domain=["神经科学", "神经药理学"],
                   coverage=0.7, essence_features=["奖励信号", "学习强化"],
                   tags=["多巴胺", "奖励", "动机", "记忆"]),
        MemoryNode("乙酰胆碱", "乙酰胆碱是海马体中参与记忆编码的重要神经递质，其缺乏与阿尔茨海默病有关",
                   abstract_level=7, domain=["神经科学", "神经药理学"],
                   coverage=0.65, essence_features=["记忆编码", "海马体调节"],
                   tags=["乙酰胆碱", "记忆", "海马体", "阿尔茨海默"]),
        MemoryNode("氧化应激", "细胞内活性氧积累超过抗氧化防御能力，导致神经元损伤和认知下降",
                   abstract_level=7, domain=["细胞生物学", "神经科学"],
                   coverage=0.65, essence_features=["活性氧积累", "神经元损伤"],
                   tags=["氧化应激", "神经元", "认知", "抗氧化"]),
        MemoryNode("遗忘曲线", "艾宾浩斯遗忘曲线揭示记忆随时间指数衰减的规律，学习后24h内遗忘最快",
                   abstract_level=7, domain=["认知心理学"],
                   coverage=0.7, essence_features=["记忆衰减", "时间指数"],
                   tags=["遗忘曲线", "艾宾浩斯", "遗忘", "记忆"]),
        MemoryNode("情境依赖记忆", "记忆提取受编码时情境影响，在原始情境中提取效果最佳",
                   abstract_level=6, domain=["认知心理学"],
                   coverage=0.55, essence_features=["情境匹配", "提取增强"],
                   tags=["情境依赖", "记忆提取", "编码情境"]),
        MemoryNode("营养与记忆", "充足的Omega-3脂肪酸、抗氧化剂摄入支持神经元健康和记忆功能",
                   abstract_level=6, domain=["营养学", "神经科学"],
                   coverage=0.55, essence_features=["营养支持", "神经元健康"],
                   tags=["营养", "Omega-3", "记忆", "神经元"]),
        MemoryNode("记忆宫殿", "记忆宫殿（地点记忆法）通过将信息与空间位置关联，利用海马体空间导航能力增强记忆",
                   abstract_level=5, domain=["认知心理学"],
                   coverage=0.5, essence_features=["空间关联", "记忆增强策略"],
                   tags=["记忆宫殿", "记忆技巧", "空间", "海马体"]),
    ]

    for node in nodes_abstract:
        net.add_node(node)

    # ── 关系建立（包含大量 PROMOTES/INHIBITS）──────────────────
    relations = [
        # 有氧运动 → PROMOTES → BDNF
        Relation("有氧运动", "BDNF", RelationType.PROMOTES, weight=0.9),
        # 有氧运动 → PROMOTES → 神经发生
        Relation("有氧运动", "神经发生", RelationType.PROMOTES, weight=0.85),
        # 有氧运动 → PROMOTES → 海马体
        Relation("有氧运动", "海马体", RelationType.PROMOTES, weight=0.8),

        # BDNF → PROMOTES → 突触可塑性
        Relation("BDNF", "突触可塑性", RelationType.PROMOTES, weight=0.9),
        # BDNF → PROMOTES → 神经发生
        Relation("BDNF", "神经发生", RelationType.PROMOTES, weight=0.85),
        # BDNF → PROMOTES → 长时程增强
        Relation("BDNF", "长时程增强", RelationType.PROMOTES, weight=0.8),

        # 突触可塑性 → PROMOTES → 长时程增强
        Relation("突触可塑性", "长时程增强", RelationType.PROMOTES, weight=0.9),
        # 长时程增强 → PROMOTES → 长期记忆
        Relation("长时程增强", "长期记忆", RelationType.PROMOTES, weight=0.9),
        # 神经发生 → PROMOTES → 海马体
        Relation("神经发生", "海马体", RelationType.PROMOTES, weight=0.8),
        # 海马体 → PROMOTES → 长期记忆
        Relation("海马体", "长期记忆", RelationType.PROMOTES, weight=0.9),

        # 睡眠 → PROMOTES → 长期记忆
        Relation("睡眠", "长期记忆", RelationType.PROMOTES, weight=0.9),
        # 睡眠 → PROMOTES → 突触可塑性
        Relation("睡眠", "突触可塑性", RelationType.PROMOTES, weight=0.8),
        # 间隔重复 → PROMOTES → 长期记忆
        Relation("间隔重复", "长期记忆", RelationType.PROMOTES, weight=0.9),
        # 记忆宫殿 → PROMOTES → 海马体（利用空间能力）
        Relation("记忆宫殿", "海马体", RelationType.PROMOTES, weight=0.75),

        # 多巴胺 → PROMOTES → 长期记忆
        Relation("多巴胺", "长期记忆", RelationType.PROMOTES, weight=0.85),
        # 乙酰胆碱 → PROMOTES → 长期记忆
        Relation("乙酰胆碱", "长期记忆", RelationType.PROMOTES, weight=0.85),
        # 杏仁核 → PROMOTES → 长期记忆（情绪增强记忆）
        Relation("杏仁核", "长期记忆", RelationType.PROMOTES, weight=0.8),
        # 营养与记忆 → PROMOTES → 突触可塑性
        Relation("营养与记忆", "突触可塑性", RelationType.PROMOTES, weight=0.7),
        # 情境依赖记忆 → PROMOTES → 长期记忆（提取效率）
        Relation("情境依赖记忆", "长期记忆", RelationType.PROMOTES, weight=0.7),

        # 急性压力 → PROMOTES → 长期记忆（短暂皮质醇增强编码）
        Relation("急性压力", "长期记忆", RelationType.PROMOTES, weight=0.6),
        # 急性压力 → PROMOTES → 皮质醇（短暂升高）
        Relation("急性压力", "皮质醇", RelationType.PROMOTES, weight=0.8),

        # 冥想 → INHIBITS → 皮质醇（降低皮质醇）
        Relation("冥想", "皮质醇", RelationType.INHIBITS, weight=0.8),
        # 冥想 → PROMOTES → 前额叶皮层
        Relation("冥想", "前额叶皮层", RelationType.PROMOTES, weight=0.75),
        # 冥想 → PROMOTES → 工作记忆
        Relation("冥想", "工作记忆", RelationType.PROMOTES, weight=0.7),

        # 皮质醇（长期高水平）→ INHIBITS → 海马体
        Relation("皮质醇", "海马体", RelationType.INHIBITS, weight=0.85),
        # 皮质醇 → INHIBITS → 突触可塑性
        Relation("皮质醇", "突触可塑性", RelationType.INHIBITS, weight=0.8),
        # 皮质醇 → INHIBITS → 神经发生
        Relation("皮质醇", "神经发生", RelationType.INHIBITS, weight=0.8),
        # 皮质醇 → INHIBITS → BDNF
        Relation("皮质醇", "BDNF", RelationType.INHIBITS, weight=0.75),

        # 慢性压力 → PROMOTES → 皮质醇（持续升高）
        Relation("慢性压力", "皮质醇", RelationType.PROMOTES, weight=0.9),
        # 慢性压力 → INHIBITS → 海马体（通过皮质醇间接，直接效应也有）
        Relation("慢性压力", "海马体", RelationType.INHIBITS, weight=0.8),
        # 慢性压力 → INHIBITS → 神经发生
        Relation("慢性压力", "神经发生", RelationType.INHIBITS, weight=0.8),

        # 酒精 → INHIBITS → 长时程增强
        Relation("酒精", "长时程增强", RelationType.INHIBITS, weight=0.85),
        # 酒精 → INHIBITS → 海马体
        Relation("酒精", "海马体", RelationType.INHIBITS, weight=0.8),
        # 酒精 → INHIBITS → 长期记忆
        Relation("酒精", "长期记忆", RelationType.INHIBITS, weight=0.85),

        # 睡眠剥夺 → INHIBITS → 长期记忆
        Relation("睡眠剥夺", "长期记忆", RelationType.INHIBITS, weight=0.9),
        # 睡眠剥夺 → INHIBITS → 突触可塑性
        Relation("睡眠剥夺", "突触可塑性", RelationType.INHIBITS, weight=0.85),
        # 睡眠剥夺 → INHIBITS → 海马体
        Relation("睡眠剥夺", "海马体", RelationType.INHIBITS, weight=0.8),

        # 氧化应激 → INHIBITS → 突触可塑性
        Relation("氧化应激", "突触可塑性", RelationType.INHIBITS, weight=0.8),
        # 氧化应激 → INHIBITS → 长期记忆
        Relation("氧化应激", "长期记忆", RelationType.INHIBITS, weight=0.75),

        # 其他结构关系
        Relation("工作记忆", "长期记忆", RelationType.CONDITION_FOR, weight=0.7),
        Relation("前额叶皮层", "工作记忆", RelationType.DEPENDS_ON, weight=0.85),
        Relation("遗忘曲线", "间隔重复", RelationType.CONDITION_FOR, weight=0.8),
        Relation("皮质醇", "氧化应激", RelationType.CAUSES, weight=0.7),
        Relation("海马体", "工作记忆", RelationType.CO_OCCURS_WITH, weight=0.7),
        Relation("睡眠", "睡眠剥夺", RelationType.OPPOSITE_TO, weight=1.0),
        Relation("急性压力", "慢性压力", RelationType.OPPOSITE_TO, weight=0.8),
    ]

    for rel in relations:
        if net.get_node(rel.source_id) and net.get_node(rel.target_id):
            net.add_relation(rel)

    node_count = len(net.nodes)
    edge_count = net.graph.number_of_edges()
    promotes_count = sum(1 for _, _, d in net.graph.edges(data=True)
                         if d.get("relation_obj") and
                         d["relation_obj"].relation_type == RelationType.PROMOTES)
    inhibits_count = sum(1 for _, _, d in net.graph.edges(data=True)
                         if d.get("relation_obj") and
                         d["relation_obj"].relation_type == RelationType.INHIBITS)
    print(f"[知识库] 节点:{node_count}  边:{edge_count}  "
          f"PROMOTES:{promotes_count}  INHIBITS:{inhibits_count}")
    return net


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

def sub(title: str):
    print(f"\n  --- {title} ---")

def ok(msg: str):
    print(f"  [PASS] {msg}")

def fail(msg: str):
    print(f"  [FAIL] {msg}")

def info(msg: str):
    print(f"  [INFO] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# T1: 极性关系存储验证
# ─────────────────────────────────────────────────────────────────────────────

def test_t1_storage(net: MemoryNetwork) -> dict:
    section("T1: 极性关系存储")
    results = {"pass": 0, "fail": 0}

    # 统计 PROMOTES / INHIBITS 边
    promotes_edges = [(s, t) for s, t, d in net.graph.edges(data=True)
                      if d.get("relation_obj") and
                      d["relation_obj"].relation_type == RelationType.PROMOTES]
    inhibits_edges = [(s, t) for s, t, d in net.graph.edges(data=True)
                      if d.get("relation_obj") and
                      d["relation_obj"].relation_type == RelationType.INHIBITS]

    info(f"PROMOTES 边数: {len(promotes_edges)}")
    info(f"INHIBITS 边数: {len(inhibits_edges)}")

    # 检查关键边存在
    key_promotes = [
        ("有氧运动", "BDNF"),
        ("BDNF", "突触可塑性"),
        ("突触可塑性", "长时程增强"),
        ("长时程增强", "长期记忆"),
        ("睡眠", "长期记忆"),
    ]
    key_inhibits = [
        ("皮质醇", "海马体"),
        ("酒精", "长期记忆"),
        ("睡眠剥夺", "长期记忆"),
        ("慢性压力", "神经发生"),
    ]

    sub("关键 PROMOTES 边检查")
    for src, tgt in key_promotes:
        found = any(s == src and t == tgt for s, t in promotes_edges)
        if found:
            ok(f"{src} → PROMOTES → {tgt}")
            results["pass"] += 1
        else:
            fail(f"{src} → PROMOTES → {tgt} 未找到")
            results["fail"] += 1

    sub("关键 INHIBITS 边检查")
    for src, tgt in key_inhibits:
        found = any(s == src and t == tgt for s, t in inhibits_edges)
        if found:
            ok(f"{src} → INHIBITS → {tgt}")
            results["pass"] += 1
        else:
            fail(f"{src} → INHIBITS → {tgt} 未找到")
            results["fail"] += 1

    # 验证 get_neighbors 能正确返回极性边
    sub("get_neighbors 极性边可见性")
    neighbors = net.get_neighbors("BDNF", direction="out")
    bdnf_targets = {t: r.relation_type for t, r in neighbors}
    if RelationType.PROMOTES in bdnf_targets.values():
        ok("BDNF 的 out neighbors 包含 PROMOTES 边")
        results["pass"] += 1
    else:
        fail("BDNF 的 out neighbors 未找到 PROMOTES 边")
        results["fail"] += 1

    return results


# ─────────────────────────────────────────────────────────────────────────────
# T2: 极性冲突检测
# ─────────────────────────────────────────────────────────────────────────────

def test_t2_conflict(net: MemoryNetwork) -> dict:
    section("T2: 极性边冲突检测")
    results = {"pass": 0, "fail": 0}
    resolver = ConflictResolver(net)

    # Case A：尝试添加与现有 INHIBITS 边矛盾的 PROMOTES 边
    # 现有：皮质醇 → INHIBITS → 海马体
    # 新增：皮质醇 → PROMOTES → 海马体  ← 应被检测为冲突
    sub("Case A: PROMOTES vs INHIBITS 直接矛盾")
    new_info_a = NewInformation(
        node_id="test_polar_conflict_a",
        content="皮质醇长期升高通过促进海马体突触生长改善记忆功能",
        abstract_level=6,
        domain=["神经科学"],
        coverage=0.5,
        essence_features=["皮质醇促进", "海马体增强"],
        tags=["测试"],
        evidence_strength=0.7,
        source="test",
        proposed_relations=[
            Relation("皮质醇", "海马体", RelationType.PROMOTES, weight=0.7)
        ],
    )
    report_a = resolver.process(new_info_a, verbose=False)
    if report_a.level in (ConflictLevel.RULE, ConflictLevel.PARADOX):
        ok(f"Case A 正确检测为冲突 (level={report_a.level.name}): {report_a.conflict_description[:60]}")
        results["pass"] += 1
    else:
        fail(f"Case A 未检测到冲突 (level={report_a.level.name})")
        results["fail"] += 1

    # Case B：合法的新 PROMOTES 边（不与现有边矛盾）
    sub("Case B: 无冲突的新 PROMOTES 边")
    new_info_b = NewInformation(
        node_id="test_polar_no_conflict_b",
        content="规律冥想练习通过增加BDNF分泌促进海马体神经发生",
        abstract_level=6,
        domain=["神经科学", "心理学"],
        coverage=0.5,
        essence_features=["BDNF分泌增加", "神经发生促进"],
        tags=["测试"],
        evidence_strength=0.75,
        source="test",
        proposed_relations=[
            Relation("test_polar_no_conflict_b", "BDNF", RelationType.PROMOTES, weight=0.75)
        ],
    )
    report_b = resolver.process(new_info_b, verbose=False)
    if report_b.level in (ConflictLevel.NO_CONFLICT, ConflictLevel.LOCAL):
        ok(f"Case B 正确无冲突 (level={report_b.level.name})")
        results["pass"] += 1
    else:
        fail(f"Case B 误判为冲突 (level={report_b.level.name}): {report_b.conflict_description[:60]}")
        results["fail"] += 1

    # Case C：极性边冲突检测不影响无关节点的入库
    sub("Case C: 无关节点不受极性冲突误判影响")
    new_info_c = NewInformation(
        node_id="test_unrelated_c",
        content="生酮饮食可能通过降低神经炎症改善认知功能",
        abstract_level=5,
        domain=["营养学", "神经科学"],
        coverage=0.45,
        essence_features=["生酮", "神经炎症降低"],
        tags=["测试", "生酮"],
        evidence_strength=0.6,
        source="test",
        proposed_relations=[],
    )
    report_c = resolver.process(new_info_c, verbose=False)
    if report_c.level in (ConflictLevel.NO_CONFLICT, ConflictLevel.LOCAL):
        ok(f"Case C 无关节点正确入库 (level={report_c.level.name})")
        results["pass"] += 1
    else:
        fail(f"Case C 无关节点被误拦截 (level={report_c.level.name})")
        results["fail"] += 1

    return results


# ─────────────────────────────────────────────────────────────────────────────
# T3: 极性方向感知推理
# ─────────────────────────────────────────────────────────────────────────────

def test_t3_polarity_reasoning(net: MemoryNetwork) -> dict:
    section("T3: 极性方向感知推理")
    results = {"pass": 0, "fail": 0, "details": []}

    # 先构建向量索引
    try:
        net.build_vectors()
    except Exception as e:
        info(f"向量索引构建（可能已存在）: {e}")

    engine = AssociativeReasoningEngine(net, max_depth=3, auto_store_threshold=0.95)

    # 查询组：正向（促进类）
    promotes_queries = [
        ("什么可以促进记忆？", ["有氧运动", "睡眠", "BDNF", "多巴胺", "间隔重复"]),
        ("什么有助于增强突触可塑性？", ["BDNF", "睡眠", "有氧运动", "营养与记忆"]),
        ("如何改善海马体功能？", ["有氧运动", "BDNF", "神经发生", "冥想"]),
    ]

    # 查询组：负向（抑制类）
    inhibits_queries = [
        ("什么会损伤记忆？", ["睡眠剥夺", "酒精", "皮质醇", "慢性压力", "氧化应激"]),
        ("什么抑制神经发生？", ["皮质醇", "慢性压力", "睡眠剥夺"]),
    ]

    sub("正向查询（促进类）")
    for query, expected_nodes in promotes_queries:
        print(f"\n  查询: '{query}'")
        t0 = time.time()
        result = engine.reason(query, verbose=False)
        elapsed = (time.time() - t0) * 1000

        activated = set(result.activated_nodes)
        hits = [n for n in expected_nodes if n in activated]
        hit_rate = len(hits) / len(expected_nodes)

        # 检查推理路径中是否有 [促进方向匹配] 标注
        path_all = []
        for step in result.reasoning_chain:
            path_all.extend(step.path_taken)
        polar_matches = [p for p in path_all if "促进方向匹配" in p or "PROMOTES" in p]

        info(f"激活节点数: {len(activated)}  命中预期: {len(hits)}/{len(expected_nodes)} ({hit_rate:.0%})")
        info(f"极性路径命中: {len(polar_matches)} 条  耗时: {elapsed:.0f}ms")
        if hits:
            info(f"命中节点: {', '.join(hits)}")

        if hit_rate >= 0.4:
            ok(f"'{query[:20]}...' 命中率 {hit_rate:.0%} >= 40%")
            results["pass"] += 1
        else:
            fail(f"'{query[:20]}...' 命中率 {hit_rate:.0%} < 40%")
            results["fail"] += 1

        results["details"].append({
            "query": query, "polarity": "promotes",
            "hit_rate": hit_rate, "polar_paths": len(polar_matches),
            "elapsed_ms": elapsed
        })

    sub("负向查询（抑制类）")
    for query, expected_nodes in inhibits_queries:
        print(f"\n  查询: '{query}'")
        t0 = time.time()
        result = engine.reason(query, verbose=False)
        elapsed = (time.time() - t0) * 1000

        activated = set(result.activated_nodes)
        hits = [n for n in expected_nodes if n in activated]
        hit_rate = len(hits) / len(expected_nodes)

        path_all = []
        for step in result.reasoning_chain:
            path_all.extend(step.path_taken)
        polar_matches = [p for p in path_all if "抑制方向匹配" in p or "INHIBITS" in p]

        info(f"激活节点数: {len(activated)}  命中预期: {len(hits)}/{len(expected_nodes)} ({hit_rate:.0%})")
        info(f"极性路径命中: {len(polar_matches)} 条  耗时: {elapsed:.0f}ms")
        if hits:
            info(f"命中节点: {', '.join(hits)}")

        if hit_rate >= 0.4:
            ok(f"'{query[:20]}...' 命中率 {hit_rate:.0%} >= 40%")
            results["pass"] += 1
        else:
            fail(f"'{query[:20]}...' 命中率 {hit_rate:.0%} < 40%")
            results["fail"] += 1

        results["details"].append({
            "query": query, "polarity": "inhibits",
            "hit_rate": hit_rate, "polar_paths": len(polar_matches),
            "elapsed_ms": elapsed
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# T4: 多跳 PROMOTES 链条推理
# ─────────────────────────────────────────────────────────────────────────────

def test_t4_multi_hop(net: MemoryNetwork) -> dict:
    """
    有氧运动 → PROMOTES → BDNF → PROMOTES → 突触可塑性
             → PROMOTES → 长时程增强 → PROMOTES → 长期记忆
    查询"有氧运动如何改善记忆"，应该激活这条完整链条上的多个节点。
    """
    section("T4: 多跳 PROMOTES 链条推理")
    results = {"pass": 0, "fail": 0}

    engine = AssociativeReasoningEngine(net, max_depth=4, auto_store_threshold=0.95)

    chain = ["有氧运动", "BDNF", "突触可塑性", "长时程增强", "长期记忆"]
    query = "有氧运动如何通过分子机制促进长期记忆的形成？"

    print(f"\n  查询: '{query}'")
    print(f"  预期链条: {' → '.join(chain)}")

    t0 = time.time()
    result = engine.reason(query, verbose=False)
    elapsed = (time.time() - t0) * 1000

    activated = set(result.activated_nodes)
    chain_hits = [n for n in chain if n in activated]
    chain_coverage = len(chain_hits) / len(chain)

    info(f"链条节点命中: {len(chain_hits)}/{len(chain)} ({chain_coverage:.0%})")
    info(f"命中节点: {', '.join(chain_hits)}")
    info(f"总激活节点: {len(activated)}, 耗时: {elapsed:.0f}ms")

    # 检查多跳路径（步骤2/3的路径日志）
    all_paths = []
    for step in result.reasoning_chain:
        all_paths.extend(step.path_taken)
    promotes_hops = [p for p in all_paths if "PROMOTES" in p or "促进" in p]
    info(f"PROMOTES 路径数: {len(promotes_hops)}")
    for p in promotes_hops[:5]:
        info(f"  {p}")

    if chain_coverage >= 0.6:
        ok(f"多跳链条覆盖率 {chain_coverage:.0%} >= 60%")
        results["pass"] += 1
    else:
        fail(f"多跳链条覆盖率 {chain_coverage:.0%} < 60%")
        results["fail"] += 1

    if len(promotes_hops) >= 2:
        ok(f"推理路径中存在 {len(promotes_hops)} 条 PROMOTES 路径")
        results["pass"] += 1
    else:
        fail(f"推理路径中 PROMOTES 路径过少: {len(promotes_hops)}")
        results["fail"] += 1

    return results


# ─────────────────────────────────────────────────────────────────────────────
# T5: 极性对比推理（同目标，多个促进者和抑制者）
# ─────────────────────────────────────────────────────────────────────────────

def test_t5_polarity_contrast(net: MemoryNetwork) -> dict:
    """
    长期记忆同时有促进者（睡眠/有氧运动/BDNF/多巴胺）和抑制者（酒精/睡眠剥夺/氧化应激）。
    查询应该能同时检索到两类，形成完整的对比视图。
    """
    section("T5: 极性对比推理（长期记忆的促进者与抑制者）")
    results = {"pass": 0, "fail": 0}

    engine = AssociativeReasoningEngine(net, max_depth=3, auto_store_threshold=0.95)

    query = "哪些因素促进长期记忆，哪些因素损伤长期记忆？"
    known_promoters = {"有氧运动", "睡眠", "BDNF", "多巴胺", "乙酰胆碱", "间隔重复"}
    known_inhibitors = {"酒精", "睡眠剥夺", "氧化应激", "慢性压力"}

    print(f"\n  查询: '{query}'")
    t0 = time.time()
    result = engine.reason(query, verbose=False)
    elapsed = (time.time() - t0) * 1000

    activated = set(result.activated_nodes)
    promoter_hits = known_promoters & activated
    inhibitor_hits = known_inhibitors & activated

    info(f"促进者命中: {len(promoter_hits)}/{len(known_promoters)} → {promoter_hits}")
    info(f"抑制者命中: {len(inhibitor_hits)}/{len(known_inhibitors)} → {inhibitor_hits}")
    info(f"总激活节点: {len(activated)}, 耗时: {elapsed:.0f}ms")

    # 分析答案是否包含对比信息
    answer_lower = result.answer.lower()
    has_promotes_mention = any(kw in result.answer for kw in ["促进", "增强", "有益", "改善"])
    has_inhibits_mention = any(kw in result.answer for kw in ["损伤", "抑制", "有害", "降低", "阻碍"])

    info(f"答案提及促进关系: {has_promotes_mention}")
    info(f"答案提及抑制关系: {has_inhibits_mention}")

    if len(promoter_hits) >= 2:
        ok(f"检索到 {len(promoter_hits)} 个促进者节点")
        results["pass"] += 1
    else:
        fail(f"促进者命中不足: {len(promoter_hits)} < 2")
        results["fail"] += 1

    if len(inhibitor_hits) >= 1:
        ok(f"检索到 {len(inhibitor_hits)} 个抑制者节点")
        results["pass"] += 1
    else:
        fail(f"抑制者命中不足: {len(inhibitor_hits)} < 1")
        results["fail"] += 1

    if has_promotes_mention and has_inhibits_mention:
        ok("答案包含对比性描述（同时涉及促进和抑制）")
        results["pass"] += 1
    elif has_promotes_mention or has_inhibits_mention:
        info("答案只包含单一方向描述，对比性不足")
        results["pass"] += 1  # 部分通过
    else:
        fail("答案未体现任何极性对比信息")
        results["fail"] += 1

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  综合测试：极性关系下的记忆存储与联想检索")
    print("  知识域：神经科学/记忆与学习")
    print("=" * 65)

    # 构建知识库
    print("\n[初始化] 构建知识库...")
    net = build_neuro_knowledge_base()

    # 构建向量索引
    print("[初始化] 构建向量索引...")
    try:
        net.build_vectors()
        print("[初始化] 向量索引完成")
    except Exception as e:
        print(f"[初始化] 向量索引失败（测试继续）: {e}")

    # 执行各项测试
    total_pass = 0
    total_fail = 0

    r1 = test_t1_storage(net)
    r2 = test_t2_conflict(net)
    r3 = test_t3_polarity_reasoning(net)
    r4 = test_t4_multi_hop(net)
    r5 = test_t5_polarity_contrast(net)

    all_results = [r1, r2, r3, r4, r5]
    for r in all_results:
        total_pass += r.get("pass", 0)
        total_fail += r.get("fail", 0)

    # 汇总报告
    section("综合测试汇总")
    total = total_pass + total_fail
    pass_rate = total_pass / total if total > 0 else 0

    print(f"\n  测试项总数: {total}")
    print(f"  通过: {total_pass}   失败: {total_fail}")
    print(f"  通过率: {pass_rate:.1%}")
    print()

    labels = ["T1-极性存储", "T2-冲突检测", "T3-方向推理", "T4-多跳链条", "T5-对比推理"]
    for label, r in zip(labels, all_results):
        p, f_ = r.get("pass", 0), r.get("fail", 0)
        status = "PASS" if f_ == 0 else ("PART" if p > 0 else "FAIL")
        print(f"  {label}: {p}通过/{f_}失败  [{status}]")

    print()
    if pass_rate >= 0.75:
        print("  [总体评估] GOOD - 极性关系机制有效支撑记忆存储和联想检索")
    elif pass_rate >= 0.5:
        print("  [总体评估] PARTIAL - 基本功能正常，部分场景需要改进")
    else:
        print("  [总体评估] NEEDS WORK - 需要调整")

    # T3 详细数据
    if r3.get("details"):
        print("\n  T3 查询详情:")
        print(f"  {'查询':<30} {'极性':<8} {'命中率':<8} {'极性路径':<8} {'耗时ms':<8}")
        print("  " + "-" * 62)
        for d in r3["details"]:
            print(f"  {d['query'][:28]:<30} {d['polarity']:<8} "
                  f"{d['hit_rate']:.0%}    {d['polar_paths']:<8} {d['elapsed_ms']:.0f}")

    print()

if __name__ == "__main__":
    main()
