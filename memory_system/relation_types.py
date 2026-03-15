"""
元关系类型定义（固定集合，约20种）
这是联想推理的导航骨架，描述信息之间可能存在的连接方式
"""

from enum import Enum
from dataclasses import dataclass


class RelationType(Enum):
    # ── 纵向关系（抽象层级）──────────────────────────
    BELONGS_TO      = "归属"        # A 是 B 的一种（男人 → 人类）
    CONTAINS        = "包含"        # A 包含 B 作为组成部分（人体 → 器官）
    INSTANCE_OF     = "实例"        # A 是 B 的一个具体例子（张三 → 男人）

    # ── 横向关系（同层关联）──────────────────────────
    ANALOGOUS_TO    = "类比"        # A 和 B 在某维度上相似（猴子行为 ≈ 人类行为）
    OPPOSITE_TO     = "对立"        # A 和 B 在某维度上相反（掠食者 ↔ 猎物）
    CO_OCCURS_WITH  = "共生"        # A 和 B 同时出现频率高（危险环境 & 警觉行为）

    # ── 因果关系（时间维度）──────────────────────────
    CAUSES          = "前因"        # A 导致 B（危险环境 → 进化出感知能力）
    CAUSED_BY       = "后果"        # B 由 A 产生（警觉行为 ← 危险环境）
    CONDITION_FOR   = "条件"        # 满足 A 时才有 B（有危险 → 才需要感知）

    # ── 制约关系（空间/结构维度）────────────────────
    CONSTRAINS      = "限制"        # A 的存在约束 B 的范围（物理规律 限制 可能行为）
    DEPENDS_ON      = "依赖"        # B 的存在需要 A（行为表现 依赖 神经系统）
    COMPETES_WITH   = "竞争"        # A 和 B 争夺同一资源（物种间的竞争）

    # ── 功能关系（目的维度）──────────────────────────
    TOOL_FOR        = "工具"        # A 是实现 B 的手段（感知能力 是 生存的手段）
    GOAL_OF         = "目标"        # B 是 A 的目的（生存 是 进化的目标）
    SUBSTITUTES     = "替代"        # A 可以替换 B 实现同样功能

    # ── 极性功能关系（方向明确的影响）──────────────
    PROMOTES        = "促进"        # A 促进/增强/激活 B（运动 → 促进 海马体神经新生）
    INHIBITS        = "抑制"        # A 抑制/削弱/阻碍 B（长期压力 → 抑制 免疫功能）

    # ── 进化/溯源关系（特殊纵向）────────────────────
    EVOLVED_FROM    = "进化溯源"    # A 进化自 B（人类 进化自 灵长类）
    DERIVED_FROM    = "派生"        # A 概念派生自 B（外显行为 派生自 本能）

    # ── 时间关系 ─────────────────────────────────────
    PRECEDES        = "时序前"      # A 在时间上先于 B
    FOLLOWS         = "时序后"      # A 在时间上后于 B

    # ── 属性关系 ─────────────────────────────────────
    HAS_PROPERTY    = "具有属性"    # A 具有属性 B（人类 具有属性 社会性）


@dataclass
class Relation:
    """一条关系边"""
    source_id: str           # 源节点ID
    target_id: str           # 目标节点ID
    relation_type: RelationType
    weight: float = 1.0      # 关系强度
    is_shortcut: bool = False  # 是否为自动生成的快捷边
    use_count: int = 0       # 被使用次数（用于生成快捷边的依据）
    context: str = ""        # 附加语境说明（如 "EXCEPT_CASE" / "auto_detected"）

    def to_dict(self):
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.relation_type.value,
            "weight": self.weight,
            "is_shortcut": self.is_shortcut,
            "context": self.context,
        }


# 联想推理时的关系优先级（值越小优先级越高）
RELATION_PRIORITY = {
    RelationType.BELONGS_TO:     1,
    RelationType.EVOLVED_FROM:   1,
    RelationType.CAUSES:         2,
    RelationType.CAUSED_BY:      2,
    RelationType.HAS_PROPERTY:   2,
    RelationType.CONTAINS:       3,
    RelationType.DEPENDS_ON:     3,
    RelationType.CONDITION_FOR:  3,
    RelationType.PROMOTES:       2,   # 与因果同级，方向明确，推理价值高
    RelationType.INHIBITS:       2,   # 同上
    RelationType.ANALOGOUS_TO:   4,
    RelationType.CO_OCCURS_WITH: 4,
    RelationType.TOOL_FOR:       4,
    RelationType.GOAL_OF:        4,
    RelationType.DERIVED_FROM:   4,
    RelationType.CONSTRAINS:     5,
    RelationType.PRECEDES:       5,
    RelationType.FOLLOWS:        5,
    RelationType.INSTANCE_OF:    5,
    RelationType.OPPOSITE_TO:    6,
    RelationType.COMPETES_WITH:  6,
    RelationType.SUBSTITUTES:    6,
}
