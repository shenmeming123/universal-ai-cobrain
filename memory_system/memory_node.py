"""
记忆单元定义
每个记忆节点包含：核心内容、抽象层级、领域归属、标签、权重、验证历史、
认识论状态（epistemic_status）、来源可信度追踪
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time


class EpistemicStatus(Enum):
    """
    节点的认识论状态——描述该知识的当前"知识性质"。

    confirmed   : 已确认为正确（有充分证据支持，或经过归谬验证）
    hypothesis  : 假设/待验证（有迹象但无直接确认，类型一可能性）
    potential   : 潜在可能性（条件具备但尚未发生，类型二可能性）
    false       : 已确认为错误（归谬验证后确认为假，保留用于识别重复错误信息）
    paradox_pending : 悖论暂存（与现有公理级节点矛盾，等待人工裁决）

    状态流转示意：
      hypothesis → confirmed  （证据充分时升级）
      hypothesis → false      （归谬验证确认为假）
      potential  → confirmed  （条件触发、事件发生后升级）
      potential  → false      （条件不成立时降级）
      paradox_pending → confirmed / false / hypothesis（人工裁决后）
    """
    CONFIRMED       = "confirmed"
    HYPOTHESIS      = "hypothesis"
    POTENTIAL       = "potential"
    FALSE           = "false"
    PARADOX_PENDING = "paradox_pending"


@dataclass
class MemoryNode:
    """记忆单元"""

    # 唯一标识
    node_id: str

    # 核心内容（自然语言描述）
    content: str

    # 静态坐标（存储时确定）
    abstract_level: int          # 抽象层级 0=最具体 10=最抽象
    domain: List[str]            # 领域归属，如 ["生物学", "进化论"]
    coverage: float              # 覆盖度：能映射到的外延信息数量估计（0-1）

    # 本质特征列表（该节点所属的抽象概念）
    essence_features: List[str] = field(default_factory=list)

    # 标签（用于快速跳跃检索）
    tags: List[str] = field(default_factory=list)

    # 动态属性（运行时更新）
    weight: float = 1.0          # 当前信任权重（被验证影响）
    activation_count: int = 0    # 被激活次数
    created_at: float = field(default_factory=time.time)
    last_activated_at: float = field(default_factory=time.time)  # 最近一次激活时间戳

    # 认识论状态（默认已确认，存量节点不受影响）
    epistemic_status: EpistemicStatus = EpistemicStatus.CONFIRMED

    # 类型二可能性节点专用：触发该可能性所需的条件节点ID列表
    trigger_conditions: List[str] = field(default_factory=list)

    # 来源可信度追踪：{source_id: trust_score}
    # 记录哪些信息来源提供了本节点的支持信息，以及来源的信任得分
    source_trust: Dict[str, float] = field(default_factory=dict)

    # 若 epistemic_status=FALSE，记录驳回原因
    refutation_reason: Optional[str] = None

    # 验证历史
    validation_history: List[Dict[str, Any]] = field(default_factory=list)

    def effective_weight(self, decay_half_life_days: float = 30.0) -> float:
        """
        带时间衰减的有效权重（修复"只增不减"问题）。

        设计原则：
          - 真实信任权重（self.weight）永远不改变，避免信息丢失
          - effective_weight 是推理时实际使用的"当前影响力"
          - 衰减只影响推理排序，不影响四重验证等逻辑判断

        衰减公式（指数半衰期）：
          age_days = (now - last_activated_at) / 86400
          decay    = 0.5 ^ (age_days / half_life_days)
          result   = weight × decay，下限为 weight × 0.1（最多衰减90%）

        参数：
          decay_half_life_days : 衰减半衰期（天）。
            默认 30 天：激活后30天权重降至50%，90天后降至12.5%。
            值越大衰减越慢（保守），值越小衰减越快（激进）。

        典型效果：
          - 近期高频激活节点（0~7天）：effective_weight ≈ weight（基本无影响）
          - 1个月未激活节点：effective_weight ≈ weight × 0.5
          - 3个月未激活节点：effective_weight ≈ weight × 0.125
          - FALSE/PARADOX_PENDING 节点：返回 0（直接屏蔽，不参与排序）
        """
        if self.epistemic_status in (EpistemicStatus.FALSE,
                                     EpistemicStatus.PARADOX_PENDING):
            return 0.0

        age_days = (time.time() - self.last_activated_at) / 86400.0
        decay = 0.5 ** (age_days / max(decay_half_life_days, 0.1))
        # 衰减下限：不低于原权重的10%，保留最低存在感
        min_effective = self.weight * 0.1
        return max(min_effective, self.weight * decay)

    def update_weight(self, delta: float, reason: str):
        """渐进式更新权重（不一刀切）"""
        # 渐进式：每次只调整 delta 的 30%，避免单向趋势
        actual_delta = delta * 0.3
        self.weight = max(0.1, min(2.0, self.weight + actual_delta))
        # 更新最近激活时间（凡是调整权重，均视为"被使用"）
        self.last_activated_at = time.time()
        self.validation_history.append({
            "time": time.time(),
            "delta": actual_delta,
            "reason": reason,
            "new_weight": self.weight
        })

    def mark_false(self, reason: str):
        """将节点标记为已确认错误，记录驳回原因"""
        self.epistemic_status = EpistemicStatus.FALSE
        self.refutation_reason = reason
        # 权重降至最低，但不删除节点（保留用于识别重复错误信息）
        self.weight = 0.1
        self.validation_history.append({
            "time": time.time(),
            "delta": 0,
            "reason": f"标记为FALSE: {reason}",
            "new_weight": self.weight
        })

    def mark_confirmed(self, reason: str = ""):
        """将节点升级为已确认状态"""
        self.epistemic_status = EpistemicStatus.CONFIRMED
        self.validation_history.append({
            "time": time.time(),
            "delta": 0,
            "reason": f"升级为CONFIRMED: {reason}",
            "new_weight": self.weight
        })

    def update_source_trust(self, source_id: str, delta: float):
        """
        更新来源可信度。
        当某来源提供了已知错误信息时，调用此方法降低该来源的信任分。
        当来源提供了被验证正确的信息时，调用此方法提升信任分。
        信任分范围：0.0（完全不信任）~ 1.0（完全信任），默认 0.8
        """
        current = self.source_trust.get(source_id, 0.8)
        self.source_trust[source_id] = max(0.0, min(1.0, current + delta))

    def is_reliable(self, min_weight: float = 0.3) -> bool:
        """
        判断本节点是否可信（用于推理时过滤）。
        FALSE 状态的节点在正向推理中应被跳过，
        但在"识别错误来源"的特殊推理中仍可被读取。
        """
        return (self.epistemic_status != EpistemicStatus.FALSE
                and self.weight >= min_weight)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "content": self.content,
            "abstract_level": self.abstract_level,
            "domain": self.domain,
            "coverage": self.coverage,
            "essence_features": self.essence_features,
            "tags": self.tags,
            "weight": self.weight,
            "activation_count": self.activation_count,
            "last_activated_at": self.last_activated_at,
            "epistemic_status": self.epistemic_status.value,
            "trigger_conditions": self.trigger_conditions,
            "source_trust": self.source_trust,
            "refutation_reason": self.refutation_reason,
        }
