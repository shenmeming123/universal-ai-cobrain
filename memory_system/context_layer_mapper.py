"""
改进2：语境-层级映射表
核心思想：同一概念在不同语境下应激活不同的抽象层级
例：
  "猫" 在 "日常对话" 语境 → 激活具体实例层（abstract_level 0-3）
  "猫" 在 "生物进化" 语境 → 激活规律层（abstract_level 6-10）
  "猫" 在 "工程应用" 语境 → 激活中间层（abstract_level 3-7）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import re


# ─────────────────────────────────────────────────────────────────────────────
# 语境类型定义
# ─────────────────────────────────────────────────────────────────────────────

CONTEXT_TYPES = {
    "daily":        "日常对话",
    "science":      "科学分析",
    "engineering":  "工程应用",
    "philosophy":   "哲学抽象",
    "education":    "教育解释",
    "medical":      "医学诊断",
    "history":      "历史叙述",
    "creative":     "创意联想",
}


@dataclass
class ContextProfile:
    """一次查询的语境画像"""
    context_type: str           # 语境类型（对应 CONTEXT_TYPES 的 key）
    level_range: Tuple[int, int]  # 推荐激活的抽象层级范围 [min, max]
    confidence: float           # 语境识别置信度
    matched_signals: List[str] = field(default_factory=list)  # 触发识别的信号词


# ─────────────────────────────────────────────────────────────────────────────
# 语境-层级映射表（核心配置）
# ─────────────────────────────────────────────────────────────────────────────

# 格式：语境类型 -> (层级范围下限, 层级范围上限, 权重衰减函数描述)
CONTEXT_LEVEL_MAP: Dict[str, Tuple[int, int]] = {
    "daily":        (0, 4),    # 日常对话：具体实例，不要太抽象
    "education":    (1, 6),    # 教育解释：从具体到原理，中等抽象
    "engineering":  (2, 7),    # 工程应用：偏中层，需要可操作的原理
    "medical":      (2, 7),    # 医学诊断：具体症状+机制原理
    "history":      (1, 5),    # 历史叙述：具体事件为主
    "science":      (5, 10),   # 科学分析：规律层，需要高抽象
    "creative":     (0, 10),   # 创意联想：全层级开放，跨层联想
    "philosophy":   (7, 10),   # 哲学抽象：最高层，本质规律
}

# 各语境的触发信号词（用于语境识别）
CONTEXT_SIGNALS: Dict[str, List[str]] = {
    "daily": [
        "怎么", "为什么", "是什么", "好不好", "可以吗", "怎样",
        "有没有", "能不能", "会不会", "平时", "一般", "通常",
        "我想", "我要", "帮我", "告诉我",
    ],
    "science": [
        "机制", "原理", "规律", "理论", "本质", "证明", "实验",
        "假设", "验证", "分析", "研究", "数据", "模型", "算法",
        "进化", "演化", "基因", "神经", "量子", "热力学",
    ],
    "engineering": [
        "实现", "设计", "架构", "系统", "方案", "优化", "效率",
        "部署", "开发", "代码", "框架", "接口", "模块", "性能",
        "如何做", "怎么实现", "怎么设计",
    ],
    "medical": [
        "症状", "诊断", "治疗", "病因", "药物", "手术", "检查",
        "患者", "医院", "疾病", "健康", "预防", "康复",
    ],
    "history": [
        "历史", "古代", "战争", "朝代", "年代", "事件", "人物",
        "发生了", "那时候", "过去", "曾经",
    ],
    "creative": [
        "联想", "类比", "想象", "创意", "如果", "假如", "像什么",
        "有什么相似", "启发", "灵感",
    ],
    "philosophy": [
        "本质", "存在", "意义", "价值", "道德", "伦理", "真理",
        "认识论", "形而上", "为何存在", "终极",
    ],
    "education": [
        "解释", "举例", "说明", "理解", "学习", "教", "入门",
        "基础", "概念", "简单来说", "通俗",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 语境识别器
# ─────────────────────────────────────────────────────────────────────────────

class ContextLayerMapper:
    """
    语境-层级映射器
    功能：
      1. 识别查询语境（identify_context）
      2. 给定语境，过滤/排序候选节点（filter_by_context）
      3. 计算节点在当前语境下的相关度加权（context_weight）
    """

    def __init__(self):
        self.context_level_map = CONTEXT_LEVEL_MAP
        self.context_signals = CONTEXT_SIGNALS
        # 默认语境：日常对话
        self.default_context = "daily"

    # ── 语境识别 ─────────────────────────────────────────────────────────────

    def identify_context(self, query: str,
                         explicit_context: Optional[str] = None) -> ContextProfile:
        """
        识别查询语境
        :param query: 用户查询文本
        :param explicit_context: 外部显式指定的语境（如有则直接使用）
        :return: ContextProfile
        """
        # 外部显式指定，直接返回
        if explicit_context and explicit_context in self.context_level_map:
            level_range = self.context_level_map[explicit_context]
            return ContextProfile(
                context_type=explicit_context,
                level_range=level_range,
                confidence=1.0,
                matched_signals=["explicit"],
            )

        query_lower = query.lower()
        scores: Dict[str, float] = {}
        matched: Dict[str, List[str]] = {}

        # 信号词匹配打分
        for ctx_type, signals in self.context_signals.items():
            hit_signals = [s for s in signals if s in query_lower]
            if hit_signals:
                scores[ctx_type] = len(hit_signals) / len(signals)
                matched[ctx_type] = hit_signals

        if not scores:
            # 无匹配，使用默认语境
            level_range = self.context_level_map[self.default_context]
            return ContextProfile(
                context_type=self.default_context,
                level_range=level_range,
                confidence=0.3,
                matched_signals=[],
            )

        # 取得分最高的语境
        best_ctx = max(scores, key=scores.__getitem__)
        confidence = min(0.95, scores[best_ctx] * 5 + 0.3)  # 归一化到合理范围

        level_range = self.context_level_map[best_ctx]
        return ContextProfile(
            context_type=best_ctx,
            level_range=level_range,
            confidence=confidence,
            matched_signals=matched.get(best_ctx, []),
        )

    # ── 节点过滤与加权 ────────────────────────────────────────────────────────

    def context_weight(self, node_abstract_level: int,
                       context_profile: ContextProfile) -> float:
        """
        计算节点在当前语境下的权重系数
        - 节点抽象层级在推荐范围内：权重 1.0（满分）
        - 轻微偏离：权重线性衰减
        - 严重偏离：权重接近 0.1（不完全排除，允许跨层联想）

        :param node_abstract_level: 节点的 abstract_level（0-10）
        :param context_profile: 当前语境画像
        :return: 权重系数 [0.1, 1.0]
        """
        low, high = context_profile.level_range
        level = node_abstract_level

        if low <= level <= high:
            # 在范围内，满权重（范围中心略高）
            center = (low + high) / 2
            distance_to_center = abs(level - center)
            span = (high - low) / 2 + 1e-6
            return 1.0 - 0.1 * (distance_to_center / span)  # 最多衰减10%

        # 偏离范围，线性衰减
        if level < low:
            distance = low - level
        else:
            distance = level - high

        # 每偏离1级，衰减15%，最低0.1
        weight = max(0.1, 1.0 - distance * 0.15)
        return weight

    def filter_nodes_by_context(self,
                                 node_list: List[Tuple],
                                 context_profile: ContextProfile,
                                 min_weight: float = 0.2) -> List[Tuple]:
        """
        对候选节点列表按语境权重重新排序
        :param node_list: [(node_id, score, node_obj), ...]
        :param context_profile: 语境画像
        :param min_weight: 语境权重低于此值的节点被过滤
        :return: 重排序后的列表
        """
        weighted = []
        for item in node_list:
            node_id, score = item[0], item[1]
            node = item[2] if len(item) > 2 else None

            abstract_level = 5  # 默认中等层级
            if node and hasattr(node, 'abstract_level'):
                abstract_level = node.abstract_level

            ctx_w = self.context_weight(abstract_level, context_profile)
            if ctx_w >= min_weight:
                # 最终得分 = 原始得分 × 语境权重
                final_score = score * ctx_w
                weighted.append((node_id, final_score, node, ctx_w))

        weighted.sort(key=lambda x: x[1], reverse=True)
        # 返回格式保持与输入一致
        return [(item[0], item[1], item[2]) for item in weighted]

    def describe_context(self, profile: ContextProfile) -> str:
        """输出语境画像的可读描述（用于日志/论文记录）"""
        ctx_name = CONTEXT_TYPES.get(profile.context_type, profile.context_type)
        low, high = profile.level_range
        return (f"[语境] {ctx_name} | 推荐层级: {low}-{high} | "
                f"置信度: {profile.confidence:.2f} | "
                f"触发词: {profile.matched_signals[:3]}")
