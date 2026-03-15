"""
外挂式元数据模板库（改进B-A步）

解决问题：
  新信息进入系统时，MemoryNode 的元字段（essence_features / abstract_level / domain）
  通常需要人工填写，大模型输入时无法自动补全。
  本模块通过"预置领域知识模板 + 语义最近邻匹配"，自动推断并填充元字段。

使用方式：
  from metadata_template_library import MetadataTemplateLibrary
  lib = MetadataTemplateLibrary(memory_network)     # 传入现有 MemoryNetwork（用于向量相似度）
  filled = lib.fill(content="压力激素升高会导致睡眠障碍")
  # 返回：{
  #   "abstract_level": 6,
  #   "domain": ["神经科学", "心理学"],
  #   "essence_features": ["压力激素", "睡眠障碍", "因果关系"],
  #   "coverage": 0.4,
  #   "match_source": "template:神经内分泌-因果",  # 匹配来源
  #   "confidence": 0.73
  # }

设计：
  1. 预置模板库（TEMPLATES）：覆盖常见领域的元数据样板
     每条模板包含：样例句、domain、abstract_level、essence_features 特征词
  2. 文本相似度匹配：用关键词+领域词命中最佳模板（轻量，不依赖 embedding）
     若 MemoryNetwork 可用，也支持 embedding 距离精匹配（可选）
  3. 特征提取：从内容中提取关键词作为 essence_features 补充（基于词表）
  4. 全程退化友好：无模板命中时，返回合理默认值，不报错
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import re


# ─────────────────────────────────────────────────────────────────────────────
# 模板定义
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetadataTemplate:
    """一条元数据模板"""
    template_id: str
    description: str                    # 该模板适用的场景描述
    domain: List[str]                   # 推荐领域
    abstract_level: int                 # 推荐抽象层级（0~10）
    coverage: float                     # 推荐覆盖度（0~1）
    essence_keywords: List[str]         # 触发该模板的关键词（命中越多越好）
    essence_feature_hints: List[str]    # 建议填入 essence_features 的词
    tags_hint: List[str] = field(default_factory=list)  # 建议标签


# 预置模板库
TEMPLATES: List[MetadataTemplate] = [

    # ── 神经科学 / 心理学 ─────────────────────────────────────────────────

    MetadataTemplate(
        template_id="neuro_anatomy",
        description="神经解剖结构（脑区/神经核团名称及其功能）",
        domain=["神经科学", "生物学"],
        abstract_level=6,
        coverage=0.5,
        essence_keywords=["大脑", "皮层", "杏仁核", "海马体", "前额叶",
                          "小脑", "基底节", "纹状体", "丘脑", "脑干",
                          "神经元", "突触", "轴突", "树突", "神经回路"],
        essence_feature_hints=["神经解剖结构", "脑区功能"],
        tags_hint=["神经科学", "解剖"],
    ),

    MetadataTemplate(
        template_id="neuro_causal",
        description="神经/心理因果关系（X导致Y的神经机制）",
        domain=["神经科学", "心理学"],
        abstract_level=6,
        coverage=0.4,
        essence_keywords=["导致", "引起", "影响", "激活", "抑制", "促进",
                          "皮质醇", "多巴胺", "血清素", "压力", "焦虑",
                          "神经递质", "受体", "信号传导"],
        essence_feature_hints=["神经因果机制", "神经调质"],
        tags_hint=["因果", "神经机制"],
    ),

    MetadataTemplate(
        template_id="neuro_plasticity",
        description="神经可塑性与学习记忆",
        domain=["神经科学", "认知科学"],
        abstract_level=7,
        coverage=0.45,
        essence_keywords=["可塑性", "长时程", "LTP", "学习", "记忆",
                          "神经新生", "突触增强", "BDNF", "记忆巩固",
                          "海马体", "编码", "提取", "遗忘"],
        essence_feature_hints=["神经可塑性", "学习记忆机制"],
        tags_hint=["可塑性", "学习", "记忆"],
    ),

    MetadataTemplate(
        template_id="psychology_behavior",
        description="心理行为与情绪（行为模式/情绪反应）",
        domain=["心理学", "行为科学"],
        abstract_level=5,
        coverage=0.4,
        essence_keywords=["行为", "情绪", "恐惧", "奖励", "动机", "认知",
                          "压力反应", "应对", "本能", "条件反射",
                          "习得", "强化", "消退"],
        essence_feature_hints=["行为机制", "情绪调节"],
        tags_hint=["行为", "情绪"],
    ),

    # ── 生物学 / 进化 ────────────────────────────────────────────────────

    MetadataTemplate(
        template_id="bio_evolution",
        description="进化生物学（自然选择/适应/物种演化）",
        domain=["生物学", "进化论"],
        abstract_level=8,
        coverage=0.6,
        essence_keywords=["进化", "自然选择", "适应", "物种", "基因",
                          "遗传", "突变", "繁殖", "生存优势", "淘汰",
                          "本能", "先天"],
        essence_feature_hints=["进化适应", "自然选择压力"],
        tags_hint=["进化", "生物学"],
    ),

    MetadataTemplate(
        template_id="bio_instinct",
        description="本能行为与先天机制",
        domain=["生物学", "行为科学"],
        abstract_level=7,
        coverage=0.5,
        essence_keywords=["本能", "先天", "固定行为模式", "反射", "不需要学习",
                          "天生", "基因决定", "物种特异性"],
        essence_feature_hints=["本能行为", "先天固有"],
        tags_hint=["本能", "先天"],
    ),

    MetadataTemplate(
        template_id="bio_physiology",
        description="生理学（器官功能/生理指标/代谢）",
        domain=["生物学", "生理学"],
        abstract_level=5,
        coverage=0.4,
        essence_keywords=["激素", "分泌", "免疫", "代谢", "心率", "血压",
                          "体温", "睡眠", "昼夜节律", "炎症", "氧化应激"],
        essence_feature_hints=["生理调节机制", "生理指标"],
        tags_hint=["生理学", "代谢"],
    ),

    # ── 认知科学 / 人工智能 ──────────────────────────────────────────────

    MetadataTemplate(
        template_id="cog_reasoning",
        description="推理与决策（认知过程/推理模式）",
        domain=["认知科学", "哲学"],
        abstract_level=7,
        coverage=0.45,
        essence_keywords=["推理", "逻辑", "归纳", "演绎", "归谬", "悖论",
                          "假设", "验证", "矛盾", "一致性", "真值",
                          "认知偏差", "启发式"],
        essence_feature_hints=["推理模式", "认知过程"],
        tags_hint=["推理", "认知"],
    ),

    MetadataTemplate(
        template_id="ai_knowledge",
        description="知识表示与知识图谱",
        domain=["人工智能", "知识工程"],
        abstract_level=7,
        coverage=0.5,
        essence_keywords=["知识图谱", "本体", "关系", "实体", "向量",
                          "embedding", "检索", "推理引擎", "节点",
                          "图谱", "知识库", "记忆"],
        essence_feature_hints=["知识表示", "图谱结构"],
        tags_hint=["知识图谱", "AI"],
    ),

    # ── 物理 / 基础科学 ──────────────────────────────────────────────────

    MetadataTemplate(
        template_id="physics_general",
        description="物理规律（能量/力/运动等基础物理）",
        domain=["物理学"],
        abstract_level=9,
        coverage=0.7,
        essence_keywords=["能量", "力", "质量", "速度", "加速度",
                          "守恒", "热力学", "熵", "波", "量子"],
        essence_feature_hints=["物理规律", "基础定律"],
        tags_hint=["物理", "规律"],
    ),

    # ── 通用模式 ─────────────────────────────────────────────────────────

    MetadataTemplate(
        template_id="general_rule",
        description="通用规律/原则（适用于多个领域的抽象规律）",
        domain=["通用"],
        abstract_level=8,
        coverage=0.6,
        essence_keywords=["规律", "原则", "定律", "普遍", "一般来说",
                          "通常", "往往", "趋势", "模式"],
        essence_feature_hints=["普遍规律", "抽象原则"],
        tags_hint=["规律", "原则"],
    ),

    MetadataTemplate(
        template_id="general_exception",
        description="例外/特殊情形（对规律的例外）",
        domain=["通用"],
        abstract_level=3,
        coverage=0.2,
        essence_keywords=["但是", "除了", "除非", "例外", "特殊情况",
                          "然而", "尽管", "虽然", "不同于", "某些"],
        essence_feature_hints=["例外情形", "特殊条件"],
        tags_hint=["例外", "特殊"],
    ),

    MetadataTemplate(
        template_id="general_instance",
        description="具体实例（某个具体事物/事件的描述）",
        domain=["通用"],
        abstract_level=2,
        coverage=0.1,
        essence_keywords=["某个", "这个", "张三", "李四", "具体",
                          "案例", "实验", "观察到", "测量结果"],
        essence_feature_hints=["具体实例", "个别情形"],
        tags_hint=["实例", "具体"],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 特征词提取辅助
# ─────────────────────────────────────────────────────────────────────────────

# 关系性质词（出现这些词时，essence_features 应包含对应关系描述）
RELATION_SIGNAL_WORDS: Dict[str, str] = {
    "导致": "因果关系",    "引起": "因果关系",    "造成": "因果关系",
    "促进": "促进关系",    "增强": "增强效果",    "激活": "激活机制",
    "抑制": "抑制机制",    "阻碍": "阻碍效应",    "削弱": "削弱效果",
    "依赖": "依赖关系",    "需要": "前提依赖",    "条件": "条件关系",
    "先天": "先天固有",    "后天": "后天习得",    "学习": "学习获得",
    "进化": "进化适应",    "遗传": "遗传决定",
    "增加": "正向调节",    "减少": "负向调节",
    "有益": "正面效果",    "有害": "负面效果",
    "存在": "结构存在",    "缺失": "结构缺失",
}

# 抽象层级关键词（出现这些词时可推断层级范围）
ABSTRACT_LEVEL_HINTS: Dict[str, Tuple[int, int]] = {
    "所有生物": (9, 10),   "宇宙": (9, 10),      "物理定律": (9, 10),
    "普遍规律": (8, 9),    "进化压力": (8, 9),   "自然选择": (8, 9),
    "哺乳动物": (7, 8),    "灵长类": (7, 8),     "本能": (7, 8),
    "类群": (6, 7),        "机制": (5, 7),        "系统": (5, 7),
    "猫": (3, 5),          "狗": (3, 5),          "人类": (4, 6),
    "实验": (2, 4),        "观察": (2, 4),        "个体": (2, 3),
    "某次": (1, 2),        "这个": (1, 2),
}


# ─────────────────────────────────────────────────────────────────────────────
# 主类
# ─────────────────────────────────────────────────────────────────────────────

class MetadataTemplateLibrary:
    """
    外挂式元数据模板库。

    功能：
      1. fill(content) → 自动推断 domain / abstract_level / essence_features / coverage
      2. match_template(content) → 返回最匹配的模板及得分
      3. extract_essence_features(content) → 从内容中提取关键特征词
    """

    def __init__(self, memory_network=None):
        """
        memory_network : 可选，传入 MemoryNetwork 后，可用 embedding 相似度做精匹配。
                         若不传，退化为关键词命中模式（无需 Qdrant 服务）。
        """
        self.net = memory_network
        self.templates: List[MetadataTemplate] = TEMPLATES

    # ─────────────────────────────────────────────────
    # 主接口
    # ─────────────────────────────────────────────────

    def fill(self, content: str, verbose: bool = False) -> Dict[str, Any]:
        """
        对输入文本自动推断元字段，返回填充结果字典。

        返回字段：
          domain           : List[str]
          abstract_level   : int
          coverage         : float
          essence_features : List[str]
          tags_hint        : List[str]
          match_source     : str     （"template:xxx" 或 "default"）
          confidence       : float   （0~1，模板匹配置信度）
        """
        # 1. 匹配模板
        template, score = self.match_template(content)

        if template is not None and score >= 0.15:
            domain = list(template.domain)
            abstract_level = template.abstract_level
            coverage = template.coverage
            tags_hint = list(template.tags_hint)
            match_source = f"template:{template.template_id}"
            confidence = min(1.0, score / 0.5)   # 归一化：0.5命中率 → 1.0置信度
        else:
            # 无模板命中，使用保守默认值
            domain = ["通用"]
            abstract_level = 5
            coverage = 0.3
            tags_hint = []
            match_source = "default"
            confidence = 0.1

        # 2. 从内容中提取 essence_features
        features = self.extract_essence_features(content)
        if template is not None:
            features = list(set(features + template.essence_feature_hints))

        # 3. 用内容中的层级暗示词微调 abstract_level
        adjusted_level = self._adjust_abstract_level(content, abstract_level)

        # 4. 若 memory_network 可用，尝试从现有节点推断 domain
        inferred_domain = self._infer_domain_from_network(content, domain)

        result = {
            "domain": inferred_domain,
            "abstract_level": adjusted_level,
            "coverage": coverage,
            "essence_features": features,
            "tags_hint": tags_hint,
            "match_source": match_source,
            "confidence": round(confidence, 3),
        }

        if verbose:
            print(f"[元数据填充] 内容: '{content[:40]}'")
            print(f"  模板: {match_source}  置信度: {confidence:.2f}")
            print(f"  domain={inferred_domain}  "
                  f"level={adjusted_level}  "
                  f"features={features[:3]}")

        return result

    def match_template(
            self, content: str) -> Tuple[Optional[MetadataTemplate], float]:
        """
        基于关键词命中率寻找最匹配的模板。
        返回 (最佳模板 or None, 命中率得分)

        得分 = 命中关键词数 / max(模板关键词总数, 1)
        """
        best_template: Optional[MetadataTemplate] = None
        best_score = 0.0

        for tmpl in self.templates:
            hit = sum(1 for kw in tmpl.essence_keywords if kw in content)
            score = hit / max(len(tmpl.essence_keywords), 1)
            if score > best_score:
                best_score = score
                best_template = tmpl

        return best_template, best_score

    def extract_essence_features(self, content: str) -> List[str]:
        """
        从内容文本中提取关键 essence_features：
        1. 关系性质词（因果/促进/抑制等）
        2. 领域关键实体（从所有模板的关键词中过滤）
        3. 去重、去空
        """
        features: List[str] = []

        # 关系性质词
        for signal, feature in RELATION_SIGNAL_WORDS.items():
            if signal in content and feature not in features:
                features.append(feature)

        # 领域关键实体（从模板关键词中找到出现在内容中的词，取最长的3个）
        entity_hits: List[str] = []
        seen = set()
        for tmpl in self.templates:
            for kw in tmpl.essence_keywords:
                if kw in content and len(kw) >= 2 and kw not in seen:
                    entity_hits.append(kw)
                    seen.add(kw)
        # 按词长排序，取最具体的词（最长的）
        entity_hits.sort(key=len, reverse=True)
        features.extend(entity_hits[:4])

        return list(dict.fromkeys(features))[:8]  # 去重保序，最多8个

    def _adjust_abstract_level(self, content: str, base_level: int) -> int:
        """
        根据内容中的层级暗示词微调 abstract_level。
        只调整，不覆盖（最多±2）。
        """
        level_sum = 0
        count = 0
        for hint_word, (lo, hi) in ABSTRACT_LEVEL_HINTS.items():
            if hint_word in content:
                mid = (lo + hi) / 2
                level_sum += mid
                count += 1
        if count == 0:
            return base_level
        inferred = level_sum / count
        # 微调：在 base_level 基础上最多偏移2
        adjusted = base_level + max(-2, min(2, int(inferred - base_level)))
        return max(0, min(10, adjusted))

    def _infer_domain_from_network(
            self, content: str, fallback_domain: List[str]) -> List[str]:
        """
        若 MemoryNetwork 可用，用向量相似度在现有节点中找到最近的节点，
        参考其 domain 作为补充。
        不可用时，直接返回 fallback_domain。
        """
        if self.net is None:
            return fallback_domain

        try:
            similar = self.net.vector_search(content, top_k=3)
            domain_count: Dict[str, int] = {}
            for nid, score in similar:
                if score < 0.5:
                    continue
                node = self.net.get_node(nid)
                if node:
                    for d in node.domain:
                        domain_count[d] = domain_count.get(d, 0) + 1
            if not domain_count:
                return fallback_domain
            # 取最高频的领域，最多2个
            top_domains = sorted(
                domain_count, key=domain_count.__getitem__, reverse=True)[:2]
            # 与模板领域合并，优先保留网络推断的领域
            merged = list(dict.fromkeys(top_domains + fallback_domain))[:3]
            return merged
        except Exception:
            return fallback_domain

    # ─────────────────────────────────────────────────
    # 便利方法
    # ─────────────────────────────────────────────────

    def fill_new_information(self, content: str, source: str = "",
                             evidence_strength: float = 0.8,
                             node_id: Optional[str] = None,
                             verbose: bool = False) -> Dict[str, Any]:
        """
        快捷方法：填充后直接返回可用于构建 NewInformation 的完整字典。
        包含所有 NewInformation 必需字段。
        """
        import uuid
        filled = self.fill(content, verbose=verbose)
        return {
            "node_id": node_id or f"auto_{uuid.uuid4().hex[:8]}",
            "content": content,
            "abstract_level": filled["abstract_level"],
            "domain": filled["domain"],
            "coverage": filled["coverage"],
            "essence_features": filled["essence_features"],
            "tags": filled["tags_hint"] + ([f"来源:{source}"] if source else []),
            "evidence_strength": evidence_strength,
            "source": source or "auto_filled",
            "proposed_relations": [],
            # 元数据填充附加信息（非 NewInformation 标准字段，供调试）
            "_fill_meta": {
                "match_source": filled["match_source"],
                "confidence": filled["confidence"],
            },
        }

    def add_template(self, template: MetadataTemplate):
        """动态添加自定义模板（运行时扩展）"""
        self.templates.append(template)

    def list_templates(self) -> List[str]:
        """列出所有模板ID和描述"""
        return [f"{t.template_id}: {t.description}" for t in self.templates]
