"""
改进3：新信息冲突分级处理器
Level 0 - 补充（无冲突）：直接添加
Level 1 - 细化（局部/实例级冲突）：添加条件限定，保留双方
Level 2 - 修正（规律级冲突）：降低原知识置信度，存储新信息并标注
Level 3 - 悖论（根本矛盾）：标记悖论，不修改，等待人工裁决

极性冲突检测方案（语义轴投影）：
  不再维护词对列表，而是定义若干"语义轴"（每条轴由一对锚点词向量的差定义）。
  任意 essence_feature 在某轴上的投影值（点积）表示其在该维度的极性得分：
    > +AXIS_CONFLICT_THRESHOLD  → 正向
    < -AXIS_CONFLICT_THRESHOLD  → 负向
  两个特征在同一轴上极性相反且绝对值均超过阈值，判定为极性冲突。
  词对列表（CONTRADICTION_PAIRS）保留为 fallback，在 embedding 服务不可用时生效。

悖论池设计（2026-03-15 新增）：
  悖论进入之前先做"条件域检测"——
  若两个矛盾命题的 domain/essence_features 存在可分离的适用域，
  则自动判定为"条件不同、双方共存"，降级为 L1 处理，不进悖论池。
  只有适用域完全重叠的真正根本矛盾才进入悖论池。

  悖论池三路出口：
    resolve_paradox(report_id, decision)
      decision="new_wins"  : 新信息胜，旧公理降权，新节点写入
      decision="old_wins"  : 旧信息胜，新信息标记为 FALSE，记录驳回原因
      decision="coexist"   : 条件不同双方共存，降级为 L1 例外处理，移出悖论池
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Literal
from enum import Enum
import time
import uuid
import numpy as np

from memory_node import MemoryNode, EpistemicStatus
from relation_types import Relation, RelationType


class ConflictLevel(Enum):
    NO_CONFLICT = 0
    LOCAL       = 1
    RULE        = 2
    PARADOX     = 3


@dataclass
class NewInformation:
    node_id: str
    content: str
    abstract_level: int
    domain: List[str]
    coverage: float
    essence_features: List[str]
    tags: List[str]
    evidence_strength: float      # 0=未知来源, 1=高可信
    source: str
    proposed_relations: List[Relation] = field(default_factory=list)


@dataclass
class ConflictReport:
    level: ConflictLevel
    new_node_id: str
    conflicting_node_ids: List[str]
    conflict_description: str
    resolution_action: str
    evidence_strength: float
    human_review_required: bool = False
    timestamp: float = field(default_factory=time.time)
    # 悖论池专用：唯一ID，用于 resolve_paradox 定位
    paradox_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    # 悖论池专用：存储新信息的快照（供三路出口重新处理时使用）
    new_info_snapshot: Optional["NewInformation"] = None


# ─────────────────────────────────────────────────────────────────────────────
# 语义轴引擎
# ─────────────────────────────────────────────────────────────────────────────

class SemanticAxisEngine:
    """
    语义轴极性判断引擎。

    原理：
      定义 N 条"极性轴"，每条轴由一对锚点词（正极词, 负极词）定义。
      轴向量 = embed(正极词) - embed(负极词)，归一化后存储。
      任意文本片段在轴上的投影 = dot(embed(文本), 轴向量)：
        > +threshold  → 正极
        < -threshold  → 负极
        其余          → 中性/无关
      两个特征在同一轴上极性相反 → 极性冲突。

    轴定义覆盖了生物/心理/神经科学领域最常见的极性维度，
    新增轴只需在 AXIS_ANCHORS 里加一行，无需维护词对列表。
    """

    # 语义轴锚点定义：(轴名, 正极短语, 负极短语)
    # 用完整语境短语而非单词，让 embedding 模型有足够区分度
    AXIS_ANCHORS: List[Tuple[str, str, str]] = [
        ("增减",   "促进增加激活增强有利于提升",     "阻碍减少抑制削弱有害于降低"),
        ("好坏",   "对健康有益带来正面效果改善功能", "对健康有害带来负面效果损伤功能"),
        ("先后天", "先天遗传基因决定本能固有",       "后天习得通过学习经历获得"),
        ("强弱",   "功能增强信号放大效果加强",       "功能减弱信号衰减效果降低"),
        ("固变",   "结构稳定保持固定不随环境改变",   "结构可变动态适应随环境变化"),
        ("存在",   "结构存在具有该功能正常运作",     "结构缺失功能丧失无法运作"),
        ("兴抑",   "神经元兴奋激活唤醒活动增加",     "神经元抑制压制活动减少"),
        ("适应",   "进化适应提高生存优势被自然选择", "不适应导致淘汰降低生存率"),
        ("主动",   "主动发起自发产生内在驱动",       "被动响应外部刺激依赖触发"),
        ("快慢",   "过程加速时间缩短快速完成",       "过程减速时间延长缓慢进行"),
    ]

    # 两个特征被判定为极性冲突所需的最小投影绝对值
    # 设置较高阈值（0.15）避免短词/语义模糊词的误判
    AXIS_CONFLICT_THRESHOLD = 0.15

    def __init__(self, memory_network):
        self.net = memory_network
        # 轴向量缓存：{轴名: np.ndarray (归一化)}
        self._axis_vectors: Dict[str, np.ndarray] = {}
        # 特征向量缓存：{文本: np.ndarray (归一化)}
        self._feat_cache: Dict[str, np.ndarray] = {}
        self._ready = False   # 首次使用时懒初始化

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """获取文本的 embedding 向量（归一化），优先走缓存"""
        if text in self._feat_cache:
            return self._feat_cache[text]
        try:
            vec = np.array(self.net._get_embed_vector(text), dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                return None
            vec = vec / norm
            self._feat_cache[text] = vec
            return vec
        except Exception:
            return None

    def _build_axes(self):
        """懒加载：首次调用时构建所有轴向量"""
        if self._ready:
            return
        built = 0
        for axis_name, pos_text, neg_text in self.AXIS_ANCHORS:
            vp = self._embed(pos_text)
            vn = self._embed(neg_text)
            if vp is None or vn is None:
                continue
            axis_vec = vp - vn
            norm = np.linalg.norm(axis_vec)
            if norm < 1e-9:
                continue
            self._axis_vectors[axis_name] = axis_vec / norm
            built += 1
        self._ready = True
        if built > 0:
            print(f"[SemanticAxis] 已构建 {built}/{len(self.AXIS_ANCHORS)} 条语义轴")

    def polarity_scores(self, text: str) -> Dict[str, float]:
        """
        返回文本在各语义轴上的投影得分字典。
        得分 > 0 → 正极，< 0 → 负极，≈ 0 → 中性。
        """
        self._build_axes()
        vec = self._embed(text)
        if vec is None or not self._axis_vectors:
            return {}
        return {
            axis: float(np.dot(vec, axis_vec))
            for axis, axis_vec in self._axis_vectors.items()
        }

    def are_polar_opposite(self, feat_a: str, feat_b: str) -> Tuple[bool, str]:
        """
        判断两个特征文本是否在某条语义轴上极性相反。
        返回 (是否冲突, 冲突说明)
        """
        self._build_axes()
        if not self._axis_vectors:
            return False, ""

        scores_a = self.polarity_scores(feat_a)
        scores_b = self.polarity_scores(feat_b)

        for axis in self._axis_vectors:
            sa = scores_a.get(axis, 0.0)
            sb = scores_b.get(axis, 0.0)
            # 两者绝对值都超过阈值，且符号相反
            if (abs(sa) >= self.AXIS_CONFLICT_THRESHOLD and
                    abs(sb) >= self.AXIS_CONFLICT_THRESHOLD and
                    sa * sb < 0):
                direction_a = "正" if sa > 0 else "负"
                direction_b = "正" if sb > 0 else "负"
                return True, (
                    f"[{axis}轴] '{feat_a}'({direction_a},{sa:+.3f}) vs "
                    f"'{feat_b}'({direction_b},{sb:+.3f})"
                )
        return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# 主类
# ─────────────────────────────────────────────────────────────────────────────

class ConflictResolver:

    # Fallback 词对列表（仅在 embedding 服务不可用时使用）
    CONTRADICTION_PAIRS_FALLBACK = [
        ("先天", "后天"), ("本能", "学习"), ("固定", "可变"),
        ("有益", "有害"), ("增加", "减少"), ("存在", "不存在"),
        ("活", "死"), ("正", "负"), ("激活", "抑制"), ("增强", "削弱"),
    ]

    def __init__(self, memory_network):
        self.net = memory_network
        self.conflict_history: List[ConflictReport] = []
        self.paradox_pool: List[ConflictReport] = []
        # 语义轴引擎（懒初始化，首次极性判断时构建轴向量）
        self._axis_engine: Optional[SemanticAxisEngine] = None

    def _get_axis_engine(self) -> SemanticAxisEngine:
        """懒加载语义轴引擎"""
        if self._axis_engine is None:
            self._axis_engine = SemanticAxisEngine(self.net)
        return self._axis_engine

    # ─────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────

    def process(self, new_info: NewInformation,
                verbose: bool = True) -> ConflictReport:
        if verbose:
            print(f"\n[冲突处理] 新信息: {new_info.content[:50]}")
            print(f"  层级={new_info.abstract_level} 证据={new_info.evidence_strength:.2f}")

        level, conflict_ids, desc = self._detect(new_info)

        # ── 悖论前置：条件域检测（尝试自动降级为 L1）──────────────
        if level == ConflictLevel.PARADOX:
            can_coexist, coexist_reason = self._check_domain_coexistence(
                new_info, conflict_ids)
            if can_coexist:
                level = ConflictLevel.LOCAL
                desc = f"[自动降级 PARADOX→L1] {coexist_reason} | 原因:{desc}"
                if verbose:
                    print(f"  [条件域检测] 检测到适用域差异，自动降级为L1共存: {coexist_reason}")

        if level == ConflictLevel.NO_CONFLICT:
            action = self._act_add(new_info, verbose)
        elif level == ConflictLevel.LOCAL:
            action = self._act_refine(new_info, conflict_ids, verbose)
        elif level == ConflictLevel.RULE:
            action = self._act_revise(new_info, conflict_ids, verbose)
        else:
            action = self._act_paradox(new_info, conflict_ids, verbose)

        report = ConflictReport(
            level=level,
            new_node_id=new_info.node_id,
            conflicting_node_ids=conflict_ids,
            conflict_description=desc,
            resolution_action=action,
            evidence_strength=new_info.evidence_strength,
            human_review_required=(
                level == ConflictLevel.PARADOX or
                (level == ConflictLevel.RULE and new_info.evidence_strength < 0.7)
            ),
            new_info_snapshot=new_info if level == ConflictLevel.PARADOX else None,
        )
        self.conflict_history.append(report)
        if level == ConflictLevel.PARADOX:
            self.paradox_pool.append(report)

        if verbose:
            flag = "[!需人工]" if report.human_review_required else ""
            print(f"  结果: {level.name} | {action} {flag}")
        return report

    # ─────────────────────────────────────────
    # 冲突检测
    # ─────────────────────────────────────────

    def _detect(self, new_info: NewInformation
                ) -> Tuple[ConflictLevel, List[str], str]:
        conflict_ids = []
        descs = []

        # ── 第1步：检查 proposed_relations 中的 source 节点是否在图中已有相反极性边 ──
        # 这是最精确的结构性冲突检测，直接查边类型，不依赖向量相似度
        for pr in new_info.proposed_relations:
            if pr.relation_type not in (RelationType.PROMOTES, RelationType.INHIBITS):
                continue
            src_id = pr.source_id
            tgt_id = pr.target_id
            if not self.net.get_node(src_id) or not self.net.get_node(tgt_id):
                continue
            reverse_type = (RelationType.INHIBITS
                            if pr.relation_type == RelationType.PROMOTES
                            else RelationType.PROMOTES)
            for _, existing_tgt, data in self.net.graph.out_edges(src_id, data=True):
                rel = data.get("relation_obj")
                if rel and rel.relation_type == reverse_type and existing_tgt == tgt_id:
                    desc = (f"边类型冲突: '{src_id}'→{reverse_type.value}→'{tgt_id}' "
                            f"vs 新增'{src_id}'→{pr.relation_type.value}→'{tgt_id}'")
                    if src_id not in conflict_ids:
                        conflict_ids.append(src_id)
                        descs.append(desc)

        # ── 第2步：向量相似度检索 + 语义级冲突检测 ──
        similar = self.net.vector_search(new_info.content, top_k=5)
        for node_id, score in similar:
            if score < 0.6:
                continue
            if node_id in conflict_ids:
                continue
            existing = self.net.get_node(node_id)
            if not existing:
                continue

            # 检查对立关系（OPPOSITE_TO 边）
            if self._has_opposite_rel(node_id, new_info):
                conflict_ids.append(node_id)
                descs.append(f"'{existing.content[:25]}' 存在对立关系")
                continue

            # 检查极性边类型冲突（existing_id 视角，补充第1步未覆盖的情况）
            pec = self._has_polar_edge_conflict(node_id, new_info)
            if pec:
                conflict_ids.append(node_id)
                descs.append(pec)
                continue

            # 推理结论豁免
            if new_info.source == "associative_engine":
                continue

            # 检查本质特征冲突（词对 + 语义轴兜底）
            ec = self._essence_conflict(existing, new_info)
            if ec:
                conflict_ids.append(node_id)
                descs.append(ec)

        if not conflict_ids:
            return ConflictLevel.NO_CONFLICT, [], "无冲突"

        # 根据冲突节点最高层级定级
        max_node = max(
            (self.net.get_node(nid) for nid in conflict_ids if self.net.get_node(nid)),
            key=lambda n: n.abstract_level, default=None
        )
        if max_node is None:
            return ConflictLevel.NO_CONFLICT, [], "无冲突"

        ca = max_node.abstract_level
        if new_info.abstract_level <= 3 and ca <= 3:
            lvl = ConflictLevel.LOCAL
        elif ca >= 8 or new_info.abstract_level >= 8:
            lvl = ConflictLevel.RULE if new_info.evidence_strength >= 0.8 else ConflictLevel.PARADOX
        else:
            lvl = ConflictLevel.RULE

        return lvl, conflict_ids, " ; ".join(descs)

    def _has_opposite_rel(self, existing_id: str,
                          new_info: NewInformation) -> bool:
        for _, target, data in self.net.graph.out_edges(existing_id, data=True):
            rel = data.get("relation_obj")
            if rel and rel.relation_type == RelationType.OPPOSITE_TO:
                tn = self.net.get_node(target)
                if tn:
                    for feat in new_info.essence_features:
                        if feat in tn.content or feat in tn.tags:
                            return True
        return False

    def _has_polar_edge_conflict(self, existing_id: str,
                                  new_info: NewInformation) -> Optional[str]:
        """
        结构性极性冲突检测（基于关系边类型）：

        如果现有节点 existing 通过 PROMOTES 边指向某目标 T，
        而新信息 proposed_relations 中包含 existing → INHIBITS → T（或反向），
        则判定为结构性极性矛盾，无需词对匹配。

        同样检查反向：新信息通过 PROMOTES/INHIBITS 指向 existing，
        而现有图中对同一方向存在相反边。
        """
        # 收集现有节点的所有 PROMOTES/INHIBITS 出边
        existing_promotes: set = set()
        existing_inhibits: set = set()
        for _, target, data in self.net.graph.out_edges(existing_id, data=True):
            rel = data.get("relation_obj")
            if rel:
                if rel.relation_type == RelationType.PROMOTES:
                    existing_promotes.add(target)
                elif rel.relation_type == RelationType.INHIBITS:
                    existing_inhibits.add(target)

        # 检查新信息的 proposed_relations 是否与现有边冲突
        for pr in new_info.proposed_relations:
            if pr.source_id == existing_id:
                if pr.relation_type == RelationType.INHIBITS and pr.target_id in existing_promotes:
                    return (f"边类型冲突: '{existing_id}'→PROMOTES→'{pr.target_id}' "
                            f"vs 新增'{existing_id}'→INHIBITS→'{pr.target_id}'")
                if pr.relation_type == RelationType.PROMOTES and pr.target_id in existing_inhibits:
                    return (f"边类型冲突: '{existing_id}'→INHIBITS→'{pr.target_id}' "
                            f"vs 新增'{existing_id}'→PROMOTES→'{pr.target_id}'")

        # 也检查新节点本身被现有图中已有相反边指向的情况
        for _, target, data in self.net.graph.out_edges(existing_id, data=True):
            rel = data.get("relation_obj")
            if not rel:
                continue
            if (rel.relation_type == RelationType.PROMOTES
                    and target == new_info.node_id):
                # 现有 existing→PROMOTES→new_node，看新信息里是否 existing→INHIBITS→new_node
                for pr in new_info.proposed_relations:
                    if (pr.source_id == existing_id
                            and pr.target_id == new_info.node_id
                            and pr.relation_type == RelationType.INHIBITS):
                        return (f"边类型冲突: 现有PROMOTES vs 新增INHIBITS "
                                f"({existing_id}→{new_info.node_id})")
            if (rel.relation_type == RelationType.INHIBITS
                    and target == new_info.node_id):
                for pr in new_info.proposed_relations:
                    if (pr.source_id == existing_id
                            and pr.target_id == new_info.node_id
                            and pr.relation_type == RelationType.PROMOTES):
                        return (f"边类型冲突: 现有INHIBITS vs 新增PROMOTES "
                                f"({existing_id}→{new_info.node_id})")
        return None

    def _essence_conflict(self, existing: MemoryNode,
                          new_info: NewInformation) -> Optional[str]:
        """
        本质特征极性冲突检测（两层）：

        第1层（主）：essence_features 词对字符串匹配
          → 精确、无误判，是当前最可靠路径
          → 覆盖有限（需要词对在列表里），但不会产生噪音干扰

        第2层（语义轴补充）：对 essence_features 短词做轴投影
          → 目前 nomic-embed-text 对极短词区分度不足（相反词 cosine≈1.0）
          → 保留架构，随时可换精度更高的 embedding 模型激活此路径

        注：content 级轴投影在测试中发现会产生误判（语义相近的正确概念
        被噪音维度误判为冲突），暂不启用，等待更精确的极性模型。
        """
        axis_engine = self._get_axis_engine()

        for f1 in new_info.essence_features:
            for f2 in existing.essence_features:
                # 第1层：词对 fallback（主要路径，最可靠）
                for a, b in self.CONTRADICTION_PAIRS_FALLBACK:
                    if (a in f1 and b in f2) or (b in f1 and a in f2):
                        return f"极性冲突(词对): '{f1}' vs '{f2}'"

                # 第2层：语义轴投影（补充，当前 embedding 精度有限）
                try:
                    conflict, desc = axis_engine.are_polar_opposite(f1, f2)
                    if conflict:
                        return f"极性冲突(轴投影): {desc}"
                except Exception:
                    pass

        return None

    def _rule_contra(self, existing: MemoryNode,
                     new_info: NewInformation) -> bool:
        """
        规律级冲突检测：
        推理引擎自动生成的结论（source=associative_engine）直接豁免——
        推理结论天然派生自现有节点，不应反过来降权父节点。
        其余情况依赖上游 _essence_conflict 已覆盖，此处不再重复检测。
        """
        return new_info.source != "associative_engine"

    # ─────────────────────────────────────────
    # 四级处理动作
    # ─────────────────────────────────────────

    def _act_add(self, new_info: NewInformation, verbose: bool) -> str:
        """Level 0: 直接添加"""
        node = MemoryNode(
            node_id=new_info.node_id,
            content=new_info.content,
            abstract_level=new_info.abstract_level,
            domain=new_info.domain,
            coverage=new_info.coverage,
            essence_features=new_info.essence_features,
            tags=new_info.tags,
            weight=new_info.evidence_strength,
        )
        # 注册来源可信度（初始值0.8）
        if new_info.source:
            node.source_trust[new_info.source] = 0.8
        self.net.add_node(node)
        for rel in new_info.proposed_relations:
            if self.net.get_node(rel.source_id) and self.net.get_node(rel.target_id):
                self.net.add_relation(rel)
        if verbose:
            print(f"  [L0-补充] 直接添加节点 {new_info.node_id}")
        return f"ADDED:{new_info.node_id}"

    def _act_refine(self, new_info: NewInformation,
                    conflict_ids: List[str], verbose: bool) -> str:
        """
        Level 1: 局部冲突 - 细化处理
        保留原节点，添加新节点，用 EXCEPT 关系链接（例外关系）
        相当于：规律层保留，添加例外实例
        """
        node = MemoryNode(
            node_id=new_info.node_id,
            content=new_info.content,
            abstract_level=new_info.abstract_level,
            domain=new_info.domain,
            coverage=new_info.coverage,
            essence_features=new_info.essence_features,
            tags=new_info.tags + ["例外", "局部冲突"],
            weight=new_info.evidence_strength * 0.8,  # 冲突节点初始权重略低
        )
        # P1-1修复：注册来源可信度（与 L0/L2/L3 保持一致）
        if new_info.source:
            node.source_trust[new_info.source] = 0.8
        self.net.add_node(node)

        # 建立 EXCEPT 关系（用 OPPOSITE_TO 的弱化版，这里用 ANALOGOUS_TO 标注）
        for cid in conflict_ids:
            except_rel = Relation(
                source_id=cid,
                target_id=new_info.node_id,
                relation_type=RelationType.ANALOGOUS_TO,
                weight=0.5,
                context="EXCEPT_CASE",   # 标记为例外关系
            )
            self.net.add_relation(except_rel)

        for rel in new_info.proposed_relations:
            if self.net.get_node(rel.source_id) and self.net.get_node(rel.target_id):
                self.net.add_relation(rel)

        if verbose:
            print(f"  [L1-细化] 添加例外节点 {new_info.node_id}，关联 {len(conflict_ids)} 个冲突节点")
        return f"REFINED:{new_info.node_id} with EXCEPT links to {conflict_ids}"

    def _act_revise(self, new_info: NewInformation,
                    conflict_ids: List[str], verbose: bool) -> str:
        """
        Level 2: 规律级冲突 - 修正处理
        降低原规律节点权重 + 存储新信息 + 标注待验证
        """
        # 降低冲突节点的权重（不删除，渐进式修正）
        for cid in conflict_ids:
            node = self.net.get_node(cid)
            if node:
                node.update_weight(-0.3, f"被新信息 {new_info.node_id} 挑战")
                node.tags.append("待验证")

        # 添加新节点，标注来源证据
        node = MemoryNode(
            node_id=new_info.node_id,
            content=new_info.content,
            abstract_level=new_info.abstract_level,
            domain=new_info.domain,
            coverage=new_info.coverage,
            essence_features=new_info.essence_features,
            tags=new_info.tags + ["修正候选", f"来源:{new_info.source}"],
            weight=new_info.evidence_strength * 0.9,
        )
        # 注册来源可信度（初始值0.8）
        if new_info.source:
            node.source_trust[new_info.source] = 0.8
        self.net.add_node(node)

        for rel in new_info.proposed_relations:
            if self.net.get_node(rel.source_id) and self.net.get_node(rel.target_id):
                self.net.add_relation(rel)

        if verbose:
            print(f"  [L2-修正] 已降低 {len(conflict_ids)} 个冲突节点权重，添加修正候选 {new_info.node_id}")
        return f"REVISED:{conflict_ids} weight down; ADDED:{new_info.node_id} as candidate"

    def _act_paradox(self, new_info: NewInformation,
                     conflict_ids: List[str], verbose: bool) -> str:
        """
        Level 3: 悖论 - 不修改知识库，记录矛盾等待人工裁决
        新信息节点以 epistemic_status=PARADOX_PENDING 写入（不参与正向推理）
        """
        # 写入一个 PARADOX_PENDING 状态的节点（权重极低，不参与正向推理）
        node = MemoryNode(
            node_id=new_info.node_id,
            content=new_info.content,
            abstract_level=new_info.abstract_level,
            domain=new_info.domain,
            coverage=new_info.coverage,
            essence_features=new_info.essence_features,
            tags=new_info.tags + ["悖论暂存", "待人工裁决"],
            weight=0.1,
            epistemic_status=EpistemicStatus.PARADOX_PENDING,
        )
        # 注册来源可信度（初始值0.8，裁决后可调整）
        if new_info.source:
            node.source_trust[new_info.source] = 0.8
        self.net.add_node(node)
        if verbose:
            print(f"  [L3-悖论] 检测到根本矛盾，新节点以PARADOX_PENDING状态写入，不参与推理")
            for cid in conflict_ids:
                n = self.net.get_node(cid)
                if n:
                    print(f"    冲突节点: {n.content[:40]}")
        return f"PARADOX:new={new_info.node_id} conflicts={conflict_ids} PENDING_REVIEW"

    # ─────────────────────────────────────────
    # 条件域检测（悖论自动降级）
    # ─────────────────────────────────────────

    def _check_domain_coexistence(
            self, new_info: NewInformation,
            conflict_ids: List[str]) -> Tuple[bool, str]:
        """
        检测两个矛盾命题是否属于"条件不同、适用域不同"的情况。
        若是，悖论可自动降级为 L1（共存），无需人工裁决。

        判断逻辑（按优先级）：
        0. 图结构包含关系前置检查：若两节点之间存在 BELONGS_TO / INSTANCE_OF 边
           → 一方是另一方的下位实例，天然不矛盾，直接降级（根本修复，不依赖词表/阈值）
        1. domain 字段无交集 → 适用域完全不同，可共存
        2. abstract_level 差异 ≥ 2 → 一个是规律，一个是实例/例外，可共存
           （注：原阈值3过宽，调整为2以覆盖"相差两层"的父子概念对）

        原规则3（个体限定词词表硬编码）已废弃：
        - 词表治标不治本，无法规模化扩展
        - 正确解法是在图谱中建立完整 BELONGS_TO/INSTANCE_OF 关系，由规则0覆盖
        - 若图谱关系不完善，规则2的阈值2也能覆盖大部分父子概念对
        """
        from relation_types import RelationType as RT

        # 包含关系判断所需的关系类型
        inclusion_types = {RT.BELONGS_TO, RT.INSTANCE_OF}

        for cid in conflict_ids:
            existing = self.net.get_node(cid)
            if not existing:
                continue

            # 规则0（前置）：图结构包含关系检测
            # 检查 new_info.node_id ↔ cid 之间是否有 BELONGS_TO / INSTANCE_OF 边（双向）
            new_node_id = new_info.node_id
            has_inclusion = False
            inclusion_desc = ""
            for src, tgt, data in self.net.graph.edges([new_node_id, cid], data=True):
                rel = data.get("relation_obj")
                if rel and rel.relation_type in inclusion_types:
                    # src BELONGS_TO/INSTANCE_OF tgt → src 是 tgt 的下位实例
                    has_inclusion = True
                    inclusion_desc = (
                        f"节点'{src}'通过[{rel.relation_type.value}]关系归属于节点'{tgt}'，"
                        f"属于规律+实例天然共存，无矛盾"
                    )
                    break
            if has_inclusion:
                return True, inclusion_desc

            # 规则1：domain 无交集
            new_domains = set(new_info.domain)
            ext_domains = set(existing.domain)
            if new_domains and ext_domains and new_domains.isdisjoint(ext_domains):
                return True, (
                    f"domain完全不同({new_domains} vs {ext_domains})，"
                    f"适用域互不重叠")

            # 规则2：抽象层级差 ≥ 2（阈值从3降为2，覆盖"相差两层"的父子概念对）
            level_diff = abs(new_info.abstract_level - existing.abstract_level)
            if level_diff >= 2:
                higher = ("新信息" if new_info.abstract_level > existing.abstract_level
                          else f"节点'{cid}'")
                lower = ("新信息" if new_info.abstract_level < existing.abstract_level
                         else f"节点'{cid}'")
                return True, (
                    f"抽象层级差={level_diff}({higher}为规律层，{lower}为实例层)，"
                    f"属于规律+例外共存")

        return False, ""

    # ─────────────────────────────────────────
    # 悖论池三路出口
    # ─────────────────────────────────────────

    def resolve_paradox(
            self,
            paradox_id: str,
            decision: Literal["new_wins", "old_wins", "coexist"],
            reason: str = "",
            verbose: bool = True) -> bool:
        """
        人工裁决悖论池中的一条记录。

        decision 参数：
          "new_wins"  : 新信息正确，旧公理被推翻
                        → 旧冲突节点权重大幅降低，标记为 hypothesis
                        → 新节点升级为 confirmed，权重恢复正常
          "old_wins"  : 旧信息正确，新信息是错误的
                        → 新节点标记为 FALSE，记录驳回原因
                        → 旧冲突节点维持不变（权重略微提升）
          "coexist"   : 条件不同，双方共存
                        → 降级为 L1 例外处理
                        → 两个节点均保留，用 EXCEPT 关系链接
                        → 新节点 epistemic_status 改为 confirmed

        返回：True=处理成功，False=未找到对应悖论ID
        """
        # 查找悖论报告
        report = next(
            (r for r in self.paradox_pool if r.paradox_id == paradox_id), None)
        if report is None:
            if verbose:
                print(f"[resolve_paradox] 未找到 paradox_id={paradox_id}")
            return False

        new_node = self.net.get_node(report.new_node_id)
        conflict_nodes = [self.net.get_node(cid)
                          for cid in report.conflicting_node_ids
                          if self.net.get_node(cid)]

        if decision == "new_wins":
            # 旧公理降权 + 标记为 hypothesis（不删除，降级为待验证）
            for old_node in conflict_nodes:
                old_node.update_weight(-0.5, f"被悖论裁决推翻: {reason}")
                old_node.epistemic_status = EpistemicStatus.HYPOTHESIS
                if "公理" in old_node.tags:
                    old_node.tags.remove("公理")
                old_node.tags.append("被修正")
            # 新节点升级
            if new_node:
                new_node.epistemic_status = EpistemicStatus.CONFIRMED
                new_node.weight = max(new_node.weight, 0.8)
                new_node.tags = [t for t in new_node.tags
                                 if t not in ("悖论暂存", "待人工裁决")]
                new_node.tags.append("悖论裁决:新信息胜")
            if verbose:
                print(f"[resolve_paradox:{paradox_id}] new_wins → "
                      f"旧节点降权/降为HYPOTHESIS，新节点升级为CONFIRMED")

        elif decision == "old_wins":
            # 新节点标记为 FALSE
            if new_node:
                new_node.mark_false(
                    reason=reason or f"悖论裁决:旧信息胜，新节点{report.new_node_id}被驳回")
                new_node.tags = [t for t in new_node.tags
                                 if t not in ("悖论暂存", "待人工裁决")]
                new_node.tags.append("悖论裁决:被驳回")
                # ── source_trust 断路修复：降低提供了错误信息的来源可信度 ──
                # 新节点的 source_trust 记录了为该节点提供过信息的来源
                # 该节点被裁定为 FALSE，说明这些来源提供了错误信息，应降分
                for source_id in new_node.source_trust:
                    new_node.update_source_trust(source_id, -0.15)
                    if verbose:
                        new_score = new_node.source_trust.get(source_id, 0.0)
                        print(f"  [source_trust] 来源 '{source_id}' "
                              f"信任分降低 -0.15 → {new_score:.2f}（提供了被驳回的错误信息）")
            # 旧节点轻微加权（被挑战后经验证依然正确）
            for old_node in conflict_nodes:
                old_node.update_weight(+0.1, f"悖论裁决验证正确: {reason}")
            if verbose:
                print(f"[resolve_paradox:{paradox_id}] old_wins → "
                      f"新节点标记为FALSE，旧节点轻微加权")

        elif decision == "coexist":
            # 降级为 L1：两节点共存，添加 EXCEPT 关系
            if new_node:
                new_node.epistemic_status = EpistemicStatus.CONFIRMED
                new_node.weight = max(new_node.weight,
                                      report.evidence_strength * 0.8)
                new_node.tags = [t for t in new_node.tags
                                 if t not in ("悖论暂存", "待人工裁决")]
                new_node.tags.extend(["例外", "条件共存",
                                      f"悖论裁决:共存({reason[:20]})"])
            for old_node in conflict_nodes:
                except_rel = Relation(
                    source_id=old_node.node_id,
                    target_id=report.new_node_id,
                    relation_type=RelationType.ANALOGOUS_TO,
                    weight=0.5,
                    context=f"EXCEPT_CASE:{reason[:40]}",
                )
                if self.net.get_node(old_node.node_id) and new_node:
                    self.net.add_relation(except_rel)
            if verbose:
                print(f"[resolve_paradox:{paradox_id}] coexist → "
                      f"新节点升级为CONFIRMED，添加EXCEPT关系")

        # 从悖论池移除
        self.paradox_pool = [r for r in self.paradox_pool
                             if r.paradox_id != paradox_id]
        # 在 conflict_history 中更新记录
        for r in self.conflict_history:
            if r.paradox_id == paradox_id:
                r.resolution_action += f" | RESOLVED:{decision}({reason[:30]})"
                r.human_review_required = False
                break

        return True

    def get_paradox_pool(self) -> List[ConflictReport]:
        """
        返回当前未处理的悖论列表，供人工或上层决策模块查阅。
        每条记录包含：paradox_id、new_node_id、conflicting_node_ids、
        conflict_description、evidence_strength、timestamp
        """
        return list(self.paradox_pool)

    def paradox_stats(self) -> Dict:
        """
        悖论池统计信息。
        返回：池大小、平均等待时间、最老记录时间戳
        """
        if not self.paradox_pool:
            return {
                "pool_size": 0,
                "avg_wait_seconds": 0,
                "oldest_timestamp": None,
            }
        now = time.time()
        waits = [now - r.timestamp for r in self.paradox_pool]
        return {
            "pool_size": len(self.paradox_pool),
            "avg_wait_seconds": round(sum(waits) / len(waits), 1),
            "oldest_timestamp": min(r.timestamp for r in self.paradox_pool),
            "oldest_wait_seconds": round(max(waits), 1),
        }

    # ─────────────────────────────────────────
    # 报告与统计
    # ─────────────────────────────────────────

    def summary(self) -> Dict:
        counts = {lvl.name: 0 for lvl in ConflictLevel}
        for r in self.conflict_history:
            counts[r.level.name] += 1
        return {
            "total_processed": len(self.conflict_history),
            "by_level": counts,
            "paradox_pending": len(self.paradox_pool),
        }

    def get_pending_reviews(self) -> List[ConflictReport]:
        return [r for r in self.conflict_history if r.human_review_required]
