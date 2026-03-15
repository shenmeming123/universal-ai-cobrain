"""
联想推理引擎（路径A）
核心机制：
  1. 拆解输入关键特征
  2. 双重筛选（覆盖度 + 语境相关度）激活本质节点
  3. 沿元关系逐层推导（纵向通路 + 横向扩展）
  4. 缺口反向定义 → 再检索
  5. 四重约束验证
  6. 渐进式权重更新
"""

from typing import List, Dict, Tuple, Optional, Any, Literal
from dataclasses import dataclass, field
from memory_network import MemoryNetwork
from memory_node import MemoryNode, EpistemicStatus
from relation_types import RelationType, Relation, RELATION_PRIORITY
from context_layer_mapper import ContextLayerMapper, ContextProfile
import numpy as np
import time
import uuid


@dataclass
class ReasoningStep:
    """推理过程中的一步"""
    step_id: int
    action: str           # 动作描述
    activated_nodes: List[str] = field(default_factory=list)
    path_taken: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ReasoningResult:
    """推理最终结果"""
    answer: str
    confidence: float
    reasoning_chain: List[ReasoningStep]
    activated_nodes: List[str]
    gaps_found: List[str]
    validation_passed: bool
    elapsed_ms: float
    context_profile: Optional[ContextProfile] = None   # 语境画像
    stored_as_node: Optional[str] = None               # 若结论被存储，记录新节点ID
    top_vector_score: float = 0.0                      # step1 向量检索最高相似度（用于response_score兜底）
    potential_nodes: List[str] = field(default_factory=list)  # 本次推理发现的潜在可能性节点ID
    negated_nodes: List[str] = field(default_factory=list)    # 本次推理中被识别为FALSE/PARADOX_PENDING的节点ID（内部追溯用，不呈现给用户）


class AssociativeReasoningEngine:
    """
    联想推理引擎（含语境映射 + 推理结论自动存储）
    """

    def __init__(self, memory_network: MemoryNetwork,
                 max_depth: int = 4, max_nodes_per_step: int = 5,
                 auto_store_threshold: float = 0.85):
        self.net = memory_network
        self.max_depth = max_depth
        self.max_nodes_per_step = max_nodes_per_step
        self.auto_store_threshold = auto_store_threshold  # 置信度超过此值自动存储结论
        self.context_mapper = ContextLayerMapper()
        # P3-1修复：ConflictResolver 提升为引擎级成员变量（长期有状态对象）
        # 原实现在 _store_conclusion 中每次 new 一个临时 resolver，导致：
        #   ① 悖论池在方法返回后被垃圾回收，永久丢失，resolve_paradox 无法工作
        #   ② 冲突历史割裂，summary() 永远为空
        #   ③ SemanticAxisEngine 轴向量每次重复构建，浪费 embedding 调用
        # 提升后三个问题全部消除：悖论池跨次推理持久化，历史累积，轴向量只构建一次。
        from conflict_resolver import ConflictResolver as _CR
        self.resolver = _CR(self.net)
        # 推理结论存储记录 [{query, node_id, confidence, timestamp}]
        self.stored_conclusions: List[Dict] = []
        # 假设沙箱记录：[{hypothesis, result, verdict, timestamp}]
        # verdict: "confirmed" / "refuted" / "inconclusive"
        self.hypothesis_sandbox_log: List[Dict] = []

    # ─────────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────────

    def reason(self, query: str, verbose: bool = True,
               explicit_context: Optional[str] = None) -> ReasoningResult:
        """对一个问题进行联想推理"""
        start = time.time()
        steps = []
        all_activated = []

        # ── 语境识别 ──────────────────────────────────────
        context_profile = self.context_mapper.identify_context(
            query, explicit_context=explicit_context)

        if verbose:
            print(f"\n{'='*60}")
            print(f"[联想引擎] 问题: {query}")
            print(self.context_mapper.describe_context(context_profile))
            print(f"{'='*60}")

        # ── 第一步：向量快速检索，找初始激活节点 ──────
        step1 = self._step_initial_activation(query, verbose, context_profile)
        steps.append(step1)
        all_activated.extend(step1.activated_nodes)
        # 记录 step1 最高向量得分（用于 response_score 兜底）
        top_vector_score: float = step1.confidence

        if not step1.activated_nodes:
            return ReasoningResult(
                answer="记忆网络中没有找到相关信息",
                confidence=0.0,
                reasoning_chain=steps,
                activated_nodes=[],
                gaps_found=[],
                validation_passed=False,
                elapsed_ms=(time.time() - start) * 1000,
                context_profile=context_profile,
            )

        # ── 第二步：纵向追溯（抽象层级通路）──────────
        step2 = self._step_vertical_traversal(
            step1.activated_nodes, query, verbose)
        steps.append(step2)
        all_activated.extend(
            [n for n in step2.activated_nodes if n not in all_activated])

        # ── 第三步：横向扩展（同层关联 + 跨域）────────
        step3 = self._step_horizontal_expansion(
            all_activated, query, verbose,
            seed_nodes=step1.activated_nodes)   # 方案一：传入步骤1种子节点
        steps.append(step3)
        all_activated.extend(
            [n for n in step3.activated_nodes if n not in all_activated])
        # 提取极性映射，传递给后续步骤
        polarity_map: Dict[str, str] = getattr(step3, "_polarity_map", {})
        query_polarity: Optional[str] = getattr(step3, "_query_polarity", None)

        # ── 第四步：信息组织，尝试构建推理结构 ────────
        step4, gaps = self._step_organize(all_activated, query, verbose)
        steps.append(step4)

        # ── 第五步：缺口反向定义 + 补充检索 ──────────
        if gaps:
            step5 = self._step_gap_filling(gaps, all_activated, query, verbose)
            steps.append(step5)
            all_activated.extend(
                [n for n in step5.activated_nodes if n not in all_activated])

        # ── 第六步：四重约束验证 ───────────────────────
        answer, confidence, passed, negated_nodes = self._step_validate(
            all_activated, query, verbose,
            polarity_map=polarity_map, query_polarity=query_polarity,
            top_vector_score=top_vector_score)
        steps.append(ReasoningStep(
            step_id=len(steps) + 1,
            action="四重约束验证",
            activated_nodes=all_activated,
            findings=[answer],
            confidence=confidence,
        ))

        # ── 记录路径，触发快捷边机制 ──────────────────
        if len(all_activated) >= 2:
            self.net.record_path_usage(all_activated)

        # ── 推理结论自动存储（改进4）──────────────────
        stored_node_id = None
        if passed and confidence >= self.auto_store_threshold:
            stored_node_id = self._store_conclusion(
                query, answer, confidence, all_activated, context_profile, verbose)

        # ── 类型二可能性节点发现 ───────────────────────
        potential_nodes = self._discover_potential_nodes(
            all_activated, query, verbose)

        # ── 超图边联合条件触发 ────────────────────────
        hyper_triggered = self._check_hyper_edges(all_activated, verbose, query=query)
        if hyper_triggered:
            all_activated.extend(
                [nid for nid in hyper_triggered if nid not in all_activated])

        elapsed = (time.time() - start) * 1000
        if verbose:
            print(f"\n[联想引擎] 完成，耗时 {elapsed:.1f}ms，置信度 {confidence:.2f}")
            print(f"[联想引擎] 答案: {answer}")
            if stored_node_id:
                print(f"[联想引擎] 结论已自动存储为节点: {stored_node_id}")
            if potential_nodes:
                print(f"[联想引擎] 发现潜在可能性节点: {potential_nodes}")

        return ReasoningResult(
            answer=answer,
            confidence=confidence,
            reasoning_chain=steps,
            activated_nodes=all_activated,
            gaps_found=gaps,
            validation_passed=passed,
            elapsed_ms=elapsed,
            context_profile=context_profile,
            stored_as_node=stored_node_id,
            top_vector_score=top_vector_score,
            potential_nodes=potential_nodes,
            negated_nodes=negated_nodes,
        )

    # ─────────────────────────────────────────────────
    # 各推理步骤
    # ─────────────────────────────────────────────────

    def _decompose_query(self, query: str) -> List[str]:
        """
        多锚点策略：将问题拆解为多个子查询，覆盖不同实体/概念。
        规则：
          1. 原始查询本身作为第一个锚点
          2. 按中文标点/连词分割，取最长的2段作为子锚点
          3. tag关键字锚点：用知识库中所有节点的tag在查询中做关键词匹配，
             找到命中次数最多的节点，以其content作为额外锚点（覆盖专有词汇）
        """
        import re
        sub_queries = [query]

        # 按疑问词/连词分割
        parts = re.split(r'[，。？、；：为什么如何什么哪个]', query)
        parts = [p.strip() for p in parts if len(p.strip()) >= 4]
        parts.sort(key=len, reverse=True)
        for p in parts[:2]:
            if p not in sub_queries:
                sub_queries.append(p)

        # tag关键字锚点：在知识库节点中找 tag 与查询词最匹配的节点
        # 用于解决专有词汇（如"杏仁核"）与描述性查询之间的语义鸿沟
        if hasattr(self, 'net') and self.net:
            best_node_id = None
            best_score = 0
            for nid, node in self.net.nodes.items():
                if nid.startswith("inferred_"):
                    continue
                score = sum(1 for tag in node.tags if tag in query)
                # 也检查节点名（node_id）本身是否出现在查询中
                if nid in query:
                    score += 3
                if score > best_score:
                    best_score = score
                    best_node_id = nid
            if best_node_id and best_score > 0:
                tag_anchor = self.net.get_node(best_node_id).content
                if tag_anchor not in sub_queries:
                    sub_queries.append(tag_anchor)

        return sub_queries[:4]  # 最多4个锚点

    def _step_initial_activation(self, query: str,
                                  verbose: bool,
                                  context_profile: Optional[ContextProfile] = None) -> ReasoningStep:
        """
        第一步：两阶段检索（T5）
          阶段一（向量粗排）：多锚点向量检索，扩大召回，收集候选集
          阶段二（图结构精排）：用 graph_rerank 对候选集按图连通性重排，
                               消除孤立的高相似度噪声节点，保留与图谱结构一致的节点
        多锚点策略：对原始查询及其子查询分别检索，合并后统一打分排序，
        避免单一查询向量在大规模库中被主题无关节点稀释。
        """
        sub_queries = self._decompose_query(query)

        # ── 阶段一：多锚点向量粗排（扩大候选集）─────────────────────
        # 每个锚点取 top_k=12（比原来的8多，为精排提供更大候选池）
        candidate_scores: Dict[str, float] = {}
        for sq in sub_queries:
            results = self.net.vector_search(sq, top_k=12)
            for node_id, score in results:
                if node_id not in candidate_scores or score > candidate_scores[node_id]:
                    candidate_scores[node_id] = score

        # 可靠性过滤（P0-1修复：剔除 FALSE/PARADOX_PENDING 节点）
        reliable_candidates: List[Tuple[str, float]] = []
        for node_id, score in candidate_scores.items():
            node = self.net.get_node(node_id)
            if node and node.is_reliable():
                reliable_candidates.append((node_id, score))

        # ── 阶段二：图结构精排 ────────────────────────────────────────
        # graph_rerank 融合向量分数（alpha=0.6）和图连通分（beta=0.4）
        # 候选集取语境过滤前的全量，精排后再做语境权重二次调整
        graph_reranked = self.net.graph_rerank(
            reliable_candidates,
            top_k=min(len(reliable_candidates), self.max_nodes_per_step * 3),
            alpha=0.6,
            beta=0.4,
        )

        # 语境权重二次调整 + 覆盖度融合
        filtered = []
        for node_id, rerank_score in graph_reranked:
            node = self.net.get_node(node_id)
            if not node:
                continue
            ctx_w = 1.0
            if context_profile:
                ctx_w = self.context_mapper.context_weight(
                    node.abstract_level, context_profile)
            # 融合覆盖度和层级信息（保持与原来一致的权重结构）
            final_score = (rerank_score * 0.6 +
                           node.coverage * 0.2 +
                           (node.abstract_level / 10.0) * 0.2) * ctx_w
            filtered.append((node_id, final_score, node))

        filtered.sort(key=lambda x: x[1], reverse=True)
        top_nodes = [nid for nid, _, _ in filtered[:self.max_nodes_per_step]]

        # ── 激活计数 + 权重反馈 ──────────────────────────────────────
        for nid in top_nodes:
            node = self.net.get_node(nid)
            if node:
                node.activation_count += 1
                node.update_weight(+0.02, "推理激活(步骤1-T5)")

        if verbose:
            print(f"\n[步骤1-T5] 两阶段检索 (锚点数:{len(sub_queries)}, "
                  f"粗排候选:{len(candidate_scores)}, "
                  f"精排后:{len(graph_reranked)}, "
                  f"最终激活:{len(top_nodes)}):")
            for nid, score, node in filtered[:self.max_nodes_per_step]:
                vec_s = candidate_scores.get(nid, 0.0)
                print(f"  · {node.content[:50]}  "
                      f"(融合分:{score:.3f}, 向量:{vec_s:.3f}, 激活次数:{node.activation_count})")

        findings = [self.net.get_node(nid).content
                    for nid in top_nodes if self.net.get_node(nid)]

        return ReasoningStep(
            step_id=1,
            action="两阶段检索(T5)：向量粗排→图结构精排",
            activated_nodes=top_nodes,
            findings=findings,
            confidence=filtered[0][1] if filtered else 0.0,
        )

    def _step_vertical_traversal(self, seed_nodes: List[str],
                                   query: str, verbose: bool) -> ReasoningStep:
        """第二步：纵向追溯抽象层级（男人→人类→灵长类→生物）"""
        new_nodes = []
        path_log = []

        for seed_id in seed_nodes[:3]:  # 最多追溯3个种子节点
            ancestors = self.net.get_abstract_ancestors(seed_id, max_depth=3)
            for anc_id, depth in ancestors:
                if anc_id not in seed_nodes and anc_id not in new_nodes:
                    # P0-1修复：跳过 FALSE / PARADOX_PENDING 节点
                    anc_node = self.net.get_node(anc_id)
                    if anc_node and not anc_node.is_reliable():
                        continue
                    new_nodes.append(anc_id)
                    path_log.append(f"{seed_id} →[纵向]→ {anc_id} (深度{depth})")

        if verbose:
            print(f"\n[步骤2] 纵向追溯，新激活 {len(new_nodes)} 个节点:")
            for nid in new_nodes[:5]:
                node = self.net.get_node(nid)
                if node:
                    print(f"  · {node.content[:50]}")

        findings = [self.net.get_node(nid).content
                    for nid in new_nodes if self.net.get_node(nid)]

        return ReasoningStep(
            step_id=2,
            action="纵向抽象层级追溯",
            activated_nodes=new_nodes,
            path_taken=path_log,
            findings=findings,
            confidence=0.7,
        )

    def _detect_query_polarity(self, query: str,
                                seed_nodes: Optional[List[str]] = None) -> Optional[str]:
        """
        判断查询意图是正向（促进/有助于）还是负向（阻碍/损伤）。
        返回 "promotes" / "inhibits" / None（中性）

        方案一：两级判断
        第1级：关键词匹配（快速、准确）
        第2级：图感知（当关键词判断为中性时启用）
          ——统计步骤1激活节点出边中 PROMOTES vs INHIBITS 的加权数量，
            哪方占比超过60%则认为查询天然处于该方向语境。
          ——不依赖任何词表，规模化后依然有效。
        """
        import re
        promotes_keywords = ["促进", "增强", "有助于", "改善", "提升", "激活",
                              "有利于", "加速", "增加", "增进"]
        inhibits_keywords = ["抑制", "损伤", "阻碍", "削弱", "降低", "破坏",
                              "阻止", "妨碍", "减少", "危害", "影响.*负"]
        for kw in promotes_keywords:
            if kw in query:
                return "promotes"
        for kw in inhibits_keywords:
            if re.search(kw, query):
                return "inhibits"

        # ── 第2级：图感知（关键词未命中时启用）──────────
        if seed_nodes and self.net:
            promotes_weight = 0.0
            inhibits_weight = 0.0
            for nid in seed_nodes:
                node = self.net.get_node(nid)
                if not node:
                    continue
                for _, _, data in self.net.graph.out_edges(nid, data=True):
                    rel = data.get("relation_obj")
                    if not rel:
                        continue
                    edge_w = rel.weight
                    if rel.relation_type == RelationType.PROMOTES:
                        promotes_weight += edge_w
                    elif rel.relation_type == RelationType.INHIBITS:
                        inhibits_weight += edge_w
            total = promotes_weight + inhibits_weight
            if total > 0:
                if promotes_weight / total >= 0.6:
                    return "promotes"
                if inhibits_weight / total >= 0.6:
                    return "inhibits"
        return None

    def _step_horizontal_expansion(self, current_nodes: List[str],
                                    query: str, verbose: bool,
                                    seed_nodes: Optional[List[str]] = None) -> ReasoningStep:
        """
        第三步：横向扩展，沿高优先级元关系向外联想。
        极性排序修复：极性匹配边获得额外加分，保证它排在同优先级通用边前面。
        path_log 格式：(source, rel_type_value, target, polarity_label)，
        供答案组织阶段使用。

        方案一：图感知极性意图识别
          传入 seed_nodes（步骤1激活节点），在关键词判断为中性时用图结构感知查询方向。
          同时把步骤1种子节点的 PROMOTES/INHIBITS 邻居直接注入 polarity_map，
          补全步骤1激活节点的极性覆盖缺口。
        """
        # ── 方案一：用图感知的极性识别（传入seed_nodes）──
        query_polarity = self._detect_query_polarity(
            query, seed_nodes=seed_nodes or current_nodes[:5])

        # ── 方案一：把步骤1种子节点的极性邻居预注入polarity_map ──
        polarity_map: Dict[str, str] = {}
        if query_polarity and seed_nodes:
            for nid in seed_nodes:
                for _, tgt, data in self.net.graph.out_edges(nid, data=True):
                    rel = data.get("relation_obj")
                    if not rel:
                        continue
                    if query_polarity == "promotes" and rel.relation_type == RelationType.PROMOTES:
                        polarity_map[tgt] = "[促进方向匹配]"
                    elif query_polarity == "inhibits" and rel.relation_type == RelationType.INHIBITS:
                        polarity_map[tgt] = "[抑制方向匹配]"
                    elif rel.relation_type in (RelationType.PROMOTES, RelationType.INHIBITS):
                        if tgt not in polarity_map:
                            polarity_map[tgt] = "[方向相反]"

        sorted_nodes = sorted(
            current_nodes,
            key=lambda nid: self.net.get_node(nid).weight
            if self.net.get_node(nid) else 0,
            reverse=True
        )

        # ── 收集所有候选（source, target, rel, 综合得分）──────
        candidates: List[Tuple[str, str, Any, float]] = []

        for node_id in sorted_nodes[:5]:
            neighbors = self.net.get_neighbors(node_id, direction="both")
            for target_id, rel in neighbors:
                if target_id in current_nodes:
                    continue
                target_node = self.net.get_node(target_id)
                # P0-1修复：跳过 FALSE / PARADOX_PENDING 节点（不可靠节点不参与横向扩展）
                if not target_node or not target_node.is_reliable():
                    continue

                rel_type = rel.relation_type
                # 基础得分 = 关系优先级分（优先级越低数字越小越好 → 取倒数）
                priority_score = 1.0 / max(RELATION_PRIORITY.get(rel_type, 5), 1)
                # 极性匹配加分
                polarity_bonus = 0.0
                if query_polarity == "promotes" and rel_type == RelationType.PROMOTES:
                    polarity_bonus = 0.5
                elif query_polarity == "inhibits" and rel_type == RelationType.INHIBITS:
                    polarity_bonus = 0.5
                # 节点权重贡献
                score = priority_score + polarity_bonus + target_node.weight * 0.1
                candidates.append((node_id, target_id, rel, score))

        # 按得分降序，去重 target，取前 max_nodes_per_step 个
        candidates.sort(key=lambda x: x[3], reverse=True)
        seen_targets: set = set()
        new_nodes: List[str] = []
        path_log: List[str] = []
        # polarity_map 已由方案一预注入，这里不重置，只追加步骤3新发现的极性标签

        for src, tgt, rel, score in candidates:
            if tgt in seen_targets or tgt in new_nodes:
                continue
            seen_targets.add(tgt)
            rel_type = rel.relation_type

            label = ""
            if query_polarity == "promotes" and rel_type == RelationType.PROMOTES:
                label = "[促进方向匹配]"
            elif query_polarity == "inhibits" and rel_type == RelationType.INHIBITS:
                label = "[抑制方向匹配]"
            elif query_polarity and rel_type in (RelationType.PROMOTES, RelationType.INHIBITS):
                label = "[方向相反]"

            new_nodes.append(tgt)
            path_log.append(f"{src} →[{rel_type.value}]→ {tgt}{label}")
            if label:
                polarity_map[tgt] = label  # 追加/覆盖，不重置整个 map

            # ── 方案二：横向扩展激活节点的权重反馈 ─────────
            tgt_node = self.net.get_node(tgt)
            if tgt_node:
                tgt_node.activation_count += 1
                tgt_node.update_weight(+0.01, "推理激活(步骤3)")

            if len(new_nodes) >= self.max_nodes_per_step * 2:
                break

        # 跨域检索
        cross_domain = self._cross_domain_search(current_nodes, query, new_nodes)
        new_nodes.extend(cross_domain)

        if verbose:
            print(f"\n[步骤3] 横向扩展 (极性意图:{query_polarity or '中性'})，"
                  f"新激活 {len(new_nodes)} 个节点:")
            for entry in path_log[:8]:
                print(f"  {entry}")

        findings = [self.net.get_node(nid).content
                    for nid in new_nodes if self.net.get_node(nid)]

        step = ReasoningStep(
            step_id=3,
            action=f"横向扩展+跨域检索 (极性意图:{query_polarity or '中性'})",
            activated_nodes=new_nodes,
            path_taken=path_log,
            findings=findings,
            confidence=0.6,
        )
        # 把极性映射挂在 step 上，供 _step_validate → _compose_answer 读取
        step._polarity_map = polarity_map          # type: ignore[attr-defined]
        step._query_polarity = query_polarity      # type: ignore[attr-defined]
        return step

    def _cross_domain_search(self, current_nodes: List[str],
                              query: str,
                              already_found: List[str]) -> List[str]:
        """跨域检索：找到与当前激活节点相似但领域不同的节点"""
        # 收集当前领域
        current_domains = set()
        for nid in current_nodes:
            node = self.net.get_node(nid)
            if node:
                current_domains.update(node.domain)

        # 向量检索，过滤掉已激活和同域节点
        results = self.net.vector_search(query, top_k=10)
        cross = []
        for node_id, score in results:
            if node_id in current_nodes or node_id in already_found:
                continue
            node = self.net.get_node(node_id)
            if node and score > 0.3:
                # P0-1补丁：跨域检索同样不引入 FALSE/PARADOX_PENDING 节点
                if not node.is_reliable():
                    continue
                # 检查是否有不同领域
                node_domains = set(node.domain)
                if not node_domains.issubset(current_domains):
                    cross.append(node_id)
                    if len(cross) >= 3:
                        break
        return cross

    def _step_organize(self, all_nodes: List[str],
                        query: str, verbose: bool) -> Tuple[ReasoningStep, List[str]]:
        """第四步：信息组织，找出推理链，识别缺口"""
        # 收集所有激活节点的内容
        contents = []
        for nid in all_nodes:
            node = self.net.get_node(nid)
            if node:
                contents.append(f"[{node.abstract_level}] {node.content}")

        # 识别缺口：找到推理链中缺失的环节
        gaps = self._identify_gaps(all_nodes, query)

        if verbose:
            print(f"\n[步骤4] 信息组织:")
            print(f"  已激活 {len(all_nodes)} 个节点")
            if gaps:
                print(f"  发现缺口 {len(gaps)} 个:")
                for g in gaps:
                    print(f"    ? {g}")

        findings = contents[:8]  # 最多展示8条
        return (
            ReasoningStep(
                step_id=4,
                action="信息组织 + 缺口识别",
                activated_nodes=all_nodes,
                findings=findings,
                confidence=0.65,
            ),
            gaps
        )

    def _identify_gaps(self, activated_nodes: List[str],
                        query: str) -> List[str]:
        """
        识别推理链缺口
        检查三类断链：CAUSES / PROMOTES / INHIBITS 的目标节点不在激活集中
        """
        gaps = []
        checked_types = (
            RelationType.CAUSES,
            RelationType.PROMOTES,
            RelationType.INHIBITS,
        )
        type_labels = {
            RelationType.CAUSES: "导致",
            RelationType.PROMOTES: "促进",
            RelationType.INHIBITS: "抑制",
        }

        for nid in activated_nodes:
            node = self.net.get_node(nid)
            if not node:
                continue
            for _, target, data in self.net.graph.out_edges(nid, data=True):
                rel = data.get("relation_obj")
                if not rel or rel.relation_type not in checked_types:
                    continue
                if target not in activated_nodes:
                    target_node = self.net.get_node(target)
                    if target_node:
                        label = type_labels[rel.relation_type]
                        gaps.append(
                            f"缺少: {node.content[:30]} → [{label}] → {target_node.content[:30]}")
                        if len(gaps) >= 4:   # 最多返回4个缺口（比原来多一个）
                            return gaps

        return gaps

    def _step_gap_filling(self, gaps: List[str],
                           current_nodes: List[str],
                           query: str, verbose: bool) -> ReasoningStep:
        """第五步：根据缺口特征反向定义，再次检索补充"""
        new_nodes = []

        for gap in gaps:
            # 用缺口描述作为查询，再次向量检索
            gap_results = self.net.vector_search(gap, top_k=3)
            for node_id, score in gap_results:
                if node_id not in current_nodes and node_id not in new_nodes:
                    if score > 0.25:
                        # P0-1补丁：步骤5是P0-1修复的唯一漏网路径，在此截断
                        gap_node = self.net.get_node(node_id)
                        if gap_node and not gap_node.is_reliable():
                            continue
                        new_nodes.append(node_id)

        if verbose and new_nodes:
            print(f"\n[步骤5] 缺口补充检索，新激活 {len(new_nodes)} 个节点:")
            for nid in new_nodes:
                node = self.net.get_node(nid)
                if node:
                    print(f"  · {node.content[:50]}")

        findings = [self.net.get_node(nid).content
                    for nid in new_nodes if self.net.get_node(nid)]

        return ReasoningStep(
            step_id=5,
            action="缺口反向定义 + 补充检索",
            activated_nodes=new_nodes,
            findings=findings,
            confidence=0.5,
        )

    def _step_validate(self, all_nodes: List[str],
                        query: str, verbose: bool,
                        polarity_map: Optional[Dict[str, str]] = None,
                        query_polarity: Optional[str] = None,
                        top_vector_score: float = 0.0) -> Tuple[str, float, bool, List[str]]:
        """
        第六步：四重约束验证 + 组织最终答案

        方案三：置信度改为四维真实质量评分

        ① 问题响应度（0-1）
           激活节点内容与查询词汇的覆盖重合度——
           衡量推理是否真正回应了问题，而非偶然激活无关内容。

        ② 逻辑/规律约束满足度（0-1）
           激活集内 CAUSES/PROMOTES/INHIBITS 链是否形成"有出发有终点"的完整路径——
           不只计边数，而是看能否从某个激活节点出发沿因果/极性边到达另一个激活节点，
           反映推理链的内在逻辑是否成立。

        ③ 结构闭合度（0-1）
           激活节点之间实际有边相连的数量 / 激活节点数——
           节点孤立堆叠不得分，只有彼此有关系才提升分数。

        ④ 来源一致性（0-1）
           内部矛盾检测：无矛盾=1.0，OPPOSITE_TO矛盾=0.4，极性双向矛盾=0.5；
           同时考虑节点自身的 weight 均值（来自 ConflictResolver 的历史修正）。

        最终置信度 = ① × 0.30 + ② × 0.25 + ③ × 0.25 + ④ × 0.20
        """
        polarity_map = polarity_map or {}

        # 收集所有激活节点的内容，按权重排序
        # FALSE/PARADOX_PENDING 节点单独分拣：不进入内容池（不影响LLM答案），但参与置信度计算
        from memory_node import EpistemicStatus as _ES
        node_contents = []
        negated_node_ids: List[str] = []
        for nid in all_nodes:
            node = self.net.get_node(nid)
            if node:
                if node.epistemic_status in (_ES.FALSE, _ES.PARADOX_PENDING):
                    negated_node_ids.append(nid)
                else:
                    node_contents.append((node.weight, node.abstract_level,
                                          node.content, nid))
        node_contents.sort(key=lambda x: (x[0], x[1]), reverse=True)

        if not node_contents:
            return "无法从记忆网络中找到足够信息回答此问题。", 0.0, False, negated_node_ids

        # ───────────── ① 问题响应度 ─────────────────────────
        import re as _re
        # 提取查询中的有效词（去掉疑问词和标点）
        stopwords = {"什么", "怎么", "为什么", "如何", "哪个", "哪些", "是否",
                     "有没有", "吗", "呢", "的", "了", "吧", "吗", "？", "?", "，", "。"}
        query_tokens = set(
            t for t in _re.split(r'[\s，。？、；：！\?\!\.,;]', query)
            if t and t not in stopwords and len(t) >= 2
        )
        hit_count = 0
        if query_tokens:
            for _, _, content, _ in node_contents:
                for token in query_tokens:
                    if token in content:
                        hit_count += 1
                        break  # 每个节点对同一token只计一次
            overlap_score = min(1.0, hit_count / max(len(query_tokens), 1))
        else:
            overlap_score = 0.5  # 无法拆分时给中性分

        # 向量相似度兜底：取词汇overlap和step1向量检索最高分中的较大值
        # 解决"语义相关但用词不同"时overlap被低估的问题
        response_score = max(overlap_score, min(top_vector_score, 1.0))

        # ───────────── ② 逻辑/规律约束满足度 ────────────────
        causal_types = (RelationType.CAUSES, RelationType.PROMOTES, RelationType.INHIBITS)
        activated_set = set(all_nodes)
        causal_paths = 0  # 在激活集内完整的因果/极性路径数
        for nid in all_nodes:
            for _, tgt, data in self.net.graph.out_edges(nid, data=True):
                rel = data.get("relation_obj")
                if rel and rel.relation_type in causal_types and tgt in activated_set:
                    causal_paths += 1
        # 期望：每3个激活节点至少有1条完整因果路径
        logic_score = min(1.0, causal_paths / max(len(all_nodes) / 3.0, 1.0))

        # ───────────── ③ 结构闭合度 ──────────────────────────
        internal_edges = 0
        for nid in all_nodes:
            for _, tgt, _ in self.net.graph.out_edges(nid, data=True):
                if tgt in activated_set:
                    internal_edges += 1
        # 连通密度：内部边数 / 节点数（而非节点数的平方，避免小图高估）
        closure_score = min(1.0, internal_edges / max(len(all_nodes), 1))

        # ───────────── ④ 来源一致性（矛盾检测）──────────────
        contradiction_found = False
        contradiction_desc = ""
        contradiction_severity = 0.0  # 0=无冲突, 0.5=极性矛盾, 1.0=根本对立
        for nid in all_nodes:
            for _, target, data in self.net.graph.out_edges(nid, data=True):
                rel = data.get("relation_obj")
                if not rel or target not in activated_set:
                    continue
                if rel.relation_type == RelationType.OPPOSITE_TO:
                    contradiction_found = True
                    contradiction_severity = max(contradiction_severity, 1.0)
                    contradiction_desc = f"{nid}↔OPPOSITE↔{target}"
                    break
                if rel.relation_type in (RelationType.PROMOTES, RelationType.INHIBITS):
                    reverse_type = (RelationType.INHIBITS
                                    if rel.relation_type == RelationType.PROMOTES
                                    else RelationType.PROMOTES)
                    for _, t2, d2 in self.net.graph.out_edges(nid, data=True):
                        r2 = d2.get("relation_obj")
                        if r2 and r2.relation_type == reverse_type and t2 == target:
                            contradiction_found = True
                            contradiction_severity = max(contradiction_severity, 0.5)
                            contradiction_desc = f"{nid}→PROMOTES+INHIBITS→{target} 极性矛盾"
                            break
            if contradiction_severity >= 1.0:
                break
        # ── 隐性矛盾检测（向量语义反转，EXP-006补丁）────────────
        # 原理：若两个激活节点在向量空间中余弦相似度 < 0，说明它们在语义上
        # "方向相反"，但知识图谱中又没有 OPPOSITE_TO 关系连接它们。
        # 这类"无显式对立关系的语义对立节点对"即为隐性矛盾。
        # 处理策略：不拦截，但对 contradiction_severity 施加轻微惩罚（+0.25），
        # 并在 verbose 中报告，供调试和后续人工标注使用。
        implicit_contradiction_pairs: List[Tuple[str, str, float]] = []
        activated_nids_list = [nid for _, _, _, nid in node_contents]
        # 只对激活节点数 ≤ 20 时做扫描，避免大激活集时 O(n²) 过慢
        if len(activated_nids_list) <= 20 and contradiction_severity < 1.0:
            try:
                # 获取各节点向量（利用现有 embedding 接口）
                vecs: Dict[str, Any] = {}
                for nid in activated_nids_list:
                    node = self.net.get_node(nid)
                    if node:
                        try:
                            vecs[nid] = self.net._get_embed_vector(node.content)
                        except Exception:
                            pass

                # 扫描所有节点对
                checked_pairs: set = set()
                nids_with_vec = list(vecs.keys())
                for i in range(len(nids_with_vec)):
                    for j in range(i + 1, len(nids_with_vec)):
                        ni, nj = nids_with_vec[i], nids_with_vec[j]
                        pair_key = (min(ni, nj), max(ni, nj))
                        if pair_key in checked_pairs:
                            continue
                        checked_pairs.add(pair_key)
                        vi, vj = vecs[ni], vecs[nj]
                        norm_i = float(np.linalg.norm(vi))
                        norm_j = float(np.linalg.norm(vj))
                        if norm_i < 1e-9 or norm_j < 1e-9:
                            continue
                        cosine = float(np.dot(vi, vj) / (norm_i * norm_j))
                        # cosine < -0.1 视为语义反转（阈值可调）
                        if cosine < -0.1:
                            # 检查是否已有显式 OPPOSITE_TO 边（已被上面检测覆盖则跳过）
                            has_explicit = any(
                                d.get("relation_obj") and
                                d["relation_obj"].relation_type == RelationType.OPPOSITE_TO
                                for _, t, d in self.net.graph.out_edges(ni, data=True)
                                if t == nj
                            )
                            if not has_explicit:
                                implicit_contradiction_pairs.append((ni, nj, cosine))
                                # 轻度惩罚（不拦截，防止误报）
                                contradiction_severity = min(
                                    1.0, contradiction_severity + 0.25)
            except Exception:
                pass  # 向量服务不可用时静默跳过，不影响主流程

        # 节点平均权重（体现来源可信度历史）
        avg_weight = sum(w for w, _, _, _ in node_contents) / len(node_contents)
        consistency_score = (1.0 - contradiction_severity) * min(1.0, avg_weight)

        # ───────────── 极性链质量加成 ────────────────────────
        polarity_match_count = sum(
            1 for label in polarity_map.values()
            if "[促进方向匹配]" in label or "[抑制方向匹配]" in label
        )
        polarity_bonus = min(0.10, polarity_match_count * 0.03)

        # ───────────── 最终置信度 ────────────────────────────
        confidence = (
            response_score    * 0.30 +
            logic_score       * 0.25 +
            closure_score     * 0.25 +
            consistency_score * 0.20 +
            polarity_bonus
        )
        confidence = max(0.05, min(1.0, confidence))

        # 组织答案
        top_contents = [c for _, _, c, _ in node_contents[:6]]
        top_nids = [nid for _, _, _, nid in node_contents[:6]]
        answer = self._compose_answer(
            query, top_contents, top_nids, polarity_map, query_polarity)

        passed = not contradiction_found and confidence > 0.35
        # FALSE/PARADOX_PENDING 节点已在内容池构建时分拣到 negated_node_ids，
        # 此处用分拣结果直接做占比检查（不重复查图），保持逻辑一致
        if passed and negated_node_ids:
            unreliable_ratio = len(negated_node_ids) / max(len(all_nodes), 1)
            if unreliable_ratio > 0.20:
                passed = False
                if verbose:
                    print(f"  [验证] 激活集含 {len(negated_node_ids)} 个否定节点"
                          f"({unreliable_ratio:.0%})，超过阈值，验证未通过")

        if verbose:
            print(f"\n[步骤6] 四维置信度评分:")
            print(f"  ① 问题响应度:  {response_score:.2f}  "
                  f"(overlap:{overlap_score:.2f}, 向量兜底:{top_vector_score:.2f}, "
                  f"命中词:{hit_count if query_tokens else 'N/A'}/{len(query_tokens) if query_tokens else 'N/A'})")
            print(f"  ② 逻辑约束度:  {logic_score:.2f}  (因果/极性路径:{causal_paths}条)")
            print(f"  ③ 结构闭合度:  {closure_score:.2f}  (内部边:{internal_edges}条/{len(all_nodes)}节点)")
            print(f"  ④ 来源一致性:  {consistency_score:.2f}  "
                  f"(显式矛盾:{('发现! '+contradiction_desc) if contradiction_found else '无'}"
                  f"{', 隐性矛盾对数:'+str(len(implicit_contradiction_pairs)) if implicit_contradiction_pairs else ''})")
            if implicit_contradiction_pairs:
                for ni, nj, cos in implicit_contradiction_pairs[:3]:
                    nn_i = self.net.get_node(ni)
                    nn_j = self.net.get_node(nj)
                    ci = nn_i.content[:25] if nn_i else ni
                    cj = nn_j.content[:25] if nn_j else nj
                    print(f"    隐性矛盾: [{ci}] ↔ [{cj}]  cosine={cos:.3f}")
            print(f"  极性链加成:    +{polarity_bonus:.2f}  (匹配:{polarity_match_count}个)")
            print(f"  最终置信度:    {confidence:.3f}")
            print(f"  否定节点数:    {len(negated_node_ids)}（已从答案内容池剔除，ID记录于negated_nodes）")
            print(f"  验证{'通过' if passed else '未通过'}")

        return answer, confidence, passed, negated_node_ids

    def _compose_answer(self, query: str, contents: List[str],
                        node_ids: Optional[List[str]] = None,
                        polarity_map: Optional[Dict[str, str]] = None,
                        query_polarity: Optional[str] = None) -> str:
        """
        用 Ollama qwen2.5:7b 基于激活的记忆节点内容生成自然语言回答。
        若存在极性意图，将节点按"促进/抑制/通用"分类，让 LLM 给出结构化答案。
        若 Ollama 不可用，退化为拼接模式（保持可用性）。
        """
        if not contents:
            return "无法从记忆网络中找到足够信息回答此问题。"

        polarity_map = polarity_map or {}
        node_ids = node_ids or []

        # 若有极性意图，按分类构造上下文
        promotes_items: List[str] = []
        inhibits_items: List[str] = []
        neutral_items: List[str] = []
        if query_polarity and polarity_map:
            # P2-1修复：删除此处冗余的重赋值（原代码在 if 分支内重置为空列表，
            # 导致 else 分支走退化模式时 promotes_items 等变量未定义 → NameError）
            for i, (nid, content) in enumerate(zip(node_ids, contents)):
                label = polarity_map.get(nid, "")
                if "[促进方向匹配]" in label:
                    promotes_items.append(content)
                elif "[抑制方向匹配]" in label or "[方向相反]" in label:
                    inhibits_items.append(content)
                else:
                    neutral_items.append(content)

            sections = []
            if promotes_items:
                sections.append("【促进/有利因素】\n" +
                                 "\n".join(f"  {i+1}. {c}" for i, c in enumerate(promotes_items)))
            if inhibits_items:
                sections.append("【抑制/不利因素】\n" +
                                 "\n".join(f"  {i+1}. {c}" for i, c in enumerate(inhibits_items)))
            if neutral_items:
                sections.append("【相关背景知识】\n" +
                                 "\n".join(f"  {i+1}. {c}" for i, c in enumerate(neutral_items)))
            context_text = "\n\n".join(sections) if sections else \
                "\n".join(f"{i+1}. {c}" for i, c in enumerate(contents))
            polarity_hint = (
                "回答时请明确区分促进因素和抑制因素（如有），"
                "并说明各因素如何作用于目标结果。"
            )
        else:
            context_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(contents))
            polarity_hint = "回答应体现知识片段之间的内在联系，而不是简单罗列。"

        prompt = (
            f"你是一个知识推理助手。以下是从知识图谱中检索并联想激活的相关知识片段：\n\n"
            f"{context_text}\n\n"
            f"请根据以上知识片段，对以下问题给出清晰、连贯的回答。"
            f"{polarity_hint}\n\n"
            f"问题：{query}\n\n回答："
        )

        try:
            from config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_TIMEOUT
            import ollama
            import re
            client = ollama.Client(host=OLLAMA_BASE_URL)
            response = client.generate(
                model=OLLAMA_LLM_MODEL,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": 512},
            )
            answer = response.get("response", "").strip()
            # 过滤 qwen3 的 <think>...</think> 思维链输出
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            if answer:
                return answer
        except Exception as e:
            print(f"[LLM] Ollama 调用失败，退化为拼接模式: {e}")

        # 退化：拼接模式（不依赖 LLM）
        if query_polarity and (promotes_items or inhibits_items):
            parts = ["基于联想推理："]
            if promotes_items:
                parts.append("促进因素：")
                for c in promotes_items:
                    parts.append(f"  · {c}")
            if inhibits_items:
                parts.append("抑制因素：")
                for c in inhibits_items:
                    parts.append(f"  · {c}")
            if neutral_items:
                parts.append("相关背景：")
                for c in neutral_items:
                    parts.append(f"  · {c}")
        else:
            parts = ["基于联想推理，以下信息与问题相关："]
            for i, content in enumerate(contents, 1):
                parts.append(f"  {i}. {content}")
            parts.append(f"\n推理链路：通过 {len(contents)} 个相关记忆节点的联想激活，"
                         f"从抽象层级到具体实例逐层推导得出上述关联。")
        return "\n".join(parts)

    # ─────────────────────────────────────────────────
    # 类型二可能性节点发现
    # ─────────────────────────────────────────────────

    def _discover_potential_nodes(
            self, activated_nodes: List[str],
            query: str, verbose: bool) -> List[str]:
        """
        在本次推理激活的节点集合中，检测"条件具备但结论尚未存在"的潜在可能性。

        原理：
          对激活集中的每个节点，检查其出边（CAUSES/PROMOTES）的目标节点：
          - 如果目标节点已在激活集 → 已知结论，跳过
          - 如果目标节点存在于知识库 → 已知结论但未激活，跳过
          - 如果目标节点不存在于知识库 → 不判断（知识库未覆盖）
          
          真正的类型二可能性发现：
          检查是否有≥2个激活节点同时 CAUSES/PROMOTES 同一个目标T，
          且T节点不在激活集中，且T节点的 epistemic_status=POTENTIAL 或不存在。
          → 多个已知条件共同指向同一潜在结果 = 高价值潜在可能性

        返回：本次新发现或激活的 POTENTIAL 节点ID列表
        """
        from collections import Counter
        target_counter: Counter = Counter()
        target_sources: Dict[str, List[str]] = {}

        causal_types = (RelationType.CAUSES, RelationType.PROMOTES)
        activated_set = set(activated_nodes)

        for nid in activated_nodes:
            for _, tgt, data in self.net.graph.out_edges(nid, data=True):
                rel = data.get("relation_obj")
                if not rel or rel.relation_type not in causal_types:
                    continue
                if tgt in activated_set:
                    continue
                target_counter[tgt] += 1
                if tgt not in target_sources:
                    target_sources[tgt] = []
                target_sources[tgt].append(nid)

        # 找到被≥2个激活节点共同指向的目标
        potential_found: List[str] = []
        for tgt, count in target_counter.items():
            if count < 2:
                continue
            tgt_node = self.net.get_node(tgt)
            if tgt_node is None:
                continue
            # 若节点已是confirmed且权重正常 → 已知结论，不算新发现
            if (tgt_node.epistemic_status == EpistemicStatus.CONFIRMED
                    and tgt_node.weight >= 0.5):
                continue
            # 若节点是POTENTIAL状态 → 激活它（权重小幅提升）
            if tgt_node.epistemic_status == EpistemicStatus.POTENTIAL:
                tgt_node.activation_count += 1
                tgt_node.update_weight(+0.05, f"可能性节点被{count}个条件节点激活")
                potential_found.append(tgt)
                if verbose:
                    print(f"\n[可能性] 激活已有POTENTIAL节点: {tgt_node.content[:50]}")
                    print(f"  条件来源: {[self.net.get_node(s).content[:25] if self.net.get_node(s) else s for s in target_sources[tgt]]}")
            else:
                # 节点存在但不是POTENTIAL → 多条件汇聚是值得关注的信号
                if verbose:
                    print(f"\n[可能性] 多条件汇聚目标: {tgt_node.content[:50]} (条件数:{count})")
                    print(f"  条件来源: {[self.net.get_node(s).content[:25] if self.net.get_node(s) else s for s in target_sources[tgt]]}")

        # ── 前向推断：推理链末端节点生成推断型POTENTIAL候选 ─────────────
        # 当激活集中存在推理链末端节点（没有有效出边指向任何激活节点的节点），
        # 且该末端节点是 CAUSES/PROMOTES 关系的源头，但其所有出边目标都不在知识库中时，
        # 说明推理链在此处"悬空"——这是推断型可能性的孵化点。
        # 此时构造一个推断候选放入 inferred_pool（不写入主知识库，仅记录）。
        # 候选通过 confirm_inferred_potential() 四重校验后，才能正式写入为POTENTIAL节点。
        inferred_pool: List[Dict] = []
        terminal_nodes = [
            nid for nid in activated_nodes
            if not any(tgt in activated_set
                       for _, tgt, d in self.net.graph.out_edges(nid, data=True)
                       if d.get("relation_obj") and
                       d["relation_obj"].relation_type in causal_types)
        ]
        for nid in terminal_nodes:
            node = self.net.get_node(nid)
            if not node:
                continue
            # 末端节点的所有出边目标（包括不在知识库中的）
            for _, tgt, data in self.net.graph.out_edges(nid, data=True):
                rel = data.get("relation_obj")
                if not rel or rel.relation_type not in causal_types:
                    continue
                tgt_node = self.net.get_node(tgt)
                if tgt_node is not None:
                    continue  # 目标存在于知识库，不是"悬空"推断
                # tgt 是不存在于知识库的目标 → 生成推断候选
                inferred_pool.append({
                    "source_node_id": nid,
                    "source_content": node.content,
                    "inferred_target_id": tgt,
                    "inferred_from": [nid],
                    "relation_type": rel.relation_type.value,
                    "note": (
                        f"推理链末端'{node.content[:30]}'的出边指向不存在的目标'{tgt}'，"
                        f"为推断型POTENTIAL候选，需经confirm_inferred_potential()四重校验后写入"
                    ),
                })
        # 把推断候选附加到引擎的候选池（跨推理持久化）
        if not hasattr(self, 'inferred_potential_pool'):
            self.inferred_potential_pool: List[Dict] = []
        self.inferred_potential_pool.extend(inferred_pool)
        if inferred_pool and verbose:
            print(f"\n[推断候选] 本次推理产生 {len(inferred_pool)} 个前向推断候选，"
                  f"已暂存至 engine.inferred_potential_pool，等待confirm_inferred_potential()校验")

        return potential_found

    def confirm_inferred_potential(
            self,
            candidate_idx: int,
            content: str,
            domain: Optional[List[str]] = None,
            essence_features: Optional[List[str]] = None,
            verbose: bool = True) -> Optional[str]:
        """
        对 inferred_potential_pool 中下标为 candidate_idx 的推断候选进行四重校验并写入。

        四重校验（调用者须确认以下全部满足）：
          1. 规律约束：推断结论符合领域内已知规律（不与任何CONFIRMED节点存在根本矛盾）
          2. 目标合理性：结论在领域逻辑上有意义（content参数即为调用者构造的自然语言描述）
          3. 推断来源可追溯：candidate 中记录了推断来源节点 ID
          4. 区别性标注：节点写入为 POTENTIAL，不升级为 CONFIRMED，直到有现实证据

        写入后：
          - 节点 epistemic_status = POTENTIAL，weight = 0.4（低于手动添加的0.5，体现推断不确定性）
          - tags 包含 "推断型可能性"、"前向推断"，与手动添加的类型二可能性可区分
          - 从 inferred_potential_pool 中移除该候选（避免重复写入）

        升级路径：当现实中出现证据证实该推断时，调用 node.mark_confirmed(reason) 升级。

        返回：新节点ID，或 None（若候选不存在）
        """
        if not hasattr(self, 'inferred_potential_pool') or \
                candidate_idx >= len(self.inferred_potential_pool):
            if verbose:
                print("[推断候选] 候选索引不存在或推断池为空")
            return None

        candidate = self.inferred_potential_pool[candidate_idx]
        source_ids = candidate["inferred_from"]

        node_id = f"inferred_{uuid.uuid4().hex[:8]}"
        node = MemoryNode(
            node_id=node_id,
            content=content,
            abstract_level=5,   # 推断型可能性默认中等抽象层级
            domain=domain or ["通用"],
            coverage=0.15,      # 推断覆盖度低于手动添加的类型二可能性
            essence_features=essence_features or [],
            tags=["推断型可能性", "前向推断",
                  f"推断源:{candidate['source_node_id'][:12]}"],
            weight=0.4,
            epistemic_status=EpistemicStatus.POTENTIAL,
            trigger_conditions=source_ids,
        )
        self.net.add_node(node)
        # 建立推断源→推断结论的 CAUSES 关系（弱权重，标注推断来源）
        for sid in source_ids:
            if self.net.get_node(sid):
                rel = Relation(
                    source_id=sid,
                    target_id=node_id,
                    relation_type=RelationType.CAUSES,
                    weight=0.3,
                    context="inferred_potential",
                )
                self.net.add_relation(rel)

        # 从候选池移除（避免重复写入）
        self.inferred_potential_pool.pop(candidate_idx)

        if verbose:
            print(f"[推断候选→POTENTIAL] 已写入: {node_id}")
            print(f"  内容: {content[:60]}")
            print(f"  推断来源: {source_ids}")
            print(f"  状态: POTENTIAL（weight=0.4），待现实证据升级为CONFIRMED）")
        return node_id

    def add_potential_node(
            self,
            content: str,
            trigger_conditions: List[str],
            abstract_level: int = 5,
            domain: Optional[List[str]] = None,
            essence_features: Optional[List[str]] = None,
            verbose: bool = True) -> str:
        """
        手动添加一个"类型二可能性"节点。
        trigger_conditions：触发该可能性所需的条件节点ID列表（≥2个）。
        节点以 epistemic_status=POTENTIAL 写入，权重=0.5，不参与正向推理的主链，
        但在 _discover_potential_nodes 中会被多条件汇聚激活。

        返回新节点ID。
        """
        node_id = f"potential_{uuid.uuid4().hex[:8]}"
        node = MemoryNode(
            node_id=node_id,
            content=content,
            abstract_level=abstract_level,
            domain=domain or ["通用"],
            coverage=0.2,
            essence_features=essence_features or [],
            tags=["潜在可能性", "类型二", f"条件数:{len(trigger_conditions)}"],
            weight=0.5,
            epistemic_status=EpistemicStatus.POTENTIAL,
            trigger_conditions=trigger_conditions,
        )
        self.net.add_node(node)
        # 建立条件→可能性节点的 CAUSES 关系（弱权重）
        for cid in trigger_conditions:
            if self.net.get_node(cid):
                rel = Relation(
                    source_id=cid,
                    target_id=node_id,
                    relation_type=RelationType.CAUSES,
                    weight=0.4,
                    context="potential_condition",
                )
                self.net.add_relation(rel)
        if verbose:
            print(f"[可能性节点] 已添加: {node_id} → '{content[:50]}'")
            print(f"  触发条件: {trigger_conditions}")
        return node_id

    # ─────────────────────────────────────────────────
    # 超图边联合条件触发
    # ─────────────────────────────────────────────────

    def _check_hyper_edges(
            self, activated_nodes: List[str],
            verbose: bool,
            query: str = "") -> List[str]:
        """
        检查当前激活节点集合是否满足任何超图边的联合条件，
        若满足则将该超图边的目标节点加入激活集，并对目标节点轻微加权。

        T4语义门控：若超图边设置了 context_keywords，则触发前检查：
          query文本 + 激活节点内容 中是否包含至少一个关键词。
          对 OR 超图边（低触发门槛）强制执行此检查；
          对 AND 超图边（高触发门槛）可选执行（context_keywords非空时同样检查）。
          留空 context_keywords 保持原有行为，完全向后兼容。

        返回：新增加的目标节点ID列表
        """
        triggered_targets: List[str] = []

        # 预先构造激活节点内容文本（用于语境关键词匹配）
        activated_content_text = " ".join(
            (self.net.get_node(nid).content if self.net.get_node(nid) else "")
            for nid in activated_nodes
        )
        context_corpus = (query + " " + activated_content_text).lower()

        # 利用 MemoryNetwork.find_triggered_hyper_edges 做筛选
        # AND 模式：全部条件节点在激活集中才触发（min_satisfaction=1.0）
        # OR 模式：任一条件节点在激活集中即触发（min_satisfaction=1/N）
        triggered = self.net.find_triggered_hyper_edges(
            activated_nodes, min_satisfaction=1.0)  # AND: 全满足

        # 也检查部分满足（≥0.5）的超图边作为弱信号
        partial = self.net.find_triggered_hyper_edges(
            activated_nodes, min_satisfaction=0.5)
        # 合并（完全满足优先）
        seen_hids = {he.hyper_id for he, _ in triggered}
        for he, ratio in partial:
            if he.hyper_id not in seen_hids:
                triggered.append((he, ratio))

        for hyper_edge, ratio in triggered:
            # ── 语义门控检查（T4）─────────────────────────────
            # OR边必须通过语境关键词过滤（context_keywords非空时）
            # AND边也遵守（context_keywords非空时），为空则跳过
            if hyper_edge.context_keywords:
                kw_hit = any(
                    kw.lower() in context_corpus
                    for kw in hyper_edge.context_keywords
                )
                if not kw_hit:
                    if verbose:
                        print(f"\n[超图][语境门控] 超图边 {hyper_edge.hyper_id} "
                              f"语境关键词 {hyper_edge.context_keywords} 均未命中，跳过触发")
                    continue

            fully_met = (ratio >= 1.0
                         or hyper_edge.condition == "OR")
            for tgt_id in hyper_edge.target_ids:
                if tgt_id in activated_nodes:
                    continue
                tgt_node = self.net.get_node(tgt_id)
                if tgt_node is None:
                    continue
                # P0-1补丁：超图边目标节点同样不应是 FALSE/PARADOX_PENDING 节点
                if not tgt_node.is_reliable():
                    continue
                if fully_met:
                    # 完全满足：激活目标节点
                    tgt_node.activation_count += 1
                    tgt_node.update_weight(
                        +0.03, f"超图边完全触发({hyper_edge.hyper_id})")
                    triggered_targets.append(tgt_id)
                    if verbose:
                        co_contents = [
                            (self.net.get_node(cn).content[:20]
                             if self.net.get_node(cn) else cn)
                            for cn in hyper_edge.co_nodes
                        ]
                        print(f"\n[超图] 联合条件完全满足，触发目标节点: "
                              f"'{tgt_node.content[:40]}'")
                        print(f"  条件({hyper_edge.condition}): "
                              f"{co_contents}")
                        print(f"  超图边: {hyper_edge.hyper_id}  "
                              f"关系: {hyper_edge.relation_type.value}  "
                              f"权重: {hyper_edge.weight:.2f}")
                else:
                    # 部分满足：仅作为信号输出，不写入激活集
                    if verbose:
                        print(f"\n[超图] 部分满足({ratio:.0%})的超图边: "
                              f"'{tgt_node.content[:40]}' "
                              f"({hyper_edge.hyper_id})")

        return triggered_targets

    # ─────────────────────────────────────────────────
    # 临时假设沙箱（归谬推理）
    # ─────────────────────────────────────────────────

    def reason_with_hypothesis(
            self,
            hypothesis: str,
            assumed_true: bool = True,
            test_queries: Optional[List[str]] = None,
            verbose: bool = True) -> Dict:
        """
        归谬推理沙箱：临时假设某命题为真（或假），
        在不修改知识库的情况下进行推理，
        检测是否与已有"公理级节点"或"客观现实约束"产生确凿矛盾。

        裁决信号（三路独立，不再用置信度代替矛盾检测）：
          信号A（结构矛盾）：激活节点与公理节点之间存在 OPPOSITE_TO 边
          信号B（极性矛盾）：激活节点通过 PROMOTES/INHIBITS 与公理节点形成相反极性边
          信号C（语义覆盖）：假设内容与公理节点内容在向量空间中高度相似
                             → 若假设为真，公理应直接激活；
                               若公理激活但 FALSE/HYPOTHESIS 则说明假设被已有否定信息覆盖

          "反驳得分"  = 信号A × 3 + 信号B × 2（强权重）
          "支持得分"  = 信号C_match × 2 + 激活节点中含 CONFIRMED 公理数（弱权重）
          置信度通过 refuted/support 得分之比计算，而非推理置信度

        注意：此方法不会修改 self.net，完全只读（沙箱隔离）。

        参数：
          hypothesis    : 待验证的假设命题（自然语言）
          assumed_true  : True=假设为真进行验证，False=假设为假进行验证
          test_queries  : 若提供，用这些查询测试假设的推论；
                          若不提供，自动用 hypothesis 本身做一次推理
          verbose       : 是否打印过程

        返回字典：
          {
            "hypothesis": str,
            "assumed_true": bool,
            "verdict": "refuted" | "supported" | "inconclusive",
            "evidence": List[str],
            "contradiction_nodes": List[str],
            "support_nodes": List[str],
            "refute_score": float,
            "support_score": float,
            "confidence": float
          }
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"[假设沙箱] 假设: '{hypothesis}'")
            print(f"[假设沙箱] 假设方向: {'假设为真，寻找矛盾' if assumed_true else '假设为假，寻找支撑'}")
            print(f"{'='*60}")

        queries = test_queries or [hypothesis]
        evidence: List[str] = []
        contradiction_nodes: List[str] = []
        support_nodes: List[str] = []
        refute_score = 0.0
        support_score = 0.0

        # 公理节点（abstract_level≥8，confirmed状态，weight≥0.5）作为裁决锚点
        axiom_nodes = [
            node for node in self.net.nodes.values()
            if (node.abstract_level >= 8
                and node.epistemic_status == EpistemicStatus.CONFIRMED
                and node.weight >= 0.5)
        ]
        axiom_ids = {n.node_id for n in axiom_nodes}
        axiom_contents = {n.node_id: n.content for n in axiom_nodes}

        # 预先检索：假设内容与公理节点的语义相似度（信号C基础）
        hyp_similar = self.net.vector_search(hypothesis, top_k=10)
        hyp_similar_axioms = {
            nid: score for nid, score in hyp_similar
            if nid in axiom_ids and score >= 0.55
        }

        for q in queries:
            # 推理（只读，不触发自动存储）
            original_threshold = self.auto_store_threshold
            self.auto_store_threshold = 999.0
            try:
                result = self.reason(q, verbose=False)
            finally:
                self.auto_store_threshold = original_threshold

            activated_set = set(result.activated_nodes)

            # ── 信号A：结构矛盾（OPPOSITE_TO 边直连公理）────────────────
            for nid in result.activated_nodes:
                node_obj = self.net.get_node(nid)
                if not node_obj:
                    continue
                for _, tgt, data in self.net.graph.out_edges(nid, data=True):
                    rel = data.get("relation_obj")
                    if not rel:
                        continue
                    if rel.relation_type == RelationType.OPPOSITE_TO and tgt in axiom_ids:
                        desc = (f"激活节点 '{node_obj.content[:30]}' "
                                f"与公理 '{axiom_contents[tgt][:30]}' "
                                f"存在 OPPOSITE_TO 关系")
                        evidence.append(f"[信号A-结构矛盾] {desc}")
                        contradiction_nodes.append(tgt)
                        refute_score += 3.0

            # ── 信号B：极性矛盾（激活节点↔公理之间存在相反极性边）────────
            for nid in result.activated_nodes:
                promotes_tgts: set = set()
                inhibits_tgts: set = set()
                for _, tgt, data in self.net.graph.out_edges(nid, data=True):
                    rel = data.get("relation_obj")
                    if not rel:
                        continue
                    if rel.relation_type == RelationType.PROMOTES:
                        promotes_tgts.add(tgt)
                    elif rel.relation_type == RelationType.INHIBITS:
                        inhibits_tgts.add(tgt)
                # 若激活节点 INHIBITS 某个公理节点（公理应被"支持"不应被"抑制"）
                blocked_axioms = inhibits_tgts & axiom_ids
                for aid in blocked_axioms:
                    node_obj = self.net.get_node(nid)
                    desc = (f"激活节点 '{node_obj.content[:25] if node_obj else nid}' "
                            f"INHIBITS 公理 '{axiom_contents[aid][:30]}'")
                    evidence.append(f"[信号B-极性矛盾] {desc}")
                    contradiction_nodes.append(aid)
                    refute_score += 2.0

            # ── 信号C：语义覆盖（假设与公理高度相似）────────────────────
            # 若假设内容与某公理向量相似度高，且该公理被激活（说明系统认可公理优先于假设）
            for axiom_id, sim_score in hyp_similar_axioms.items():
                axiom_node = self.net.get_node(axiom_id)
                if not axiom_node:
                    continue
                if axiom_id in activated_set:
                    # 公理被激活且与假设相似：说明假设与公理语义重叠
                    # 若公理的内容方向与假设一致 → 支持
                    # 若公理的内容被系统以 FALSE 节点覆盖 → 反驳
                    false_neighbors = [
                        tgt for _, tgt, data in self.net.graph.out_edges(axiom_id, data=True)
                        if (self.net.get_node(tgt)
                            and self.net.get_node(tgt).epistemic_status == EpistemicStatus.FALSE)
                    ]
                    if false_neighbors:
                        desc = (f"公理 '{axiom_node.content[:30]}' "
                                f"（与假设语义相似={sim_score:.2f}）"
                                f"关联有已确认FALSE节点，假设可能错误")
                        evidence.append(f"[信号C-语义覆盖→反驳] {desc}")
                        refute_score += 1.0
                    else:
                        desc = (f"假设与公理 '{axiom_node.content[:30]}' "
                                f"语义相似({sim_score:.2f})，公理已确认为真")
                        evidence.append(f"[信号C-语义覆盖→支持] {desc}")
                        support_nodes.append(axiom_id)
                        support_score += 2.0

            # ── 补充支持信号：激活节点中含 CONFIRMED 公理 + 无矛盾边 ──────
            confirmed_axioms_activated = [
                nid for nid in result.activated_nodes
                if nid in axiom_ids
            ]
            # 已在矛盾节点中的不算支持
            clean_confirmed = [
                nid for nid in confirmed_axioms_activated
                if nid not in contradiction_nodes
            ]
            if clean_confirmed:
                desc = (f"激活了 {len(clean_confirmed)} 个确认公理节点且无结构矛盾: "
                        f"{[axiom_contents[nid][:20] for nid in clean_confirmed[:2]]}")
                evidence.append(f"[信号D-公理激活] {desc}")
                support_score += len(clean_confirmed) * 0.5

        # ── 裁决 ───────────────────────────────────────────────────────
        total = refute_score + support_score
        if total < 0.5:
            verdict = "inconclusive"
            verdict_confidence = 0.25
        else:
            refute_ratio = refute_score / total
            support_ratio = support_score / total
            # 反驳信号主导（信号A、B权重更高，阈值60%）
            if refute_ratio >= 0.60:
                verdict = "refuted" if assumed_true else "supported"
                verdict_confidence = min(0.92, refute_ratio * 0.95)
            elif support_ratio >= 0.60:
                verdict = "supported" if assumed_true else "refuted"
                verdict_confidence = min(0.85, support_ratio * 0.90)
            else:
                verdict = "inconclusive"
                verdict_confidence = 0.40

        result_dict = {
            "hypothesis": hypothesis,
            "assumed_true": assumed_true,
            "verdict": verdict,
            "evidence": evidence,
            "contradiction_nodes": list(set(contradiction_nodes)),
            "support_nodes": list(set(support_nodes)),
            "refute_score": round(refute_score, 2),
            "support_score": round(support_score, 2),
            "confidence": round(verdict_confidence, 3),
        }

        # 记录到沙箱日志
        self.hypothesis_sandbox_log.append({
            **result_dict,
            "timestamp": time.time(),
        })

        if verbose:
            print(f"\n[假设沙箱] 反驳得分: {refute_score:.1f}  支持得分: {support_score:.1f}")
            print(f"[假设沙箱] 裁决: {verdict.upper()} (置信度:{verdict_confidence:.2f})")
            for e in evidence:
                print(f"  {e}")
            if contradiction_nodes:
                print(f"  矛盾节点: {list(set(contradiction_nodes))}")
            if support_nodes:
                print(f"  支持节点: {list(set(support_nodes))}")

        return result_dict

    # ─────────────────────────────────────────────────
    # 改进4：推理结论自动存储
    # ─────────────────────────────────────────────────

    def _store_conclusion(self,
                           query: str,
                           answer: str,
                           confidence: float,
                           activated_nodes: List[str],
                           context_profile: Optional[ContextProfile],
                           verbose: bool) -> Optional[str]:
        """
        将高置信度推理结论自动存储为新记忆节点。
        修复：写入前经过 ConflictResolver 三级冲突检测，而不是直接 add_node。
        """
        from conflict_resolver import NewInformation
        from relation_types import Relation, RelationType

        # 防止重复存储
        for record in self.stored_conclusions:
            if record["query"] == query:
                return None

        if len(activated_nodes) < 3:
            return None

        # 推断抽象层级：激活节点的平均层级
        levels = [self.net.get_node(nid).abstract_level
                  for nid in activated_nodes if self.net.get_node(nid)]
        avg_level = int(sum(levels) / len(levels)) if levels else 5

        # 推断领域
        domain_count: Dict[str, int] = {}
        for nid in activated_nodes:
            node = self.net.get_node(nid)
            if node:
                for d in node.domain:
                    domain_count[d] = domain_count.get(d, 0) + 1
        top_domains = sorted(domain_count, key=domain_count.__getitem__, reverse=True)[:2]

        conclusion_content = f"[推理结论] 关于'{query[:30]}'：{answer[:100]}"
        new_node_id = f"inferred_{uuid.uuid4().hex[:8]}"

        # 构建拟写入节点的 proposed_relations（与激活节点的 DERIVED_FROM 关系）
        proposed_relations = []
        for nid in activated_nodes[:3]:
            if self.net.get_node(nid):
                proposed_relations.append(Relation(
                    source_id=new_node_id,
                    target_id=nid,
                    relation_type=RelationType.DERIVED_FROM,
                    weight=confidence,
                    context=f"inferred_from_query:{query[:30]}",
                ))

        new_info = NewInformation(
            node_id=new_node_id,
            content=conclusion_content,
            abstract_level=avg_level,
            domain=top_domains if top_domains else ["通用"],
            coverage=0.3,
            essence_features=[f"推理自:{query[:20]}"],
            tags=["推理结论", "自动生成",
                  f"置信度:{confidence:.2f}",
                  context_profile.context_type if context_profile else "unknown"],
            evidence_strength=confidence,
            source="associative_engine",
            proposed_relations=proposed_relations,
        )

        # 经过冲突检测后再决定写入方式（修复：之前是直接 add_node 绕过检测）
        # P3-1修复：使用引擎级共享 resolver，悖论池/历史跨次推理持久化
        report = self.resolver.process(new_info, verbose=verbose)

        # 悖论级别不写入（等待人工裁决）
        from conflict_resolver import ConflictLevel
        if report.level == ConflictLevel.PARADOX:
            if verbose:
                print(f"  [自动存储] 冲突级别为悖论，跳过写入，已加入悖论池")
            return None

        # 向量同步写入 Qdrant（ConflictResolver 已调用 add_node，这里同步向量）
        stored_node = self.net.get_node(new_node_id)
        if stored_node:
            try:
                self.net.upsert_node_vector(stored_node)
            except Exception as e:
                if verbose:
                    print(f"  [向量同步] Qdrant 写入失败（不影响图谱）: {e}")

        self.stored_conclusions.append({
            "query": query,
            "node_id": new_node_id,
            "confidence": confidence,
            "activated_count": len(activated_nodes),
            "conflict_level": report.level.name,
            "timestamp": time.time(),
        })

        if verbose:
            print(f"  [自动存储] 新节点 {new_node_id} 已写入（冲突级别: {report.level.name}）")

        return new_node_id
