"""
记忆网络核心
图谱 + 向量双索引
向量层：Ollama nomic-embed-text（中文支持）
持久化：Qdrant（HNSW 近似最近邻，O(log n)）

2026-03-15 新增：超图轻量实现
  超图边（HyperEdge）：表达多个节点共同决定某个结论的 N→1 联合关系。
  例："压力激素升高" AND "睡眠剥夺" → 共同导致 "免疫功能受损"
  普通二元边无法表达联合因果，超图边专门处理此类语义。
  超图与现有二元边完全兼容，按需使用，不影响原有推理流程。
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from memory_node import MemoryNode, EpistemicStatus
from relation_types import Relation, RelationType, RELATION_PRIORITY
import json
import uuid
import time


# ─────────────────────────────────────────────────────────────────────────────
# 超图边定义
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HyperEdge:
    """
    超图边：表达多节点联合关系（N→1 或 N→M）。

    co_nodes    : 联合参与该关系的所有节点ID列表（"共同主语"）
    target_ids  : 结论节点ID列表（"共同结论"，通常为1个，也可多个）
    relation_type : 超图边的语义类型（与 RelationType 共用枚举）
    weight      : 关系强度（联合条件均满足时的置信度）
    condition   : "AND"（所有co_nodes同时成立）/ "OR"（任一co_node成立）
    context     : 附加说明（如来源、领域等）
    hyper_id    : 唯一标识符
    created_at  : 创建时间戳
    context_keywords : 语义门控关键词列表（可选）。
        对于 OR 超图边，触发前需要查询文本或当前激活节点内容中至少包含一个关键词，
        否则即便节点相似度匹配也不触发——防止语义无关节点通过相似度误触发。
        AND 超图边本身约束已强，可不填（填写时同样有效）。
        留空（默认[]）表示不做语境过滤，行为与原来一致，保持向后兼容。

    使用场景举例：
      压力激素升高 + 睡眠剥夺 → CAUSES → 免疫功能受损
        co_nodes    = ["node_cortisol_high", "node_sleep_deprived"]
        target_ids  = ["node_immune_suppressed"]
        condition   = "AND"
        weight      = 0.85

      有氧运动 + 低糖饮食 → PROMOTES → 神经新生
        co_nodes    = ["node_aerobic", "node_low_sugar_diet"]
        target_ids  = ["node_neurogenesis"]
        condition   = "AND"
        weight      = 0.75
        context_keywords = ["神经", "运动", "代谢", "大脑"]
    """
    co_nodes: List[str]
    target_ids: List[str]
    relation_type: RelationType
    weight: float = 1.0
    condition: str = "AND"          # "AND" | "OR"
    context: str = ""
    hyper_id: str = field(default_factory=lambda: f"he_{uuid.uuid4().hex[:8]}")
    created_at: float = field(default_factory=time.time)
    context_keywords: List[str] = field(default_factory=list)  # 语义门控关键词，OR边防误触发

    def to_dict(self) -> Dict:
        return {
            "hyper_id": self.hyper_id,
            "co_nodes": self.co_nodes,
            "target_ids": self.target_ids,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "condition": self.condition,
            "context": self.context,
            "created_at": self.created_at,
            "context_keywords": self.context_keywords,
        }





class MemoryNetwork:
    """
    记忆网络：图谱（关系导航）+ 向量索引（语义相似度）双层结构
    向量检索由 Qdrant 提供，embedding 由 Ollama nomic-embed-text 提供
    超图边（HyperEdge）由内部字典存储，与二元边并列
    """

    def __init__(self):
        # 有向图：节点=MemoryNode，边=Relation
        self.graph = nx.DiGraph()

        # 节点存储 {node_id: MemoryNode}
        self.nodes: Dict[str, MemoryNode] = {}

        # 超图边存储：{hyper_id: HyperEdge}
        # key: hyper_id, value: HyperEdge
        self.hyper_edges: Dict[str, HyperEdge] = {}
        # 倒排索引：{node_id: [hyper_id, ...]} 用于快速查询某节点参与的超图边
        self._hyper_index: Dict[str, List[str]] = {}

        # 懒加载的客户端（首次使用时初始化）
        self._ollama_client = None
        self._qdrant_client = None
        self._qdrant_collection_ready = False

        # 快捷边使用阈值（一条路径被走N次后生成快捷边）
        self.shortcut_threshold = 3

    # ─────────────────────────────────────────────────
    # 客户端懒加载
    # ─────────────────────────────────────────────────

    def _get_ollama(self):
        """获取 Ollama 客户端（懒加载）"""
        if self._ollama_client is None:
            import ollama
            from config import OLLAMA_BASE_URL
            self._ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
            print(f"[Ollama] 已连接: {OLLAMA_BASE_URL}")
        return self._ollama_client

    def _get_qdrant(self):
        """获取 Qdrant 客户端（懒加载）"""
        if self._qdrant_client is None:
            from qdrant_client import QdrantClient
            from config import QDRANT_HOST, QDRANT_PORT
            self._qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            print(f"[Qdrant] 已连接: {QDRANT_HOST}:{QDRANT_PORT}")
        return self._qdrant_client

    def _ensure_qdrant_collection(self):
        """确保 Qdrant collection 存在（懒加载，只创建一次）"""
        if self._qdrant_collection_ready:
            return
        from qdrant_client.models import Distance, VectorParams
        from config import QDRANT_COLLECTION, QDRANT_VECTOR_SIZE
        client = self._get_qdrant()
        existing = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION not in existing:
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=QDRANT_VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[Qdrant] 已创建 collection: {QDRANT_COLLECTION}")
        else:
            print(f"[Qdrant] 已加载 collection: {QDRANT_COLLECTION}")
        self._qdrant_collection_ready = True

    # ─────────────────────────────────────────────────
    # 添加节点和关系
    # ─────────────────────────────────────────────────

    def add_node(self, node: MemoryNode):
        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            abstract_level=node.abstract_level,
            coverage=node.coverage,
            weight=node.weight,
        )

    def add_relation(self, relation: Relation):
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            relation_type=relation.relation_type,
            weight=relation.weight,
            is_shortcut=relation.is_shortcut,
            use_count=relation.use_count,
            relation_obj=relation,
        )

    # ─────────────────────────────────────────────────
    # Embedding（Ollama）
    # ─────────────────────────────────────────────────

    def _encode(self, text: str) -> np.ndarray:
        """用 Ollama nomic-embed-text 生成向量"""
        from config import OLLAMA_EMBED_MODEL
        client = self._get_ollama()
        response = client.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
        return np.array(response["embedding"], dtype=np.float32)

    def _get_embed_vector(self, text: str) -> np.ndarray:
        """公开的向量获取接口，供外部模块（如语义轴引擎）使用"""
        return self._encode(text)

    # ─────────────────────────────────────────────────
    # Qdrant 持久化
    # ─────────────────────────────────────────────────

    def build_vectors(self):
        """
        为所有节点生成 embedding 并写入 Qdrant。
        若节点已存在（point_id 相同）则跳过，实现增量更新。
        """
        from qdrant_client.models import PointStruct
        from config import QDRANT_COLLECTION
        self._ensure_qdrant_collection()
        client = self._get_qdrant()

        print(f"[向量引擎] 开始构建向量索引（共 {len(self.nodes)} 个节点）...")

        # 查询已存在的 point ids
        existing_ids = set()
        scroll_result = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        for point in scroll_result[0]:
            existing_ids.add(point.payload.get("node_id") if point.payload else None)
            # Qdrant point id 是整数，用 node_id 的 hash 映射
        # 重新用 payload 里的 node_id 字段来判断已存在
        existing_node_ids = set()
        scroll_result2 = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        for point in scroll_result2[0]:
            if point.payload and "node_id" in point.payload:
                existing_node_ids.add(point.payload["node_id"])

        points = []
        skipped = 0
        for node_id, node in self.nodes.items():
            if node_id in existing_node_ids:
                skipped += 1
                continue
            vec = self._encode(node.content)
            # Qdrant point id 必须是整数或 UUID，用 node_id 的绝对哈希值
            point_id = abs(hash(node_id)) % (2**53)
            payload = node.to_dict()
            payload["node_id"] = node_id  # 确保 payload 里有 node_id
            points.append(PointStruct(id=point_id, vector=vec.tolist(), payload=payload))

        if points:
            client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            print(f"[向量引擎] 写入 {len(points)} 个节点，跳过已存在 {skipped} 个")
        else:
            print(f"[向量引擎] 所有 {skipped} 个节点已存在，无需重写")

    def upsert_node_vector(self, node: MemoryNode):
        """单节点增量写入 Qdrant（新增节点时调用）"""
        from qdrant_client.models import PointStruct
        from config import QDRANT_COLLECTION
        self._ensure_qdrant_collection()
        client = self._get_qdrant()
        vec = self._encode(node.content)
        point_id = abs(hash(node.node_id)) % (2**53)
        payload = node.to_dict()
        payload["node_id"] = node.node_id
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[PointStruct(id=point_id, vector=vec.tolist(), payload=payload)],
        )

    # ─────────────────────────────────────────────────
    # 向量检索（Qdrant HNSW，O(log n)）
    # ─────────────────────────────────────────────────

    def vector_search(self, query: str, top_k: int = 5,
                      min_weight: float = 0.3) -> List[Tuple[str, float]]:
        """
        向量相似度检索，返回 [(node_id, score)]
        使用 Qdrant HNSW 索引，O(log n)
        兼容 qdrant-client >= 1.7（query_points API）
        """
        from config import QDRANT_COLLECTION, QDRANT_TOP_K_DEFAULT
        self._ensure_qdrant_collection()
        client = self._get_qdrant()

        query_vec = self._encode(query)
        # 多取一些候选，再按 min_weight 过滤
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vec.tolist(),
            limit=max(top_k * 3, QDRANT_TOP_K_DEFAULT),
            with_payload=True,
        )
        # query_points 返回 QueryResponse，points 在 .points 属性里
        hits = response.points

        results = []
        for hit in hits:
            node_id = hit.payload.get("node_id") if hit.payload else None
            if node_id is None:
                continue
            node = self.nodes.get(node_id)
            if node is None:
                continue
            if node.weight < min_weight:
                continue
            cos_sim = float(hit.score)
            weighted_score = cos_sim * node.weight
            results.append((node_id, weighted_score))
            if len(results) >= top_k:
                break

        return results

    def graph_rerank(
            self,
            candidates: List[Tuple[str, float]],
            top_k: int = 5,
            alpha: float = 0.6,
            beta: float = 0.4) -> List[Tuple[str, float]]:
        """
        T5 两阶段检索：图结构精排。

        输入：向量粗排候选集 [(node_id, vector_score), ...]
        输出：融合图连通性的重排结果 [(node_id, final_score), ...]

        算法：
          对每个候选节点 u，计算其"图内聚分"：
            graph_score(u) = (在候选集内与 u 有边相连的邻居数) / max(候选集大小-1, 1)
                           + 高优先级关系边加成（CAUSES/IS_A/PART_OF 额外+0.1/边）

          最终分数 = alpha × vector_score + beta × graph_score
          其中 alpha + beta ≈ 1，默认 alpha=0.6（向量相似度权重），beta=0.4（图结构权重）

        设计原则：
          1. 向量粗排（Qdrant HNSW）负责语义召回，图精排负责结构一致性
          2. 孤立高相似度节点（无图邻居）会被降权，避免无关节点误入推理链
          3. 图中彼此连通的节点组合得分更高，自然形成"语义+结构"双重保证
          4. 完全向后兼容：alpha=1.0 时退化为纯向量排序

        参数：
          candidates : 向量粗排结果，[(node_id, score)]
          top_k      : 返回节点数
          alpha      : 向量分数权重（0~1）
          beta       : 图结构权重（0~1）

        返回：[(node_id, final_score)]，按 final_score 降序，最多 top_k 条
        """
        if not candidates:
            return []

        candidate_ids = {nid for nid, _ in candidates}
        vec_score_map = {nid: score for nid, score in candidates}

        # 高优先级关系类型（获得额外图加成）
        high_priority_types = {
            RelationType.CAUSES,
            RelationType.IS_A,
            RelationType.PART_OF,
            RelationType.PROMOTES,
            RelationType.INHIBITS,
        }

        reranked: List[Tuple[str, float]] = []
        for nid, vec_score in candidates:
            # 统计在候选集内的图邻居
            neighbor_count = 0
            bonus = 0.0
            # 出边
            for _, tgt, data in self.graph.out_edges(nid, data=True):
                if tgt in candidate_ids and tgt != nid:
                    neighbor_count += 1
                    rel = data.get("relation_obj")
                    if rel and rel.relation_type in high_priority_types:
                        bonus += 0.1
            # 入边
            for src, _, data in self.graph.in_edges(nid, data=True):
                if src in candidate_ids and src != nid:
                    neighbor_count += 1
                    rel = data.get("relation_obj")
                    if rel and rel.relation_type in high_priority_types:
                        bonus += 0.05  # 入边加成略低于出边

            max_neighbors = max(len(candidate_ids) - 1, 1)
            graph_score = min(1.0, neighbor_count / max_neighbors + bonus)
            final_score = alpha * vec_score + beta * graph_score
            reranked.append((nid, final_score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def apply_weight_decay(self, decay_half_life_days: float = 30.0) -> int:
        """
        全局激活权重时间衰减（修复"只增不减"问题）。

        原理：
          使用 MemoryNode.effective_weight() 提供动态衰减视图，
          本方法在需要时（如长时间运行后、持久化前）将 effective_weight
          写回 node.weight，实现永久性衰减（适用于长期部署场景）。

          注意：
            - 推理排序时应直接调用 node.effective_weight()（不修改原始权重）
            - 本方法仅在需要"永久降权"时调用，如系统重启前/每周定期维护
            - FALSE/PARADOX_PENDING 节点跳过（其权重已由状态控制）
            - 自动存储（auto_store）生成的推断节点（inferred_*）正常参与衰减

        参数：
          decay_half_life_days : 衰减半衰期，与 effective_weight() 保持一致

        返回：
          实际衰减的节点数（effective_weight < weight × 0.95 的节点数）
        """
        decayed_count = 0
        for node in self.nodes.values():
            if node.epistemic_status.value in ("false", "paradox_pending"):
                continue
            eff = node.effective_weight(decay_half_life_days)
            if eff < node.weight * 0.95:
                old_w = node.weight
                node.weight = max(0.1, eff)
                # 不写入 validation_history，避免污染人工验证记录
                decayed_count += 1
        return decayed_count

    # ─────────────────────────────────────────────────
    # 持久化：保存/加载图谱结构（节点关系边）
    # ─────────────────────────────────────────────────

    def save_graph(self, path: str = "graph_state.json"):
        """将图谱的节点和关系序列化到 JSON 文件（向量存 Qdrant，图结构存文件）"""
        data = {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [],
            "hyper_edges": [he.to_dict() for he in self.hyper_edges.values()],
            "shortcut_counts": getattr(self, "_shortcut_counts", {}),
        }
        for src, tgt, edata in self.graph.edges(data=True):
            rel = edata.get("relation_obj")
            if rel:
                data["edges"].append({
                    "source": src,
                    "target": tgt,
                    "relation_type": rel.relation_type.value,
                    "weight": rel.weight,
                    "is_shortcut": rel.is_shortcut,
                    "use_count": rel.use_count,
                    "context": rel.context,
                })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[图谱] 已保存到 {path}（{len(data['nodes'])} 节点，"
              f"{len(data['edges'])} 二元边，"
              f"{len(data['hyper_edges'])} 超图边）")

    def load_graph(self, path: str = "graph_state.json") -> bool:
        """从 JSON 文件加载图谱结构（向量从 Qdrant 恢复检索能力）"""
        import os
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for nid, nd in data["nodes"].items():
            node = MemoryNode(
                node_id=nd["node_id"],
                content=nd["content"],
                abstract_level=nd["abstract_level"],
                domain=nd["domain"],
                coverage=nd["coverage"],
                essence_features=nd.get("essence_features", []),
                tags=nd.get("tags", []),
                weight=nd.get("weight", 1.0),
                # P0-2修复：恢复认识论状态及相关字段，防止重启后状态归零
                epistemic_status=EpistemicStatus(
                    nd.get("epistemic_status", "confirmed")),
                trigger_conditions=nd.get("trigger_conditions", []),
                source_trust=nd.get("source_trust", {}),
                refutation_reason=nd.get("refutation_reason", None),
            )
            # 恢复 activation_count 和 last_activated_at（不在构造函数参数里，单独赋值）
            node.activation_count = nd.get("activation_count", 0)
            node.last_activated_at = nd.get("last_activated_at", node.created_at)
            self.add_node(node)

        for ed in data["edges"]:
            rel = Relation(
                source_id=ed["source"],
                target_id=ed["target"],
                relation_type=RelationType(ed["relation_type"]),
                weight=ed["weight"],
                is_shortcut=ed.get("is_shortcut", False),
                use_count=ed.get("use_count", 0),
                context=ed.get("context", ""),
            )
            self.add_relation(rel)

        # 加载超图边
        for hed in data.get("hyper_edges", []):
            try:
                he = HyperEdge(
                    co_nodes=hed["co_nodes"],
                    target_ids=hed["target_ids"],
                    relation_type=RelationType(hed["relation_type"]),
                    weight=hed.get("weight", 1.0),
                    condition=hed.get("condition", "AND"),
                    context=hed.get("context", ""),
                    hyper_id=hed["hyper_id"],
                    created_at=hed.get("created_at", time.time()),
                )
                self.add_hyper_edge(he)
            except Exception as e:
                print(f"[图谱] 超图边加载失败（跳过）: {e}")

        self._shortcut_counts = data.get("shortcut_counts", {})
        print(f"[图谱] 已从 {path} 加载（{len(self.nodes)} 节点，"
              f"{self.graph.number_of_edges()} 二元边，"
              f"{len(self.hyper_edges)} 超图边）")
        return True

    # ─────────────────────────────────────────────────
    # 图谱检索
    # ─────────────────────────────────────────────────

    def get_neighbors(self, node_id: str,
                      relation_filter: Optional[List[RelationType]] = None,
                      direction: str = "out") -> List[Tuple[str, Relation]]:
        """
        获取邻居节点，按元关系优先级 + 节点权重排序
        direction: "out"=出边, "in"=入边, "both"=双向
        """
        results = []

        if direction in ("out", "both"):
            for _, target, data in self.graph.out_edges(node_id, data=True):
                rel = data.get("relation_obj")
                if rel and (relation_filter is None or
                            rel.relation_type in relation_filter):
                    results.append((target, rel))

        if direction in ("in", "both"):
            for source, _, data in self.graph.in_edges(node_id, data=True):
                rel = data.get("relation_obj")
                if rel and (relation_filter is None or
                            rel.relation_type in relation_filter):
                    results.append((source, rel))

        def sort_key(item):
            target_id, rel = item
            priority = RELATION_PRIORITY.get(rel.relation_type, 9)
            target_weight = self.nodes.get(target_id, MemoryNode(
                target_id, "", 0, [], 0)).weight
            return (priority, -target_weight)

        results.sort(key=sort_key)
        return results

    def get_abstract_ancestors(self, node_id: str,
                                max_depth: int = 4) -> List[Tuple[str, int]]:
        """向上追溯抽象层级"""
        visited = []
        current_level = [(node_id, 0)]

        for depth in range(1, max_depth + 1):
            next_level = []
            for nid, _ in current_level:
                neighbors = self.get_neighbors(
                    nid,
                    relation_filter=[RelationType.BELONGS_TO,
                                     RelationType.EVOLVED_FROM,
                                     RelationType.DERIVED_FROM],
                    direction="out"
                )
                for target_id, rel in neighbors:
                    if target_id not in [v[0] for v in visited]:
                        next_level.append((target_id, depth))
                        visited.append((target_id, depth))
            if not next_level:
                break
            current_level = next_level

        return visited

    # ─────────────────────────────────────────────────
    # 快捷边机制
    # ─────────────────────────────────────────────────

    def record_path_usage(self, path: List[str]):
        """
        记录一条推理路径的使用，达到阈值则生成快捷边。
        快捷边类型 = 路径中出现最多的高优先级关系类型，而非硬编码 ANALOGOUS_TO。
        优先级（数字越小越高）：1→CAUSES/PROMOTES/INHIBITS, 2→..., 最后才是 ANALOGOUS_TO。
        """
        if len(path) < 3:
            return

        source = path[0]
        target = path[-1]

        if self.graph.has_edge(source, target):
            data = self.graph[source][target]
            if data.get("is_shortcut"):
                data["use_count"] = data.get("use_count", 0) + 1
                return

        shortcut_key = f"__shortcut_{source}__{target}"
        if not hasattr(self, "_shortcut_counts"):
            self._shortcut_counts = {}
        # 记录路径上各边类型（用于选快捷边类型）
        if not hasattr(self, "_shortcut_rel_types"):
            self._shortcut_rel_types: Dict[str, Dict] = {}

        # 统计本次路径中各边的关系类型
        type_counter: Dict = self._shortcut_rel_types.setdefault(shortcut_key, {})
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                rel_obj = self.graph[u][v].get("relation_obj")
                if rel_obj:
                    rt = rel_obj.relation_type
                    type_counter[rt] = type_counter.get(rt, 0) + 1

        self._shortcut_counts[shortcut_key] = \
            self._shortcut_counts.get(shortcut_key, 0) + 1

        if self._shortcut_counts[shortcut_key] >= self.shortcut_threshold:
            # 选最高频的关系类型，若同频则选优先级最高（数字最小）的
            best_type = RelationType.ANALOGOUS_TO  # 兜底
            if type_counter:
                best_type = min(
                    type_counter,
                    key=lambda rt: (
                        -type_counter[rt],           # 频次降序
                        RELATION_PRIORITY.get(rt, 9)  # 优先级升序
                    )
                )
            shortcut = Relation(
                source_id=source,
                target_id=target,
                relation_type=best_type,
                weight=0.8,
                is_shortcut=True,
                use_count=self.shortcut_threshold,
            )
            self.add_relation(shortcut)
            print(f"[快捷边] 生成: {source} → {target} [{best_type.value}]")

    # ─────────────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────────────

    # ─────────────────────────────────────────────────
    # 超图边管理
    # ─────────────────────────────────────────────────

    def add_hyper_edge(self, hyper_edge: HyperEdge) -> str:
        """
        添加一条超图边。
        同时维护 _hyper_index（倒排索引），使按节点查超图边 O(1)。
        返回 hyper_id。
        """
        hid = hyper_edge.hyper_id
        self.hyper_edges[hid] = hyper_edge

        # 更新倒排索引：co_nodes + target_ids 均参与索引
        all_related = set(hyper_edge.co_nodes) | set(hyper_edge.target_ids)
        for nid in all_related:
            if nid not in self._hyper_index:
                self._hyper_index[nid] = []
            if hid not in self._hyper_index[nid]:
                self._hyper_index[nid].append(hid)

        return hid

    def get_hyper_edges_by_node(self, node_id: str) -> List[HyperEdge]:
        """
        获取某节点参与的所有超图边（无论是作为 co_nodes 还是 target_ids）。
        """
        hids = self._hyper_index.get(node_id, [])
        return [self.hyper_edges[hid] for hid in hids if hid in self.hyper_edges]

    def get_hyper_edges_where_source(self, node_id: str) -> List[HyperEdge]:
        """获取 node_id 作为条件节点（co_nodes 之一）的超图边。"""
        return [
            he for he in self.get_hyper_edges_by_node(node_id)
            if node_id in he.co_nodes
        ]

    def get_hyper_edges_where_target(self, node_id: str) -> List[HyperEdge]:
        """获取 node_id 作为结论节点（target_ids 之一）的超图边。"""
        return [
            he for he in self.get_hyper_edges_by_node(node_id)
            if node_id in he.target_ids
        ]

    def check_hyper_conditions(
            self, hyper_edge: HyperEdge,
            active_nodes: List[str],
            min_weight: float = 0.3) -> Tuple[bool, float]:
        """
        检查超图边的条件节点是否在 active_nodes 中满足激活要求。

        参数：
          hyper_edge   : 待检查的超图边
          active_nodes : 当前推理激活的节点列表
          min_weight   : 条件节点的最小权重（太低的节点不算"有效激活"）

        返回：
          (conditions_met: bool, satisfaction_ratio: float)
            AND 模式：所有 co_nodes 均在 active_nodes 且 weight≥min_weight → True
            OR  模式：至少一个 co_node 在 active_nodes 且 weight≥min_weight → True
            satisfaction_ratio：满足条件的 co_nodes 比例（0.0 ~ 1.0）
        """
        active_set = set(active_nodes)
        satisfied = 0
        for nid in hyper_edge.co_nodes:
            node = self.nodes.get(nid)
            if node and node.weight >= min_weight and nid in active_set:
                satisfied += 1

        total = max(len(hyper_edge.co_nodes), 1)
        ratio = satisfied / total

        if hyper_edge.condition == "AND":
            return (satisfied == total), ratio
        else:  # "OR"
            return (satisfied >= 1), ratio

    def find_triggered_hyper_edges(
            self, active_nodes: List[str],
            min_satisfaction: float = 1.0) -> List[Tuple[HyperEdge, float]]:
        """
        在当前激活节点集合中，找出所有条件满足的超图边。
        用于联想推理的"联合条件发现"步骤。

        参数：
          active_nodes     : 当前推理激活的节点列表
          min_satisfaction : 最小满足比例（AND模式固定1.0，OR模式可低至1/N）

        返回：[(HyperEdge, satisfaction_ratio), ...]，按 ratio×weight 降序
        """
        active_set = set(active_nodes)
        results: List[Tuple[HyperEdge, float]] = []

        # 只检查至少有一个 co_node 在激活集中的超图边（利用倒排索引加速）
        candidate_hids: set = set()
        for nid in active_nodes:
            for hid in self._hyper_index.get(nid, []):
                he = self.hyper_edges.get(hid)
                if he and any(cn in active_set for cn in he.co_nodes):
                    candidate_hids.add(hid)

        for hid in candidate_hids:
            he = self.hyper_edges[hid]
            conditions_met, ratio = self.check_hyper_conditions(he, active_nodes)
            if ratio >= min_satisfaction or conditions_met:
                results.append((he, ratio))

        results.sort(key=lambda x: x[1] * x[0].weight, reverse=True)
        return results

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        return self.nodes.get(node_id)

    def summary(self):
        he_count = len(self.hyper_edges)
        print(f"[记忆网络] 节点数: {len(self.nodes)}  "
              f"二元边数: {self.graph.number_of_edges()}  "
              f"超图边数: {he_count}")
