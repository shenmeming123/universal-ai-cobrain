"""
改进1：向量方向关系检测器
核心思想：两节点向量之差（方向）编码了它们之间的关系信息
类似 word2vec 的 "king - man + woman = queen"

流程：
  1. 从已知关系对中学习每种关系类型的"典型方向向量"
  2. 对新节点对，计算向量差，与各关系类型典型方向做余弦相似度
  3. 相似度最高的关系类型 -> 候选关系标注
  4. 低置信度的候选关系标记为"待验证"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from relation_types import RelationType, Relation


@dataclass
class RelationCandidate:
    """向量方向推断出的候选关系"""
    source_id: str
    target_id: str
    relation_type: RelationType
    confidence: float           # 余弦相似度得分
    is_verified: bool = False   # 是否已通过人工/规则验证
    evidence_count: int = 1     # 支持该方向的样本数


class RelationDetector:
    """
    向量方向关系检测器
    """

    # 置信度阈值
    HIGH_CONFIDENCE = 0.75    # 高置信度：自动标注
    LOW_CONFIDENCE  = 0.45    # 低置信度：标记待验证
    # 低于 LOW_CONFIDENCE 的候选直接丢弃

    def __init__(self, memory_network):
        self.net = memory_network
        # 每种关系类型的方向向量样本池 {RelationType: [delta_vec, ...]}
        self._direction_samples: Dict[RelationType, List[np.ndarray]] = defaultdict(list)
        # 每种关系类型的平均方向向量（归一化）{RelationType: unit_vec}
        self._direction_centroids: Dict[RelationType, np.ndarray] = {}
        self._is_trained = False

    # ─────────────────────────────────────────
    # 训练：从已知关系对学习方向
    # ─────────────────────────────────────────

    def train_from_existing_relations(self, verbose: bool = True) -> Dict[str, int]:
        """
        从记忆网络中所有已有关系对提取方向向量，构建典型方向库
        :return: {relation_type_name: sample_count}
        """
        if verbose:
            print("\n[关系检测器] 开始从已有关系学习方向向量...")

        if not self.net.vectors:
            self.net.build_vectors()

        count_by_type: Dict[str, int] = defaultdict(int)

        for source_id, target_id, data in self.net.graph.edges(data=True):
            rel_obj = data.get("relation_obj")
            if not rel_obj:
                continue
            if rel_obj.is_shortcut:  # 跳过快捷边
                continue

            src_vec = self.net.vectors.get(source_id)
            tgt_vec = self.net.vectors.get(target_id)
            if src_vec is None or tgt_vec is None:
                continue

            # 方向向量 = target - source
            delta = tgt_vec - src_vec
            norm = np.linalg.norm(delta)
            if norm < 1e-6:  # 向量太近，忽略
                continue

            delta_unit = delta / norm
            self._direction_samples[rel_obj.relation_type].append(delta_unit)
            count_by_type[rel_obj.relation_type.value] += 1

        # 计算每种关系类型的质心方向
        self._direction_centroids = {}
        for rel_type, samples in self._direction_samples.items():
            if len(samples) >= 1:
                centroid = np.mean(samples, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-6:
                    self._direction_centroids[rel_type] = centroid / norm

        self._is_trained = True

        if verbose:
            print(f"  学习完成，覆盖 {len(self._direction_centroids)} 种关系类型")
            for rt, cnt in count_by_type.items():
                print(f"  · {rt}: {cnt} 个样本")

        return dict(count_by_type)

    # ─────────────────────────────────────────
    # 推断：对新节点对预测关系类型
    # ─────────────────────────────────────────

    def detect_relation(self,
                        source_id: str,
                        target_id: str,
                        top_k: int = 3,
                        verbose: bool = False) -> List[RelationCandidate]:
        """
        给定两个节点ID，推断它们之间最可能的关系类型
        :return: 按置信度排序的候选关系列表
        """
        if not self._is_trained:
            self.train_from_existing_relations(verbose=False)

        src_vec = self.net.vectors.get(source_id)
        tgt_vec = self.net.vectors.get(target_id)

        if src_vec is None or tgt_vec is None:
            return []

        delta = tgt_vec - src_vec
        norm = np.linalg.norm(delta)
        if norm < 1e-6:
            return []
        delta_unit = delta / norm

        # 与每种关系类型的质心做余弦相似度
        scores: List[Tuple[RelationType, float]] = []
        for rel_type, centroid in self._direction_centroids.items():
            cos_sim = float(np.dot(delta_unit, centroid))
            # 余弦相似度范围 [-1, 1]，归一化到 [0, 1]
            normalized = (cos_sim + 1) / 2
            scores.append((rel_type, normalized))

        scores.sort(key=lambda x: x[1], reverse=True)

        candidates = []
        for rel_type, conf in scores[:top_k]:
            if conf < self.LOW_CONFIDENCE:
                break
            sample_count = len(self._direction_samples.get(rel_type, []))
            candidates.append(RelationCandidate(
                source_id=source_id,
                target_id=target_id,
                relation_type=rel_type,
                confidence=conf,
                is_verified=(conf >= self.HIGH_CONFIDENCE and sample_count >= 3),
                evidence_count=sample_count,
            ))

        if verbose and candidates:
            src_node = self.net.get_node(source_id)
            tgt_node = self.net.get_node(target_id)
            src_name = src_node.content[:20] if src_node else source_id
            tgt_name = tgt_node.content[:20] if tgt_node else target_id
            print(f"\n[关系检测] {src_name} -> {tgt_name}")
            for c in candidates:
                status = "AUTO" if c.is_verified else "PENDING"
                print(f"  [{status}] {c.relation_type.value}: {c.confidence:.3f}")

        return candidates

    def scan_all_unlinked_pairs(self,
                                 min_vector_similarity: float = 0.5,
                                 verbose: bool = True) -> List[RelationCandidate]:
        """
        扫描所有尚未有关系边的节点对，推断潜在关系
        只扫描向量相似度足够高的对（减少计算量）
        :return: 所有候选关系列表
        """
        if not self._is_trained:
            self.train_from_existing_relations(verbose=False)

        if not self.net.vectors:
            self.net.build_vectors()

        if verbose:
            print(f"\n[关系检测器] 扫描未链接节点对...")

        node_ids = list(self.net.nodes.keys())
        all_candidates = []

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                nid_a = node_ids[i]
                nid_b = node_ids[j]

                # 跳过已有关系的对
                if self.net.graph.has_edge(nid_a, nid_b):
                    continue
                if self.net.graph.has_edge(nid_b, nid_a):
                    continue

                # 预筛：向量相似度
                va = self.net.vectors.get(nid_a)
                vb = self.net.vectors.get(nid_b)
                if va is None or vb is None:
                    continue
                sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
                if sim < min_vector_similarity:
                    continue

                # 推断关系（A->B 和 B->A 两个方向）
                cands_ab = self.detect_relation(nid_a, nid_b, top_k=1)
                cands_ba = self.detect_relation(nid_b, nid_a, top_k=1)

                all_candidates.extend(cands_ab)
                all_candidates.extend(cands_ba)

        # 按置信度排序
        all_candidates.sort(key=lambda c: c.confidence, reverse=True)

        if verbose:
            auto = sum(1 for c in all_candidates if c.is_verified)
            pending = sum(1 for c in all_candidates if not c.is_verified)
            print(f"  发现候选关系 {len(all_candidates)} 条: 自动标注={auto}, 待验证={pending}")

        return all_candidates

    def apply_candidates(self,
                          candidates: List[RelationCandidate],
                          auto_only: bool = True,
                          verbose: bool = True) -> int:
        """
        将候选关系写入记忆网络
        :param auto_only: True=只写入高置信度(is_verified)的，False=全部写入
        :return: 实际写入的关系数量
        """
        added = 0
        for cand in candidates:
            if auto_only and not cand.is_verified:
                continue
            if self.net.graph.has_edge(cand.source_id, cand.target_id):
                continue  # 已存在，跳过

            rel = Relation(
                source_id=cand.source_id,
                target_id=cand.target_id,
                relation_type=cand.relation_type,
                weight=cand.confidence,
                context=f"auto_detected:conf={cand.confidence:.2f}",
            )
            self.net.add_relation(rel)
            added += 1

            if verbose:
                src = self.net.get_node(cand.source_id)
                tgt = self.net.get_node(cand.target_id)
                sn = src.content[:20] if src else cand.source_id
                tn = tgt.content[:20] if tgt else cand.target_id
                print(f"  [写入] {sn} -[{cand.relation_type.value}]-> {tn} (conf={cand.confidence:.2f})")

        return added

    # ─────────────────────────────────────────
    # 统计
    # ─────────────────────────────────────────

    def training_stats(self) -> Dict:
        return {
            rel_type.value: len(samples)
            for rel_type, samples in self._direction_samples.items()
        }
