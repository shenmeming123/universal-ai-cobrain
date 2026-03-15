"""
知识爬虫 + LLM 驱动的结构转换管道
功能：
  1. 爬取指定 URL 或关键词搜索结果的正文
  2. 用 LLM 将非结构化文本转换为 MemoryNode + Relation 结构
  3. 结构校验（字段完整性保障）+ 与现有节点的关系推断
  4. 经 ConflictResolver 冲突检测后写入知识库

关于"如何保障转换后的信息完整性"的设计决策：
  - JSON Schema 强制输出：LLM 必须按固定格式输出，缺字段即重试
  - 字段范围校验：abstract_level(1-10)、coverage(0-1)、domain/essence_features 非空等
  - 两阶段关系推断：先出节点，再专门向 LLM 询问节点对之间的关系
  - 与现有节点关联：向量检索相似节点，LLM 判断关系方向和类型
  - 最多重试3次，失败记录到日志供人工审查

用法：
  from knowledge_crawler import KnowledgeCrawler
  crawler = KnowledgeCrawler(net)
  crawler.crawl_url("https://example.com/article")
  crawler.crawl_keyword("神经可塑性 进化心理学")
"""

import re
import json
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from memory_network import MemoryNetwork
from memory_node import MemoryNode
from relation_types import Relation, RelationType
from conflict_resolver import ConflictResolver, NewInformation

logging.basicConfig(
    level=logging.INFO,
    format="[爬虫] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crawler")


# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────

VALID_RELATION_TYPES = {rt.value: rt for rt in RelationType}

# LLM 输出的节点 schema（用于校验）
NODE_SCHEMA_REQUIRED = {
    "node_id": str,
    "content": str,
    "abstract_level": int,
    "domain": list,
    "coverage": float,
    "essence_features": list,
    "tags": list,
}

NODE_SCHEMA_CONSTRAINTS = {
    "abstract_level": (1, 10),
    "coverage": (0.0, 1.0),
    "domain": {"min_len": 1},
    "essence_features": {"min_len": 1},
    "content": {"min_chars": 10},
}

@dataclass
class CrawlResult:
    url: str
    nodes_added: int
    relations_added: int
    nodes_skipped: int       # 校验失败或冲突拦截
    errors: List[str] = field(default_factory=list)
    elapsed_sec: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 爬虫主类
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeCrawler:
    """
    知识爬虫：网页正文 → MemoryNode + Relation → 写入知识库
    """

    MAX_RETRIES = 3
    # 每次爬取最多提取的节点数（避免单篇文章产生过多低质节点）
    MAX_NODES_PER_PAGE = 8
    # 与现有节点建立关系时，向量检索的候选数
    RELATION_SEARCH_TOP_K = 5

    def __init__(self, memory_network: MemoryNetwork,
                 source_tag: str = "crawler"):
        self.net = memory_network
        self.source_tag = source_tag
        self._ollama_client = None

    # ─────────────────────────────────────────────────────────────
    # 公开入口
    # ─────────────────────────────────────────────────────────────

    def crawl_url(self, url: str, verbose: bool = True) -> CrawlResult:
        """爬取单个 URL，提取正文并写入知识库"""
        start = time.time()
        result = CrawlResult(url=url, nodes_added=0, relations_added=0, nodes_skipped=0)

        logger.info(f"开始爬取: {url}")
        try:
            raw_text = self._fetch_url(url)
        except Exception as e:
            result.errors.append(f"网页获取失败: {e}")
            logger.error(f"获取失败: {e}")
            return result

        self._process_text(raw_text, source_url=url, result=result, verbose=verbose)
        result.elapsed_sec = time.time() - start
        logger.info(
            f"完成 {url} | 新增节点:{result.nodes_added} 关系:{result.relations_added} "
            f"跳过:{result.nodes_skipped} 耗时:{result.elapsed_sec:.1f}s"
        )
        return result

    def crawl_keyword(self, keyword: str, max_pages: int = 3,
                      verbose: bool = True) -> List[CrawlResult]:
        """
        用关键词搜索，爬取前 N 条结果。
        使用 DuckDuckGo（无需 API Key）或百度搜索。
        """
        logger.info(f"关键词搜索: '{keyword}' (最多 {max_pages} 页)")
        urls = self._search_urls(keyword, max_pages)
        logger.info(f"找到 {len(urls)} 个 URL")

        results = []
        for url in urls:
            r = self.crawl_url(url, verbose=verbose)
            results.append(r)
            time.sleep(1.5)  # 礼貌性延迟，避免被封

        return results

    def crawl_text(self, text: str, source_label: str = "手动输入",
                   verbose: bool = True) -> CrawlResult:
        """
        直接传入文本（不爬网页），用于手动批量导入。
        适合将已有文档/笔记批量转换入库。
        """
        result = CrawlResult(
            url=source_label, nodes_added=0, relations_added=0, nodes_skipped=0
        )
        self._process_text(text, source_url=source_label, result=result, verbose=verbose)
        return result

    # ─────────────────────────────────────────────────────────────
    # 网页获取
    # ─────────────────────────────────────────────────────────────

    def _fetch_url(self, url: str) -> str:
        """获取网页正文（优先 requests+BeautifulSoup，降级为 urllib）"""
        try:
            import requests
            from bs4 import BeautifulSoup
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
                )
            }
            resp = requests.get(url, headers=headers, timeout=15)
            resp.encoding = resp.apparent_encoding
            soup = BeautifulSoup(resp.text, "html.parser")

            # 移除脚本、样式、导航等噪音标签
            for tag in soup(["script", "style", "nav", "footer",
                              "header", "aside", "iframe", "form"]):
                tag.decompose()

            # 优先取 <article> / <main>，否则取 <body>
            main = soup.find("article") or soup.find("main") or soup.find("body")
            text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")

            # 清理多余空白
            lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 20]
            return "\n".join(lines[:200])  # 最多200行，约6000字

        except ImportError:
            # 降级：urllib（无 BeautifulSoup）
            import urllib.request
            with urllib.request.urlopen(url, timeout=15) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            # 简单去除 HTML 标签
            raw = re.sub(r"<[^>]+>", " ", raw)
            raw = re.sub(r"\s+", " ", raw)
            return raw[:8000]

    def _search_urls(self, keyword: str, max_pages: int) -> List[str]:
        """
        用 DuckDuckGo HTML 接口搜索，返回 URL 列表。
        无需 API Key，适合本地使用。
        """
        urls = []
        try:
            import requests
            from bs4 import BeautifulSoup

            search_url = f"https://html.duckduckgo.com/html/?q={keyword.replace(' ', '+')}"
            headers = {"User-Agent": "Mozilla/5.0 (compatible; KnowledgeCrawler/1.0)"}
            resp = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")

            for a in soup.select(".result__a"):
                href = a.get("href", "")
                if href.startswith("http") and href not in urls:
                    urls.append(href)
                    if len(urls) >= max_pages:
                        break

        except Exception as e:
            logger.warning(f"搜索失败（{e}），尝试备用方案")
            # 备用：直接用百度
            try:
                import requests
                search_url = (
                    f"https://www.baidu.com/s?wd={keyword.replace(' ', '+')}"
                    f"&rn={max_pages * 2}"
                )
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(search_url, headers=headers, timeout=10)
                # 简单提取 href
                found = re.findall(r'href="(https?://[^"]+)"', resp.text)
                for u in found:
                    if "baidu.com" not in u and u not in urls:
                        urls.append(u)
                        if len(urls) >= max_pages:
                            break
            except Exception as e2:
                logger.error(f"备用搜索也失败: {e2}")

        return urls[:max_pages]

    # ─────────────────────────────────────────────────────────────
    # 核心：文本 → 结构化节点
    # ─────────────────────────────────────────────────────────────

    def _process_text(self, text: str, source_url: str,
                      result: CrawlResult, verbose: bool):
        """文本 → 节点提取 → 关系推断 → 冲突检测 → 写入"""
        if not text or len(text) < 50:
            result.errors.append("文本过短，跳过")
            return

        # 第一阶段：提取节点（带重试+校验）
        raw_nodes = self._extract_nodes_with_retry(text, source_url)
        if not raw_nodes:
            result.errors.append("LLM 提取节点失败")
            return

        if verbose:
            print(f"\n[爬虫] 从文本提取到 {len(raw_nodes)} 个候选节点")

        # 第二阶段：为每个节点推断与现有库的关系
        valid_nodes: List[Tuple[MemoryNode, List[Relation]]] = []
        for raw in raw_nodes:
            node = self._build_node(raw, source_url)
            if node is None:
                result.nodes_skipped += 1
                continue
            relations = self._infer_relations(node, raw_nodes, verbose)
            valid_nodes.append((node, relations))

        # 第三阶段：经 ConflictResolver 写入
        resolver = ConflictResolver(self.net)
        for node, relations in valid_nodes:
            # 检查是否已存在同 node_id
            if self.net.get_node(node.node_id):
                if verbose:
                    print(f"  [跳过] 节点已存在: {node.node_id}")
                result.nodes_skipped += 1
                continue

            new_info = NewInformation(
                node_id=node.node_id,
                content=node.content,
                abstract_level=node.abstract_level,
                domain=node.domain,
                coverage=node.coverage,
                essence_features=node.essence_features,
                tags=node.tags,
                evidence_strength=0.75,  # 爬取内容默认中等可信度
                source=self.source_tag,
                proposed_relations=relations,
            )
            report = resolver.process(new_info, verbose=verbose)

            from conflict_resolver import ConflictLevel
            if report.level == ConflictLevel.PARADOX:
                result.nodes_skipped += 1
                if verbose:
                    print(f"  [拦截] {node.node_id} 触发悖论，已进入人工审查池")
            else:
                # 同步向量到 Qdrant
                stored = self.net.get_node(node.node_id)
                if stored:
                    try:
                        self.net.upsert_node_vector(stored)
                        result.nodes_added += 1
                        result.relations_added += len(relations)
                        if verbose:
                            print(f"  [写入] {node.node_id} ({node.content[:40]})")
                    except Exception as e:
                        result.errors.append(f"向量写入失败 {node.node_id}: {e}")

    # ─────────────────────────────────────────────────────────────
    # LLM：提取节点（结构化 + 校验 + 重试）
    # ─────────────────────────────────────────────────────────────

    def _extract_nodes_with_retry(self, text: str,
                                   source_url: str) -> List[Dict]:
        """
        调用 LLM 提取节点，最多重试 MAX_RETRIES 次。
        每次重试会在 prompt 中附上上次失败的错误信息，引导 LLM 修正。
        """
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                raw = self._call_llm_extract(text, source_url, last_error)
                nodes = self._parse_and_validate_nodes(raw)
                if nodes:
                    logger.info(f"第{attempt+1}次尝试成功，提取 {len(nodes)} 个节点")
                    return nodes
                else:
                    last_error = "输出的 JSON 为空或所有节点校验失败"
            except Exception as e:
                last_error = str(e)
                logger.warning(f"第{attempt+1}次提取失败: {e}")
                time.sleep(1)
        logger.error(f"提取节点失败，已重试 {self.MAX_RETRIES} 次")
        return []

    def _call_llm_extract(self, text: str, source_url: str,
                          last_error: Optional[str]) -> str:
        """构造提取 prompt，调用 Ollama"""
        from config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL
        import ollama

        # 关系类型提示（告诉 LLM 有哪些合法值）
        rel_types = "、".join([rt.value for rt in RelationType])

        error_hint = ""
        if last_error:
            error_hint = f"\n\n上次输出有误（{last_error}），请修正后重新输出。"

        prompt = f"""你是一个知识图谱构建助手。请从以下文章中提取 3-{self.MAX_NODES_PER_PAGE} 个核心知识概念，
将每个概念转换为结构化的知识节点，并以 JSON 数组格式输出。

## 输出格式要求（必须严格遵守）
输出一个 JSON 数组，每个元素包含以下字段（不得缺少）：
{{
  "node_id": "概念名称（简短中文，作为唯一ID，如'神经可塑性'）",
  "content": "一句完整的陈述句，描述该概念的本质（≥15个汉字）",
  "abstract_level": 整数1-10（1=最具体实例，10=最抽象原理，如'某实验'=2，'进化'=8）,
  "domain": ["领域1", "领域2"],（如["神经科学","心理学"]，至少1个）
  "coverage": 0到1之间的小数（该概念的外延广度，越基础的概念越高，如'生物'=0.9，'杏仁核'=0.6）,
  "essence_features": ["特征1", "特征2"],（2-4个核心属性词，用于判断概念间的语义极性）,
  "tags": ["标签1", "标签2"]（3-6个检索关键词）
}}

## 关系字段（可选，如能判断则加上）
每个节点可附加 "relations" 数组，描述该节点与其他提取节点的关系：
{{
  "target": "目标节点的node_id",
  "type": "关系类型（从以下选择）",
  "weight": 0.7
}}
合法关系类型：{rel_types}

## 注意事项
- 只提取文章中实质性的知识概念，不提取人名、书名等专有名词实体
- content 必须是对该概念本质的陈述，不是对文章的描述
- essence_features 要体现概念的核心属性极性（如"增加"/"减少"，"先天"/"后天"），用于冲突检测
- 直接输出 JSON 数组，不要有任何解释文字{error_hint}

## 文章来源
{source_url}

## 文章内容
{text[:4000]}

## 输出（JSON数组）："""

        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.generate(
            model=OLLAMA_LLM_MODEL,
            prompt=prompt,
            options={"temperature": 0.1, "num_predict": 2048},
        )
        answer = response.get("response", "").strip()
        # 过滤 think 标签
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        return answer

    def _parse_and_validate_nodes(self, llm_output: str) -> List[Dict]:
        """
        解析 LLM 输出的 JSON，逐字段校验。
        校验失败的节点记录警告并跳过，不影响其他节点。
        这是保障信息完整性的关键环节。
        """
        # 提取 JSON 数组（LLM 可能在前后加了说明文字）
        json_match = re.search(r'\[.*\]', llm_output, re.DOTALL)
        if not json_match:
            raise ValueError(f"输出中未找到 JSON 数组，原始输出前200字: {llm_output[:200]}")

        try:
            nodes = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败: {e}")

        valid = []
        for i, node in enumerate(nodes):
            errors = self._validate_node_dict(node, index=i)
            if errors:
                logger.warning(f"节点[{i}] '{node.get('node_id','?')}' 校验失败: {'; '.join(errors)}")
                continue
            valid.append(node)

        return valid[:self.MAX_NODES_PER_PAGE]

    def _validate_node_dict(self, node: Dict, index: int) -> List[str]:
        """
        字段完整性 + 范围校验。
        返回错误列表，空列表表示通过。
        """
        errors = []

        # 必填字段类型检查
        for field_name, expected_type in NODE_SCHEMA_REQUIRED.items():
            if field_name not in node:
                errors.append(f"缺少字段 '{field_name}'")
                continue
            if not isinstance(node[field_name], expected_type):
                errors.append(
                    f"'{field_name}' 类型错误，期望 {expected_type.__name__}，"
                    f"实际 {type(node[field_name]).__name__}"
                )

        if errors:  # 基础类型错误直接返回，后续约束无法检查
            return errors

        # 范围约束
        al = node["abstract_level"]
        if not (1 <= al <= 10):
            errors.append(f"abstract_level={al} 超出范围 [1,10]")

        cov = node["coverage"]
        if not (0.0 <= cov <= 1.0):
            errors.append(f"coverage={cov} 超出范围 [0,1]")

        if len(node["domain"]) < 1:
            errors.append("domain 不能为空列表")

        if len(node["essence_features"]) < 1:
            errors.append("essence_features 不能为空列表")

        if len(node["content"]) < 10:
            errors.append(f"content 过短（{len(node['content'])}字）")

        # node_id 不能含特殊字符（用作字典 key）
        nid = node["node_id"]
        if not nid or len(nid) > 30:
            errors.append(f"node_id 为空或过长: '{nid}'")

        return errors

    # ─────────────────────────────────────────────────────────────
    # 关系推断（两阶段）
    # ─────────────────────────────────────────────────────────────

    def _infer_relations(self, node: MemoryNode,
                         raw_nodes: List[Dict],
                         verbose: bool) -> List[Relation]:
        """
        两阶段关系推断：
        阶段1：批文内关系（同一篇文章提取的节点之间）——从 LLM 输出的 relations 字段读取
        阶段2：与现有库的关系——向量检索相似节点，LLM 判断关系类型
        """
        relations = []

        # 阶段1：文内关系（已由 LLM 在 raw 里生成）
        for raw in raw_nodes:
            if raw.get("node_id") != node.node_id:
                continue
            for rel_dict in raw.get("relations", []):
                target_id = rel_dict.get("target", "")
                rel_type_str = rel_dict.get("type", "")
                weight = float(rel_dict.get("weight", 0.7))
                rt = VALID_RELATION_TYPES.get(rel_type_str)
                if rt and target_id:
                    relations.append(Relation(
                        source_id=node.node_id,
                        target_id=target_id,
                        relation_type=rt,
                        weight=weight,
                        context="crawler_intra_doc",
                    ))

        # 阶段2：与现有库的关系
        try:
            existing_relations = self._infer_relations_to_existing(node, verbose)
            relations.extend(existing_relations)
        except Exception as e:
            logger.warning(f"与现有库关系推断失败 ({node.node_id}): {e}")

        return relations

    def _infer_relations_to_existing(self, node: MemoryNode,
                                      verbose: bool) -> List[Relation]:
        """
        向量检索最相似的现有节点，然后让 LLM 判断关系类型。
        这确保新节点能被正确嵌入现有图谱，而不是孤立存在。
        """
        if not self.net.nodes:
            return []

        # 向量检索相似节点
        candidates = self.net.vector_search(node.content, top_k=self.RELATION_SEARCH_TOP_K)
        if not candidates:
            return []

        relations = []
        for existing_id, score in candidates:
            if score < 0.4:
                continue
            existing_node = self.net.get_node(existing_id)
            if not existing_node:
                continue

            # 用 LLM 判断关系类型
            rel_type, weight, direction = self._llm_judge_relation(node, existing_node)
            if rel_type is None:
                continue

            if direction == "forward":
                relations.append(Relation(
                    source_id=node.node_id,
                    target_id=existing_id,
                    relation_type=rel_type,
                    weight=weight,
                    context=f"crawler_to_existing|sim={score:.2f}",
                ))
            elif direction == "backward":
                relations.append(Relation(
                    source_id=existing_id,
                    target_id=node.node_id,
                    relation_type=rel_type,
                    weight=weight,
                    context=f"crawler_from_existing|sim={score:.2f}",
                ))

        return relations

    def _llm_judge_relation(
        self,
        new_node: MemoryNode,
        existing_node: MemoryNode,
    ) -> Tuple[Optional[RelationType], float, str]:
        """
        让 LLM 判断两个节点之间的关系类型和方向。
        返回 (RelationType | None, weight, direction)
        direction: "forward"(new→existing) / "backward"(existing→new) / "none"
        """
        from config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL
        import ollama

        rel_options = "\n".join(
            f'  "{rt.value}" - {rt.name}' for rt in RelationType
        )

        prompt = f"""判断以下两个知识概念之间的关系。

概念A（新）：{new_node.content}
概念B（已有）：{existing_node.content}

请从以下关系类型中选择最合适的一个（如果无明显关系请选"无关系"）：
{rel_options}
  "无关系" - 两者没有明显逻辑关系

特别注意：
- 如果A对B有促进/增强/激活/改善作用，请选"促进"
- 如果A对B有抑制/削弱/阻碍/损伤作用，请选"抑制"
- "促进"和"抑制"优先于"前因"，因为它们包含了方向信息

输出格式（JSON，只输出JSON，不要解释）：
{{"relation": "关系类型中文名", "direction": "A到B" 或 "B到A", "weight": 0.5到1.0的小数, "reason": "一句话说明"}}"""

        try:
            client = ollama.Client(host=OLLAMA_BASE_URL)
            response = client.generate(
                model=OLLAMA_LLM_MODEL,
                prompt=prompt,
                options={"temperature": 0.0, "num_predict": 200},
            )
            answer = response.get("response", "").strip()
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

            json_match = re.search(r'\{.*\}', answer, re.DOTALL)
            if not json_match:
                return None, 0.0, "none"

            data = json.loads(json_match.group())
            rel_str = data.get("relation", "无关系")
            if rel_str == "无关系":
                return None, 0.0, "none"

            rt = VALID_RELATION_TYPES.get(rel_str)
            if rt is None:
                return None, 0.0, "none"

            direction_raw = data.get("direction", "A到B")
            direction = "forward" if "A到B" in direction_raw else "backward"
            weight = float(data.get("weight", 0.7))
            return rt, weight, direction

        except Exception as e:
            logger.debug(f"关系判断 LLM 调用失败: {e}")
            return None, 0.0, "none"

    # ─────────────────────────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────────────────────────

    def _build_node(self, raw: Dict, source_url: str) -> Optional[MemoryNode]:
        """将校验通过的 raw dict 转换为 MemoryNode"""
        try:
            tags = raw.get("tags", [])
            # 自动追加来源标签，方便溯源
            if self.source_tag not in tags:
                tags = tags + [self.source_tag]

            return MemoryNode(
                node_id=raw["node_id"],
                content=raw["content"],
                abstract_level=int(raw["abstract_level"]),
                domain=raw["domain"],
                coverage=float(raw["coverage"]),
                essence_features=raw["essence_features"],
                tags=tags,
                weight=0.8,  # 爬取节点初始权重略低于手工录入（1.0）
            )
        except Exception as e:
            logger.warning(f"构建 MemoryNode 失败 ({raw.get('node_id','?')}): {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# 命令行快捷入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    命令行使用示例：
      python knowledge_crawler.py --keyword "神经可塑性"
      python knowledge_crawler.py --url "https://example.com/article"
      python knowledge_crawler.py --text "path/to/file.txt"
    """
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from knowledge_base_large import build_large_knowledge_base
    from run_demo import check_services

    parser = argparse.ArgumentParser(description="知识爬虫 - 按需获取信息写入记忆网络")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--keyword", "-k", help="搜索关键词")
    group.add_argument("--url", "-u", help="直接爬取指定 URL")
    group.add_argument("--text", "-t", help="读取本地文件内容")
    parser.add_argument("--pages", type=int, default=3, help="关键词搜索时的最大页数")
    parser.add_argument("--quiet", action="store_true", help="减少输出")
    args = parser.parse_args()

    # 检查服务
    ok, msg = check_services()
    if not ok:
        print(f"[错误] 服务未就绪: {msg}")
        sys.exit(1)

    # 加载知识库
    print("[初始化] 加载知识库...")
    net = build_large_knowledge_base()
    net.load_graph("graph_state.json")
    net.build_vectors()

    crawler = KnowledgeCrawler(net)
    verbose = not args.quiet

    if args.keyword:
        results = crawler.crawl_keyword(args.keyword, max_pages=args.pages, verbose=verbose)
        total_added = sum(r.nodes_added for r in results)
        total_skipped = sum(r.nodes_skipped for r in results)
        print(f"\n[汇总] 关键词='{args.keyword}' | 共新增节点:{total_added} 跳过:{total_skipped}")

    elif args.url:
        result = crawler.crawl_url(args.url, verbose=verbose)
        print(f"\n[汇总] 新增节点:{result.nodes_added} 关系:{result.relations_added} "
              f"跳过:{result.nodes_skipped} 耗时:{result.elapsed_sec:.1f}s")
        if result.errors:
            print(f"  错误: {result.errors}")

    elif args.text:
        with open(args.text, "r", encoding="utf-8") as f:
            text = f.read()
        result = crawler.crawl_text(text, source_label=args.text, verbose=verbose)
        print(f"\n[汇总] 新增节点:{result.nodes_added} 关系:{result.relations_added}")

    # 保存图谱
    net.save_graph("graph_state.json")
    print("[完成] 图谱已保存")


if __name__ == "__main__":
    main()
