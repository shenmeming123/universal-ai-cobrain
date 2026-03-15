"""
验证场景：构建记忆知识库
主题：人类行为的进化起源

记忆库里只存基础事实，不存答案。
问题需要跨多步推理才能得出，测试联想引擎 vs 普通RAG的差距。
"""

from memory_network import MemoryNetwork, HyperEdge
from memory_node import MemoryNode
from relation_types import Relation, RelationType


def build_knowledge_base() -> MemoryNetwork:
    net = MemoryNetwork()

    # ── 节点定义 ─────────────────────────────────────
    nodes = [
        # 抽象层（level 8-10）
        MemoryNode("物质",       "物质是一切存在的基础构成",
                   abstract_level=10, domain=["物理学", "哲学"],
                   coverage=1.0, essence_features=["存在", "基础"],
                   tags=["基础", "本质"]),

        MemoryNode("生命",       "生命是具有自我复制和代谢能力的物质组织形式",
                   abstract_level=9, domain=["生物学"],
                   coverage=0.9, essence_features=["自我复制", "代谢"],
                   tags=["生命", "生物"]),

        MemoryNode("进化",       "进化是生物种群基因频率随时间的变化过程，自然选择是其核心机制",
                   abstract_level=8, domain=["进化论", "生物学"],
                   coverage=0.85, essence_features=["自然选择", "适应", "遗传"],
                   tags=["进化", "自然选择", "达尔文"]),

        MemoryNode("自然选择",   "自然选择使有利于生存繁殖的特征在后代中比例增加",
                   abstract_level=8, domain=["进化论"],
                   coverage=0.8, essence_features=["适者生存", "遗传"],
                   tags=["自然选择", "适应"]),

        MemoryNode("生存压力",   "生存压力是环境对生物体的生存和繁殖构成的威胁",
                   abstract_level=7, domain=["生态学", "进化论"],
                   coverage=0.75, essence_features=["威胁", "环境", "竞争"],
                   tags=["危险", "威胁", "生存"]),

        # 中层（level 4-7）
        MemoryNode("生物",       "生物是具有生命特征的有机体",
                   abstract_level=7, domain=["生物学"],
                   coverage=0.9, essence_features=["生命", "有机体"],
                   tags=["生物", "有机体"]),

        MemoryNode("动物",       "动物是能够自主运动并以其他有机物为食的生物",
                   abstract_level=6, domain=["生物学", "动物学"],
                   coverage=0.8, essence_features=["运动", "捕食"],
                   tags=["动物", "运动"]),

        MemoryNode("哺乳动物",   "哺乳动物是温血、胎生、哺乳后代的脊椎动物",
                   abstract_level=6, domain=["生物学", "动物学"],
                   coverage=0.75, essence_features=["温血", "哺乳", "胎生"],
                   tags=["哺乳", "温血"]),

        MemoryNode("灵长类",     "灵长类是拥有高度发达大脑和灵活四肢的哺乳动物，包括猿类和人类",
                   abstract_level=5, domain=["生物学", "灵长类学"],
                   coverage=0.7, essence_features=["大脑发达", "社会性", "工具使用"],
                   tags=["灵长类", "猿", "社会"]),

        MemoryNode("早期人类祖先", "早期人类祖先生活在非洲草原，面临大型捕食者威胁，需要持续感知环境威胁才能生存",
                   abstract_level=5, domain=["人类学", "进化论"],
                   coverage=0.65, essence_features=["捕食者压力", "环境感知", "草原"],
                   tags=["祖先", "草原", "捕食者", "非洲"]),

        MemoryNode("危险环境感知", "在危险环境中，能快速感知威胁来源（如出口、遮蔽物、捕食者位置）的个体存活率更高",
                   abstract_level=5, domain=["进化论", "心理学"],
                   coverage=0.65, essence_features=["威胁感知", "空间定向", "存活"],
                   tags=["感知", "威胁", "出口", "空间"]),

        MemoryNode("本能行为",   "本能行为是由遗传决定、无需学习的固定行为模式，是长期自然选择的结果",
                   abstract_level=6, domain=["心理学", "行为学", "进化论"],
                   coverage=0.7, essence_features=["遗传", "固定模式", "无需学习"],
                   tags=["本能", "遗传", "固定行为"]),

        MemoryNode("人类",       "人类是灵长类中智力最发达的物种，具有语言、文化和复杂社会结构",
                   abstract_level=5, domain=["人类学", "生物学"],
                   coverage=0.7, essence_features=["智力", "语言", "文化"],
                   tags=["人", "智人", "人类"]),

        MemoryNode("男人",       "男人是人类的雄性个体，在进化历史中承担狩猎和防御角色",
                   abstract_level=4, domain=["人类学", "社会学"],
                   coverage=0.55, essence_features=["雄性", "狩猎", "防御"],
                   tags=["男人", "雄性", "狩猎"]),

        MemoryNode("狩猎采集者", "狩猎采集者是人类祖先的主要生存方式，需要在开放环境中追踪猎物同时警惕危险",
                   abstract_level=4, domain=["人类学", "考古学"],
                   coverage=0.5, essence_features=["狩猎", "环境警觉", "开放空间"],
                   tags=["狩猎", "采集", "祖先"]),

        MemoryNode("空间警觉性", "空间警觉性是对周围环境的出口、遮蔽物和潜在威胁位置保持持续监控的能力",
                   abstract_level=4, domain=["心理学", "进化心理学"],
                   coverage=0.5, essence_features=["空间定向", "威胁监控", "出口感知"],
                   tags=["警觉", "空间", "出口", "环境扫描"]),

        MemoryNode("进化心理学", "进化心理学研究人类心理特征如何通过自然选择塑造，解释现代行为的进化根源",
                   abstract_level=6, domain=["心理学", "进化论"],
                   coverage=0.7, essence_features=["心理特征", "自然选择", "行为起源"],
                   tags=["进化心理", "行为", "起源"]),

        # 旁证节点（用于跨域验证）
        MemoryNode("猴子群体行为", "猴子在陌生环境中会率先寻找撤退路线和高处瞭望点，这与其捕食者规避策略有关",
                   abstract_level=3, domain=["动物行为学", "灵长类学"],
                   coverage=0.4, essence_features=["环境扫描", "捕食者规避"],
                   tags=["猴子", "动物行为", "瞭望", "出口"]),

        MemoryNode("战斗逃跑反应", "战斗逃跑反应是面对威胁时的应激反应，包括肌肉紧张、注意力聚焦和空间定向",
                   abstract_level=5, domain=["神经科学", "心理学"],
                   coverage=0.6, essence_features=["应激", "注意力", "威胁响应"],
                   tags=["应激", "战斗", "逃跑", "威胁"]),
    ]

    for node in nodes:
        net.add_node(node)

    # ── 关系定义 ─────────────────────────────────────
    relations = [
        # 纵向归属链（核心联想通路）
        Relation("男人",         "人类",           RelationType.BELONGS_TO,    weight=1.0),
        Relation("人类",         "灵长类",         RelationType.BELONGS_TO,    weight=1.0),
        Relation("灵长类",       "哺乳动物",       RelationType.BELONGS_TO,    weight=1.0),
        Relation("哺乳动物",     "动物",           RelationType.BELONGS_TO,    weight=1.0),
        Relation("动物",         "生物",           RelationType.BELONGS_TO,    weight=1.0),
        Relation("生物",         "生命",           RelationType.BELONGS_TO,    weight=1.0),
        Relation("生命",         "物质",           RelationType.BELONGS_TO,    weight=1.0),

        # 进化溯源链
        Relation("人类",         "灵长类",         RelationType.EVOLVED_FROM,  weight=1.0),
        Relation("灵长类",       "早期人类祖先",   RelationType.EVOLVED_FROM,  weight=0.9),
        Relation("男人",         "狩猎采集者",     RelationType.EVOLVED_FROM,  weight=0.9),

        # 因果链（核心推理路径）
        Relation("早期人类祖先", "生存压力",       RelationType.CAUSES,        weight=1.0),
        Relation("生存压力",     "危险环境感知",   RelationType.CAUSES,        weight=1.0),
        Relation("危险环境感知", "自然选择",       RelationType.CAUSES,        weight=0.9),
        Relation("自然选择",     "本能行为",       RelationType.CAUSES,        weight=1.0),
        Relation("本能行为",     "空间警觉性",     RelationType.CAUSES,        weight=0.9),

        # 狩猎采集者的因果
        Relation("狩猎采集者",   "空间警觉性",     RelationType.CAUSES,        weight=0.9),
        Relation("狩猎采集者",   "危险环境感知",   RelationType.CAUSES,        weight=0.9),

        # 属性/功能关系
        Relation("空间警觉性",   "战斗逃跑反应",  RelationType.CO_OCCURS_WITH,weight=0.8),
        Relation("危险环境感知", "空间警觉性",     RelationType.TOOL_FOR,      weight=0.9),
        Relation("进化",         "自然选择",       RelationType.CONTAINS,      weight=1.0),

        # 跨域类比（猴子行为 ≈ 人类祖先行为）
        Relation("猴子群体行为", "空间警觉性",     RelationType.ANALOGOUS_TO,  weight=0.85),
        Relation("猴子群体行为", "早期人类祖先",   RelationType.ANALOGOUS_TO,  weight=0.8),

        # 进化心理学作为元框架
        Relation("进化心理学",   "本能行为",       RelationType.CONTAINS,      weight=0.9),
        Relation("进化心理学",   "空间警觉性",     RelationType.CONTAINS,      weight=0.85),

        # 条件关系
        Relation("生存压力",     "自然选择",       RelationType.CONDITION_FOR, weight=0.9),
    ]

    # 修正笔误
    relations_fixed = []
    for r in relations:
        relations_fixed.append(r)

    for rel in relations_fixed:
        # 确保节点存在
        if rel.source_id in net.nodes and rel.target_id in net.nodes:
            net.add_relation(rel)
        else:
            missing = rel.source_id if rel.source_id not in net.nodes else rel.target_id
            print(f"[警告] 节点不存在，跳过关系: {missing}")

    # ── 超图边（含 context_keywords 语义门控）────────────
    hyper_edges = [
        # AND超图边：狩猎采集者+危险环境感知 → 空间警觉性强化
        HyperEdge(
            co_nodes=["狩猎采集者", "危险环境感知"],
            target_ids=["空间警觉性"],
            relation_type=RelationType.PROMOTES,
            weight=0.85,
            condition="AND",
            context="狩猎采集者在危险环境中 → 空间警觉性进化强化",
            context_keywords=["狩猎", "危险", "环境", "感知", "生存", "祖先"],
        ),
        # AND超图边：自然选择+生存压力 → 本能行为固化
        HyperEdge(
            co_nodes=["自然选择", "生存压力"],
            target_ids=["本能行为"],
            relation_type=RelationType.CAUSES,
            weight=0.90,
            condition="AND",
            context="自然选择压力持续作用 → 本能行为遗传固化",
            context_keywords=["自然选择", "生存", "压力", "本能", "遗传", "进化"],
        ),
        # OR超图边：猴子群体行为或空间警觉性 → 战斗逃跑反应触发
        # OR门槛低，必须有 context_keywords 防止误触发
        HyperEdge(
            co_nodes=["猴子群体行为", "空间警觉性"],
            target_ids=["战斗逃跑反应"],
            relation_type=RelationType.PROMOTES,
            weight=0.75,
            condition="OR",
            context="灵长类空间监控行为 → 触发应激反应准备",
            context_keywords=["猴子", "空间", "警觉", "应激", "逃跑", "危险", "威胁"],
        ),
    ]

    he_added = 0
    for he in hyper_edges:
        all_nodes_exist = all(n in net.nodes for n in he.co_nodes + he.target_ids)
        if all_nodes_exist:
            net.add_hyper_edge(he)
            he_added += 1
        else:
            missing = [n for n in he.co_nodes + he.target_ids if n not in net.nodes]
            print(f"[警告] 超图边节点不存在，跳过: {missing}")

    print(f"[知识库] 节点:{len(net.nodes)} 关系:{net.graph.number_of_edges()} 超图边:{he_added}")
    return net
