"""
EXP-003 规模化测试知识库
规模：50节点 / ~80条关系
主题：人类行为的进化起源（扩展版）
  - 原始19节点全部保留
  - 新增21个高质量节点（神经科学/认知科学/动物行为学）
  - 新增10个噪音节点（弱相关/无关领域）
噪音节点标记：tag中含 "NOISE"
目的：测试规模扩大+噪音引入后，系统准确率和抗干扰能力的变化
"""

from memory_network import MemoryNetwork, HyperEdge
from memory_node import MemoryNode
from relation_types import Relation, RelationType


def build_large_knowledge_base() -> MemoryNetwork:
    net = MemoryNetwork()

    # ═══════════════════════════════════════════════════════════
    # 第一部分：原始19节点（完整保留）
    # ═══════════════════════════════════════════════════════════
    core_nodes = [
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

        MemoryNode("猴子群体行为", "猴子在陌生环境中会率先寻找撤退路线和高处瞭望点，这与其捕食者规避策略有关",
                   abstract_level=3, domain=["动物行为学", "灵长类学"],
                   coverage=0.4, essence_features=["环境扫描", "捕食者规避"],
                   tags=["猴子", "动物行为", "瞭望", "出口"]),

        MemoryNode("战斗逃跑反应", "战斗逃跑反应是面对威胁时的应激反应，包括肌肉紧张、注意力聚焦和空间定向",
                   abstract_level=5, domain=["神经科学", "心理学"],
                   coverage=0.6, essence_features=["应激", "注意力", "威胁响应"],
                   tags=["应激", "战斗", "逃跑", "威胁"]),
    ]

    # ═══════════════════════════════════════════════════════════
    # 第二部分：新增21个高质量节点（神经科学/认知科学/动物行为学）
    # ═══════════════════════════════════════════════════════════
    extended_nodes = [
        # 神经科学层
        MemoryNode("杏仁核",
                   "杏仁核是大脑边缘系统的核心结构，负责情绪处理和威胁评估，是恐惧反应的神经基础",
                   abstract_level=5, domain=["神经科学", "心理学"],
                   coverage=0.65, essence_features=["恐惧", "威胁评估", "情绪"],
                   tags=["大脑", "神经", "恐惧", "杏仁核"]),

        MemoryNode("前额叶皮层",
                   "前额叶皮层负责高级认知功能，包括计划、决策和对杏仁核的调控，是人类理性行为的神经基础",
                   abstract_level=6, domain=["神经科学", "认知科学"],
                   coverage=0.7, essence_features=["决策", "计划", "认知控制"],
                   tags=["大脑", "理性", "决策", "前额叶"]),

        MemoryNode("皮质醇",
                   "皮质醇是应激激素，由肾上腺分泌，在应激反应中调动能量资源，使机体处于警觉状态",
                   abstract_level=4, domain=["内分泌学", "神经科学"],
                   coverage=0.5, essence_features=["应激", "激素", "警觉"],
                   tags=["激素", "应激", "皮质醇", "肾上腺"]),

        MemoryNode("海马体",
                   "海马体负责空间记忆和情境记忆的编码与巩固，与导航和环境地图构建密切相关",
                   abstract_level=5, domain=["神经科学"],
                   coverage=0.6, essence_features=["空间记忆", "导航", "情境"],
                   tags=["记忆", "空间", "海马体", "导航"]),

        MemoryNode("多巴胺系统",
                   "多巴胺系统负责奖励预测和动机驱动，使生物趋向有利于生存的行为并形成习惯",
                   abstract_level=6, domain=["神经科学", "心理学"],
                   coverage=0.65, essence_features=["奖励", "动机", "习惯"],
                   tags=["多巴胺", "奖励", "动机", "习惯"]),

        # 认知科学层
        MemoryNode("注意力系统",
                   "注意力系统通过选择性过滤机制将认知资源集中在高优先级刺激上，威胁刺激具有注意力捕获优先权",
                   abstract_level=6, domain=["认知科学", "心理学"],
                   coverage=0.65, essence_features=["选择性注意", "资源分配", "威胁优先"],
                   tags=["注意力", "认知", "威胁", "过滤"]),

        MemoryNode("工作记忆",
                   "工作记忆是临时存储和处理信息的认知系统，容量有限（约7±2个单元），是实时推理的基础",
                   abstract_level=6, domain=["认知科学", "心理学"],
                   coverage=0.6, essence_features=["临时存储", "处理", "容量限制"],
                   tags=["记忆", "工作记忆", "认知", "容量"]),

        MemoryNode("认知地图",
                   "认知地图是大脑对空间环境的内部表征，使动物无需视觉即可规划路径和预测障碍物位置",
                   abstract_level=5, domain=["认知科学", "神经科学"],
                   coverage=0.55, essence_features=["空间表征", "路径规划", "内部模型"],
                   tags=["空间", "地图", "认知", "导航", "出口"]),

        MemoryNode("习得性恐惧",
                   "习得性恐惧是通过经典条件反射将中性刺激与威胁联结，形成对该刺激的恐惧反应",
                   abstract_level=5, domain=["心理学", "行为学"],
                   coverage=0.55, essence_features=["条件反射", "联结学习", "威胁联结"],
                   tags=["恐惧", "学习", "条件反射"]),

        MemoryNode("元认知",
                   "元认知是对自身认知过程的监控和调节能力，包括知道自己知道什么和不知道什么",
                   abstract_level=8, domain=["认知科学", "心理学"],
                   coverage=0.75, essence_features=["自我监控", "认知调节", "知识边界"],
                   tags=["元认知", "自我监控", "知识"]),

        # 动物行为学层
        MemoryNode("领地行为",
                   "领地行为是动物通过标记和防御固定区域来保证资源（食物、配偶）独占性的本能行为",
                   abstract_level=4, domain=["动物行为学", "生态学"],
                   coverage=0.5, essence_features=["资源保护", "标记", "防御"],
                   tags=["领地", "动物", "本能", "资源"]),

        MemoryNode("群体行为",
                   "群体行为是多个个体协调行动形成的集体模式，如鸟群飞行、鱼群游动，具有反捕食者功能",
                   abstract_level=5, domain=["动物行为学", "生态学"],
                   coverage=0.55, essence_features=["协调", "集体", "反捕食"],
                   tags=["群体", "集体", "动物", "协作"]),

        MemoryNode("求偶行为",
                   "求偶行为是动物通过展示、鸣叫或礼物吸引配偶的本能行为模式，受性激素调控",
                   abstract_level=4, domain=["动物行为学", "进化论"],
                   coverage=0.45, essence_features=["性选择", "展示", "配对"],
                   tags=["求偶", "繁殖", "本能", "配偶"]),

        MemoryNode("母性行为",
                   "母性行为是雌性哺乳动物对后代的护理和保护行为，由催产素系统驱动，是本能行为的典型形式",
                   abstract_level=4, domain=["动物行为学", "神经科学"],
                   coverage=0.5, essence_features=["护理", "催产素", "保护后代"],
                   tags=["母性", "本能", "哺乳动物", "后代"]),

        MemoryNode("捕食者回避策略",
                   "捕食者回避策略包括伪装、拟态、逃跑路线规划和群体警戒，是自然选择塑造的复合本能",
                   abstract_level=5, domain=["动物行为学", "进化论"],
                   coverage=0.6, essence_features=["伪装", "逃跑", "警戒"],
                   tags=["捕食者", "回避", "本能", "生存"]),

        MemoryNode("条件反射",
                   "条件反射是神经系统将刺激与反应建立联结的基本学习机制，分为经典条件反射和操作条件反射",
                   abstract_level=6, domain=["行为学", "神经科学"],
                   coverage=0.65, essence_features=["联结", "学习", "刺激-反应"],
                   tags=["条件反射", "学习", "神经", "行为"]),

        # 人类特有行为
        MemoryNode("语言能力",
                   "语言能力是人类通过符号系统传递抽象信息的能力，依赖布洛卡区和韦尼克区，是人类区别于其他灵长类的核心特征",
                   abstract_level=6, domain=["语言学", "神经科学", "人类学"],
                   coverage=0.65, essence_features=["符号", "抽象", "交流"],
                   tags=["语言", "人类", "符号", "交流"]),

        MemoryNode("工具制造",
                   "工具制造是人类和少数灵长类通过认知规划制作和使用工具的能力，反映了前额叶的高级功能",
                   abstract_level=5, domain=["人类学", "认知科学"],
                   coverage=0.55, essence_features=["规划", "制造", "工具"],
                   tags=["工具", "人类", "认知", "制造"]),

        MemoryNode("社会性学习",
                   "社会性学习是通过观察和模仿他人行为获得知识和技能的学习方式，大幅加速文化知识积累",
                   abstract_level=6, domain=["心理学", "人类学"],
                   coverage=0.6, essence_features=["模仿", "观察学习", "文化传承"],
                   tags=["学习", "社会", "模仿", "文化"]),

        MemoryNode("恐高症",
                   "恐高症是对高处的非理性恐惧，被认为是进化遗留的本能保护机制，防止坠落风险",
                   abstract_level=3, domain=["心理学", "进化心理学"],
                   coverage=0.4, essence_features=["恐惧", "高处", "保护"],
                   tags=["恐高", "恐惧", "本能", "进化"]),

        MemoryNode("蛇类恐惧",
                   "人类和灵长类对蛇的先天恐惧反应被认为是来自祖先与蛇类长期共存的进化压力，无需学习即可触发",
                   abstract_level=3, domain=["进化心理学", "动物行为学"],
                   coverage=0.4, essence_features=["先天恐惧", "蛇", "进化遗留"],
                   tags=["蛇", "恐惧", "本能", "灵长类", "先天"]),
    ]

    # ═══════════════════════════════════════════════════════════
    # 第三部分：10个噪音节点（弱相关/无关领域）
    # ═══════════════════════════════════════════════════════════
    noise_nodes = [
        MemoryNode("光合作用",
                   "光合作用是植物利用光能将二氧化碳和水转化为有机物的过程，是地球主要的能量来源",
                   abstract_level=7, domain=["植物学", "生物化学"],
                   coverage=0.7, essence_features=["光能", "有机物合成"],
                   tags=["植物", "光合", "能量", "NOISE"]),

        MemoryNode("量子纠缠",
                   "量子纠缠是两个粒子在量子态上形成关联，对一个粒子的测量会瞬间影响另一个粒子的状态",
                   abstract_level=9, domain=["量子物理"],
                   coverage=0.5, essence_features=["纠缠", "量子态", "非定域性"],
                   tags=["量子", "物理", "NOISE"]),

        MemoryNode("股票市场",
                   "股票市场是买卖公司股权的金融市场，价格由供需关系决定，受宏观经济、政策和情绪影响",
                   abstract_level=5, domain=["经济学", "金融学"],
                   coverage=0.4, essence_features=["价格", "交易", "股权"],
                   tags=["金融", "股票", "经济", "NOISE"]),

        MemoryNode("烹饪技术",
                   "烹饪技术是通过加热、调味等处理食材使其更易消化和口感更佳的人类文化行为",
                   abstract_level=3, domain=["文化", "人类学"],
                   coverage=0.3, essence_features=["食物处理", "文化", "加热"],
                   tags=["烹饪", "食物", "文化", "NOISE"]),

        MemoryNode("区块链",
                   "区块链是一种分布式账本技术，通过密码学保证数据不可篡改，是加密货币的底层技术",
                   abstract_level=6, domain=["计算机科学", "密码学"],
                   coverage=0.4, essence_features=["分布式", "不可篡改", "密码学"],
                   tags=["区块链", "技术", "加密", "NOISE"]),

        MemoryNode("气候变化",
                   "气候变化指地球长期气候模式的显著变化，当前主要由人类活动导致的温室气体排放加剧",
                   abstract_level=7, domain=["气候科学", "环境科学"],
                   coverage=0.6, essence_features=["温度上升", "温室效应", "环境"],
                   tags=["气候", "环境", "全球变暖", "NOISE"]),

        MemoryNode("音乐理论",
                   "音乐理论研究音符、节奏、和声的组织规律，是作曲和音乐分析的基础框架",
                   abstract_level=5, domain=["音乐学"],
                   coverage=0.35, essence_features=["音符", "和声", "节奏"],
                   tags=["音乐", "理论", "艺术", "NOISE"]),

        MemoryNode("建筑结构",
                   "建筑结构通过力学原理分配荷载，确保建筑物在重力、风力等外力下保持稳定",
                   abstract_level=5, domain=["建筑学", "力学"],
                   coverage=0.4, essence_features=["荷载", "稳定", "力学"],
                   tags=["建筑", "结构", "力学", "NOISE"]),

        MemoryNode("数据库索引",
                   "数据库索引是通过B树或哈希结构加速数据查询的数据结构，以额外存储空间换取查询速度",
                   abstract_level=5, domain=["计算机科学", "数据库"],
                   coverage=0.4, essence_features=["查询加速", "B树", "索引"],
                   tags=["数据库", "索引", "计算机", "NOISE"]),

        MemoryNode("中世纪历史",
                   "中世纪（5-15世纪）是欧洲封建制度主导的时期，以天主教会的文化权威和骑士制度为特征",
                   abstract_level=5, domain=["历史学"],
                   coverage=0.35, essence_features=["封建", "教会", "骑士"],
                   tags=["历史", "中世纪", "欧洲", "NOISE"]),
    ]

    # 写入所有节点
    for node in core_nodes + extended_nodes + noise_nodes:
        net.add_node(node)

    # ═══════════════════════════════════════════════════════════
    # 关系定义（原始24条 + 新增约55条）
    # ═══════════════════════════════════════════════════════════
    relations = [
        # ── 原始24条关系（完整保留）──────────────────────
        Relation("男人",           "人类",             RelationType.BELONGS_TO,    weight=1.0),
        Relation("人类",           "灵长类",           RelationType.BELONGS_TO,    weight=1.0),
        Relation("灵长类",         "哺乳动物",         RelationType.BELONGS_TO,    weight=1.0),
        Relation("哺乳动物",       "动物",             RelationType.BELONGS_TO,    weight=1.0),
        Relation("动物",           "生物",             RelationType.BELONGS_TO,    weight=1.0),
        Relation("生物",           "生命",             RelationType.BELONGS_TO,    weight=1.0),
        Relation("生命",           "物质",             RelationType.BELONGS_TO,    weight=1.0),
        Relation("人类",           "灵长类",           RelationType.EVOLVED_FROM,  weight=1.0),
        Relation("灵长类",         "早期人类祖先",     RelationType.EVOLVED_FROM,  weight=0.9),
        Relation("男人",           "狩猎采集者",       RelationType.EVOLVED_FROM,  weight=0.9),
        Relation("早期人类祖先",   "生存压力",         RelationType.CAUSES,        weight=1.0),
        Relation("生存压力",       "危险环境感知",     RelationType.CAUSES,        weight=1.0),
        Relation("危险环境感知",   "自然选择",         RelationType.CAUSES,        weight=0.9),
        Relation("自然选择",       "本能行为",         RelationType.CAUSES,        weight=1.0),
        Relation("本能行为",       "空间警觉性",       RelationType.CAUSES,        weight=0.9),
        Relation("狩猎采集者",     "空间警觉性",       RelationType.CAUSES,        weight=0.9),
        Relation("狩猎采集者",     "危险环境感知",     RelationType.CAUSES,        weight=0.9),
        Relation("空间警觉性",     "战斗逃跑反应",     RelationType.CO_OCCURS_WITH,weight=0.8),
        Relation("危险环境感知",   "空间警觉性",       RelationType.TOOL_FOR,      weight=0.9),
        Relation("进化",           "自然选择",         RelationType.CONTAINS,      weight=1.0),
        Relation("猴子群体行为",   "空间警觉性",       RelationType.ANALOGOUS_TO,  weight=0.85),
        Relation("猴子群体行为",   "早期人类祖先",     RelationType.ANALOGOUS_TO,  weight=0.8),
        Relation("进化心理学",     "本能行为",         RelationType.CONTAINS,      weight=0.9),
        Relation("进化心理学",     "空间警觉性",       RelationType.CONTAINS,      weight=0.85),
        Relation("生存压力",       "自然选择",         RelationType.CONDITION_FOR, weight=0.9),

        # ── 新增神经科学关系 ───────────────────────────
        Relation("战斗逃跑反应",   "杏仁核",           RelationType.CAUSES,        weight=0.95),
        Relation("杏仁核",         "皮质醇",           RelationType.CAUSES,        weight=0.9),
        Relation("皮质醇",         "战斗逃跑反应",     RelationType.CAUSES,        weight=0.85),
        Relation("杏仁核",         "空间警觉性",       RelationType.CAUSES,        weight=0.9),
        Relation("前额叶皮层",     "杏仁核",           RelationType.CONDITION_FOR, weight=0.85),
        Relation("海马体",         "认知地图",         RelationType.CAUSES,        weight=0.95),
        Relation("认知地图",       "空间警觉性",       RelationType.CAUSES,        weight=0.9),
        Relation("认知地图",       "危险环境感知",     RelationType.TOOL_FOR,      weight=0.85),
        Relation("多巴胺系统",     "本能行为",         RelationType.CAUSES,        weight=0.8),
        Relation("杏仁核",         "哺乳动物",         RelationType.BELONGS_TO,    weight=0.9),
        Relation("海马体",         "哺乳动物",         RelationType.BELONGS_TO,    weight=0.9),

        # ── 新增认知科学关系 ───────────────────────────
        Relation("注意力系统",     "危险环境感知",     RelationType.TOOL_FOR,      weight=0.9),
        Relation("注意力系统",     "战斗逃跑反应",     RelationType.CAUSES,        weight=0.85),
        Relation("工作记忆",       "认知地图",         RelationType.TOOL_FOR,      weight=0.8),
        Relation("元认知",         "工作记忆",         RelationType.CONTAINS,      weight=0.8),
        Relation("习得性恐惧",     "条件反射",         RelationType.BELONGS_TO,    weight=0.95),
        Relation("习得性恐惧",     "杏仁核",           RelationType.CAUSES,        weight=0.9),
        Relation("条件反射",       "本能行为",         RelationType.ANALOGOUS_TO,  weight=0.7),

        # ── 新增动物行为学关系 ────────────────────────
        Relation("领地行为",       "本能行为",         RelationType.BELONGS_TO,    weight=0.95),
        Relation("群体行为",       "捕食者回避策略",   RelationType.TOOL_FOR,      weight=0.9),
        Relation("捕食者回避策略", "空间警觉性",       RelationType.CAUSES,        weight=0.9),
        Relation("捕食者回避策略", "本能行为",         RelationType.BELONGS_TO,    weight=0.95),
        Relation("母性行为",       "本能行为",         RelationType.BELONGS_TO,    weight=0.95),
        Relation("求偶行为",       "本能行为",         RelationType.BELONGS_TO,    weight=0.9),
        Relation("猴子群体行为",   "群体行为",         RelationType.BELONGS_TO,    weight=0.9),
        Relation("猴子群体行为",   "捕食者回避策略",   RelationType.TOOL_FOR,      weight=0.85),

        # ── 新增人类特有行为关系 ──────────────────────
        Relation("语言能力",       "人类",             RelationType.BELONGS_TO,    weight=1.0),
        Relation("语言能力",       "前额叶皮层",       RelationType.CAUSES,        weight=0.85),
        Relation("工具制造",       "前额叶皮层",       RelationType.CAUSES,        weight=0.9),
        Relation("工具制造",       "狩猎采集者",       RelationType.TOOL_FOR,      weight=0.9),
        Relation("社会性学习",     "语言能力",         RelationType.TOOL_FOR,      weight=0.85),
        Relation("社会性学习",     "人类",             RelationType.BELONGS_TO,    weight=0.9),

        # ── 进化心理学延伸关系 ────────────────────────
        Relation("恐高症",         "本能行为",         RelationType.BELONGS_TO,    weight=0.9),
        Relation("恐高症",         "进化心理学",       RelationType.BELONGS_TO,    weight=0.85),
        Relation("蛇类恐惧",       "本能行为",         RelationType.BELONGS_TO,    weight=0.95),
        Relation("蛇类恐惧",       "灵长类",           RelationType.EVOLVED_FROM,  weight=0.9),
        Relation("蛇类恐惧",       "杏仁核",           RelationType.CAUSES,        weight=0.9),
        Relation("进化心理学",     "恐高症",           RelationType.CONTAINS,      weight=0.85),
        Relation("进化心理学",     "蛇类恐惧",         RelationType.CONTAINS,      weight=0.9),
    ]

    added, skipped = 0, 0
    for rel in relations:
        if rel.source_id in net.nodes and rel.target_id in net.nodes:
            net.add_relation(rel)
            added += 1
        else:
            missing = rel.source_id if rel.source_id not in net.nodes else rel.target_id
            print(f"[警告] 节点不存在，跳过关系: {missing}")
            skipped += 1

    print(f"[知识库-大规模] 节点数: {len(net.nodes)}  "
          f"关系数: {added}  跳过: {skipped}")

    # ═══════════════════════════════════════════════════════════
    # 超图边（T4语义门控，含 context_keywords）
    # N→1 联合因果关系，普通二元边无法表达
    # ═══════════════════════════════════════════════════════════
    hyper_edges = [
        # AND超图边：压力激素+睡眠剥夺 → 免疫功能受损
        # 两个条件同时成立才触发，context_keywords 指定相关语境
        HyperEdge(
            co_nodes=["皮质醇", "战斗逃跑反应"],
            target_ids=["杏仁核"],
            relation_type=RelationType.CAUSES,
            weight=0.85,
            condition="AND",
            context="压力激素升高 + 持续应激 → 杏仁核过度激活",
            context_keywords=["压力", "应激", "皮质醇", "激素", "过激", "慢性"],
        ),
        # AND超图边：杏仁核激活+感知到危险 → 战斗逃跑反应强化
        HyperEdge(
            co_nodes=["杏仁核", "危险环境感知"],
            target_ids=["战斗逃跑反应"],
            relation_type=RelationType.CAUSES,
            weight=0.90,
            condition="AND",
            context="杏仁核 + 危险感知 → 战斗逃跑反应强化激活",
            context_keywords=["危险", "威胁", "应激", "逃跑", "战斗", "恐惧"],
        ),
        # AND超图边：蛇类恐惧+灵长类本能 → 进化心理学验证案例
        HyperEdge(
            co_nodes=["蛇类恐惧", "本能行为"],
            target_ids=["进化心理学"],
            relation_type=RelationType.CAUSES,
            weight=0.80,
            condition="AND",
            context="蛇类恐惧+本能行为共现 → 支持进化心理学理论",
            context_keywords=["蛇", "恐惧", "本能", "进化", "灵长类"],
        ),
        # AND超图边：认知地图+空间警觉性 → 危险环境中存活率提升
        HyperEdge(
            co_nodes=["认知地图", "空间警觉性"],
            target_ids=["危险环境感知"],
            relation_type=RelationType.PROMOTES,
            weight=0.88,
            condition="AND",
            context="认知地图+空间警觉性联合 → 危险环境感知能力大幅提升",
            context_keywords=["认知", "地图", "空间", "警觉", "危险", "感知", "存活"],
        ),
        # OR超图边：任一捕食者回避能力激活 → 空间警觉性提升
        # OR条件门槛低，必须有 context_keywords 防止误触发
        HyperEdge(
            co_nodes=["捕食者回避策略", "习得性恐惧"],
            target_ids=["空间警觉性"],
            relation_type=RelationType.PROMOTES,
            weight=0.75,
            condition="OR",
            context="捕食者回避策略或习得性恐惧 → 空间警觉性提升",
            context_keywords=["捕食者", "回避", "恐惧", "习得", "警觉", "空间"],
        ),
        # AND超图边：工具制造+语言能力 → 社会性学习爆发（人类特有联合条件）
        HyperEdge(
            co_nodes=["工具制造", "语言能力"],
            target_ids=["社会性学习"],
            relation_type=RelationType.PROMOTES,
            weight=0.85,
            condition="AND",
            context="工具制造+语言能力协同 → 社会性学习突破性进化",
            context_keywords=["工具", "语言", "社会", "学习", "文化", "人类", "进化"],
        ),
    ]

    he_added = 0
    for he in hyper_edges:
        # 检查 co_nodes 和 target_ids 中的节点是否存在
        all_nodes_exist = all(
            n in net.nodes for n in he.co_nodes + he.target_ids
        )
        if all_nodes_exist:
            net.add_hyper_edge(he)
            he_added += 1
        else:
            missing = [n for n in he.co_nodes + he.target_ids if n not in net.nodes]
            print(f"[警告] 超图边节点不存在，跳过: {missing}")

    print(f"[知识库-大规模] 超图边: {he_added} 条（含context_keywords语义门控）")
    return net
