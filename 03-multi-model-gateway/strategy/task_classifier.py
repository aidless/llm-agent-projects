"""
任务类型分类器 - 基于关键词匹配将用户输入分类为不同任务类型
用于按任务类型路由策略
"""
import re
from enum import Enum
from typing import Optional


class TaskType(str, Enum):
    """任务类型枚举"""
    CODE_GENERATION = "code_generation"      # 代码生成/编程
    CREATIVE_WRITING = "creative_writing"    # 创意写作/文案
    DATA_ANALYSIS = "data_analysis"          # 数据分析/数学
    TRANSLATION = "translation"              # 翻译
    SUMMARY = "summary"                      # 摘要/总结
    GENERAL = "general"                      # 通用对话


# 任务类型关键词映射表
# 每个任务类型对应一组中英文关键词正则表达式
_TASK_KEYWORDS: dict[TaskType, list[str]] = {
    TaskType.CODE_GENERATION: [
        r"写?\s*(一个|段|个)?(函数|方法|类|程序|脚本|代码|算法)",
        r"(实现|编写|开发|写)?(一个)?(排序|搜索|爬虫|API|接口|网页|网站|后端|前端)",
        r"(Python|Java|JavaScript|TypeScript|Go|Rust|C\+\+|HTML|CSS|React|Vue|Node)",
        r"(函数|方法|类|模块|组件|服务)(的)?(实现|定义|代码)",
        r"(debug|调试|修复|fix|bug|error)",
        r"(code|function|class|implement|program|script|algorithm)",
        r"(写|编|敲|写一个?)(Python|Java|JS|Go|C|代码|程序)",
        r"(leetcode|力扣|算法题|编程题)",
        r"(代码|code)\s*(审查|review|优化|refactor)",
        r"(如何|怎么|怎样)?(实现|写|用)(.+)?(功能|逻辑|接口)",
    ],
    TaskType.CREATIVE_WRITING: [
        r"(写|创作|编|构思).{0,15}(诗|小说|故事|散文|歌词|剧本|文案)",
        r"(创意|构思|灵感|想象|幻想|虚构)",
        r"(角色|人物|场景|情节|故事线)",
        r"(write|create|compose|draft)\s*(a\s+)?(poem|story|novel|script|song|lyrics)",
        r"(创意|creative|story|fiction|narrative)",
        r"(品牌|营销|推广|宣传)(文案|话术|口号|标语)",
    ],
    TaskType.DATA_ANALYSIS: [
        r"(分析|统计|计算|汇总|聚合)(一下|这|这些)?(数据|信息|报表)",
        r"(图表|可视化|柱状图|折线图|饼图|散点图)",
        r"(SQL|查询|数据库|表|字段|索引)",
        r"(均值|中位数|标准差|方差|相关|回归|预测)",
        r"(pandas|numpy|matplotlib|seaborn|excel|csv)",
        r"(analyze|statistics|calculate|aggregate|data)",
        r"(KPI|指标|转化率|留存|活跃)",
    ],
    TaskType.TRANSLATION: [
        r"(翻译|translate|translation)",
        r"(把|将)(.+)?(翻译|译)(成|为|到)(英文|中文|日文|法文|德文|韩文)",
        r"(英译中|中译英|中翻英|英翻中)",
        r"(用英语|用中文|用日语|用法语)说|表达|描述",
    ],
    TaskType.SUMMARY: [
        r"(总结|摘要|概括|归纳|梳理|提炼|简述|概述)",
        r"(summarize|summary|abstract|outline|recap|brief)",
        r"(要点|重点|核心)(是|有|包括|如下)",
        r"(用)?(一两句|简短|简要|简单地)(说|描述|概括|总结)",
        r"(TL;DR|tl;dr|太长不看)",
    ],
}

# 摘要关键词需要在代码生成之前匹配，提高优先级
# 通过将摘要放在更高优先级位置实现
# 重新排列：摘要 > 翻译 > 创意写作 > 数据分析 > 代码生成
_ORDERED_KEYWORDS: list[tuple[TaskType, list[str]]] = [
    (TaskType.SUMMARY, _TASK_KEYWORDS[TaskType.SUMMARY]),
    (TaskType.TRANSLATION, _TASK_KEYWORDS[TaskType.TRANSLATION]),
    (TaskType.CREATIVE_WRITING, _TASK_KEYWORDS[TaskType.CREATIVE_WRITING]),
    (TaskType.DATA_ANALYSIS, _TASK_KEYWORDS[TaskType.DATA_ANALYSIS]),
    (TaskType.CODE_GENERATION, _TASK_KEYWORDS[TaskType.CODE_GENERATION]),
]

# 默认的通用关键词（优先级最低）
_GENERAL_KEYWORDS = [
    r"(你好|hi|hello|嗨|hey)",
    r"(谢谢|感谢|thank)",
    r"(帮助|help|怎么用|如何使用)",
    r"(是什么|什么是|解释一下|介绍一下|什么是)",
]


class TaskClassifier:
    """
    任务类型分类器
    基于关键词匹配对用户输入进行分类
    """

    def __init__(self):
        # 预编译所有正则表达式，按优先级排列
        self._compiled_rules: list[tuple[TaskType, list[re.Pattern]]] = []
        for task_type, keywords in _ORDERED_KEYWORDS:
            patterns = [re.compile(kw, re.IGNORECASE) for kw in keywords]
            self._compiled_rules.append((task_type, patterns))

        self._general_patterns = [re.compile(kw, re.IGNORECASE) for kw in _GENERAL_KEYWORDS]

    def classify(self, text: str, user_hint: Optional[str] = None) -> TaskType:
        """
        对输入文本进行任务类型分类

        参数:
            text: 用户输入的文本
            user_hint: 用户手动指定的任务类型提示（优先级最高）

        返回:
            TaskType 枚举值
        """
        # 如果用户手动指定了任务类型，直接使用
        if user_hint:
            hint = user_hint.lower().strip()
            hint_map = {
                "code": TaskType.CODE_GENERATION,
                "code_generation": TaskType.CODE_GENERATION,
                "coding": TaskType.CODE_GENERATION,
                "programming": TaskType.CODE_GENERATION,
                "creative": TaskType.CREATIVE_WRITING,
                "creative_writing": TaskType.CREATIVE_WRITING,
                "writing": TaskType.CREATIVE_WRITING,
                "analysis": TaskType.DATA_ANALYSIS,
                "data_analysis": TaskType.DATA_ANALYSIS,
                "translation": TaskType.TRANSLATION,
                "translate": TaskType.TRANSLATION,
                "summary": TaskType.SUMMARY,
                "summarize": TaskType.SUMMARY,
            }
            if hint in hint_map:
                return hint_map[hint]

        # 按优先级匹配关键词
        # 优先级: 代码生成 > 创意写作 > 数据分析 > 翻译 > 摘要 > 通用
        for task_type, patterns in self._compiled_rules:
            for pattern in patterns:
                if pattern.search(text):
                    return task_type

        return TaskType.GENERAL

    def estimate_complexity(self, text: str) -> float:
        """
        估算请求复杂度（0.0 ~ 1.0）
        用于成本优化路由：简单请求用便宜模型

        估算依据:
        - 文本长度
        - 是否包含代码块
        - 是否包含多轮对话
        - 是否包含复杂数据/公式
        """
        complexity = 0.0

        # 1. 文本长度因子（越长越复杂，最高 0.3）
        length = len(text)
        if length > 2000:
            complexity += 0.3
        elif length > 1000:
            complexity += 0.2
        elif length > 500:
            complexity += 0.1

        # 2. 代码块因子（包含代码更复杂，+0.2）
        if "```" in text or "def " in text or "function " in text or "class " in text:
            complexity += 0.2

        # 3. 多轮对话因子（消息数量多更复杂，+0.2）
        # 通过检测可能的对话标记
        dialogue_markers = text.count("\n") + text.count("\uff1a") + text.count(":")
        if dialogue_markers > 20:
            complexity += 0.2
        elif dialogue_markers > 10:
            complexity += 0.1

        # 4. 特殊内容因子（+0.2）
        special_patterns = [
            r"\$[\w\s=+\-*/()]+\$",   # 数学公式
            r"\|.*\|.*\|",              # 表格
            r"\d+\.\d+\.\d+",           # 版本号
            r"https?://\S+",            # URL
        ]
        for pat in special_patterns:
            if re.search(pat, text):
                complexity += 0.1

        return min(complexity, 1.0)


# 全局分类器实例
task_classifier = TaskClassifier()