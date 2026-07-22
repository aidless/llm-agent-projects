"""
智能路由策略模块
提供四种路由策略：
1. 按任务类型路由（task_type）- 代码生成→DeepSeek，创意写作→Claude，通用→GPT-4o
2. 按成本优化路由（cost）- 简单任务用便宜模型
3. 按延迟优化路由（latency）- 选择历史延迟最低的模型
4. 负载均衡路由（load_balance）- 多 API Key 轮询
"""
import logging
from typing import Optional

from app.models import ChatCompletionRequest
from app.config import get_config
from monitor.stats_collector import StatsCollector, stats_collector
from strategy.task_classifier import TaskClassifier, TaskType, task_classifier
from routers.manager import get_adapter_manager, AdapterManager

logger = logging.getLogger(__name__)


class RouteResult:
    """路由结果"""

    def __init__(
        self,
        model_key: str,
        strategy: str,
        task_type: str,
        fallback_models: list[str],
        reason: str = "",
    ):
        self.model_key = model_key           # 主模型 key
        self.strategy = strategy               # 使用的策略名称
        self.task_type = task_type             # 识别到的任务类型
        self.fallback_models = fallback_models  # 降级备用模型列表
        self.reason = reason                    # 路由原因说明

    def __repr__(self):
        return (
            f"RouteResult(model={self.model_key}, strategy={self.strategy}, "
            f"task={self.task_type}, fallback={self.fallback_models})"
        )


class BaseRouterStrategy:
    """路由策略基类"""

    name: str = "base"

    def __init__(
        self,
        adapter_manager: Optional[AdapterManager] = None,
        stats: Optional[StatsCollector] = None,
        classifier: Optional[TaskClassifier] = None,
    ):
        self.adapter_manager = adapter_manager or get_adapter_manager()
        self.stats = stats or stats_collector
        self.classifier = classifier or task_classifier

    def route(self, request: ChatCompletionRequest) -> RouteResult:
        """
        根据请求进行路由决策

        返回:
            RouteResult 包含主模型和备用模型列表
        """
        raise NotImplementedError


class TaskTypeRouter(BaseRouterStrategy):
    """
    按任务类型路由策略
    - 代码生成 → DeepSeek（擅长代码）
    - 创意写作 → Claude（擅长创意）
    - 数据分析 → GPT-4o（擅长推理分析）
    - 翻译 → Qwen（性价比高）
    - 通用 → GPT-4o（综合能力强）
    """

    name = "task_type"

    # 任务类型到模型的首选映射
    TASK_MODEL_MAP: dict[TaskType, str] = {
        TaskType.CODE_GENERATION: "deepseek",
        TaskType.CREATIVE_WRITING: "claude",
        TaskType.DATA_ANALYSIS: "gpt-4o",
        TaskType.TRANSLATION: "qwen",
        TaskType.SUMMARY: "qwen",
        TaskType.GENERAL: "gpt-4o",
    }

    def route(self, request: ChatCompletionRequest) -> RouteResult:
        """根据任务类型选择最合适的模型"""
        user_text = request.get_last_user_message()
        task_type = self.classifier.classify(user_text, request.user_hint)

        # 查找首选模型
        primary_model = self.TASK_MODEL_MAP.get(task_type, "gpt-4o")

        # 如果首选模型不可用，回退到 gpt-4o
        available = self.adapter_manager.get_available_models()
        if primary_model not in available:
            logger.warning(
                f"任务类型路由: 首选模型 {primary_model} 不可用，回退到 gpt-4o"
            )
            primary_model = "gpt-4o"
            if primary_model not in available:
                primary_model = available[0] if available else "gpt-4o"

        # 构建降级列表：排除主模型，优先选 GPT-4o
        fallback = [m for m in available if m != primary_model]
        # 把 gpt-4o 放在降级列表最前面
        if "gpt-4o" in fallback:
            fallback.remove("gpt-4o")
            fallback.insert(0, "gpt-4o")

        reason = f"任务类型 '{task_type.value}' 路由到 {primary_model}"
        logger.info(f"[任务类型路由] {reason}")

        return RouteResult(
            model_key=primary_model,
            strategy=self.name,
            task_type=task_type.value,
            fallback_models=fallback,
            reason=reason,
        )


class CostOptimizeRouter(BaseRouterStrategy):
    """
    按成本优化路由策略
    - 简单任务（复杂度低）→ 用便宜的模型（DeepSeek > Qwen > GPT-4o > Claude）
    - 复杂任务 → 用能力强的模型（Claude > GPT-4o > DeepSeek > Qwen）
    """

    name = "cost"

    # 模型按成本从低到高排列
    COST_ORDER: list[str] = ["deepseek", "qwen", "gpt-4o", "claude"]

    # 复杂任务按能力从强到弱排列
    CAPABILITY_ORDER: list[str] = ["claude", "gpt-4o", "deepseek", "qwen"]

    def route(self, request: ChatCompletionRequest) -> RouteResult:
        """根据请求复杂度选择最具性价比的模型"""
        user_text = request.get_last_user_message()
        task_type = self.classifier.classify(user_text, request.user_hint)
        complexity = self.classifier.estimate_complexity(user_text)

        available = self.adapter_manager.get_available_models()

        if complexity < 0.4:
            # 简单任务：选最便宜的可用模型
            order = self.COST_ORDER
            reason = f"低复杂度({complexity:.2f})，选择低成本模型"
        else:
            # 复杂任务：选能力最强的模型
            order = self.CAPABILITY_ORDER
            reason = f"高复杂度({complexity:.2f})，选择高能力模型"

        # 按顺序找到第一个可用的模型
        primary_model = None
        for model in order:
            if model in available:
                primary_model = model
                break

        if primary_model is None:
            primary_model = available[0] if available else "gpt-4o"

        fallback = [m for m in available if m != primary_model]

        logger.info(f"[成本优化路由] {reason} -> {primary_model}")
        return RouteResult(
            model_key=primary_model,
            strategy=self.name,
            task_type=task_type.value,
            fallback_models=fallback,
            reason=reason,
        )


class LatencyOptimizeRouter(BaseRouterStrategy):
    """
    按延迟优化路由策略
    选择历史平均延迟最低的模型
    如果没有历史数据，按默认优先级选择
    """

    name = "latency"

    # 默认延迟优先级（从低到高，基于一般经验）
    DEFAULT_LATENCY_ORDER: list[str] = ["deepseek", "qwen", "gpt-4o", "claude"]

    def route(self, request: ChatCompletionRequest) -> RouteResult:
        """选择历史延迟最低的可用模型"""
        user_text = request.get_last_user_message()
        task_type = self.classifier.classify(user_text, request.user_hint)

        available = self.adapter_manager.get_available_models()

        # 收集每个可用模型的历史平均延迟
        model_latencies: list[tuple[str, float]] = []
        for model_key in available:
            model_stats = self.stats.get_model_stats(model_key)
            avg_lat = model_stats.avg_latency_ms
            if avg_lat > 0:
                model_latencies.append((model_key, avg_lat))

        if model_latencies:
            # 按延迟从低到高排序
            model_latencies.sort(key=lambda x: x[1])
            primary_model = model_latencies[0][0]
            reason = (
                f"基于历史延迟选择: {primary_model}"
                f" (平均 {model_latencies[0][1]:.0f}ms)"
            )
        else:
            # 没有历史数据，按默认优先级
            for model in self.DEFAULT_LATENCY_ORDER:
                if model in available:
                    primary_model = model
                    break
            else:
                primary_model = available[0] if available else "gpt-4o"
            reason = f"无历史延迟数据，按默认优先级选择: {primary_model}"

        fallback = [m for m in available if m != primary_model]

        logger.info(f"[延迟优化路由] {reason}")
        return RouteResult(
            model_key=primary_model,
            strategy=self.name,
            task_type=task_type.value,
            fallback_models=fallback,
            reason=reason,
        )


class LoadBalanceRouter(BaseRouterStrategy):
    """
    负载均衡路由策略
    将请求均匀分配到各个模型，避免单一模型过载
    通过轮询 + 成功请求计数实现
    """

    name = "load_balance"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._counter = 0  # 轮询计数器

    def route(self, request: ChatCompletionRequest) -> RouteResult:
        """轮询选择下一个模型"""
        user_text = request.get_last_user_message()
        task_type = self.classifier.classify(user_text, request.user_hint)

        available = self.adapter_manager.get_available_models()
        if not available:
            return RouteResult(
                model_key="gpt-4o",
                strategy=self.name,
                task_type=task_type.value,
                fallback_models=[],
                reason="无可用模型",
            )

        # 简单轮询
        primary_model = available[self._counter % len(available)]
        self._counter += 1

        fallback = [m for m in available if m != primary_model]

        logger.info(f"[负载均衡路由] 轮询选择: {primary_model} (第 {self._counter} 次)")
        return RouteResult(
            model_key=primary_model,
            strategy=self.name,
            task_type=task_type.value,
            fallback_models=fallback,
            reason=f"负载均衡轮询，第 {self._counter} 次请求",
        )


class SmartRouter:
    """
    智能路由器
    整合所有路由策略，根据请求或用户指定选择策略
    """

    # 策略名称到策略类的映射
    STRATEGY_CLASSES = {
        "task_type": TaskTypeRouter,
        "cost": CostOptimizeRouter,
        "latency": LatencyOptimizeRouter,
        "load_balance": LoadBalanceRouter,
    }

    # 默认策略
    DEFAULT_STRATEGY = "task_type"

    def __init__(
        self,
        adapter_manager: Optional[AdapterManager] = None,
        stats: Optional[StatsCollector] = None,
        classifier: Optional[TaskClassifier] = None,
    ):
        self.adapter_manager = adapter_manager or get_adapter_manager()
        self.stats = stats or stats_collector
        self.classifier = classifier or task_classifier

        # 初始化所有策略
        self._strategies: dict[str, BaseRouterStrategy] = {}
        common_kwargs = {
            "adapter_manager": self.adapter_manager,
            "stats": self.stats,
            "classifier": self.classifier,
        }
        for name, cls in self.STRATEGY_CLASSES.items():
            self._strategies[name] = cls(**common_kwargs)

    def route(self, request: ChatCompletionRequest) -> RouteResult:
        """
        执行路由决策

        参数:
            request: 聊天请求

        返回:
            RouteResult
        """
        # 如果用户明确指定了模型，直接使用
        if request.model and request.model in self.adapter_manager.get_available_models():
            logger.info(f"用户指定模型: {request.model}")
            available = self.adapter_manager.get_available_models()
            fallback = [m for m in available if m != request.model]
            return RouteResult(
                model_key=request.model,
                strategy="direct",
                task_type="unknown",
                fallback_models=fallback,
                reason=f"用户直接指定模型: {request.model}",
            )

        # 选择策略
        strategy_name = request.strategy or self.DEFAULT_STRATEGY
        strategy = self._strategies.get(strategy_name)

        if strategy is None:
            logger.warning(
                f"未知策略 '{strategy_name}'，回退到默认策略 '{self.DEFAULT_STRATEGY}'"
            )
            strategy = self._strategies[self.DEFAULT_STRATEGY]
            strategy_name = self.DEFAULT_STRATEGY

        return strategy.route(request)

    def get_available_strategies(self) -> list[str]:
        """获取所有可用的策略名称"""
        return list(self.STRATEGY_CLASSES.keys())


# 全局智能路由器单例
_smart_router: Optional[SmartRouter] = None


def get_smart_router() -> SmartRouter:
    """获取全局智能路由器实例"""
    global _smart_router
    if _smart_router is None:
        _smart_router = SmartRouter()
    return _smart_router


def reset_smart_router():
    """重置全局智能路由器（主要用于测试）"""
    global _smart_router
    _smart_router = None