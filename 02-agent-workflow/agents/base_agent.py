# ============================================
# Agent 基类和 LLM 创建工厂
# 提供所有 Agent 的通用基础设施
# ============================================

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, List, Any
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from config import get_settings


def create_llm(
    temperature: float = 0.7,
    max_tokens: int = 4096,
    model: Optional[str] = None,
) -> BaseChatModel:
    """
    创建 LLM 实例（支持 OpenAI 和 DeepSeek）

    Args:
        temperature: 生成温度
        max_tokens: 最大生成 token 数
        model: 指定模型名称（可选，覆盖配置）

    Returns:
        BaseChatModel: LangChain 的 LLM 实例
    """
    settings = get_settings()

    llm = ChatOpenAI(
        api_key=settings.current_api_key,
        base_url=settings.current_base_url,
        model=model or settings.current_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60.0,
        max_retries=2,
    )

    logger.debug(f"[LLM] 创建实例: provider={settings.llm_provider}, model={model or settings.current_model}")
    return llm


class BaseAgent(ABC):
    """
    Agent 基类
    定义所有 Agent 的通用接口和基础设施
    """

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """
        初始化 Agent

        Args:
            name: Agent 名称
            description: Agent 描述
            system_prompt: 系统提示词
            temperature: LLM 生成温度
            max_tokens: 最大 token 数
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.llm = create_llm(temperature=temperature, max_tokens=max_tokens)

    async def invoke(self, message: str, context: Optional[dict] = None) -> str:
        """
        调用 Agent 处理消息

        Args:
            message: 输入消息
            context: 额外上下文信息

        Returns:
            str: Agent 的响应文本
        """
        logger.info(f"[{self.name}] 收到消息，长度: {len(message)} 字符")

        # 构建消息列表
        messages = [SystemMessage(content=self.system_prompt)]

        # 添加上下文（如果有）
        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            messages.append(SystemMessage(content=f"当前上下文信息:\n{context_str}"))

        # 添加用户消息
        messages.append(HumanMessage(content=message))

        try:
            # 调用 LLM
            response = await self.llm.ainvoke(messages)
            result = response.content
            logger.info(f"[{self.name}] 响应完成，长度: {len(result)} 字符")
            return result
        except Exception as e:
            error_msg = f"[{self.name}] 调用失败: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            return f"Agent 调用出错: {str(e)}"

    @abstractmethod
    async def execute(self, task: str, state: dict) -> dict:
        """
        执行具体任务（子类必须实现）

        Args:
            task: 任务描述
            state: 当前工作流状态

        Returns:
            dict: 执行结果，会合并到工作流状态中
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
