"""工具调用 Agent

模拟 LLM Function Calling 风格的工具选择和调用:
- 基于用户意图选择工具
- 工具选择策略（关键词匹配、标签匹配）
- 自动执行工具调用
- 错误处理和重试
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from client.mcp_client import MCPClient
from protocol.types import ToolCallResult
from registry.permission import PermissionLevel
from registry.tool_registry import ToolRegistry
from agent.chain_executor import ChainExecutor, ChainStep, ChainResult

logger = logging.getLogger(__name__)


@dataclass
class ToolSelection:
    """工具选择结果"""
    tool_name: str
    server_name: str
    confidence: float
    matched_by: str  # "keyword", "tag", "description"
    suggested_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Agent 执行结果"""
    success: bool
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    error: str = ""
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "tool_calls": self.tool_calls,
            "final_answer": self.final_answer,
            "error": self.error,
            "total_latency_ms": self.total_latency_ms,
        }


class ToolAgent:
    """工具调用 Agent

    模拟 Function Calling 风格:
    1. 接收用户意图
    2. 从注册中心选择合适的工具
    3. 调用工具并返回结果
    """

    def __init__(
        self,
        client: MCPClient,
        registry: ToolRegistry,
        max_tool_calls: int = 5,
        default_role: str = "default",
    ):
        self.client = client
        self.registry = registry
        self.max_tool_calls = max_tool_calls
        self.default_role = default_role
        self.chain_executor = ChainExecutor(client, registry)

    # ---- 工具选择 ----

    def select_tools(self, user_input: str, role: str = "default") -> list[ToolSelection]:
        """根据用户输入选择合适的工具

        选择策略:
        1. 关键词匹配工具名称
        2. 关键词匹配工具描述
        3. 标签匹配

        Args:
            user_input: 用户自然语言输入
            role: 用户角色

        Returns:
            排序后的工具选择列表
        """
        # 获取所有可用工具
        tools = self.registry.list_tools(role=role)
        if not tools:
            return []

        # 分词
        input_lower = user_input.lower()
        words = set(re.findall(r'\w+', input_lower))

        scored: list[ToolSelection] = []
        for tool_data in tools:
            tool_name = tool_data["name"]
            server_name = tool_data.get("serverName", "")
            description = tool_data.get("description", "").lower()
            tags = tool_data.get("tags", [])
            name_lower = tool_name.lower()

            confidence = 0.0
            matched_by = ""

            # 名称精确匹配
            if name_lower in input_lower:
                confidence = 0.95
                matched_by = "keyword"
            elif name_lower.replace("_", " ") in input_lower:
                confidence = 0.9
                matched_by = "keyword"

            # 名称分词匹配
            if confidence == 0:
                name_words = set(name_lower.split("_"))
                overlap = words & name_words
                if overlap:
                    confidence = 0.5 + 0.1 * len(overlap)
                    matched_by = "keyword"

            # 描述匹配
            if confidence < 0.7:
                desc_words = set(description.split())
                overlap = words & desc_words
                if overlap:
                    desc_conf = 0.3 + 0.1 * len(overlap)
                    if desc_conf > confidence:
                        confidence = desc_conf
                        matched_by = "description"

            # 标签匹配
            if confidence < 0.5:
                for tag in tags:
                    if tag.lower() in words:
                        confidence = max(confidence, 0.4)
                        matched_by = "tag"
                        break

            if confidence > 0.2:
                scored.append(ToolSelection(
                    tool_name=tool_name,
                    server_name=server_name,
                    confidence=confidence,
                    matched_by=matched_by,
                ))

        # 按置信度排序
        scored.sort(key=lambda s: s.confidence, reverse=True)
        return scored[:self.max_tool_calls]

    def _extract_arguments(self, user_input: str, tool_name: str) -> dict[str, Any]:
        """从用户输入中提取工具参数（简单的启发式方法）"""
        args = {}

        # 尝试提取数字
        numbers = re.findall(r'-?\d+\.?\d*', user_input)
        if numbers:
            if len(numbers) >= 2:
                args["a"] = float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
                args["b"] = float(numbers[1]) if '.' in numbers[1] else int(numbers[1])
            elif len(numbers) == 1:
                # 根据工具名判断参数名
                if "sqrt" in tool_name:
                    args["x"] = float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
                elif "table" in tool_name or "count" in tool_name:
                    # 尝试找表名相关的词
                    words = user_input.split()
                    args["table"] = words[-1] if words else ""

        # 尝试提取路径
        path_match = re.search(r'/[\w/._-]+', user_input)
        if path_match and "file" in tool_name:
            args["path"] = path_match.group()

        # 尝试提取引号中的字符串
        quoted = re.findall(r'"([^"]+)"', user_input)
        if quoted:
            if "query" in tool_name:
                args["table"] = quoted[0]
            elif "write" in tool_name and len(quoted) >= 2:
                args["path"] = quoted[0]
                args["content"] = quoted[1]
            elif "write" in tool_name and len(quoted) == 1:
                args["content"] = quoted[0]

        return args

    # ---- 执行 ----

    async def execute(
        self,
        user_input: str,
        role: Optional[str] = None,
        auto_call: bool = True,
    ) -> AgentResult:
        """执行 Agent 任务

        Args:
            user_input: 用户自然语言输入
            role: 用户角色
            auto_call: 是否自动调用选中的工具

        Returns:
            Agent 执行结果
        """
        role = role or self.default_role
        start_time = time.time()
        result = AgentResult(success=True)

        # 1. 选择工具
        selections = self.select_tools(user_input, role=role)
        if not selections:
            result.final_answer = "未找到匹配的工具"
            result.total_latency_ms = (time.time() - start_time) * 1000
            return result

        if not auto_call:
            result.tool_calls = [
                {
                    "tool_name": s.tool_name,
                    "server_name": s.server_name,
                    "confidence": s.confidence,
                    "matched_by": s.matched_by,
                }
                for s in selections
            ]
            result.total_latency_ms = (time.time() - start_time) * 1000
            return result

        # 2. 构建调用链
        steps = []
        for sel in selections:
            # 权限检查
            perm = self.registry.permissions.check_permission(sel.tool_name, role)
            if perm == PermissionLevel.DENY:
                result.tool_calls.append({
                    "tool_name": sel.tool_name,
                    "server_name": sel.server_name,
                    "skipped": True,
                    "reason": "Permission denied",
                })
                continue

            args = self._extract_arguments(user_input, sel.tool_name)
            step = ChainStep(
                tool_name=sel.tool_name,
                arguments=args,
                max_retries=1,
            )
            steps.append(step)

        if not steps:
            result.success = False
            result.error = "No tools available to execute"
            result.total_latency_ms = (time.time() - start_time) * 1000
            return result

        # 3. 执行调用链
        chain_result: ChainResult = await self.chain_executor.execute_chain(steps, role=role)

        # 4. 整理结果
        result.success = chain_result.success
        result.error = chain_result.error
        result.total_latency_ms = chain_result.total_latency_ms

        for sr in chain_result.steps:
            call_info = {
                "step": sr.step_index,
                "tool_name": sr.tool_name,
                "success": sr.success,
                "latency_ms": sr.latency_ms,
            }
            if sr.success and sr.raw_text:
                call_info["result"] = sr.raw_text
            if sr.error:
                call_info["error"] = sr.error
            result.tool_calls.append(call_info)

        # 提取最终答案
        if chain_result.steps and chain_result.steps[-1].raw_text:
            result.final_answer = chain_result.steps[-1].raw_text

        return result

    async def call_single_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        role: Optional[str] = None,
    ) -> AgentResult:
        """直接调用单个工具（跳过选择阶段）"""
        role = role or self.default_role
        start_time = time.time()
        result = AgentResult(success=True)

        server_name = self.registry.get_tool_server(tool_name)
        if server_name is None:
            result.success = False
            result.error = f"Tool not found: {tool_name}"
            return result

        perm = self.registry.permissions.check_permission(tool_name, role)
        if perm == PermissionLevel.DENY:
            result.success = False
            result.error = f"Permission denied: {tool_name}"
            return result

        try:
            call_result = await self.client.call_tool(
                server_name=server_name,
                tool_name=tool_name,
                arguments=arguments,
            )
            latency = (time.time() - start_time) * 1000
            result.total_latency_ms = latency

            raw_text = call_result.content[0].get("text", "") if call_result.content else ""
            result.tool_calls.append({
                "tool_name": tool_name,
                "success": not call_result.is_error,
                "result": raw_text,
                "latency_ms": latency,
            })

            if call_result.is_error:
                result.success = False
                result.error = raw_text
            else:
                result.final_answer = raw_text

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.total_latency_ms = (time.time() - start_time) * 1000

        # 记录使用
        self.registry.record_usage(
            tool_name=tool_name,
            server_name=server_name,
            success=result.success,
            latency_ms=result.total_latency_ms,
        )

        return result