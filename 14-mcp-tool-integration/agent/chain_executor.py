"""链式调用执行器

支持:
- 多步骤工具调用链
- 步骤间参数传递
- 错误处理和重试
- 执行历史
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from client.mcp_client import MCPClient
from protocol.types import ToolCallResult
from registry.permission import PermissionLevel
from registry.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ChainStep:
    """链式调用步骤"""
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    # 参数模板: 从前序步骤结果中提取参数
    # 例如 {"value": "$steps.0.result"} 表示从第0步结果中取 value
    argument_templates: dict[str, str] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 2
    timeout: Optional[float] = None


@dataclass
class StepResult:
    """步骤执行结果"""
    step_index: int
    tool_name: str
    success: bool
    result: Optional[ToolCallResult] = None
    error: str = ""
    latency_ms: float = 0.0
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "raw_text": self.raw_text,
        }


@dataclass
class ChainResult:
    """链式执行结果"""
    success: bool
    steps: list[StepResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "steps": [s.to_dict() for s in self.steps],
            "total_latency_ms": self.total_latency_ms,
            "error": self.error,
        }


class ChainExecutor:
    """链式调用执行器"""

    def __init__(
        self,
        client: MCPClient,
        registry: ToolRegistry,
    ):
        self.client = client
        self.registry = registry
        self._history: list[ChainResult] = []

    def _resolve_template_value(self, template: str, step_results: list[StepResult]) -> Any:
        """解析参数模板

        模板格式:
        - "$steps.{index}.text" - 从步骤结果中取文本内容
        - "$steps.{index}.result" - 从步骤结果中取原始结果
        - 直接值则原样返回
        """
        if not isinstance(template, str):
            return template

        if template.startswith("$steps."):
            try:
                # $steps.0.text
                parts = template.split(".")
                idx = int(parts[1])
                field_name = parts[2] if len(parts) > 2 else "text"

                if idx < len(step_results) and step_results[idx].success:
                    sr = step_results[idx]
                    if field_name == "text" and sr.raw_text:
                        return sr.raw_text
                    if sr.result and sr.result.content:
                        return sr.result.content[0].get("text", "")
            except (IndexError, ValueError, KeyError):
                pass

        # 尝试解析为数字
        try:
            return int(template)
        except ValueError:
            pass
        try:
            return float(template)
        except ValueError:
            pass

        return template

    def _build_arguments(
        self,
        step: ChainStep,
        step_results: list[StepResult],
    ) -> dict[str, Any]:
        """构建最终参数（合并固定参数和模板参数）"""
        args = dict(step.arguments)

        for param_name, template in step.argument_templates.items():
            args[param_name] = self._resolve_template_value(template, step_results)

        return args

    async def execute_chain(self, steps: list[ChainStep], role: str = "default") -> ChainResult:
        """执行工具调用链

        Args:
            steps: 步骤列表
            role: 用户角色

        Returns:
            链式执行结果
        """
        result = ChainResult(success=True)
        step_results: list[StepResult] = []
        start_time = time.time()

        for i, step in enumerate(steps):
            # 权限检查
            perm = self.registry.permissions.check_permission(step.tool_name, role)
            if perm == PermissionLevel.DENY:
                sr = StepResult(
                    step_index=i,
                    tool_name=step.tool_name,
                    success=False,
                    error=f"Permission denied for tool: {step.tool_name}",
                )
                step_results.append(sr)
                result.steps.append(sr)
                result.success = False
                result.error = f"Permission denied at step {i}"
                break

            # 查找工具所在服务器
            server_name = self.registry.get_tool_server(step.tool_name)
            if server_name is None:
                sr = StepResult(
                    step_index=i,
                    tool_name=step.tool_name,
                    success=False,
                    error=f"Tool not found: {step.tool_name}",
                )
                step_results.append(sr)
                result.steps.append(sr)
                result.success = False
                result.error = f"Tool not found at step {i}: {step.tool_name}"
                break

            # 构建参数
            args = self._build_arguments(step, step_results)

            # 执行（带重试）
            call_start = time.time()
            last_error = ""
            call_result = None

            for attempt in range(step.max_retries + 1):
                try:
                    call_result = await self.client.call_tool(
                        server_name=server_name,
                        tool_name=step.tool_name,
                        arguments=args,
                        timeout=step.timeout,
                    )

                    latency = (time.time() - call_start) * 1000

                    if call_result.is_error:
                        last_error = call_result.content[0].get("text", "Unknown error") if call_result.content else "Unknown error"
                        if attempt < step.max_retries:
                            continue
                        sr = StepResult(
                            step_index=i,
                            tool_name=step.tool_name,
                            success=False,
                            result=call_result,
                            error=last_error,
                            latency_ms=latency,
                            raw_text=call_result.content[0].get("text", "") if call_result.content else "",
                        )
                    else:
                        raw_text = call_result.content[0].get("text", "") if call_result.content else ""
                        sr = StepResult(
                            step_index=i,
                            tool_name=step.tool_name,
                            success=True,
                            result=call_result,
                            latency_ms=latency,
                            raw_text=raw_text,
                        )
                    break
                except Exception as e:
                    last_error = str(e)
                    if attempt < step.max_retries:
                        continue
                    latency = (time.time() - call_start) * 1000
                    sr = StepResult(
                        step_index=i,
                        tool_name=step.tool_name,
                        success=False,
                        error=last_error,
                        latency_ms=latency,
                    )
            else:
                # 所有重试都失败
                latency = (time.time() - call_start) * 1000
                sr = StepResult(
                    step_index=i,
                    tool_name=step.tool_name,
                    success=False,
                    error=last_error,
                    latency_ms=latency,
                )

            step_results.append(sr)
            result.steps.append(sr)

            # 记录使用统计
            self.registry.record_usage(
                tool_name=step.tool_name,
                server_name=server_name,
                success=sr.success,
                latency_ms=sr.latency_ms,
            )

            if not sr.success:
                result.success = False
                result.error = f"Step {i} failed: {sr.error}"
                break

        result.total_latency_ms = (time.time() - start_time) * 1000
        self._history.append(result)
        return result

    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """获取执行历史"""
        return [r.to_dict() for r in self._history[-limit:]]