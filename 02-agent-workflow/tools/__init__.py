# ============================================
# 自定义工具集 - Agent 可调用的工具函数
# 包含：网页搜索、文件读写、代码执行、计算器、API 调用
# ============================================
from tools.search_tool import web_search_tool
from tools.file_tool import file_read_tool, file_write_tool
from tools.code_executor import code_execute_tool
from tools.calculator import calculator_tool
from tools.api_caller import api_call_tool

# 汇总所有工具的 LangChain Tool 列表
ALL_TOOLS = [
    web_search_tool,
    file_read_tool,
    file_write_tool,
    code_execute_tool,
    calculator_tool,
    api_call_tool,
]

__all__ = [
    "ALL_TOOLS",
    "web_search_tool",
    "file_read_tool",
    "file_write_tool",
    "code_execute_tool",
    "calculator_tool",
    "api_call_tool",
]
