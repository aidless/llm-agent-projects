# ============================================
# 计算器工具
# 使用 sympy 进行安全数学表达式计算
# ============================================

from sympy import sympify, SympifyError
from loguru import logger

try:
    from langchain_core.tools import tool
except ImportError:
    from langchain.tools import tool


@tool
async def calculator_tool(expression: str) -> str:
    """
    计算器工具：安全地计算数学表达式。

    支持基本算术运算、幂运算、对数、三角函数等。
    使用 sympy 库进行安全的表达式解析和计算。

    Args:
        expression: 数学表达式字符串，例如 "2 ** 10", "sqrt(144)", "sin(pi/4)"

    Returns:
        str: 计算结果
    """
    logger.info(f"[计算器] 计算表达式: {expression}")

    # 清理表达式中的危险字符
    cleaned = expression.strip()

    # 检查是否包含危险操作
    dangerous_keywords = ["import", "exec", "eval", "open", "file", "os", "sys"]
    for kw in dangerous_keywords:
        if kw in cleaned.lower():
            return f"错误：表达式中包含禁止的关键词 '{kw}'"

    try:
        # 使用 sympy 安全解析表达式
        result = sympify(cleaned)
        # 尝试简化并转为浮点数
        try:
            float_result = float(result.evalf())
            # 如果是整数则显示为整数
            if float_result == int(float_result):
                formatted = str(int(float_result))
            else:
                formatted = f"{float_result:.10g}"
        except (TypeError, ValueError):
            # 无法转为浮点数，保留符号形式
            formatted = str(result)

        logger.debug(f"[计算器] 结果: {formatted}")
        return f"计算结果: {formatted}"

    except SympifyError as e:
        error_msg = f"无法解析表达式: {str(e)}"
        logger.warning(f"[计算器] {error_msg}")
        return f"错误：{error_msg}"
    except Exception as e:
        error_msg = f"计算异常: {type(e).__name__}: {str(e)}"
        logger.error(f"[计算器] {error_msg}")
        return f"错误：{error_msg}"
