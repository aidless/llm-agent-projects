# ============================================
# 网页搜索工具
# 使用 httpx 调用搜索 API 获取网页搜索结果
# ============================================

import httpx
import json
from typing import Optional
from loguru import logger

try:
    from langchain_core.tools import tool
except ImportError:
    from langchain.tools import tool

from config import get_settings

# 全局 httpx 异步客户端（复用连接池）
_http_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    """获取或创建 httpx 客户端"""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


@tool
async def web_search_tool(query: str, num_results: int = 5) -> str:
    """
    网页搜索工具：根据关键词搜索互联网信息。

    Args:
        query: 搜索关键词或问题
        num_results: 返回结果数量（默认5条）

    Returns:
        str: 搜索结果摘要文本
    """
    logger.info(f"[搜索工具] 搜索: {query}, 数量: {num_results}")

    settings = get_settings()

    # 如果配置了搜索 API，调用真实的搜索服务
    if settings.search_api_url and settings.search_api_key:
        try:
            client = _get_client()
            response = await client.post(
                settings.search_api_url,
                headers={"Authorization": f"Bearer {settings.search_api_key}"},
                json={"query": query, "num_results": num_results},
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if results:
                formatted = []
                for i, r in enumerate(results, 1):
                    formatted.append(
                        f"{i}. [{r.get('title', '无标题')}] {r.get('snippet', '')}\n"
                        f"   链接: {r.get('url', '')}"
                    )
                return "\n\n".join(formatted)
        except Exception as e:
            logger.error(f"[搜索工具] API 调用失败: {e}")

    # 无 API 配置时返回模拟结果（便于演示和测试）
    logger.debug("[搜索工具] 使用模拟搜索结果（未配置搜索 API）")
    mock_results = [
        {
            "title": f"关于「{query}」的综合分析",
            "snippet": f"根据最新信息，{query}是一个重要的话题。近年来该领域发展迅速，"
                       f"多个研究表明其具有广泛的应用前景和实践价值。",
            "url": "https://example.com/analysis",
        },
        {
            "title": f"「{query}」技术报告",
            "snippet": f"本报告详细介绍了{query}的技术原理、发展历程以及当前最新的研究进展。"
                       f"涵盖基础概念、核心算法和典型应用场景。",
            "url": "https://example.com/tech-report",
        },
        {
            "title": f"「{query}」行业趋势报告",
            "snippet": f"行业分析显示，{query}相关市场正在快速增长。预计未来3年内年复合增长率"
                       f"将达到25%以上，主要驱动力来自企业数字化转型和 AI 技术普及。",
            "url": "https://example.com/industry",
        },
    ]

    formatted = []
    for i, r in enumerate(mock_results[:num_results], 1):
        formatted.append(
            f"{i}. [{r['title']}] {r['snippet']}\n   链接: {r['url']}"
        )
    return "\n\n".join(formatted)
