# ============================================
# API 调用工具
# 通用的 HTTP API 调用工具，支持 GET/POST 请求
# ============================================

import httpx
import json
from typing import Optional
from loguru import logger

try:
    from langchain_core.tools import tool
except ImportError:
    from langchain.tools import tool

# 全局 httpx 客户端
_api_client: Optional[httpx.AsyncClient] = None

# 允许的域名白名单（安全限制）
ALLOWED_DOMAINS = {
    "api.example.com",
    "jsonplaceholder.typicode.com",
    "httpbin.org",
    "api.github.com",
}


def _get_api_client() -> httpx.AsyncClient:
    """获取或创建 httpx 客户端"""
    global _api_client
    if _api_client is None or _api_client.is_closed:
        _api_client = httpx.AsyncClient(timeout=30.0)
    return _api_client


def _check_url_safe(url: str) -> bool:
    """检查 URL 是否在允许的域名白名单内，并屏蔽内网/元数据端点（SSRF 防护）。

    安全策略：
    1. 若 ALLOWED_DOMAINS 为空 → 拒绝所有（fail-safe 默认）
    2. host 必须是白名单中的字面域名
    3. host 若解析为内网/loopback/link-local IP → 拒绝
    4. host 若指向常见云元数据端点 → 拒绝（即使白名单包含字面 "169.254.169.254"）
    """
    # Cloud metadata endpoints — always blocked regardless of whitelist
    BLOCKED_HOSTS = {
        "169.254.169.254",          # AWS / OpenStack / Azure
        "metadata.google.internal", # GCP
        "metadata.azure.com",        # Azure (newer)
        "100.100.100.200",           # Aliyun
        "127.0.0.1", "localhost", "0.0.0.0",  # loopback
    }

    # 1. Fail-safe: empty whitelist → deny all
    if not ALLOWED_DOMAINS:
        logger.warning("[API工具] ALLOWED_DOMAINS 为空，拒绝所有外部请求")
        return False

    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.hostname

        if not domain:
            return False

        # 2. Block metadata / loopback
        if domain.lower() in BLOCKED_HOSTS:
            logger.warning(f"[API工具] SSRF 拦截: 元数据/loopback 端点 {domain}")
            return False

        # 3. Block private IP ranges (avoid DNS rebinding)
        import ipaddress
        try:
            ip = ipaddress.ip_address(domain)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                logger.warning(f"[API工具] SSRF 拦截: 内网 IP {domain}")
                return False
        except ValueError:
            # domain is not an IP literal — it's a hostname, OK
            pass

        # 4. Whitelist match
        if domain in ALLOWED_DOMAINS:
            return True

        logger.warning(f"[API工具] SSRF 拦截: 域名 {domain} 不在白名单中")
        return False
    except Exception as e:
        logger.error(f"[API工具] URL 安全检查异常: {e}")
        return False


@tool
async def api_call_tool(
    url: str,
    method: str = "GET",
    headers: Optional[str] = None,
    body: Optional[str] = None,
    timeout: float = 30.0,
) -> str:
    """
    API 调用工具：发起 HTTP 请求调用外部 API。

    Args:
        url: 目标 API 的 URL 地址
        method: HTTP 方法（GET / POST / PUT / DELETE）
        headers: JSON 格式的请求头字符串，例如 '{"Content-Type": "application/json"}'
        body: JSON 格式的请求体字符串（POST/PUT 时使用）
        timeout: 请求超时时间（秒）

    Returns:
        str: API 响应内容
    """
    logger.info(f"[API工具] {method} 请求: {url}")

    # URL 安全检查
    if not _check_url_safe(url):
        return f"错误：URL '{url}' 不在允许的域名白名单中"

    method = method.upper()
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
        return f"错误：不支持的 HTTP 方法 '{method}'"

    # 解析请求头
    request_headers = {}
    if headers:
        try:
            request_headers = json.loads(headers)
        except json.JSONDecodeError:
            return "错误：headers 格式不正确，请使用 JSON 格式"

    # 解析请求体
    request_body = None
    if body:
        try:
            request_body = json.loads(body)
        except json.JSONDecodeError:
            return "错误：body 格式不正确，请使用 JSON 格式"

    try:
        client = _get_api_client()
        response = await client.request(
            method=method,
            url=url,
            headers=request_headers,
            json=request_body,
            timeout=timeout,
        )
        response.raise_for_status()

        # 尝试解析 JSON 响应
        try:
            data = response.json()
            result = json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            result = response.text

        # 截断过长的响应
        if len(result) > 20000:
            result = result[:20000] + f"\n\n[... 响应过长，已截断。原始长度: {len(result)} 字符 ...]"

        logger.debug(f"[API工具] 响应状态: {response.status_code}, 长度: {len(result)} 字符")
        return f"状态码: {response.status_code}\n\n{result}"

    except httpx.TimeoutException:
        error_msg = f"请求超时 ({timeout}s)"
        logger.error(f"[API工具] {error_msg}")
        return f"错误：{error_msg}"
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP 错误: {e.response.status_code} - {e.response.text[:500]}"
        logger.error(f"[API工具] {error_msg}")
        return f"错误：{error_msg}"
    except Exception as e:
        error_msg = f"请求失败: {type(e).__name__}: {str(e)}"
        logger.error(f"[API工具] {error_msg}")
        return f"错误：{error_msg}"
