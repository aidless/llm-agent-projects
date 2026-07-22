"""GitHub API 客户端。"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GitHubClient:
    """GitHub API 客户端封装。

    提供获取 PR diff、文件列表、状态等功能的接口。
    实际使用时需要提供 GitHub Token。
    测试时通过 mock 替换。
    """

    def __init__(self, token: Optional[str] = None) -> None:
        """初始化 GitHub 客户端。

        Args:
            token: GitHub Personal Access Token。为 None 时使用 mock 模式。
        """
        self._token = token
        self._mock_mode = token is None
        if self._mock_mode:
            logger.info("GitHub 客户端使用 mock 模式")

    @property
    def is_mock(self) -> bool:
        """是否为 mock 模式。"""
        return self._mock_mode

    async def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """获取 PR 的 diff 内容。

        Args:
            owner: 仓库所有者。
            repo: 仓库名称。
            pr_number: PR 编号。

        Returns:
            diff 内容字符串。
        """
        if self._mock_mode:
            return self._get_mock_diff(owner, repo, pr_number)

        # 实际调用 GitHub API
        import httpx

        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        headers = {
            "Authorization": f"token {self._token}",
            "Accept": "application/vnd.github.v3.diff",
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.text

    async def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
        """获取 PR 变更的文件列表。

        Args:
            owner: 仓库所有者。
            repo: 仓库名称。
            pr_number: PR 编号。

        Returns:
            文件信息列表。
        """
        if self._mock_mode:
            return self._get_mock_files(owner, repo, pr_number)

        import httpx

        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
        headers = {
            "Authorization": f"token {self._token}",
            "Accept": "application/vnd.github.v3+json",
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    async def get_pr_info(self, owner: str, repo: str, pr_number: int) -> dict[str, Any]:
        """获取 PR 基本信息。

        Args:
            owner: 仓库所有者。
            repo: 仓库名称。
            pr_number: PR 编号。

        Returns:
            PR 信息字典。
        """
        if self._mock_mode:
            return {
                "number": pr_number,
                "title": f"Mock PR #{pr_number}",
                "state": "open",
                "head": {"ref": "feature-branch"},
                "base": {"ref": "main"},
            }

        import httpx

        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        headers = {
            "Authorization": f"token {self._token}",
            "Accept": "application/vnd.github.v3+json",
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    async def create_review_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
        path: str,
        line: int,
        commit_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """在 PR 中创建 review comment。

        Args:
            owner: 仓库所有者。
            repo: 仓库名称。
            pr_number: PR 编号。
            body: 评论内容。
            path: 文件路径。
            line: 行号。
            commit_id: commit SHA（可选）。

        Returns:
            API 响应。
        """
        if self._mock_mode:
            return {"id": 1, "body": body, "path": path, "line": line}

        import httpx

        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
        headers = {
            "Authorization": f"token {self._token}",
            "Accept": "application/vnd.github.v3+json",
        }
        data: dict[str, Any] = {
            "body": body,
            "path": path,
            "line": line,
        }
        if commit_id:
            data["commit_id"] = commit_id

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _get_mock_diff(owner: str, repo: str, pr_number: int) -> str:
        """生成 mock diff 内容。"""
        return f"""diff --git a/example.py b/example.py
new file mode 100644
index 0000000..e69de44
--- /dev/null
+++ b/example.py
@@ -0,0 +1,15 @@
+import os
+password = "hardcoded_secret_123"
+
+def GetData(x):
+    result = []
+    for i in range(x):
+        result.append(i * 2)
+    return result
+
+def process(data):
+    try:
+        eval(data)
+    except:
+        pass
+
+if data == None:
+    print("done")
"""

    @staticmethod
    def _get_mock_files(owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
        """生成 mock 文件列表。"""
        return [
            {
                "filename": "example.py",
                "status": "added",
                "additions": 15,
                "deletions": 0,
                "patch": "@@ -0,0 +1,15 @@",
            }
        ]
